from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from pql.algo.ac_base import ActorCriticBase
from pql.replay.nstep_replay import NStepReplay
from pql.utils.noise import add_mixed_normal_noise
from pql.utils.noise import add_normal_noise
from pql.utils.schedule_util import ExponentialSchedule
from pql.utils.schedule_util import LinearSchedule
from pql.utils.torch_util import soft_update
from pql.utils.common import handle_timeout

@dataclass
class AgentDDPG(ActorCriticBase):
    def __post_init__(self):
        super().__post_init__()
        self.critic_target = deepcopy(self.critic)
        self.actor_target = deepcopy(self.actor) if not self.cfg.algo.no_tgt_actor else self.actor

        if self.cfg.algo.noise.decay == 'linear':
            self.noise_scheduler = LinearSchedule(start_val=self.cfg.algo.noise.std_max,
                                                  end_val=self.cfg.algo.noise.std_min,
                                                  total_iters=self.cfg.algo.noise.lin_decay_iters
                                                  )
        elif self.cfg.algo.noise.decay == 'exp':
            self.noise_scheduler = ExponentialSchedule(start_val=self.cfg.algo.noise.std_max,
                                                       gamma=self.cfg.algo.exp_decay_rate,
                                                       end_val=self.cfg.algo.noise.std_min)
        else:
            self.noise_scheduler = None

        self.n_step_buffer = NStepReplay(self.obs_dim,
                                         self.action_dim,
                                         self.cfg.num_envs,
                                         self.cfg.algo.nstep,
                                         device=self.device)

    def get_noise_std(self):
        if self.noise_scheduler is None:
            return self.cfg.algo.noise.std_max
        else:
            return self.noise_scheduler.val()

    def update_noise(self):
        if self.noise_scheduler is not None:
            self.noise_scheduler.step()

    def get_actions(self, obs, sample=True):
        if self.cfg.algo.obs_norm:
            obs = self.obs_rms.normalize(obs)
        actions = self.actor(obs)
        if sample:
            if self.cfg.algo.noise.type == 'fixed':
                actions = add_normal_noise(actions,
                                           std=self.get_noise_std(),
                                           out_bounds=[-1., 1.])
            elif self.cfg.algo.noise.type == 'mixed':
                actions = add_mixed_normal_noise(actions,
                                                 std_min=self.cfg.algo.noise.std_min,
                                                 std_max=self.cfg.algo.noise.std_max,
                                                 out_bounds=[-1., 1.])
            else:
                raise NotImplementedError
        return actions

    @torch.no_grad()
    def get_tgt_policy_actions(self, obs, sample=True):
        actions = self.actor_target(obs)
        if sample:
            actions = add_normal_noise(actions,
                                       std=self.cfg.algo.noise.tgt_pol_std,
                                       noise_bounds=[-self.cfg.algo.noise.tgt_pol_noise_bound,
                                                     self.cfg.algo.noise.tgt_pol_noise_bound],
                                       out_bounds=[-1., 1.])
        return actions

    @torch.no_grad()
    def explore_env(self, env, timesteps: int, random: bool = False) -> list:
        obs_dim = (self.obs_dim,) if isinstance(self.obs_dim, int) else self.obs_dim
        traj_states = torch.empty((self.cfg.num_envs, timesteps) + (*obs_dim,), device=self.device)
        traj_actions = torch.empty((self.cfg.num_envs, timesteps) + (self.action_dim,), device=self.device)
        traj_rewards = torch.empty((self.cfg.num_envs, timesteps), device=self.device)
        traj_next_states = torch.empty((self.cfg.num_envs, timesteps) + (*obs_dim,), device=self.device)
        traj_dones = torch.empty((self.cfg.num_envs, timesteps), device=self.device)

        obs = self.obs
        for i in range(timesteps):
            if self.cfg.algo.obs_norm:
                self.obs_rms.update(obs)
            if random:
                action = torch.rand((self.cfg.num_envs, self.action_dim),
                                    device=self.cfg.device) * 2.0 - 1.0
            else:
                action = self.get_actions(obs, sample=True)

            next_obs, reward, done, info = env.step(action)
            self.update_tracker(reward, done, info)
            if self.cfg.algo.handle_timeout:
                done = handle_timeout(done, info)

            traj_states[:, i] = obs
            traj_actions[:, i] = action
            traj_dones[:, i] = done
            traj_rewards[:, i] = reward
            traj_next_states[:, i] = next_obs
            obs = next_obs
        self.obs = obs

        traj_rewards = self.cfg.algo.reward_scale * traj_rewards.reshape(self.cfg.num_envs, timesteps, 1)
        traj_dones = traj_dones.reshape(self.cfg.num_envs, timesteps, 1)
        data = self.n_step_buffer.add_to_buffer(traj_states, traj_actions, traj_rewards, traj_next_states, traj_dones)

        return data, timesteps * self.cfg.num_envs

    def update_net(self, memory):
        critic_loss_list = list()
        actor_loss_list = list()
        for i in range(self.cfg.algo.update_times):
            obs, action, reward, next_obs, done = memory.sample_batch(self.cfg.algo.batch_size)
            if self.cfg.algo.obs_norm:
                obs = self.obs_rms.normalize(obs)
                next_obs = self.obs_rms.normalize(next_obs)
            critic_loss, critic_grad_norm = self.update_critic(obs, action, reward, next_obs, done)
            critic_loss_list.append(critic_loss)

            actor_loss, actor_grad_norm = self.update_actor(obs)
            actor_loss_list.append(actor_loss)

            soft_update(self.critic_target, self.critic, self.cfg.algo.tau)
            if not self.cfg.algo.no_tgt_actor:
                soft_update(self.actor_target, self.actor, self.cfg.algo.tau)

        log_info = {
            "train/critic_loss": np.mean(critic_loss_list),
            "train/actor_loss": np.mean(actor_loss_list),
            "train/return": self.return_tracker.mean(),
            "train/episode_length": self.step_tracker.mean()
        }
        self.add_info_tracker_log(log_info)
        return log_info

    def update_critic(self, obs, action, reward, next_obs, done):
        with torch.no_grad():
            next_actions = self.get_tgt_policy_actions(next_obs)
            target_Q = self.critic_target.get_q_min(next_obs, next_actions)
            target_Q = reward + (1 - done) * (self.cfg.algo.gamma ** self.cfg.algo.nstep) * target_Q

        current_Q1, current_Q2 = self.critic.get_q1_q2(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        grad_norm = self.optimizer_update(self.critic_optimizer, critic_loss)

        return critic_loss.item(), grad_norm

    def update_actor(self, obs):
        self.critic.requires_grad_(False)
        action = self.actor(obs)
        Q = self.critic.get_q_min(obs, action)
        actor_loss = -Q.mean()
        grad_norm = self.optimizer_update(self.actor_optimizer, actor_loss)
        self.critic.requires_grad_(True)
        return actor_loss.item(), grad_norm
