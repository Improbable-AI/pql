import os
import time
from copy import deepcopy

import ray
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from loguru import logger
from pql.models import model_name_to_path
from pql.replay.simple_replay import ReplayBuffer
from pql.utils.common import Tracker
from pql.utils.common import load_class_from_path
from pql.utils.common import normalize
from pql.utils.noise import add_normal_noise
from pql.utils.torch_util import soft_update
from pql.utils.distl_util import projection
from pql.utils.model_util import load_model


@ray.remote(num_gpus=0.7)
class PQLVLearner:
    def __init__(self, obs_dim, action_dim, cfg):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(cfg.available_gpus)])
        self.device = torch.device(f"cuda:{self.cfg.algo.v_learner_gpu}")

        if self.cfg.algo.distl and "Distributional" not in self.cfg.algo.cri_class:
            self.cfg.algo.cri_class = "Distributional" + self.cfg.algo.cri_class
        cri_class = load_class_from_path(self.cfg.algo.cri_class,
                                         model_name_to_path[self.cfg.algo.cri_class])
        if self.cfg.algo.distl:
            self.critic = cri_class(self.obs_dim, self.action_dim, 
                                    v_min=self.cfg.algo.v_min, 
                                    v_max=self.cfg.algo.v_max,
                                    num_atoms=self.cfg.algo.num_atoms,
                                    device=self.device).to(self.device)
            self.loss_fnc = F.binary_cross_entropy
        else:
            self.critic = cri_class(self.obs_dim, self.action_dim).to(self.device)
            self.loss_fnc = F.mse_loss
        if self.cfg.artifact is not None:
            load_model(self.critic, "critic", cfg)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), self.cfg.algo.critic_lr)
        self.critic_target = deepcopy(self.critic)
        self.actor = None

        self.memory = ReplayBuffer(capacity=int(cfg.algo.memory_size),
                                   obs_dim=self.obs_dim,
                                   action_dim=self.action_dim,
                                   device=self.device)
        self.loss_tracker = Tracker(5)
        self.update_count = 0
        self.normalize_tuple = None
        self.sleep_time = 0

    def start(self):
        return self.critic, self.update_count, self.loss_tracker.mean()

    @torch.no_grad()
    def get_tgt_policy_actions(self, obs, sample=True):
        actions = self.actor(obs)
        if sample:
            actions = add_normal_noise(actions,
                                       std=self.cfg.algo.noise.tgt_pol_std,
                                       noise_bounds=[-self.cfg.algo.noise.tgt_pol_noise_bound,
                                                     self.cfg.algo.noise.tgt_pol_noise_bound],
                                       out_bounds=[-1., 1.])
        return actions

    def learn(self):
        if self.actor is not None:
            obs, action, reward, next_obs, done = self.memory.sample_batch(self.cfg.algo.batch_size, device=self.device)
            if self.cfg.algo.obs_norm:
                obs = normalize(obs, self.normalize_tuple)
                next_obs = normalize(next_obs, self.normalize_tuple)

            with torch.no_grad():
                next_actions = self.get_tgt_policy_actions(next_obs)
                if self.cfg.algo.distl:
                    target_Q1, target_Q2 = self.critic_target.get_q1_q2(next_obs, next_actions)
                    target_Q1_projected = projection(next_dist=target_Q1,
                                                     reward=reward,
                                                     done=done,
                                                     gamma=self.cfg.algo.gamma ** self.cfg.algo.nstep,
                                                     v_min=self.cfg.algo.v_min,
                                                     v_max=self.cfg.algo.v_max,
                                                     num_atoms=self.cfg.algo.num_atoms,
                                                     support=self.critic.z_atoms,
                                                     device=self.device)
                    target_Q2_projected = projection(next_dist=target_Q2,
                                                     reward=reward,
                                                     done=done,
                                                     gamma=self.cfg.algo.gamma ** self.cfg.algo.nstep,
                                                     v_min=self.cfg.algo.v_min,
                                                     v_max=self.cfg.algo.v_max,
                                                     num_atoms=self.cfg.algo.num_atoms,
                                                     support=self.critic.z_atoms,
                                                     device=self.device)
                    target_Q = torch.min(target_Q1_projected, target_Q2_projected)
                else:
                    target_Q = self.critic_target.get_q_min(next_obs, next_actions)
                    target_Q = reward + (1 - done) * (self.cfg.algo.gamma ** self.cfg.algo.nstep) * target_Q

            current_Q1, current_Q2 = self.critic.get_q1_q2(obs, action)
            critic_loss = self.loss_fnc(current_Q1, target_Q) + self.loss_fnc(current_Q2, target_Q)
            self.optimizer_update(self.critic_optimizer, critic_loss)

            self.loss_tracker.update(critic_loss.detach().item())
            soft_update(self.critic_target, self.critic, self.cfg.algo.tau)
            self.update_count += 1

        return self.sleep_time

    def update(self, actor, trajectory, normalize_tuple, sleep_time):
        self.actor = actor
        self.memory.add_to_buffer(trajectory)
        self.normalize_tuple = normalize_tuple
        self.sleep_time = sleep_time
        return self.critic, self.loss_tracker.mean(), self.update_count

    def optimizer_update(self, optimizer, objective):
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        if self.cfg.algo.max_grad_norm is not None:
            grad_norm = clip_grad_norm_(parameters=optimizer.param_groups[0]["params"],
                                        max_norm=self.cfg.algo.max_grad_norm)
        else:
            grad_norm = None
        optimizer.step()
        return grad_norm


@ray.remote
def asyn_v_learner(critic, cfg):
    logger.warning(f"V-Learner starts running asynchronously on GPU {cfg.algo.v_learner_gpu}")
    while True:
        sleep_time = ray.get(critic.learn.remote())
        time.sleep(sleep_time)
