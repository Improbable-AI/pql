import torch

from pql.replay.nstep_replay import NStepReplay
from pql.utils.common import Tracker, handle_timeout
from pql.utils.model_util import load_model
from pql.utils.noise import add_mixed_normal_noise, add_normal_noise
from pql.utils.schedule_util import ExponentialSchedule, LinearSchedule
from pql.utils.torch_util import RunningMeanStd


class PQLActor:
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self.obs_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.shape[0]
        self.sim_device = torch.device(f"{cfg.sim_device}")
        self.v_learner_device = torch.device(f"cuda:{cfg.algo.v_learner_gpu}")
        self.p_learner_device = torch.device(f"cuda:{cfg.algo.p_learner_gpu}")
        self.actor = None
        self.obs = None

        self.return_tracker = Tracker(self.cfg.algo.tracker_len)
        self.step_tracker = Tracker(self.cfg.algo.tracker_len)
        self.current_returns = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=self.sim_device)
        self.current_lengths = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=self.sim_device)

        info_track_keys = self.cfg.info_track_keys
        if info_track_keys is not None:
            info_track_keys = [info_track_keys] if isinstance(info_track_keys, str) else info_track_keys
            self.info_trackers = {key: Tracker(self.cfg.algo.tracker_len) for key in info_track_keys}
            self.info_track_step = {key: self.cfg.info_track_step[idx] for idx, key in enumerate(info_track_keys)}
            self.traj_info_values = {key: torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=self.sim_device) for key in info_track_keys}

        if self.cfg.algo.obs_norm:
            self.obs_rms = RunningMeanStd(shape=self.obs_dim, device=self.sim_device)
            if self.cfg.artifact is not None:
                load_model(self.obs_rms, "obs_rms", cfg)
        else:
            self.obs_rms = None
        self.n_step_buffer = NStepReplay(self.obs_dim, self.action_dim, self.cfg.num_envs, self.cfg.algo.nstep,
                                         device=self.sim_device)

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

    def reset_agent(self):
        self.obs = self.env.reset()

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
    def explore_env(self, env, timesteps: int, random: bool) -> list:
        obs_dim = (self.obs_dim,) if isinstance(self.obs_dim, int) else self.obs_dim
        traj_states = torch.empty((self.cfg.num_envs, timesteps) + (*obs_dim,), device=self.sim_device)
        traj_actions = torch.empty((self.cfg.num_envs, timesteps) + (self.action_dim,), device=self.sim_device)
        traj_rewards = torch.empty((self.cfg.num_envs, timesteps), device=self.sim_device)
        traj_next_states = torch.empty((self.cfg.num_envs, timesteps) + (*obs_dim,), device=self.sim_device)
        traj_dones = torch.empty((self.cfg.num_envs, timesteps), device=self.sim_device)

        obs = self.obs
        for i in range(timesteps):
            if self.cfg.algo.obs_norm:
                self.obs_rms.update(obs)
            if random:
                action = torch.rand((self.cfg.num_envs, self.action_dim),
                                    device=self.sim_device) * 2.0 - 1.0
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
        obs, action, reward, next_obs, done = self.n_step_buffer.add_to_buffer(traj_states, traj_actions, traj_rewards, traj_next_states, traj_dones)
        act_data = obs.clone().to(self.p_learner_device)
        cri_data = (
            obs.clone().to(self.v_learner_device), action.to(self.v_learner_device), reward.to(self.v_learner_device),
            next_obs.to(self.v_learner_device),
            done.to(self.v_learner_device))
        return act_data, cri_data, timesteps * self.cfg.num_envs

    def update_tracker(self, reward, done, info):
        self.current_returns += reward
        self.current_lengths += 1
        env_done_indices = torch.where(done)[0]
        self.return_tracker.update(self.current_returns[env_done_indices])
        self.step_tracker.update(self.current_lengths[env_done_indices])
        self.current_returns[env_done_indices] = 0
        self.current_lengths[env_done_indices] = 0

        if self.cfg.info_track_keys is not None:
            for key in self.cfg.info_track_keys:
                if self.info_track_step[key] == 'last':
                    info_val = info[key]
                    self.info_trackers[key].update(info_val[env_done_indices])
                elif self.info_track_step[key] == 'all':
                    self.traj_info_values[key] += info[key]
                    self.info_trackers[key].update(self.traj_info_values[key][env_done_indices])
                    self.traj_info_values[key][env_done_indices] = 0
        return done
    def add_info_tracker_log(self, log_info):
        if self.cfg.info_track_keys is not None:
            for key in self.cfg.info_track_keys:
                log_info[key] = self.info_trackers[key].mean()
