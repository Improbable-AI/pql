import multiprocessing as mp
import sys
import time
from copy import deepcopy
from loguru import logger
import cloudpickle
import torch

from pql.utils.common import Tracker

from pql.utils.model_util import save_model


class Evaluator:
    def __init__(self, cfg, wandb_run, rollout_callback=None, create_task_env_func=None):
        cfg = deepcopy(cfg)
        self.cfg = cfg
        self.parent, self.child = mp.Pipe()
        if rollout_callback is None:
            rollout_callback = default_rollout
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        p = mp.Process(target=rollout_callback, args=(cfg, wandb_run, self.child, create_task_env_func), daemon=True)
        p.start()

        logger.warning("Created evaluaton process!")
        self.start_time = time.time()

    def eval_policy(self, policy, value, step=0, normalizer=None):
        self.parent.send([cloudpickle.dumps(policy), cloudpickle.dumps(value), step, normalizer])

    def check_if_should_stop(self, step=None):
        if self.cfg.max_step is not None:
            return step > self.cfg.max_step
        else:
            return (time.time() - self.start_time) > self.cfg.max_time


def default_rollout(cfg, wandb_run, child, create_task_env_func=None):
    if sys.version_info.minor >= 8:
        import pickle
    else:
        import pickle5 as pickle

    cfg.headless = cfg.eval_headless
    if create_task_env_func is None:
        from pql.utils.isaacgym_util import create_task_env as create_task_env_func
    env = create_task_env_func(cfg, num_envs=cfg.eval_num_envs)
    num_envs = cfg.eval_num_envs
    max_step = env.max_episode_length
    ret_max = float('-inf')

    tracker_capacity = num_envs
    info_track_keys = cfg.info_track_keys
    if info_track_keys is not None:
        info_track_keys = [info_track_keys] if isinstance(info_track_keys, str) else info_track_keys
        info_trackers = {key: Tracker(tracker_capacity) for key in info_track_keys}
        info_track_step = {key: cfg.info_track_step[idx] for idx, key in enumerate(info_track_keys)}
        traj_info_values = {key: torch.zeros(cfg.num_envs, dtype=torch.float32, device='cpu') for key in info_track_keys}

    with torch.inference_mode():
        while True:
            [actor, critic, step, normalizer] = child.recv()
            actor = pickle.loads(actor)
            critic = pickle.loads(critic)
            if actor is None:
                break
            return_tracker = Tracker(tracker_capacity)
            step_tracker = Tracker(tracker_capacity)
            current_returns = torch.zeros(num_envs, dtype=torch.float32, device=cfg.device)
            current_lengths = torch.zeros(num_envs, dtype=torch.float32, device=cfg.device)
            obs = env.reset()
            for i_step in range(max_step):  # run an episode
                if cfg.algo.obs_norm:
                    action = actor(normalizer.normalize(obs))
                else:
                    action = actor(obs)
                next_obs, reward, done, info = env.step(action)
                current_returns += reward
                current_lengths += 1
                env_done_indices = torch.where(done)[0]

                return_tracker.update(current_returns[env_done_indices])
                step_tracker.update(current_lengths[env_done_indices])
                current_returns[env_done_indices] = 0
                current_lengths[env_done_indices] = 0
                if cfg.info_track_keys is not None:
                    env_done_indices = env_done_indices.cpu()
                    for key in cfg.info_track_keys:
                        if key not in info:
                            continue
                        if info_track_step[key] == 'last':
                            info_val = info[key]
                            info_trackers[key].update(info_val[env_done_indices].cpu())
                        elif info_track_step[key] == 'all-episode':
                            traj_info_values[key] += info[key].cpu()
                            info_trackers[key].update(traj_info_values[key][env_done_indices])
                            traj_info_values[key][env_done_indices] = 0
                        elif info_track_step[key] == 'all-step':
                            info_trackers[key].update(info[key].cpu())

                obs = next_obs

            ret_mean = return_tracker.mean()
            step_mean = step_tracker.mean()
            return_dict = {'eval/return': ret_mean, 'eval/episode_length': step_mean}
            if cfg.info_track_keys is not None:
                for key in cfg.info_track_keys:
                    return_dict[f'eval/{key}'] = info_trackers[key].mean()
            if ret_mean > ret_max:
                ret_max = ret_mean
                save_model(path=f"{wandb_run.dir}/model.pth",
                            actor=actor.state_dict(),
                            critic=critic.state_dict(),
                            rms=normalizer.get_states() if cfg.algo.obs_norm else None,
                            wandb_run=wandb_run,
                            ret_max=ret_max)
            child.send(return_dict)
    child.close()
