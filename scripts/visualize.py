

import isaacgym
import torch
import wandb
import isaacgym
import hydra
from omegaconf import DictConfig
from loguru import logger

import pql
from pql.utils.common import set_random_seed
from pql.utils.isaacgym_util import create_task_env 
from pql.utils.common import capture_keyboard_interrupt
from pql.utils.common import load_class_from_path
from pql.models import model_name_to_path
from pql.utils.common import Tracker
from pql.utils.torch_util import RunningMeanStd 
from pql.utils.model_util import load_model


@hydra.main(config_path=pql.LIB_PATH.joinpath('cfg').as_posix(), config_name="pql")
def main(cfg: DictConfig):
    set_random_seed(cfg.seed)
    capture_keyboard_interrupt()
    env = create_task_env(cfg)
    device = torch.device(cfg.device)
    obs_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    assert cfg.artifact is not None
    act_class = load_class_from_path(cfg.algo.act_class,
                                            model_name_to_path[cfg.algo.act_class])
    actor = act_class(obs_dim, action_dim).to(device)
    load_model(actor, "actor", cfg)
    obs_rms = RunningMeanStd(shape=obs_dim, device=device)
    load_model(obs_rms, "obs_rms", cfg)

    return_tracker = Tracker(cfg.num_envs)
    step_tracker = Tracker(cfg.num_envs)
    current_rewards = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)
    current_lengths = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)

    obs = env.reset()
    for i_step in range(env.max_episode_length):  # run an episode
        with torch.no_grad():
            action = actor(obs_rms.normalize(obs))
        next_obs, reward, done, _ = env.step(action)
        current_rewards += reward
        current_lengths += 1
        env_done_indices = torch.where(done)[0]
        return_tracker.update(current_rewards[env_done_indices])
        step_tracker.update(current_lengths[env_done_indices])
        current_rewards[env_done_indices] = 0
        current_lengths[env_done_indices] = 0
        obs = next_obs

    r_exp = return_tracker.mean()
    step_exp = step_tracker.mean()
    logger.warning(f"Cumulative return: {r_exp}, Episode length: {step_exp}")


if __name__ == '__main__':
    main()