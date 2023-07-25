from itertools import count
import isaacgym
import hydra
import wandb
from omegaconf import DictConfig

import pql
from pql.algo import alg_name_to_path
from pql.replay.simple_replay import ReplayBuffer
from pql.utils.common import init_wandb
from pql.utils.common import load_class_from_path
from pql.utils.common import set_random_seed
from pql.utils.evaluator import Evaluator
from pql.utils.isaacgym_util import create_task_env
from pql.utils.common import preprocess_cfg
from pql.utils.common import capture_keyboard_interrupt
from pql.utils.model_util import load_model

@hydra.main(config_path=pql.LIB_PATH.joinpath('cfg').as_posix(), config_name="default")
def main(cfg: DictConfig):
    set_random_seed(cfg.seed)
    capture_keyboard_interrupt()
    preprocess_cfg(cfg)
    wandb_run = init_wandb(cfg)
    env = create_task_env(cfg)

    algo_name = cfg.algo.name
    if 'Agent' not in algo_name:
        algo_name = 'Agent' + algo_name
    agent_class = load_class_from_path(algo_name, alg_name_to_path[algo_name])
    agent = agent_class(env=env, cfg=cfg)

    if cfg.artifact is not None:
        load_model(agent.actor, "actor", cfg)
        load_model(agent.critic, "critic", cfg)
        if cfg.algo.obs_norm:
            load_model(agent.obs_rms, "obs_rms", cfg)

    global_steps = 0
    evaluator = Evaluator(cfg=cfg, wandb_run=wandb_run)

    agent.reset_agent()
    is_off_policy = cfg.algo.name != 'PPO'
    if is_off_policy:
        memory = ReplayBuffer(capacity=int(cfg.algo.memory_size),
                              obs_dim=agent.obs_dim,
                              action_dim=agent.action_dim,
                              device=cfg.device)
        trajectory, steps = agent.explore_env(env, cfg.algo.warm_up, random=True)
        memory.add_to_buffer(trajectory)
        global_steps += steps

    for iter_t in count():
        trajectory, steps = agent.explore_env(env, cfg.algo.horizon_len, random=False)
        global_steps += steps
        if is_off_policy:
            memory.add_to_buffer(trajectory)
            log_info = agent.update_net(memory)
        else:
            log_info = agent.update_net(trajectory)

        if evaluator.parent.poll():
            return_dict = evaluator.parent.recv()
            wandb.log(return_dict, step=global_steps)
        if iter_t % cfg.algo.log_freq == 0:
            log_info['global_steps'] = global_steps
            wandb.log(log_info, step=global_steps)
        if iter_t % cfg.algo.eval_freq == 0:
            evaluator.eval_policy(agent.actor, agent.critic, normalizer=agent.obs_rms,
                                  step=global_steps)
        if evaluator.check_if_should_stop(global_steps):
            break


if __name__ == '__main__':
    main()
