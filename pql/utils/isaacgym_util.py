from isaacgymenvs.tasks import isaacgym_task_map
from omegaconf import OmegaConf

from pql.wrappers.flatten_ob import FlatObEnvWrapper
from pql.wrappers.reset import ResetEnvWrapper


def create_task_env(cfg, num_envs=None):
    task_cfg = OmegaConf.to_container(cfg.task, resolve=True, throw_on_missing=True)
    if num_envs is not None:
        task_cfg['env']['numEnvs'] = num_envs

    env = isaacgym_task_map[cfg.task.name](
        cfg=task_cfg,
        rl_device=cfg.rl_device,
        sim_device=cfg.sim_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
        virtual_screen_capture=False,
        force_render=not cfg.headless
    )
    env = ResetEnvWrapper(env)
    env = FlatObEnvWrapper(env)
    return env
