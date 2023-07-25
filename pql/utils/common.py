import ast
import platform
import random
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import gym
import numpy as np
import torch
import wandb
from loguru import logger
from omegaconf import OmegaConf, open_dict


def init_wandb(cfg):
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True,
                                       throw_on_missing=True)
    wandb_cfg['hostname'] = platform.node()
    wandb_kwargs = cfg.logging.wandb
    wandb_tags = wandb_kwargs.get('tags', None)
    if wandb_tags is not None and isinstance(wandb_tags, str):
        wandb_kwargs['tags'] = [wandb_tags]
    if cfg.artifact is not None:
        wandb_id = cfg.artifact.split("/")[-1].split(":")[0]
        wandb_run = wandb.init(**wandb_kwargs, config=wandb_cfg, id=wandb_id, resume="must")
    else:
        wandb_run = wandb.init(**wandb_kwargs, config=wandb_cfg)
    logger.warning(f'Wandb run dir:{wandb_run.dir}')
    logger.warning(f'Project name:{wandb_run.project_name()}')
    return wandb_run


def load_class_from_path(cls_name, path):
    mod_name = 'MOD%s' % cls_name
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[cls_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)


def set_random_seed(seed=None):
    if seed is None:
        max_seed_value = np.iinfo(np.uint32).max
        min_seed_value = np.iinfo(np.uint32).min
        seed = random.randint(min_seed_value, max_seed_value)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    logger.info(f'Setting random seed to:{seed}')
    return seed


def set_print_formatting():
    """ formats numpy print """
    configs = dict(
        precision=6,
        edgeitems=30,
        linewidth=1000,
        threshold=5000,
    )
    np.set_printoptions(suppress=True,
                        formatter=None,
                        **configs)
    torch.set_printoptions(sci_mode=False, **configs)


def pathlib_file(file_name):
    if isinstance(file_name, str):
        file_name = Path(file_name)
    elif not isinstance(file_name, Path):
        raise TypeError(f'Please check the type of the filename:{file_name}')
    return file_name


def list_class_names(dir_path):
    """
    Return the mapping of class names in all files
    in dir_path to their file path.
    Args:
        dir_path (str): absolute path of the folder.
    Returns:
        dict: mapping from the class names in all python files in the
        folder to their file path.
    """
    dir_path = pathlib_file(dir_path)
    py_files = list(dir_path.rglob('*.py'))
    py_files = [f for f in py_files if f.is_file() and f.name != '__init__.py']
    cls_name_to_path = dict()
    for py_file in py_files:
        with py_file.open() as f:
            node = ast.parse(f.read())
        classes_in_file = [n for n in node.body if isinstance(n, ast.ClassDef)]
        cls_names_in_file = [c.name for c in classes_in_file]
        for cls_name in cls_names_in_file:
            cls_name_to_path[cls_name] = py_file
    return cls_name_to_path


class Tracker:
    def __init__(self, max_len):
        self.moving_average = deque([0 for _ in range(max_len)], maxlen=max_len)
        self.max_len = max_len

    def __repr__(self):
        return self.moving_average.__repr__()

    def update(self, value):
        if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
            self.moving_average.extend(value.tolist())
        elif isinstance(value, Sequence):
            self.moving_average.extend(value)
        else:
            self.moving_average.append(value)

    def mean(self):
        return np.mean(self.moving_average)

    def std(self):
        return np.std(self.moving_average)

    def max(self):
        return np.max(self.moving_average)


def get_action_dim(action_space):
    if isinstance(action_space, gym.spaces.Discrete):
        act_size = action_space.n
    elif isinstance(action_space, gym.spaces.Box):
        act_size = action_space.shape[0]
    else:
        raise TypeError
    return act_size


def normalize(input, normalize_tuple):
    if normalize_tuple is not None:
        current_mean, current_var, epsilon = normalize_tuple
        y = (input - current_mean.float()) / torch.sqrt(current_var.float() + epsilon)
        y = torch.clamp(y, min=-5.0, max=5.0)
        return y
    return input


def preprocess_cfg(cfg):
    with open_dict(cfg):
        cfg.available_gpus = torch.cuda.device_count()
        
    if cfg.algo.name == 'PPO':
        if cfg.isaac_param:
            peprocess_PPO_cfg(cfg)
    elif cfg.algo.name == 'PQL':
        check_device(cfg)
        
    task_name = cfg.task.name
    task_reward_scale = dict(
        AllegroHand=0.01,
        Ant=0.01,
        Humanoid=0.01,
        Anymal=1.,
        FrankaCubeStack=0.1,
        ShadowHand=0.01,
        BallBalance=0.1
    )
    # only change the scale if the user does not pass in a new scale (default is 1.0)
    if task_name in task_reward_scale and cfg.algo.reward_scale == 1:
        cfg.algo.reward_scale = task_reward_scale[task_name]

    task_max_time = dict(
        AllegroHand=4800,
        Ant=3600,
        Humanoid=3600,
        Anymal=1800,
        FrankaCubeStack=3600,
        ShadowHand=4800,
        BallBalance=3600
    )
    if task_name in task_max_time and cfg.max_time == 3600:
        cfg.max_time = task_max_time[task_name]


def capture_keyboard_interrupt():
    import signal
    import sys
    def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)


def handle_timeout(dones, info):
    timeout_key = 'TimeLimit.truncated'
    timeout_envs = None
    if timeout_key in info:
        timeout_envs = info[timeout_key]
    if timeout_envs is not None:
        dones = dones * (~timeout_envs)
    return dones


def aggregate_traj_info(infos, key, single_info=False):
    if single_info:
        infos = [infos]
    if isinstance(infos[0], Sequence):
        out = []
        for info in infos:
            time_out = []
            for env_info in info:
                time_out.append(env_info[key])
            out.append(np.stack(time_out))
        out = stack_data(out)
    elif isinstance(infos[0], dict):
        out = []
        for info in infos:
            tensor = info[key]
            out.append(tensor)
        out = stack_data(out)
    else:
        raise NotImplementedError
    if single_info:
        out = out.squeeze(0)
    return out


def stack_data(data, torch_to_numpy=False, dim=0):
    if isinstance(data[0], dict):
        out = dict()
        for key in data[0].keys():
            out[key] = stack_data([x[key] for x in data], dim=dim)
        return out
    try:
        ret = torch.stack(data, dim=dim)
        if torch_to_numpy:
            ret = ret.cpu().numpy()
    except:
        # if data is a list of arrays that do not have same shapes (such as point cloud)
        ret = data
    return ret


# PPO uses different hyperparameters per task. See IsaacGymEnvs for details.
def peprocess_PPO_cfg(cfg):
    if cfg.task.name == 'Ant':
        cfg.num_envs = 4096
        cfg.algo.batch_size = 32768
        cfg.algo.horizon_len = 16
        cfg.algo.update_times = 4
    elif cfg.task.name == 'Humanoid':
        cfg.num_envs = 4096
        cfg.algo.batch_size = 32768
        cfg.algo.horizon_len = 32
        cfg.algo.update_times = 5
        cfg.algo.value_norm = True
    elif cfg.task.name == 'Anymal':
        cfg.num_envs = 4096
        cfg.algo.batch_size = 32768
        cfg.algo.horizon_len = 16
        cfg.algo.update_times = 5
    elif cfg.task.name == 'AllegroHand' or cfg.task == 'ShadowHand':
        cfg.num_envs = 16384
        cfg.algo.batch_size = 32768
        cfg.algo.horizon_len = 8
        cfg.algo.update_times = 5
        cfg.algo.value_norm = True
    elif cfg.task.name == 'FrankaCubeStack':
        cfg.num_envs = 8192
        cfg.algo.batch_size = 16384
        cfg.algo.horizon_len = 32
        cfg.algo.update_times = 5
    else:
        logger.warning(f'Cannot find config for PPO on task:{cfg.task}. Using default config.')


# check PQL device
def check_device(cfg):
    # sim device is always 0
    device_set = set([0, int(cfg.algo.p_learner_gpu), int(cfg.algo.v_learner_gpu)])
    if len(device_set) > cfg.available_gpus:
        assert 'Invalid CUDA device: id out of range'
    for gpu_id in device_set:
        if gpu_id >= cfg.available_gpus:
            assert f'Invalid CUDA device {gpu_id}: id out of range'
    # need more check
        
