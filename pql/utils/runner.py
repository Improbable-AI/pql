from pql.algo import alg_name_to_path
from pql.replay.nstep_replay import NStepReplay
from pql.utils.common import load_class_from_path
from pql.utils.isaacgym_util import create_task_env

obs_dim: int
action_dim: int
act_class: Any
cri_class: Any
cfg: DictConfig


def train_and_evaluate(cfg):
    env = create_task_env(cfg)
    algo_name = cfg.algo.name
    if 'Agent' not in algo_name:
        algo_name = 'Agent' + algo_name
    agent_class = load_class_from_path(algo_name, alg_name_to_path[algo_name])
    agent = agent_class()
    self.memory = NStepReplay(
        capacity=self.cfg.algo.memory_size,
        obs_dim=self.obs_dim,
        action_dim=self.action_dim,
        device=self.cfg.algo.device,
        nstep=self.cfg.algo.nstep,
        num_envs=self.cfg.num_envs,
        gamma=self.cfg.algo.gamma
    )
