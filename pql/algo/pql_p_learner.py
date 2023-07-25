import os
import time

import ray
import torch
from torch.nn.utils import clip_grad_norm_
from loguru import logger
from pql.models import model_name_to_path
from pql.utils.common import Tracker
from pql.utils.common import load_class_from_path
from pql.utils.common import normalize
from pql.utils.model_util import load_model


@ray.remote(num_gpus=0.3)
class PQLPLearner:
    def __init__(self, obs_dim, action_dim, cfg):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(cfg.available_gpus)])
        self.device = torch.device(f"cuda:{self.cfg.algo.p_learner_gpu}")

        act_class = load_class_from_path(self.cfg.algo.act_class,
                                         model_name_to_path[self.cfg.algo.act_class])
        self.actor = act_class(self.obs_dim, self.action_dim).to(self.device)
        if self.cfg.artifact is not None:
            load_model(self.actor, "actor", cfg)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), self.cfg.algo.actor_lr)
        self.critic = None

        obs_dim = (self.obs_dim,) if isinstance(self.obs_dim, int) else self.obs_dim
        self.memory_size = int(self.cfg.algo.memory_size)
        self.memory = torch.empty((self.memory_size,) + (*obs_dim,), dtype=torch.float32, device=self.device)
        self.next_p = 0
        self.if_full = False
        self.cur_capacity = 0

        self.loss_tracker = Tracker(5)
        self.update_count = 0
        self.normalize_tuple = None
        self.sleep_time = 0.01

    def start(self):
        return self.actor, self.update_count, self.loss_tracker.mean()

    def learn(self):
        if self.critic is not None:
            indices = torch.randint(self.cur_capacity, size=(self.cfg.algo.batch_size,), device=self.device)
            obs = self.memory[indices].clone()
            if self.cfg.algo.obs_norm:
                obs = normalize(obs, self.normalize_tuple)

            self.critic.requires_grad_(False)
            action = self.actor(obs)
            actor_Q = self.critic.get_q_min(obs, action)
            actor_loss = -actor_Q.mean()
            self.optimizer_update(self.actor_optimizer, actor_loss)
            self.critic.requires_grad_(True)

            self.update_count += 1
            self.loss_tracker.update(actor_loss.detach().item())

        return self.sleep_time

    def update(self, critic, obs, normalize_tuple, sleep_time):
        self.critic = critic
        self.sleep_time = sleep_time
        self.normalize_tuple = normalize_tuple

        obs_dim = (self.obs_dim,) if isinstance(self.obs_dim, int) else self.obs_dim
        obs = obs.reshape(-1, *obs_dim)
        self.add_capacity = obs.shape[0]
        p = self.next_p + self.add_capacity
        if p > self.memory_size:
            self.memory[self.next_p:self.memory_size] = obs[:self.memory_size - self.next_p]
            p = p - self.memory_size
            self.memory[0:p] = obs[-p:]
            self.if_full = True
        else:
            self.memory[self.next_p:p] = obs
        self.next_p = p  # update pointer
        self.cur_capacity = self.memory_size if self.if_full else self.next_p

        return self.actor, self.loss_tracker.mean(), self.update_count

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
def asyn_p_learner(p_learner, cfg):
    logger.warning(f"P-Learner starts running asynchronously on GPU {cfg.algo.p_learner_gpu}")
    while True:
        sleep_time = ray.get(p_learner.learn.remote())
        time.sleep(sleep_time)
