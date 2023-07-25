from dataclasses import dataclass
from typing import Any


@dataclass
class ResetEnvWrapper:
    env: Any

    def __post_init__(self):
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.max_episode_length = self.env.max_episode_length

    def reset(self):
        zero_actions = self.env.zero_actions()
        self.env.reset_buf.fill_(1)
        # step the simulator
        self.env.step(zero_actions)
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)
