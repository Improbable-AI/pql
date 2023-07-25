from dataclasses import dataclass
from typing import Any

from gym import spaces


@dataclass
class FlatObEnvWrapper:
    env: Any
    ob_key: str = 'obs'

    def __post_init__(self):
        self.observation_space = self.env.observation_space
        if isinstance(self.observation_space, spaces.Dict):
            self.observation_space = self.observation_space[self.ob_key]
        self.action_space = self.env.action_space
        self.max_episode_length = self.env.max_episode_length

    def reset(self):
        ob = self.env.reset()
        return ob[self.ob_key]

    def step(self, actions):
        next_obs, rewards, dones, info = self.env.step(actions)
        return next_obs[self.ob_key], rewards, dones, info
