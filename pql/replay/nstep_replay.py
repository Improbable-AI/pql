import torch

from pql.replay.simple_replay import create_buffer


class NStepReplay:
    def __init__(self, obs_dim: int,
                 action_dim: int,
                 num_envs: int = 1,
                 nstep: int = 3,
                 device: str = 'cuda',
                 gamma: float = 0.99
                 ):
        self.num_envs = num_envs
        self.nstep = nstep
        ret = create_buffer(capacity=(self.num_envs, self.nstep),
                            obs_dim=obs_dim, action_dim=action_dim,
                            device=device)
        self.nstep_buf_obs, self.nstep_buf_action, self.nstep_buf_next_obs, \
        self.nstep_buf_reward, self.nstep_buf_done = ret
        self.nstep_count = 0
        self.gamma = gamma
        self.gamma_array = torch.tensor([self.gamma ** i for i in range(self.nstep)]).to(device).view(-1, 1)

    @torch.no_grad()
    def add_to_buffer(self, obs, actions, rewards, next_obs, dones):
        if self.nstep > 1:
            obs_list, action_list, reward_list, next_obs_list, done_list = list(), list(), list(), list(), list()
            for i in range(obs.shape[1]):
                self.nstep_buf_obs = self.fifo_shift(self.nstep_buf_obs, obs[:, i])
                self.nstep_buf_next_obs = self.fifo_shift(self.nstep_buf_next_obs, next_obs[:, i])
                self.nstep_buf_done = self.fifo_shift(self.nstep_buf_done, dones[:, i])
                self.nstep_buf_action = self.fifo_shift(self.nstep_buf_action, actions[:, i])
                self.nstep_buf_reward = self.fifo_shift(self.nstep_buf_reward, rewards[:, i])
                self.nstep_count += 1
                if self.nstep_count < self.nstep:
                    continue

                obs_list.append(self.nstep_buf_obs[:, 0])
                action_list.append(self.nstep_buf_action[:, 0])
                reward, next_ob, done = compute_nstep_return(nstep_buf_next_obs=self.nstep_buf_next_obs,
                                                             nstep_buf_done=self.nstep_buf_done,
                                                             nstep_buf_reward=self.nstep_buf_reward,
                                                             gamma_array=self.gamma_array)
                reward_list.append(reward)
                next_obs_list.append(next_ob)
                done_list.append(done)

            return torch.cat(obs_list), torch.cat(action_list), torch.cat(reward_list), torch.cat(next_obs_list), torch.cat(done_list)
        else:
            return obs, actions, rewards, next_obs, dones

    def fifo_shift(self, queue, new_tensor):
        queue = torch.cat((queue[:, 1:], new_tensor.unsqueeze(1)), dim=1)
        return queue


@torch.jit.script
def compute_nstep_return(nstep_buf_next_obs, nstep_buf_done, nstep_buf_reward, gamma_array):
    buf_done = nstep_buf_done.squeeze(-1)
    buf_done_ids = torch.where(buf_done)
    buf_done_envs = torch.unique_consecutive(buf_done_ids[0])
    buf_done_steps = buf_done.argmax(dim=1)

    done = nstep_buf_done[:, -1].clone()
    done[buf_done_envs] = True

    next_obs = nstep_buf_next_obs[:, -1].clone()
    next_obs[buf_done_envs] = nstep_buf_next_obs[buf_done_envs, buf_done_steps[buf_done_envs]].clone()

    mask = torch.ones(buf_done.shape, device=buf_done.device, dtype=torch.bool)
    mask[buf_done_envs] = torch.arange(mask.shape[1],
                                       device=buf_done.device) <= buf_done_steps[buf_done_envs][:, None]
    discounted_rewards = nstep_buf_reward * gamma_array
    discounted_rewards = (discounted_rewards * mask.unsqueeze(-1)).sum(1)
    return discounted_rewards, next_obs, done
