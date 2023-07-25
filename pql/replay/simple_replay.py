import torch


def create_buffer(capacity, obs_dim, action_dim, device='cuda'):
    if isinstance(capacity, int):
        capacity = (capacity,)
    buf_obs_size = (*capacity, obs_dim) if isinstance(obs_dim, int) else (*capacity, *obs_dim)
    buf_obs = torch.empty(buf_obs_size,
                          dtype=torch.float32, device=device)
    buf_action = torch.empty((*capacity, int(action_dim)),
                             dtype=torch.float32, device=device)
    buf_reward = torch.empty((*capacity, 1),
                             dtype=torch.float32, device=device)
    buf_next_obs = torch.empty(buf_obs_size,
                               dtype=torch.float32, device=device)
    buf_done = torch.empty((*capacity, 1),
                           dtype=torch.bool, device=device)
    return buf_obs, buf_action, buf_next_obs, buf_reward, buf_done


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device='cpu'):
        self.obs_dim = obs_dim
        if isinstance(obs_dim, int):
            self.obs_dim = (self.obs_dim,)
        self.action_dim = action_dim
        self.device = device
        self.next_p = 0  # next pointer
        self.if_full = False
        self.cur_capacity = 0  # current capacity
        self.capacity = int(capacity)

        ret = create_buffer(capacity=self.capacity, obs_dim=obs_dim, action_dim=action_dim, device=device)
        self.buf_obs, self.buf_action, self.buf_next_obs, self.buf_reward, self.buf_done = ret

    @torch.no_grad()
    def add_to_buffer(self, trajectory):
        obs, actions, rewards, next_obs, dones = trajectory
        obs = obs.reshape(-1, *self.obs_dim)
        actions = actions.reshape(-1, self.action_dim)
        rewards = rewards.reshape(-1, 1)
        next_obs = next_obs.reshape(-1, *self.obs_dim)
        dones = dones.reshape(-1, 1).bool()
        p = self.next_p + rewards.shape[0]

        if p > self.capacity:
            self.if_full = True

            self.buf_obs[self.next_p:self.capacity] = obs[:self.capacity - self.next_p]
            self.buf_action[self.next_p:self.capacity] = actions[:self.capacity - self.next_p]
            self.buf_reward[self.next_p:self.capacity] = rewards[:self.capacity - self.next_p]
            self.buf_next_obs[self.next_p:self.capacity] = next_obs[:self.capacity - self.next_p]
            self.buf_done[self.next_p:self.capacity] = dones[:self.capacity - self.next_p]

            p = p - self.capacity
            self.buf_obs[0:p] = obs[-p:]
            self.buf_action[0:p] = actions[-p:]
            self.buf_reward[0:p] = rewards[-p:]
            self.buf_next_obs[0:p] = next_obs[-p:]
            self.buf_done[0:p] = dones[-p:]
        else:
            self.buf_obs[self.next_p:p] = obs
            self.buf_action[self.next_p:p] = actions
            self.buf_reward[self.next_p:p] = rewards
            self.buf_next_obs[self.next_p:p] = next_obs
            self.buf_done[self.next_p:p] = dones

        self.next_p = p  # update pointer
        self.cur_capacity = self.capacity if self.if_full else self.next_p

    @torch.no_grad()
    def sample_batch(self, batch_size, device='cuda'):
        indices = torch.randint(self.cur_capacity, size=(batch_size,), device=device)
        return (
            self.buf_obs[indices].to(device),
            self.buf_action[indices].to(device),
            self.buf_reward[indices].to(device),
            self.buf_next_obs[indices].to(device),
            self.buf_done[indices].to(device).float()
        )
