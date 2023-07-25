import torch


def projection(next_dist, reward, done, gamma, v_min=-10, v_max=10, num_atoms=51, support=None, device="cuda:0"):
    delta_z = (v_max - v_min) / (num_atoms - 1)
    batch_size = reward.shape[0]

    target_z = (reward + (1 - done) * gamma * support).clamp(min=v_min, max=v_max)
    b = (target_z - v_min) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    l[torch.logical_and((u > 0), (l == u))] -= 1
    u[torch.logical_and((l < (num_atoms - 1)), (l == u))] += 1

    proj_dist = torch.zeros_like(next_dist)
    offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size, device=device).unsqueeze(1).expand(batch_size, num_atoms).long()
    proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
    return proj_dist
