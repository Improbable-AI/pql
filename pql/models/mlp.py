from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Independent
from torch.distributions import Normal

from pql.utils.torch_util import SquashedNormal


def create_simple_mlp(in_dim, out_dim, hidden_layers, act=nn.ELU):
    layer_nums = [in_dim, *hidden_layers, out_dim]
    model = []
    for idx, (in_f, out_f) in enumerate(zip(layer_nums[:-1], layer_nums[1:])):
        model.append(nn.Linear(in_f, out_f))
        if idx < len(layer_nums) - 2:
            model.append(act())
    return nn.Sequential(*model)


class MLPNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers=None):
        super().__init__()
        if isinstance(in_dim, Sequence):
            in_dim = in_dim[0]
        if hidden_layers is None:
            hidden_layers = [512, 256, 128]
        self.net = create_simple_mlp(in_dim=in_dim,
                                     out_dim=out_dim,
                                     hidden_layers=hidden_layers)

    def forward(self, x):
        return self.net(x)


class DiagGaussianMLPPolicy(MLPNet):
    def __init__(self, state_dim, act_dim, hidden_layers=None,
                 init_log_std=0.):
        super().__init__(in_dim=state_dim,
                         out_dim=act_dim,
                         hidden_layers=hidden_layers)
        self.logstd = nn.Parameter(torch.full((act_dim,), init_log_std))

    def forward(self, x, sample=True):
        return self.get_actions(x, sample=sample)[0]

    def get_actions(self, x, sample=True):
        mean = self.net(x)
        log_std = self.logstd.expand_as(mean)
        std = torch.exp(log_std)
        action_dist = Independent(Normal(loc=mean, scale=std), 1)
        if sample:
            actions = action_dist.rsample()
        else:
            actions = mean
        return actions, action_dist

    def get_actions_logprob_entropy(self, state, sample=True):
        actions, action_dist = self.get_actions(state, sample=sample)
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return actions, action_dist, log_prob, entropy

    def logprob_entropy(self, state, actions):
        _, action_dist = self.get_actions(state)
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return actions, action_dist, log_prob, entropy


class TanhDiagGaussianMLPPolicy(MLPNet):
    def __init__(self, state_dim, act_dim, hidden_layers=None):
        super().__init__(in_dim=state_dim,
                         out_dim=act_dim * 2,
                         hidden_layers=hidden_layers)
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.log_std_min = -5
        self.log_std_max = 5

    def forward(self, state: Tensor, sample: bool = False) -> Tensor:
        return self.get_actions(state, sample=sample)

    def get_actions(self, state: Tensor, sample=True) -> Tensor:
        dist = self.get_action_dist(state)
        if sample:
            actions = dist.rsample()
        else:
            actions = dist.mean
        return actions

    def get_action_dist(self, state: Tensor):
        mu, log_std = self.net(state).chunk(2, dim=-1)
        std = log_std.clamp(self.log_std_min, self.log_std_max).exp()
        dist = SquashedNormal(mu, std)
        return dist

    def get_actions_logprob(self, state: Tensor):
        dist = self.get_action_dist(state)
        actions = dist.rsample()
        log_prob = dist.log_prob(actions).sum(-1, keepdim=True)
        return actions, dist, log_prob


class TanhMLPPolicy(MLPNet):
    def forward(self, state):
        return super().forward(state).tanh()


class DoubleQ(nn.Module):
    def __init__(self, state_dim, act_dim):
        super().__init__()
        if isinstance(state_dim, Sequence):
            state_dim = state_dim[0]
        self.net_q1 = MLPNet(in_dim=state_dim + act_dim, out_dim=1)
        self.net_q2 = MLPNet(in_dim=state_dim + act_dim, out_dim=1)

    def get_q_min(self, state: Tensor, action: Tensor) -> Tensor:
        return torch.min(*self.get_q1_q2(state, action))  # min Q value

    def get_q1_q2(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        input_x = torch.cat((state, action), dim=1)
        return self.net_q1(input_x), self.net_q2(input_x)  # two Q values

    def get_q1(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        input_x = torch.cat((state, action), dim=1)
        return self.net_q1(input_x)


class DistributionalDoubleQ(nn.Module):
    def __init__(self, state_dim, act_dim, v_min=-10, v_max=10, num_atoms=51, device="cuda"):
        super().__init__()
        if isinstance(state_dim, Sequence):
            state_dim = state_dim[0]
        self.device = device
        self.net_q1 = MLPNet(in_dim=state_dim + act_dim, out_dim=num_atoms)
        self.net_q2 = MLPNet(in_dim=state_dim + act_dim, out_dim=num_atoms)

        self.z_atoms = torch.linspace(v_min, v_max, num_atoms, device=device)

    def get_q_min(self, state: Tensor, action: Tensor) -> Tensor:
        Q1, Q2 = self.get_q1_q2(state, action)
        Q1 = torch.sum(Q1 * self.z_atoms.to(self.device), dim=1)
        Q2 = torch.sum(Q2 * self.z_atoms.to(self.device), dim=1)
        return torch.min(Q1, Q2)  # min Q value

    def get_q1_q2(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        input_x = torch.cat((state, action), dim=1)
        return torch.softmax(self.net_q1(input_x), dim=1), torch.softmax(self.net_q2(input_x), dim=1)  # two Q values

    def get_q1(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        input_x = torch.cat((state, action), dim=1)
        return torch.softmax(self.net_q1(input_x), dim=1)


class MLPCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        if isinstance(state_dim, Sequence):
            state_dim = state_dim[0]
        self.critic = MLPNet(in_dim=state_dim, out_dim=1)

    def forward(self, state: Tensor) -> Tensor:
        return self.critic(state)  # advantage value
