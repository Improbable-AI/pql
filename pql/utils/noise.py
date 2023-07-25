import torch
from torch.distributions import Normal


class FixedNormalActionNoise:
    def __init__(self, mean, std, bounds=None):
        self._mu = mean
        self._std = std
        self._bounds = bounds
        self.dist = Normal(self._mu, self._std)

    def __call__(self, num=torch.Size(), truncated=False):
        sample = self.dist.rsample((num,))
        if truncated:
            sample.clamp(self._bounds[0], self._bounds[1])
        return sample


def add_normal_noise(x, std, noise_bounds=None, out_bounds=None):
    noise = torch.normal(torch.zeros(x.shape, dtype=x.dtype, device=x.device),
                         torch.full(x.shape, std, dtype=x.dtype, device=x.device))
    if noise_bounds is not None:
        noise = noise.clamp(noise_bounds[0], noise_bounds[1])
    out = x + noise
    if out_bounds is not None:
        out = out.clamp(out_bounds[0], out_bounds[1])
    return out


def add_mixed_normal_noise(x, std_max, std_min, noise_bounds=None, out_bounds=None):
    std_seq = torch.linspace(std_min, std_max,
                             x.shape[0]).to(x.device).unsqueeze(-1).expand(x.shape)

    noise = torch.normal(torch.zeros(x.shape, dtype=x.dtype, device=x.device),
                         std_seq)
    if noise_bounds is not None:
        noise = noise.clamp(noise_bounds[0], noise_bounds[1])
    out = x + noise
    if out_bounds is not None:
        out = out.clamp(out_bounds[0], out_bounds[1])
    return out
