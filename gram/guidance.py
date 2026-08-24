"""Learnable stochastic guidance (Eq. 4-5 / 8-9 of the paper).

At every latent transition the recursive core first produces a deterministic
update ``u_t``; GRAM then samples a state-dependent perturbation

    eps_t ~ N(mu_theta(u_t), sigma_theta(u_t)^2 I),      z_t = u_t + eps_t.

The prior head sees only ``u_t``.  The variational posterior used during
training sees ``u_t`` together with an embedding of the target ``y``, giving
``q_phi(eps_t | u_t, y)``.  Table 4 of the paper specifies one SwiGLU MLP per
distribution parameter, which is what ``GaussianHead`` implements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .layers import SwiGLU


@dataclass
class Gaussian:
    """A diagonal Gaussian over the perturbation ``eps``.

    ``degenerate`` marks the ``sigma = 0`` case produced by the
    ``guidance="guide_only"`` ablation, so that the KL can take its Dirac limit
    without inspecting tensor values (which would force a device sync).
    """

    mean: Tensor
    std: Tensor
    degenerate: bool = False

    def rsample(self, temperature: float = 1.0) -> Tensor:
        if temperature == 0.0:
            return self.mean
        noise = torch.randn_like(self.mean)
        return self.mean + noise * self.std * temperature

    def log_prob(self, value: Tensor) -> Tensor:
        var = self.std.pow(2)
        return -0.5 * (
            ((value - self.mean) ** 2) / var + 2 * self.std.log() + math.log(2 * math.pi)
        )

    def detach(self) -> "Gaussian":
        return Gaussian(self.mean.detach(), self.std.detach(), self.degenerate)


def gaussian_kl(q: Gaussian, p: Gaussian) -> Tensor:
    """Elementwise ``KL(q || p)`` for two diagonal Gaussians."""
    var_ratio = (q.std / p.std).pow(2)
    t1 = ((q.mean - p.mean) / p.std).pow(2)
    return 0.5 * (var_ratio + t1 - 1.0 - var_ratio.log())


class GaussianHead(nn.Module):
    """Predicts ``(mu, sigma)`` of the stochastic guidance from a latent state.

    Parameters
    ----------
    guidance:
        ``"full"``          -- learned mean and learned std (GRAM).
        ``"stochastic_only"``-- mean forced to zero, ``N(0, sigma^2 I)``.
        ``"guide_only"``    -- std forced to zero, deterministic ``u + mu``.
        ``"none"``          -- no perturbation at all (deterministic RRM).
    """

    def __init__(self, hidden_size: int, ffn_hidden_size: int, guidance: str = "full",
                 min_std: float = 1e-3, max_std: float = 1.0, init_std: float = 0.1):
        super().__init__()
        self.guidance = guidance
        self.min_std = min_std
        self.max_std = max_std
        self.learn_mean = guidance in ("full", "guide_only")
        self.learn_std = guidance in ("full", "stochastic_only")

        if self.learn_mean:
            # Zero-initialised output so training starts from the deterministic
            # update and learns to steer away from it.
            self.mu_head = SwiGLU(hidden_size, ffn_hidden_size, hidden_size,
                                  zero_init_out=True)
        if self.learn_std:
            self.log_std_head = SwiGLU(hidden_size, ffn_hidden_size, hidden_size,
                                       zero_init_out=True)
            self.register_buffer("std_bias", torch.tensor(self._inverse_std(init_std)),
                                 persistent=True)

    def _inverse_std(self, std: float) -> float:
        """Invert ``min_std + (max_std - min_std) * sigmoid(x)``."""
        span = self.max_std - self.min_std
        frac = min(max((std - self.min_std) / span, 1e-6), 1 - 1e-6)
        return math.log(frac / (1 - frac))

    def forward(self, u: Tensor, cond: Optional[Tensor] = None) -> Gaussian:
        x = u if cond is None else u + cond
        if self.learn_mean:
            mean = self.mu_head(x)
        else:
            mean = torch.zeros_like(u)
        if self.learn_std:
            raw = self.log_std_head(x) + self.std_bias
            std = self.min_std + (self.max_std - self.min_std) * torch.sigmoid(raw)
        else:
            std = torch.zeros_like(u)
        return Gaussian(mean, std, degenerate=not self.learn_std)


class StochasticTransition(nn.Module):
    """Bundles the prior ``p_theta(eps|u)`` and posterior ``q_phi(eps|u, y)``."""

    def __init__(self, hidden_size: int, ffn_hidden_size: int, guidance: str = "full",
                 min_std: float = 1e-3, max_std: float = 1.0, init_std: float = 0.1):
        super().__init__()
        self.guidance = guidance
        self.deterministic = guidance == "none"
        if self.deterministic:
            return
        self.prior = GaussianHead(hidden_size, ffn_hidden_size, guidance,
                                  min_std, max_std, init_std)
        self.posterior = GaussianHead(hidden_size, ffn_hidden_size, guidance,
                                      min_std, max_std, init_std)

    def forward(self, u: Tensor, y_embed: Optional[Tensor] = None,
                use_posterior: bool = False, temperature: float = 1.0):
        """Return ``(z, prior_dist, posterior_dist)``.

        ``posterior_dist`` is ``None`` outside training (``y`` unavailable).
        """
        if self.deterministic:
            return u, None, None

        prior = self.prior(u)
        posterior = None
        if use_posterior:
            if y_embed is None:
                raise ValueError("posterior sampling requires the target embedding")
            posterior = self.posterior(u, y_embed)
            eps = posterior.rsample(temperature=1.0)
        else:
            eps = prior.rsample(temperature=temperature)
        return u + eps, prior, posterior


__all__ = ["Gaussian", "GaussianHead", "StochasticTransition", "gaussian_kl"]
