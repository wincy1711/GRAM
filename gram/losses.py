"""Training objectives for GRAM.

The main objective is the truncated ELBO surrogate of Eq. (14):

    L_GRAM^(n) = E_q[ log p(y | z_T^(n), x) ] - KL( q_phi(eps_T | u_T, y) || p_theta(eps_T | u_T) )

evaluated once per supervision step, with gradients flowing only through the
final transition of that step.  ``LGRAM`` is a *biased but memory-constant*
approximation to the full trajectory ELBO (Eq. 13), which is available for
diagnostics via ``gram.evaluate.full_elbo``.

Auxiliary objectives: the ACT halting Q-head (Eq. 15) and the Latent Process
Reward Model value head (Eq. 16).  Both read detached latents and therefore do
not propagate gradients into the recursive core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .core import TransitionOutput
from .guidance import Gaussian, gaussian_kl

IGNORE_INDEX = -100


# --------------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------------- #
def reconstruction_loss(logits: Tensor, targets: Tensor,
                        ignore_index: int = IGNORE_INDEX) -> Tensor:
    """Mean token-level cross entropy, i.e. ``-log p(y | z_T, x)`` up to a constant."""
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


# --------------------------------------------------------------------------- #
# KL term
# --------------------------------------------------------------------------- #
def balanced_kl(posterior: Gaussian, prior: Gaussian, kl_balance: float = 0.8,
                free_bits: float = 0.0) -> Tensor:
    """KL with DreamerV2-style balancing.

    ``kl_balance`` is the weight on the term that trains the *prior* towards the
    posterior; ``1 - kl_balance`` trains the posterior towards the prior.  The
    paper uses ``0.8`` to prevent posterior collapse (Appendix B.2).
    """
    if posterior is None or prior is None:
        return torch.zeros((), device=_any_device(posterior, prior))

    if prior.degenerate or posterior.degenerate:
        # ``guidance="guide_only"`` makes both distributions Dirac deltas; the KL
        # is then only defined in the limit.  We use the squared mean distance,
        # which is the sigma -> 1 limit of the Gaussian KL.
        return 0.5 * (posterior.mean - prior.mean).pow(2).sum(-1).mean()

    if kl_balance is None or kl_balance == 0.5:
        kl = gaussian_kl(posterior, prior)
    else:
        kl_prior = gaussian_kl(posterior.detach(), prior)      # trains p_theta
        kl_post = gaussian_kl(posterior, prior.detach())       # trains q_phi
        kl = kl_balance * kl_prior + (1.0 - kl_balance) * kl_post

    if free_bits > 0.0:
        kl = torch.clamp(kl, min=free_bits)
    # Sum over latent dimensions, average over batch and sequence positions.
    return kl.sum(-1).mean()


def _any_device(*dists) -> torch.device:
    for d in dists:
        if d is not None:
            return d.mean.device
    return torch.device("cpu")


def transition_kl(transitions: Sequence[TransitionOutput], kl_balance: float = 0.8,
                  free_bits: float = 0.0) -> Tensor:
    """Sum the KL contributions of a set of transitions."""
    total: Optional[Tensor] = None
    for tr in transitions:
        if tr.prior is None or tr.posterior is None:
            continue
        kl = balanced_kl(tr.posterior, tr.prior, kl_balance, free_bits)
        total = kl if total is None else total + kl
    if total is None:
        return torch.zeros(())
    return total


# --------------------------------------------------------------------------- #
# Auxiliary heads
# --------------------------------------------------------------------------- #
def act_loss(q_halt_logits: List[Tensor], q_continue_logits: List[Tensor],
             is_correct: List[Tensor], mode: str = "halt_only") -> Tensor:
    """ACT Q-learning loss (Eq. 15).

    ``halt_only`` trains only the halt head against ``1[y_hat == y]`` (the
    simplified variant the paper says it releases).  ``q_learning`` adds the
    bootstrapped continue target ``max(q_halt_{n+1}, q_continue_{n+1})``.
    """
    if not q_halt_logits:
        return torch.zeros(())
    device = q_halt_logits[0].device
    total = torch.zeros((), device=device)
    n_steps = len(q_halt_logits)
    for n in range(n_steps):
        target = is_correct[n].to(device=device, dtype=q_halt_logits[n].dtype)
        total = total + F.binary_cross_entropy_with_logits(q_halt_logits[n], target)
        if mode == "q_learning" and q_continue_logits[n] is not None:
            if n + 1 < n_steps:
                with torch.no_grad():
                    bootstrap = torch.maximum(
                        q_halt_logits[n + 1], q_continue_logits[n + 1]
                    ).sigmoid()
            else:
                # Terminal step: continuing cannot beat the final halt value.
                bootstrap = target
            total = total + F.binary_cross_entropy_with_logits(
                q_continue_logits[n], bootstrap
            )
    return total / n_steps


def lprm_loss(values: List[Tensor], reward: Tensor) -> Tensor:
    """Latent process reward model regression (Eq. 16)."""
    if not values:
        return torch.zeros(())
    device = values[0].device
    reward = reward.to(device=device, dtype=values[0].dtype)
    total = torch.zeros((), device=device)
    for v in values:
        total = total + F.mse_loss(v, reward)
    return total / len(values)


# --------------------------------------------------------------------------- #
# Aggregated loss for one supervision step
# --------------------------------------------------------------------------- #
@dataclass
class StepLoss:
    total: Tensor
    reconstruction: Tensor
    kl: Tensor


def supervision_step_loss(logits: Tensor, targets: Tensor,
                          transitions: Sequence[TransitionOutput],
                          beta: float = 0.1, kl_balance: float = 0.8,
                          free_bits: float = 0.0,
                          ignore_index: int = IGNORE_INDEX) -> StepLoss:
    """``-L_GRAM^(n)`` from Eq. (14) (a loss, so the ELBO is negated)."""
    recon = reconstruction_loss(logits, targets, ignore_index)
    kl = transition_kl(transitions, kl_balance, free_bits).to(recon.device)
    return StepLoss(recon + beta * kl, recon.detach(), kl.detach())


__all__ = [
    "IGNORE_INDEX",
    "StepLoss",
    "act_loss",
    "balanced_kl",
    "lprm_loss",
    "reconstruction_loss",
    "supervision_step_loss",
    "transition_kl",
    "supervision_step_loss",
]
