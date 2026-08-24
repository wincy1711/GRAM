"""The GRAM recursive core: hierarchical latent state and stochastic transitions.

A *latent transition* ``z_{t-1} -> z_t`` (Eq. 6-9) is

    l_{t,k} = f_L(h_{t-1}, l_{t,k-1}, e_x)        k = 1..K   (deterministic)
    u_t     = f_H(h_{t-1}, l_t)                              (deterministic)
    eps_t   ~ p_theta(eps_t | u_t)                           (stochastic)
    h_t     = u_t + eps_t

``T`` such transitions form one *supervision step*; ``N_sup`` supervision steps
form the full recursive computation (Eq. 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .guidance import Gaussian, StochasticTransition
from .layers import BlockStack, RotaryEmbedding


@dataclass
class LatentState:
    """The latent state ``z = (h, l)``; ``l`` is ``None`` for the flat variant."""

    h: Tensor
    l: Optional[Tensor] = None

    def detach(self) -> "LatentState":
        return LatentState(self.h.detach(), None if self.l is None else self.l.detach())

    def clone(self) -> "LatentState":
        return LatentState(self.h.clone(), None if self.l is None else self.l.clone())

    def index_select(self, index: Tensor) -> "LatentState":
        return LatentState(
            self.h.index_select(0, index),
            None if self.l is None else self.l.index_select(0, index),
        )

    def repeat_interleave(self, repeats: int) -> "LatentState":
        return LatentState(
            self.h.repeat_interleave(repeats, dim=0),
            None if self.l is None else self.l.repeat_interleave(repeats, dim=0),
        )


@dataclass
class TransitionOutput:
    """One latent transition, with the distributions needed for the KL term."""

    state: LatentState
    prior: Optional[Gaussian]
    posterior: Optional[Gaussian]


class RecursiveCore(nn.Module):
    """Shared transition functions ``f_L`` / ``f_H`` plus stochastic guidance."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        seq_len = config.total_seq_len

        def make_stack() -> BlockStack:
            return BlockStack(
                num_layers=config.num_layers,
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                ffn_hidden_size=config.ffn_hidden_size,
                mixer=config.mixer,
                seq_len=seq_len,
                seq_mixer_hidden=config.seq_mixer_hidden,
            )

        self.f_H = make_stack()
        self.f_L = make_stack() if config.hierarchical else None

        self.transition = StochasticTransition(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.ffn_hidden_size,
            guidance=config.guidance,
            min_std=config.min_std,
            max_std=config.max_std,
            init_std=config.init_std,
        )

        if config.pos_encoding == "rope":
            self.rotary = RotaryEmbedding(
                config.hidden_size // config.num_heads, seq_len, config.rope_theta
            )
        else:
            self.rotary = None

        # z_0 is drawn once from N(0, I) and then kept fixed (Appendix B.1).
        init_h = torch.randn(seq_len, config.hidden_size)
        self.register_buffer("h_init", init_h, persistent=True)
        if config.hierarchical:
            self.register_buffer(
                "l_init", torch.randn(seq_len, config.hidden_size), persistent=True
            )
        else:
            self.register_buffer("l_init", torch.zeros(0), persistent=False)

    # ------------------------------------------------------------------ #
    def initial_state(self, batch_size: int, device=None, dtype=None) -> LatentState:
        device = device or self.h_init.device
        dtype = dtype or self.h_init.dtype
        h = self.h_init.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        l = None
        if self.config.hierarchical:
            l = self.l_init.to(device=device, dtype=dtype).unsqueeze(0).expand(
                batch_size, -1, -1
            )
        return LatentState(h.contiguous(), None if l is None else l.contiguous())

    def _rope(self, seq_len: int):
        return None if self.rotary is None else self.rotary(seq_len)

    # ------------------------------------------------------------------ #
    def deterministic_update(self, state: LatentState, x_embed: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute ``u_t`` (and the refined ``l_t``) without adding noise."""
        rope = self._rope(state.h.shape[1])
        if self.config.hierarchical:
            l = state.l
            for _ in range(self.config.low_level_steps):
                l = self.f_L(l + state.h + x_embed, rope)
            u = self.f_H(state.h + l, rope)
            return u, l
        # Flat variant (Looped-Transformer style): K applications of the shared
        # stack per transition so that per-transition compute matches the
        # hierarchical model.
        u = state.h
        for _ in range(self.config.low_level_steps):
            u = self.f_H(u + x_embed, rope)
        return u, None

    def step(self, state: LatentState, x_embed: Tensor,
             y_embed: Optional[Tensor] = None, use_posterior: bool = False,
             temperature: float = 1.0) -> TransitionOutput:
        """One stochastic latent transition ``z_{t-1} -> z_t``."""
        u, l = self.deterministic_update(state, x_embed)
        h, prior, posterior = self.transition(
            u, y_embed=y_embed, use_posterior=use_posterior, temperature=temperature
        )
        return TransitionOutput(LatentState(h, l), prior, posterior)

    # ------------------------------------------------------------------ #
    def supervision_step(self, state: LatentState, x_embed: Tensor,
                         y_embed: Optional[Tensor] = None, use_posterior: bool = False,
                         temperature: float = 1.0, grad_last_only: bool = True,
                         collect_all: bool = False) -> Tuple[LatentState, List[TransitionOutput]]:
        """Run ``T`` transitions.

        With ``grad_last_only`` (the paper's truncated surrogate, Eq. 14) the
        first ``T-1`` transitions are executed under ``torch.no_grad`` and only
        the final one carries gradients.
        """
        outputs: List[TransitionOutput] = []
        n_steps = self.config.high_level_steps
        for t in range(n_steps):
            is_last = t == n_steps - 1
            no_grad = grad_last_only and not is_last and torch.is_grad_enabled()
            if no_grad:
                with torch.no_grad():
                    out = self.step(state, x_embed, y_embed, use_posterior, temperature)
                out = TransitionOutput(out.state.detach(), out.prior, out.posterior)
            else:
                out = self.step(state, x_embed, y_embed, use_posterior, temperature)
            state = out.state
            if collect_all or is_last:
                outputs.append(out)
        return state, outputs


__all__ = ["LatentState", "RecursiveCore", "TransitionOutput"]
