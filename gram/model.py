"""The GRAM model: encoder, recursive core, decoder and auxiliary heads.

``GRAM.forward`` runs a single *supervision step* on a carried latent state, so
the training loop (``gram.train``) and the sampler (``gram.inference``) can both
drive the recursion one step at a time, keeping memory constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor, nn

from .config import ModelConfig, PatchEncoderConfig
from .core import LatentState, RecursiveCore, TransitionOutput
from .layers import CastedLinear, SwiGLU, trunc_normal_init_


# --------------------------------------------------------------------------- #
# Image patch encoder / decoder (Table 5)
# --------------------------------------------------------------------------- #
class PatchEncoder(nn.Module):
    """Conv stem + patchify for image tasks (binarised MNIST)."""

    def __init__(self, cfg: PatchEncoderConfig, hidden_size: int, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        c_mid = hidden_size // 2
        pad = cfg.conv_kernel // 2
        groups = min(cfg.group_norm_groups, c_mid)
        self.conv = nn.Sequential(
            nn.Conv2d(cfg.in_channels, c_mid, cfg.conv_kernel, padding=pad),
            nn.SiLU(),
            nn.GroupNorm(groups, c_mid),
            nn.Conv2d(c_mid, c_mid, cfg.conv_kernel, padding=pad),
            nn.SiLU(),
            nn.GroupNorm(groups, c_mid),
        )
        patch_dim = c_mid * cfg.patch_size ** 2
        self.proj = CastedLinear(patch_dim, hidden_size)
        self.num_patches = (cfg.image_size // cfg.patch_size) ** 2

    def forward(self, tokens: Tensor) -> Tensor:
        """``tokens`` are pixel token ids in ``[0, vocab)`` flattened row-major."""
        cfg = self.cfg
        bsz = tokens.shape[0]
        img = tokens.view(bsz, cfg.in_channels, cfg.image_size, cfg.image_size).float()
        # Map token ids {pad=0, black=1, white=2} to the range [-1, 1].
        img = (img - 1.0).clamp(-1.0, 1.0)
        feat = self.conv(img)  # [B, D/2, H, W]
        p = cfg.patch_size
        c = feat.shape[1]
        feat = feat.view(bsz, c, cfg.image_size // p, p, cfg.image_size // p, p)
        feat = feat.permute(0, 2, 4, 1, 3, 5).reshape(bsz, self.num_patches, c * p * p)
        return self.proj(feat)


class PatchDecoder(nn.Module):
    """Unpatchify head producing per-pixel logits."""

    def __init__(self, cfg: PatchEncoderConfig, hidden_size: int, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.proj = CastedLinear(hidden_size, cfg.patch_size ** 2 * vocab_size)

    def forward(self, hidden: Tensor) -> Tensor:
        cfg = self.cfg
        bsz, n_patches, _ = hidden.shape
        p, v = cfg.patch_size, self.vocab_size
        side = cfg.image_size // p
        out = self.proj(hidden).view(bsz, side, side, p, p, v)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(bsz, cfg.image_size * cfg.image_size, v)
        return out


# --------------------------------------------------------------------------- #
# Encoder / decoder
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    """Token (or patch) embedding + optional puzzle prefix + position encoding."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.scale = math.sqrt(config.hidden_size)

        if config.patch_encoder is not None and config.patch_encoder.enabled:
            self.patch_encoder = PatchEncoder(
                config.patch_encoder, config.hidden_size, config.vocab_size
            )
            self.token_embed = None
        else:
            self.patch_encoder = None
            self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
            trunc_normal_init_(self.token_embed.weight, std=1.0)

        if config.puzzle_emb_tokens > 0:
            self.puzzle_embed = nn.Embedding(
                config.num_puzzle_identifiers,
                config.puzzle_emb_tokens * config.hidden_size,
            )
            nn.init.zeros_(self.puzzle_embed.weight)
        else:
            self.puzzle_embed = None

        if config.pos_encoding == "learned":
            self.pos_embed = nn.Parameter(
                torch.empty(config.total_seq_len, config.hidden_size)
            )
            trunc_normal_init_(self.pos_embed, std=1.0 / self.scale)
        else:
            self.pos_embed = None

    def forward(self, inputs: Tensor, puzzle_ids: Optional[Tensor] = None) -> Tensor:
        cfg = self.config
        if self.patch_encoder is not None:
            embed = self.patch_encoder(inputs)
        else:
            embed = self.token_embed(inputs)
        embed = self.scale * embed

        if self.puzzle_embed is not None:
            if puzzle_ids is None:
                puzzle_ids = inputs.new_zeros(inputs.shape[0], dtype=torch.long)
            prefix = self.puzzle_embed(puzzle_ids).view(
                -1, cfg.puzzle_emb_tokens, cfg.hidden_size
            )
            embed = torch.cat((prefix, embed), dim=1)

        if self.pos_embed is not None:
            embed = embed + self.pos_embed.to(embed.dtype)
        return embed


class Decoder(nn.Module):
    """Maps the terminal high-level state to output logits (puzzle prefix dropped)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        if config.patch_encoder is not None and config.patch_encoder.enabled:
            self.head = PatchDecoder(
                config.patch_encoder, config.hidden_size, config.vocab_size
            )
        elif config.lm_head == "swiglu":
            self.head = SwiGLU(config.hidden_size, config.ffn_hidden_size,
                               config.vocab_size)
        else:
            self.head = CastedLinear(config.hidden_size, config.vocab_size)

    def forward(self, h: Tensor) -> Tensor:
        content = h[:, self.config.puzzle_emb_tokens:]
        return self.head(content)


# --------------------------------------------------------------------------- #
# Model output containers
# --------------------------------------------------------------------------- #
@dataclass
class StepOutput:
    """Everything produced by one supervision step."""

    logits: Tensor
    state: LatentState
    transitions: List[TransitionOutput]
    q_halt_logits: Optional[Tensor] = None
    q_continue_logits: Optional[Tensor] = None
    value: Optional[Tensor] = None


# --------------------------------------------------------------------------- #
# GRAM
# --------------------------------------------------------------------------- #
class GRAM(nn.Module):
    """Generative Recursive Reasoning Model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.core = RecursiveCore(config)
        self.decoder = Decoder(config)

        # Target encoder feeding the variational posterior q_phi(eps | u, y).
        if config.is_stochastic:
            self.target_embed = nn.Embedding(config.vocab_size, config.hidden_size)
            trunc_normal_init_(self.target_embed.weight, std=1.0)
            self.target_proj = (
                SwiGLU(config.hidden_size, config.ffn_hidden_size, config.hidden_size)
                if config.posterior_target_proj else None
            )
        else:
            self.target_embed = None
            self.target_proj = None

        # Auxiliary heads operate on detached latents (Appendix A.1/A.2), so they
        # never propagate gradients into the recursive core.
        self.q_head = CastedLinear(config.hidden_size, 2, zero_init=True) if config.use_act else None
        self.v_head = CastedLinear(config.hidden_size, 1, zero_init=True) if config.use_lprm else None

    # ------------------------------------------------------------------ #
    @property
    def device(self) -> torch.device:
        return self.core.h_init.device

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    # ------------------------------------------------------------------ #
    def encode_input(self, inputs: Tensor, puzzle_ids: Optional[Tensor] = None) -> Tensor:
        return self.encoder(inputs, puzzle_ids)

    def encode_target(self, targets: Tensor) -> Optional[Tensor]:
        """Embed ``y`` for the posterior; zero-padded over the puzzle prefix."""
        if self.target_embed is None:
            return None
        embed = self.target_embed(targets)
        if self.target_proj is not None:
            embed = self.target_proj(embed)
        prefix = self.config.puzzle_emb_tokens
        if prefix > 0:
            pad = embed.new_zeros(embed.shape[0], prefix, embed.shape[-1])
            embed = torch.cat((pad, embed), dim=1)
        if self.config.patch_encoder is not None and self.config.patch_encoder.enabled:
            # Pool pixel-level target embeddings down to the patch grid.
            cfg = self.config.patch_encoder
            bsz, _, dim = embed.shape
            p, side = cfg.patch_size, cfg.image_size // cfg.patch_size
            embed = embed.view(bsz, side, p, side, p, dim).mean(dim=(2, 4))
            embed = embed.reshape(bsz, side * side, dim)
        return embed

    def initial_state(self, batch_size: int) -> LatentState:
        return self.core.initial_state(batch_size, device=self.device)

    # ------------------------------------------------------------------ #
    def forward(self, state: LatentState, x_embed: Tensor,
                y_embed: Optional[Tensor] = None, use_posterior: bool = False,
                temperature: float = 1.0, grad_last_only: bool = True,
                collect_all: bool = False, with_heads: bool = True) -> StepOutput:
        """Run one supervision step (``T`` latent transitions) and decode."""
        state, transitions = self.core.supervision_step(
            state,
            x_embed,
            y_embed=y_embed,
            use_posterior=use_posterior,
            temperature=temperature,
            grad_last_only=grad_last_only,
            collect_all=collect_all,
        )
        logits = self.decoder(state.h)

        q_halt = q_continue = value = None
        if with_heads:
            summary = state.h[:, 0].detach()
            if self.q_head is not None:
                q = self.q_head(summary)
                q_halt, q_continue = q[..., 0], q[..., 1]
            if self.v_head is not None:
                value = self.v_head(summary).squeeze(-1)
        return StepOutput(logits, state, transitions, q_halt, q_continue, value)


__all__ = ["Decoder", "Encoder", "GRAM", "PatchDecoder", "PatchEncoder", "StepOutput"]
