"""Transformer building blocks used by the GRAM recursive core.

The backbone follows the HRM/TRM lineage that GRAM builds on: post-norm blocks
with a parameter-free RMSNorm, rotary position embeddings, SwiGLU MLPs and no
biases anywhere.  For fixed-length symbolic tasks such as Sudoku the paper
replaces self-attention with a second SwiGLU acting along the sequence axis
(``[SwiGLU + SwiGLU]`` instead of ``[Attention + SwiGLU]``, Appendix B.1); that
variant is available through ``mixer="mlp"``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# --------------------------------------------------------------------------- #
# Initialisation
# --------------------------------------------------------------------------- #
def trunc_normal_init_(tensor: Tensor, std: float = 1.0, lower: float = -2.0,
                       upper: float = 2.0) -> Tensor:
    """Truncated normal initialisation (matches the HRM/TRM reference code)."""
    if std == 0:
        return nn.init.zeros_(tensor)
    with torch.no_grad():
        nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=lower * std, b=upper * std)
    return tensor


class CastedLinear(nn.Linear):
    """``nn.Linear`` without bias and with fan-in truncated-normal init."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 zero_init: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        if zero_init:
            nn.init.zeros_(self.weight)
        else:
            trunc_normal_init_(self.weight, std=1.0 / math.sqrt(in_features))
        if bias:
            nn.init.zeros_(self.bias)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def rms_norm(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Parameter-free RMS normalisation over the last dimension."""
    dtype = x.dtype
    x = x.float()
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return out.to(dtype)


# --------------------------------------------------------------------------- #
# Rotary position embedding
# --------------------------------------------------------------------------- #
class RotaryEmbedding(nn.Module):
    """Precomputed rotary embeddings (RoPE, Su et al. 2024)."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[Tensor, Tensor]:
        return self.cos[:seq_len], self.sin[:seq_len]


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
                         ) -> Tuple[Tensor, Tensor]:
    """Apply RoPE to ``q``/``k`` of shape ``[B, H, L, Dh]``."""
    cos = torch.cat((cos, cos), dim=-1)[None, None]  # [1, 1, L, Dh]
    sin = torch.cat((sin, sin), dim=-1)[None, None]
    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return q_out.to(q.dtype), k_out.to(k.dtype)


# --------------------------------------------------------------------------- #
# Attention / MLP mixers
# --------------------------------------------------------------------------- #
class SelfAttention(nn.Module):
    """Bidirectional multi-head self-attention (no causal mask)."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv_proj = CastedLinear(hidden_size, 3 * hidden_size)
        self.o_proj = CastedLinear(hidden_size, hidden_size)

    def forward(self, x: Tensor, rope: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each [B, H, L, Dh]
        if rope is not None:
            q, k = apply_rotary_pos_emb(q, k, rope[0], rope[1])
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Shazeer, 2020)."""

    def __init__(self, in_features: int, hidden_features: int,
                 out_features: Optional[int] = None, zero_init_out: bool = False):
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        self.gate_up_proj = CastedLinear(in_features, 2 * hidden_features)
        self.down_proj = CastedLinear(hidden_features, out_features, zero_init=zero_init_out)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class SequenceSwiGLU(nn.Module):
    """SwiGLU applied along the *sequence* axis (attention-free token mixing).

    Used for the Sudoku recursive core, where the paper replaces
    ``[Attention + SwiGLU]`` with ``[SwiGLU + SwiGLU]``.
    """

    def __init__(self, seq_len: int, hidden_features: int):
        super().__init__()
        self.seq_len = seq_len
        self.mix = SwiGLU(seq_len, hidden_features, seq_len)

    def forward(self, x: Tensor, rope: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        del rope  # positions are handled by learned embeddings in this variant
        if x.shape[1] != self.seq_len:
            raise ValueError(
                f"SequenceSwiGLU expects sequence length {self.seq_len}, got {x.shape[1]}"
            )
        return self.mix(x.transpose(1, 2)).transpose(1, 2)


class Block(nn.Module):
    """Post-norm transformer block: ``x = N(x + mix(x)); x = N(x + ffn(x))``."""

    def __init__(self, hidden_size: int, num_heads: int, ffn_hidden_size: int,
                 mixer: str = "attention", seq_len: Optional[int] = None,
                 seq_mixer_hidden: Optional[int] = None, norm_eps: float = 1e-5):
        super().__init__()
        self.norm_eps = norm_eps
        if mixer == "attention":
            self.mixer = SelfAttention(hidden_size, num_heads)
        elif mixer == "mlp":
            if seq_len is None:
                raise ValueError("mixer='mlp' requires seq_len")
            self.mixer = SequenceSwiGLU(seq_len, seq_mixer_hidden or seq_len)
        else:
            raise ValueError(f"unknown mixer {mixer!r}")
        self.mlp = SwiGLU(hidden_size, ffn_hidden_size)

    def forward(self, x: Tensor, rope: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        x = rms_norm(x + self.mixer(x, rope), eps=self.norm_eps)
        x = rms_norm(x + self.mlp(x), eps=self.norm_eps)
        return x


class BlockStack(nn.Module):
    """A stack of ``num_layers`` blocks; one application = one ``f_L``/``f_H`` call."""

    def __init__(self, num_layers: int, hidden_size: int, num_heads: int,
                 ffn_hidden_size: int, mixer: str = "attention",
                 seq_len: Optional[int] = None, seq_mixer_hidden: Optional[int] = None):
        super().__init__()
        self.layers = nn.ModuleList(
            Block(hidden_size, num_heads, ffn_hidden_size, mixer, seq_len, seq_mixer_hidden)
            for _ in range(num_layers)
        )

    def forward(self, x: Tensor, rope: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, rope)
        return x


__all__ = [
    "Block",
    "BlockStack",
    "CastedLinear",
    "RotaryEmbedding",
    "SelfAttention",
    "SequenceSwiGLU",
    "SwiGLU",
    "apply_rotary_pos_emb",
    "rms_norm",
    "trunc_normal_init_",
]
