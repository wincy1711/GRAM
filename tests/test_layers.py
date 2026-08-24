import torch

from gram.layers import (
    BlockStack,
    RotaryEmbedding,
    SelfAttention,
    SequenceSwiGLU,
    SwiGLU,
    apply_rotary_pos_emb,
    rms_norm,
)


def test_rms_norm_unit_scale():
    x = torch.randn(4, 7, 16) * 5.0
    out = rms_norm(x)
    assert torch.allclose(out.pow(2).mean(-1), torch.ones(4, 7), atol=1e-3)


def test_rms_norm_is_scale_invariant():
    x = torch.randn(2, 3, 8)
    assert torch.allclose(rms_norm(x), rms_norm(x * 10.0), atol=1e-4)


def test_rope_preserves_norm_and_relative_structure():
    rope = RotaryEmbedding(8, 16)
    cos, sin = rope(5)
    q = torch.randn(1, 2, 5, 8)
    k = torch.randn(1, 2, 5, 8)
    q_r, k_r = apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.allclose(q_r.norm(dim=-1), q.norm(dim=-1), atol=1e-4)
    # A shared shift leaves the dot product between two positions unchanged.
    base = (q_r[0, 0, 1] * k_r[0, 0, 3]).sum()
    cos2, sin2 = rope(6)
    q2, k2 = apply_rotary_pos_emb(
        torch.cat([torch.zeros(1, 2, 1, 8), q], dim=2),
        torch.cat([torch.zeros(1, 2, 1, 8), k], dim=2),
        cos2, sin2,
    )
    shifted = (q2[0, 0, 2] * k2[0, 0, 4]).sum()
    assert torch.allclose(base, shifted, atol=1e-4)


def test_attention_is_permutation_equivariant():
    attn = SelfAttention(16, 4).eval()
    x = torch.randn(1, 6, 16)
    perm = torch.randperm(6)
    with torch.no_grad():
        a = attn(x)[:, perm]
        b = attn(x[:, perm])
    assert torch.allclose(a, b, atol=1e-5)


def test_swiglu_shapes():
    layer = SwiGLU(8, 16, 4)
    assert layer(torch.randn(3, 5, 8)).shape == (3, 5, 4)


def test_sequence_swiglu_mixes_positions():
    layer = SequenceSwiGLU(5, 10)
    x = torch.zeros(1, 5, 4)
    x[0, 0] = 1.0
    out = layer(x)
    # Information from position 0 must reach the other positions.
    assert out[0, 1:].abs().sum() > 0


def test_block_stack_variants_run():
    for mixer in ("attention", "mlp"):
        stack = BlockStack(2, 16, 4, 16, mixer, seq_len=6, seq_mixer_hidden=6)
        rope = RotaryEmbedding(4, 6)(6) if mixer == "attention" else None
        assert stack(torch.randn(2, 6, 16), rope).shape == (2, 6, 16)


def test_no_biases_anywhere():
    stack = BlockStack(2, 16, 4, 16, "attention", seq_len=6)
    assert all("bias" not in name for name, _ in stack.named_parameters())
