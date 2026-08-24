import pytest
import torch

from gram.config import ModelConfig, PatchEncoderConfig
from gram.model import GRAM


def make(**overrides) -> GRAM:
    base = dict(vocab_size=6, seq_len=12, hidden_size=32, num_heads=4,
                ffn_hidden_size=32, num_layers=1, low_level_steps=2,
                high_level_steps=2, n_supervision=3)
    base.update(overrides)
    return GRAM(ModelConfig(**base))


def test_forward_shapes():
    model = make()
    x = torch.randint(0, 6, (3, 12))
    y = torch.randint(0, 6, (3, 12))
    out = model(model.initial_state(3), model.encode_input(x), model.encode_target(y),
                use_posterior=True)
    assert out.logits.shape == (3, 12, 6)
    assert out.state.h.shape == (3, 12, 32)
    assert out.q_halt_logits.shape == (3,) and out.value.shape == (3,)


def test_initial_state_is_fixed_across_calls_and_saved():
    model = make()
    a = model.initial_state(2).h
    b = model.initial_state(2).h
    assert torch.equal(a, b)
    assert "core.h_init" in model.state_dict()


def test_transitions_are_stochastic_under_the_prior():
    model = make().eval()
    x = torch.randint(0, 6, (2, 12))
    x_embed = model.encode_input(x)
    with torch.no_grad():
        a = model(model.initial_state(2), x_embed).logits
        b = model(model.initial_state(2), x_embed).logits
    assert not torch.allclose(a, b)


def test_deterministic_variant_is_reproducible():
    model = make(guidance="none").eval()
    x = torch.randint(0, 6, (2, 12))
    x_embed = model.encode_input(x)
    with torch.no_grad():
        a = model(model.initial_state(2), x_embed).logits
        b = model(model.initial_state(2), x_embed).logits
    assert torch.allclose(a, b)


def test_flat_variant_has_no_low_level_state():
    model = make(hierarchical=False)
    x = torch.randint(0, 6, (2, 12))
    out = model(model.initial_state(2), model.encode_input(x))
    assert out.state.l is None
    assert model.core.f_L is None


def test_puzzle_prefix_is_dropped_by_the_decoder():
    model = make(puzzle_emb_tokens=4, num_puzzle_identifiers=3)
    x = torch.randint(0, 6, (2, 12))
    x_embed = model.encode_input(x, torch.tensor([1, 2]))
    assert x_embed.shape[1] == 16  # 12 content + 4 puzzle tokens
    out = model(model.initial_state(2), x_embed)
    assert out.logits.shape == (2, 12, 6)


def test_patch_encoder_roundtrip_shapes():
    patch = PatchEncoderConfig(image_size=8, patch_size=2, group_norm_groups=4)
    model = make(vocab_size=3, seq_len=16, patch_encoder=patch)
    x = torch.randint(0, 3, (2, 64))
    x_embed = model.encode_input(x)
    assert x_embed.shape == (2, 16, 32)
    out = model(model.initial_state(2), x_embed)
    assert out.logits.shape == (2, 64, 3)


def test_auxiliary_heads_do_not_touch_the_core():
    model = make()
    x = torch.randint(0, 6, (2, 12))
    out = model(model.initial_state(2), model.encode_input(x))
    (out.q_halt_logits.sum() + out.value.sum()).backward()
    core_grads = [p.grad for p in model.core.parameters() if p.grad is not None]
    assert all(g.abs().sum() == 0 for g in core_grads)
    assert model.q_head.weight.grad.abs().sum() > 0


def test_truncated_gradients_reach_only_the_last_transition():
    model = make(high_level_steps=3)
    x = torch.randint(0, 6, (2, 12))
    y = torch.randint(0, 6, (2, 12))
    out = model(model.initial_state(2), model.encode_input(x), model.encode_target(y),
                use_posterior=True, grad_last_only=True)
    assert len(out.transitions) == 1  # only the differentiated transition is kept
    out.logits.sum().backward()
    assert model.core.f_H.layers[0].mlp.down_proj.weight.grad.abs().sum() > 0


def test_collect_all_returns_every_transition():
    model = make(high_level_steps=3)
    x = torch.randint(0, 6, (2, 12))
    y = torch.randint(0, 6, (2, 12))
    out = model(model.initial_state(2), model.encode_input(x), model.encode_target(y),
                use_posterior=True, grad_last_only=False, collect_all=True)
    assert len(out.transitions) == 3


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        ModelConfig(hidden_size=10, num_heads=4)
    with pytest.raises(ValueError):
        ModelConfig(guidance="magic")


def test_mlp_mixer_forces_learned_positions():
    config = ModelConfig(mixer="mlp", pos_encoding="rope", seq_len=8)
    assert config.pos_encoding == "learned"
