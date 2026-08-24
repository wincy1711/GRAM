import json

import numpy as np
import pytest
import torch

from gram.config import EvalConfig, ExperimentConfig, ModelConfig, TrainConfig, apply_overrides
from gram.data import graph_coloring as gc, nqueens as nq
from gram.data.base import PuzzleDataset
from gram.ema import ModelEMA
from gram.evaluate import evaluate, full_elbo, scaling_sweep
from gram.model import GRAM
from gram.train import Trainer, build_optimizer, lr_at, train_batch


@pytest.fixture
def nqueens_dir(tmp_path):
    nq.build(tmp_path / "nq6", n=6, remove=(3, 4), num_instances=40, seed=0)
    return tmp_path / "nq6"


def tiny_experiment(data_dir, output_dir, **model_overrides) -> ExperimentConfig:
    model = dict(vocab_size=3, seq_len=36, hidden_size=32, num_heads=4, ffn_hidden_size=32,
                 num_layers=1, low_level_steps=2, high_level_steps=2, n_supervision=2)
    model.update(model_overrides)
    return ExperimentConfig(
        name="test", task="nqueens", data_dir=str(data_dir), output_dir=str(output_dir),
        model=ModelConfig(**model),
        train=TrainConfig(epochs=2, batch_size=16, lr=1e-3, warmup_steps=1,
                          eval_interval=2, log_interval=1000, checkpoint_interval=0,
                          ema_decay=0.9, num_workers=0),
        eval=EvalConfig(num_samples=2, batch_size=16),
    )


def test_lr_schedule_warms_up_then_holds():
    config = TrainConfig(lr=1e-3, warmup_steps=10, lr_min_ratio=1.0)
    assert lr_at(0, config) == pytest.approx(1e-4)
    assert lr_at(9, config) == pytest.approx(1e-3)
    assert lr_at(100, config) == pytest.approx(1e-3)


def test_lr_schedule_cosine_decay():
    config = TrainConfig(lr=1e-3, warmup_steps=0, lr_min_ratio=0.1)
    config.total_steps = 100
    assert lr_at(0, config) == pytest.approx(1e-3, rel=1e-2)
    assert lr_at(99, config) < 2e-4


def test_optimizer_excludes_embeddings_from_weight_decay():
    model = GRAM(ModelConfig(vocab_size=5, seq_len=8, hidden_size=16, num_heads=4,
                             ffn_hidden_size=16, num_layers=1))
    optimizer = build_optimizer(model, TrainConfig(weight_decay=0.5))
    decayed = optimizer.param_groups[0]
    undecayed = optimizer.param_groups[1]
    assert decayed["weight_decay"] == 0.5 and undecayed["weight_decay"] == 0.0
    embedding = model.encoder.token_embed.weight
    assert any(p is embedding for p in undecayed["params"])


def test_train_batch_updates_parameters_and_reports_stats(nqueens_dir, tmp_path):
    config = tiny_experiment(nqueens_dir, tmp_path / "run")
    model = GRAM(config.model)
    optimizer = build_optimizer(model, config.train)
    dataset = PuzzleDataset(nqueens_dir, "train")
    batch = {k: v[:8] for k, v in {
        "inputs": dataset.inputs, "targets": dataset.targets,
        "puzzle_ids": dataset.puzzle_ids}.items()}
    before = model.core.f_H.layers[0].mlp.down_proj.weight.detach().clone()
    stats = train_batch(model, batch, config, optimizer, torch.device("cpu"), 0)
    after = model.core.f_H.layers[0].mlp.down_proj.weight
    assert not torch.allclose(before, after)
    assert stats.loss > 0 and stats.grad_norm > 0
    assert 0.0 <= stats.token_accuracy <= 1.0


def test_training_reduces_the_loss(nqueens_dir, tmp_path):
    config = tiny_experiment(nqueens_dir, tmp_path / "run")
    model = GRAM(config.model)
    optimizer = build_optimizer(model, config.train)
    dataset = PuzzleDataset(nqueens_dir, "train")
    batch = {"inputs": dataset.inputs[:16], "targets": dataset.targets[:16],
             "puzzle_ids": dataset.puzzle_ids[:16]}
    losses = [
        train_batch(model, batch, config, optimizer, torch.device("cpu"), 0).reconstruction
        for _ in range(25)
    ]
    assert np.mean(losses[-5:]) < np.mean(losses[:5])


def test_gradients_are_truncated_to_the_final_transition(nqueens_dir, tmp_path):
    """A longer recursion must not increase activation memory or graph depth."""
    config = tiny_experiment(nqueens_dir, tmp_path / "run", high_level_steps=4,
                             n_supervision=4)
    model = GRAM(config.model)
    x = torch.randint(1, 3, (2, 36))
    y = torch.randint(1, 3, (2, 36))
    out = model(model.initial_state(2), model.encode_input(x), model.encode_target(y),
                use_posterior=True, grad_last_only=True)
    # Only the last transition's distributions carry gradients.
    assert len(out.transitions) == 1
    assert out.transitions[0].posterior.mean.requires_grad


def test_trainer_fit_runs_and_checkpoints(nqueens_dir, tmp_path):
    config = tiny_experiment(nqueens_dir, tmp_path / "run")
    trainer = Trainer(config)
    metrics = trainer.fit()
    assert "exact_match" in metrics
    assert (tmp_path / "run" / "best.pt").exists()
    assert (tmp_path / "run" / "last.pt").exists()
    records = [json.loads(l) for l in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert any(r["split"] == "eval" for r in records)


def test_trainer_syncs_model_config_with_the_dataset(tmp_path):
    gc.build(tmp_path / "gc", n=6, num_instances=25, seed=1, min_solutions=2)
    config = tiny_experiment(tmp_path / "gc", tmp_path / "run")
    config.task = "graph_coloring"
    trainer = Trainer(config)
    assert trainer.model.config.vocab_size == gc.VOCAB_SIZE
    assert trainer.model.config.seq_len == 15


def test_checkpoint_round_trip(nqueens_dir, tmp_path):
    from gram.train import load_model
    config = tiny_experiment(nqueens_dir, tmp_path / "run")
    trainer = Trainer(config)
    trainer.fit()
    model = load_model(tmp_path / "run" / "best.pt")
    assert isinstance(model, GRAM)
    x = torch.randint(1, 3, (2, 36))
    with torch.no_grad():
        assert model(model.initial_state(2), model.encode_input(x)).logits.shape == (2, 36, 3)


def test_ema_tracks_a_lagged_average():
    model = GRAM(ModelConfig(vocab_size=3, seq_len=8, hidden_size=16, num_heads=4,
                             ffn_hidden_size=16, num_layers=1))
    ema = ModelEMA(model, decay=0.5)
    with torch.no_grad():
        model.decoder.head.weight.add_(1.0)
    ema.update(model)
    live = model.decoder.head.weight.detach().clone()
    with ema.average_parameters(model):
        averaged = model.decoder.head.weight.detach().clone()
    assert not torch.allclose(averaged, live)
    assert torch.allclose(model.decoder.head.weight, live)


def test_full_elbo_accumulates_every_transition(nqueens_dir, tmp_path):
    config = tiny_experiment(nqueens_dir, tmp_path / "run", n_supervision=3,
                             high_level_steps=2)
    model = GRAM(config.model)
    dataset = PuzzleDataset(nqueens_dir, "test")
    result = full_elbo(model, dataset, config, batch_size=4, max_batches=1)
    assert result["num_transitions"] == 6
    assert np.isfinite(result["neg_elbo"]) and result["kl"] >= 0


def test_evaluate_reports_task_metrics(nqueens_dir, tmp_path):
    config = tiny_experiment(nqueens_dir, tmp_path / "run")
    model = GRAM(config.model)
    metrics = evaluate(model, PuzzleDataset(nqueens_dir, "test"), config)
    for key in ("exact_match", "token_accuracy", "constraint_accuracy", "coverage"):
        assert key in metrics
    assert 0.0 <= metrics["exact_match"] <= 1.0


def test_scaling_sweep_covers_both_axes(nqueens_dir, tmp_path):
    config = tiny_experiment(nqueens_dir, tmp_path / "run")
    model = GRAM(config.model)
    results = scaling_sweep(model, PuzzleDataset(nqueens_dir, "test"), config,
                            widths=[1, 2], depths=[1, 2])
    assert len(results) == 4
    assert {(r["depth"], r["width"]) for r in results} == {(1, 1), (1, 2), (2, 1), (2, 2)}


def test_config_overrides_are_applied():
    config = ExperimentConfig()
    apply_overrides(config, {"train.epochs": 7, "model.guidance": "none",
                             "eval.num_samples": 9})
    assert config.train.epochs == 7
    assert config.model.guidance == "none"
    assert config.eval.num_samples == 9
    with pytest.raises(KeyError):
        apply_overrides(config, {"model.nope": 1})


def test_without_deep_supervision_the_recursion_still_runs_to_full_depth(
        nqueens_dir, tmp_path, monkeypatch):
    """The Looped-TF ablation applies the loss once, but at full depth."""
    calls = {"forward": 0, "backward": 0}
    config = tiny_experiment(nqueens_dir, tmp_path / "run", n_supervision=4,
                             deep_supervision=False)
    model = GRAM(config.model)
    optimizer = build_optimizer(model, config.train)

    original_forward = GRAM.forward

    def counting_forward(self, *args, **kwargs):
        calls["forward"] += 1
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(GRAM, "forward", counting_forward)
    original_backward = torch.Tensor.backward

    def counting_backward(self, *args, **kwargs):
        calls["backward"] += 1
        return original_backward(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", counting_backward)

    dataset = PuzzleDataset(nqueens_dir, "train")
    batch = {"inputs": dataset.inputs[:4], "targets": dataset.targets[:4],
             "puzzle_ids": dataset.puzzle_ids[:4]}
    train_batch(model, batch, config, optimizer, torch.device("cpu"), 0)

    # 4 supervision steps for the ELBO rollout + 4 for the prior head rollout.
    assert calls["forward"] == 8
    # One ELBO backward (not four) plus the auxiliary-head backward.
    assert calls["backward"] == 2


def test_deep_supervision_backwards_every_step(nqueens_dir, tmp_path, monkeypatch):
    calls = {"backward": 0}
    original_backward = torch.Tensor.backward

    def counting_backward(self, *args, **kwargs):
        calls["backward"] += 1
        return original_backward(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", counting_backward)
    config = tiny_experiment(nqueens_dir, tmp_path / "run", n_supervision=4)
    model = GRAM(config.model)
    optimizer = build_optimizer(model, config.train)
    dataset = PuzzleDataset(nqueens_dir, "train")
    batch = {"inputs": dataset.inputs[:4], "targets": dataset.targets[:4],
             "puzzle_ids": dataset.puzzle_ids[:4]}
    train_batch(model, batch, config, optimizer, torch.device("cpu"), 0)
    assert calls["backward"] == 5  # 4 supervision steps + auxiliary heads


def test_sequence_mixer_width_follows_the_dataset(tmp_path):
    """A stale auto-sized mixer would silently keep the config file's length."""
    gc.build(tmp_path / "gc", n=6, num_instances=25, seed=1, min_solutions=2)
    config = tiny_experiment(tmp_path / "gc", tmp_path / "run", mixer="mlp",
                             pos_encoding="learned")
    assert config.model.seq_mixer_hidden == 36  # from the config's seq_len
    trainer = Trainer(config)
    assert trainer.model.config.seq_len == 15
    assert trainer.model.config.seq_mixer_hidden == 15


def test_overrides_resize_the_sequence_mixer():
    config = ExperimentConfig(model=ModelConfig(mixer="mlp", seq_len=8))
    assert config.model.seq_mixer_hidden == 8
    apply_overrides(config, {"model.seq_len": 20})
    assert config.model.seq_mixer_hidden == 20
    apply_overrides(config, {"model.seq_len": 30, "model.seq_mixer_hidden": 64})
    assert config.model.seq_mixer_hidden == 64


def test_cosine_schedule_is_safe_without_total_steps():
    config = TrainConfig(lr=1e-3, warmup_steps=0, lr_min_ratio=0.1)
    assert lr_at(5, config) == pytest.approx(1e-3)
