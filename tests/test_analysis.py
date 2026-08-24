"""Tests for the latent trajectory analysis of Appendix D.6."""

import numpy as np
import torch

from gram.analysis import collect_trajectories, pca_project, render_svg, trajectory_spread
from gram.config import ModelConfig
from gram.model import GRAM


def make(**overrides) -> GRAM:
    base = dict(vocab_size=5, seq_len=12, hidden_size=32, num_heads=4, ffn_hidden_size=32,
                num_layers=1, low_level_steps=1, high_level_steps=2, n_supervision=5)
    base.update(overrides)
    return GRAM(ModelConfig(**base))


def test_collect_trajectories_shapes():
    model = make()
    x = torch.randint(1, 5, (12,))
    y = torch.randint(1, 5, (12,))
    trace = collect_trajectories(model, x, y, num_samples=7)
    assert trace.states.shape == (7, 5, 32)
    assert trace.losses.shape == (7, 5)
    assert trace.accuracies.shape == (7, 5)
    assert trace.predictions.shape == (7, 5, 12)
    assert trace.num_trajectories == 7 and trace.num_steps == 5


def test_losses_and_accuracies_are_consistent():
    model = make()
    x = torch.randint(1, 5, (12,))
    y = torch.randint(1, 5, (12,))
    trace = collect_trajectories(model, x, y, num_samples=4)
    assert np.all(trace.losses >= 0)
    assert np.all((trace.accuracies >= 0) & (trace.accuracies <= 1))
    # A perfect step would have to be the one with the lowest loss.
    best_step = np.argmin(trace.losses[0])
    assert trace.accuracies[0, best_step] >= trace.accuracies[0].min()


def test_analysis_works_without_targets():
    model = make()
    trace = collect_trajectories(model, torch.randint(1, 5, (12,)), num_samples=3)
    assert np.all(trace.losses == 0)


def test_pca_projection_is_centred_and_ordered():
    states = np.random.default_rng(0).normal(size=(20, 6, 16))
    projected, explained = pca_project(states)
    assert projected.shape == (20, 6, 2)
    assert np.allclose(projected.reshape(-1, 2).mean(axis=0), 0, atol=1e-8)
    assert explained[0] >= explained[1]
    assert 0 <= explained.sum() <= 1 + 1e-9


def test_pca_recovers_a_planted_two_dimensional_structure():
    rng = np.random.default_rng(1)
    latent = rng.normal(size=(30, 4, 2)) * np.array([10.0, 5.0])
    basis = np.linalg.qr(rng.normal(size=(16, 16)))[0][:, :2]
    states = latent @ basis.T
    _, explained = pca_project(states)
    assert explained.sum() > 0.99  # two components explain essentially everything


def test_spread_is_zero_for_a_deterministic_model():
    model = make(guidance="none")
    x = torch.randint(1, 5, (12,))
    trace = collect_trajectories(model, x, num_samples=6)
    projected, _ = pca_project(trace.states)
    assert np.allclose(trajectory_spread(projected), 0.0, atol=1e-6)
    # Every trajectory decodes to the same answer -- the collapse of Figure 1(a).
    assert len({row.tobytes() for row in trace.predictions[:, -1]}) == 1


def test_spread_is_positive_for_a_stochastic_model():
    model = make()
    trace = collect_trajectories(model, torch.randint(1, 5, (12,)), num_samples=12)
    projected, _ = pca_project(trace.states)
    assert trajectory_spread(projected).min() > 0


def test_spread_of_a_single_trajectory_is_zero():
    projected = np.random.default_rng(0).normal(size=(1, 5, 2))
    assert np.array_equal(trajectory_spread(projected), np.zeros(5))


def test_render_svg_is_well_formed():
    projected = np.random.default_rng(0).normal(size=(5, 4, 2))
    losses = np.random.default_rng(1).random((5, 4))
    svg = render_svg(projected, losses)
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert svg.count("<polyline") == 5
    assert svg.count("<circle") == 10  # a start and an end marker per trajectory
    assert "NaN" not in svg


def test_render_svg_handles_degenerate_extents():
    projected = np.zeros((3, 4, 2))
    losses = np.zeros((3, 4))
    svg = render_svg(projected, losses)
    assert "NaN" not in svg and "inf" not in svg
