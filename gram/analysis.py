"""Latent trajectory analysis (Appendix D.6).

The paper visualises how the high-level state ``h`` moves through latent space
over the recursion: a deterministic RRM traces a single path with no way to
escape a bad region, while GRAM spreads many trajectories, some of which reach
a valid solution (Figures 18-19).

This module collects those trajectories, projects them to 2-D with PCA, and
records the per-step loss so the projection can be overlaid on a loss landscape.
Everything is plain NumPy -- no plotting dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .model import GRAM


@dataclass
class TrajectoryTrace:
    """Latent states and per-step quality for a set of sampled trajectories."""

    states: np.ndarray  # [N, steps, D] pooled high-level state per step
    losses: np.ndarray  # [N, steps] token cross-entropy at each step
    accuracies: np.ndarray  # [N, steps] token accuracy at each step
    predictions: np.ndarray  # [N, steps, L] decoded tokens at each step

    @property
    def num_trajectories(self) -> int:
        return self.states.shape[0]

    @property
    def num_steps(self) -> int:
        return self.states.shape[1]


@torch.no_grad()
def collect_trajectories(model: GRAM, inputs: Tensor, targets: Optional[Tensor] = None,
                         puzzle_ids: Optional[Tensor] = None, num_samples: int = 50,
                         n_supervision: Optional[int] = None,
                         temperature: float = 1.0,
                         ignore_index: int = 0) -> TrajectoryTrace:
    """Sample ``num_samples`` prior trajectories for a *single* problem instance.

    ``inputs`` may be either ``[L]`` or ``[1, L]``.
    """
    model.eval()
    device = model.device
    if inputs.dim() == 1:
        inputs = inputs.unsqueeze(0)
    if inputs.shape[0] != 1:
        raise ValueError("collect_trajectories analyses one instance at a time")
    if targets is not None and targets.dim() == 1:
        targets = targets.unsqueeze(0)

    steps = n_supervision or model.config.n_supervision
    wide = inputs.to(device).repeat(num_samples, 1)
    wide_puzzle = (
        puzzle_ids.to(device).repeat(num_samples) if puzzle_ids is not None else None
    )
    x_embed = model.encode_input(wide, wide_puzzle)
    state = model.initial_state(num_samples)

    states: List[np.ndarray] = []
    losses: List[np.ndarray] = []
    accuracies: List[np.ndarray] = []
    predictions: List[np.ndarray] = []

    wide_targets = targets.to(device).repeat(num_samples, 1) if targets is not None else None
    for _ in range(steps):
        out = model(state, x_embed, use_posterior=False, temperature=temperature,
                    grad_last_only=False, with_heads=False)
        state = out.state
        # Pool over sequence positions so each step is one point in latent space.
        states.append(state.h.mean(dim=1).cpu().numpy())
        prediction = out.logits.argmax(-1)
        predictions.append(prediction.cpu().numpy())
        if wide_targets is not None:
            per_token = F.cross_entropy(
                out.logits.transpose(1, 2), wide_targets,
                ignore_index=ignore_index, reduction="none",
            )
            mask = (wide_targets != ignore_index).float()
            denominator = mask.sum(-1).clamp(min=1.0)
            losses.append(((per_token * mask).sum(-1) / denominator).cpu().numpy())
            accuracies.append(
                (((prediction == wide_targets).float() * mask).sum(-1) / denominator)
                .cpu().numpy()
            )
        else:
            losses.append(np.zeros(num_samples, dtype=np.float32))
            accuracies.append(np.zeros(num_samples, dtype=np.float32))

    return TrajectoryTrace(
        states=np.stack(states, axis=1),
        losses=np.stack(losses, axis=1),
        accuracies=np.stack(accuracies, axis=1),
        predictions=np.stack(predictions, axis=1),
    )


def pca_project(states: np.ndarray, n_components: int = 2
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Project ``[N, steps, D]`` latent states to ``[N, steps, n_components]``.

    Returns the projection and the explained-variance ratio of each component.
    """
    flat = states.reshape(-1, states.shape[-1]).astype(np.float64)
    centered = flat - flat.mean(axis=0, keepdims=True)
    # SVD is numerically kinder than forming the covariance matrix explicitly.
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    variance = singular ** 2 / max(1, len(centered) - 1)
    explained = variance / variance.sum() if variance.sum() > 0 else variance
    projected = centered @ components[:n_components].T
    return projected.reshape(*states.shape[:-1], n_components), explained[:n_components]


def trajectory_spread(projected: np.ndarray) -> np.ndarray:
    """Mean pairwise distance between trajectories at each recursion step.

    Zero at every step means the model is deterministic: all trajectories
    coincide, which is the collapse the paper illustrates in Figure 1(a).
    """
    num_trajectories, steps, _ = projected.shape
    if num_trajectories < 2:
        return np.zeros(steps)
    spread = np.empty(steps)
    for t in range(steps):
        points = projected[:, t]
        diff = points[:, None, :] - points[None, :, :]
        distances = np.sqrt((diff ** 2).sum(-1))
        iu = np.triu_indices(num_trajectories, k=1)
        spread[t] = distances[iu].mean()
    return spread


def render_svg(projected: np.ndarray, losses: np.ndarray, width: int = 720,
               height: int = 520, margin: int = 40) -> str:
    """Render the projected trajectories as a standalone SVG.

    Each trajectory is a polyline coloured by its final loss (blue = low,
    yellow = high), matching the colour convention of Figures 18-19.
    """
    points = projected.reshape(-1, 2)
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = np.maximum(hi - lo, 1e-8)

    def to_px(point: np.ndarray) -> Tuple[float, float]:
        norm = (point - lo) / span
        return (margin + norm[0] * (width - 2 * margin),
                height - margin - norm[1] * (height - 2 * margin))

    final = losses[:, -1]
    lo_loss, hi_loss = float(final.min()), float(final.max())
    loss_span = max(hi_loss - lo_loss, 1e-8)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-label="GRAM latent reasoning trajectories">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
    ]
    for i in range(projected.shape[0]):
        frac = (float(final[i]) - lo_loss) / loss_span
        red = int(40 + 215 * frac)
        green = int(90 + 130 * frac)
        blue = int(200 - 160 * frac)
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(to_px, projected[i]))
        parts.append(
            f'<polyline points="{coords}" fill="none" '
            f'stroke="rgb({red},{green},{blue})" stroke-width="1.4" opacity="0.75"/>'
        )
        start_x, start_y = to_px(projected[i, 0])
        end_x, end_y = to_px(projected[i, -1])
        parts.append(f'<circle cx="{start_x:.1f}" cy="{start_y:.1f}" r="2.5" fill="#f85149"/>')
        parts.append(f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="3" fill="#3fb950"/>')
    parts.append(
        f'<text x="{margin}" y="{height - 12}" fill="#8b949e" font-size="12" '
        f'font-family="system-ui, sans-serif">'
        f'{projected.shape[0]} trajectories x {projected.shape[1]} steps '
        f'(red = z_0, green = z_T, blue = low final loss)</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


__all__ = [
    "TrajectoryTrace",
    "collect_trajectories",
    "pca_project",
    "render_svg",
    "trajectory_spread",
]
