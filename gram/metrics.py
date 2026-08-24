"""Task metrics.

* structured reasoning -- exact-match and token accuracy (Section 4.1);
* multi-solution tasks -- constraint-satisfaction accuracy, solution coverage
  and conflict-edge count (Section 4.2, Table 1);
* generation -- Sudoku validity/uniqueness and IS/FID for binarised MNIST
  (Section 4.3).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from .data import graph_coloring as gc
from .data import nqueens as nq
from .data import sudoku as sd
from .data.base import SolutionIndex


# --------------------------------------------------------------------------- #
# Generic
# --------------------------------------------------------------------------- #
def exact_match(predictions: Tensor, targets: Tensor,
                ignore_index: Optional[int] = None) -> Tensor:
    """Fraction of sequences predicted exactly right."""
    if ignore_index is None:
        correct = predictions == targets
    else:
        mask = targets != ignore_index
        correct = (predictions == targets) | ~mask
    return correct.all(dim=-1).float()


def token_accuracy(predictions: Tensor, targets: Tensor,
                   ignore_index: Optional[int] = None) -> Tensor:
    """Per-sequence fraction of correctly predicted tokens."""
    if ignore_index is None:
        return (predictions == targets).float().mean(dim=-1)
    mask = targets != ignore_index
    correct = ((predictions == targets) & mask).float().sum(dim=-1)
    total = mask.float().sum(dim=-1).clamp(min=1.0)
    return correct / total


# --------------------------------------------------------------------------- #
# Constraint checks per task
# --------------------------------------------------------------------------- #
def check_predictions(task: str, predictions: np.ndarray, inputs: np.ndarray,
                      task_kwargs: Optional[Dict] = None) -> np.ndarray:
    """Boolean array: does each prediction satisfy the task's constraints?"""
    task_kwargs = task_kwargs or {}
    out = np.zeros(len(predictions), dtype=bool)
    if task == "nqueens":
        n = int(task_kwargs.get("board_size", int(round(predictions.shape[-1] ** 0.5))))
        for i, pred in enumerate(predictions):
            out[i] = nq.check_solution(pred, n, inputs[i])
    elif task == "graph_coloring":
        n = int(task_kwargs.get("num_vertices", predictions.shape[-1]))
        num_colors = int(task_kwargs.get("num_colors", gc.NUM_COLORS))
        for i, pred in enumerate(predictions):
            edges = gc.tokens_to_edges(inputs[i], n)
            coloring = gc.tokens_to_coloring(pred, n)
            out[i] = gc.is_valid_coloring(coloring, edges, num_colors)
    elif task == "sudoku":
        for i, pred in enumerate(predictions):
            grid = sd.tokens_to_grid(pred)
            puzzle = sd.tokens_to_grid(inputs[i])
            out[i] = sd.is_complete_valid(grid) and sd.matches_clues(grid, puzzle)
    else:
        raise ValueError(f"no constraint checker for task {task!r}")
    return out


def conflict_edges(predictions: np.ndarray, inputs: np.ndarray, n: int) -> np.ndarray:
    """Number of constraint-violating edges per graph-colouring prediction."""
    out = np.zeros(len(predictions), dtype=np.int64)
    for i, pred in enumerate(predictions):
        edges = gc.tokens_to_edges(inputs[i], n)
        coloring = gc.tokens_to_coloring(pred, n)
        # Out-of-range colours count as violating every incident edge.
        invalid = {v for v, c in enumerate(coloring) if c < 0 or c >= gc.NUM_COLORS}
        bad = sum(1 for i_, j_ in edges if i_ in invalid or j_ in invalid)
        out[i] = gc.count_conflicts(
            [c if 0 <= c < gc.NUM_COLORS else -1 for c in coloring], edges
        ) + bad
    return out


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #
@dataclass
class CoverageResult:
    coverage: float
    per_group: Dict[int, float] = field(default_factory=dict)
    num_solutions: Dict[int, int] = field(default_factory=dict)

    def by_solution_count(self) -> Dict[int, float]:
        """Mean coverage bucketed by the number of ground-truth solutions."""
        buckets: Dict[int, List[float]] = defaultdict(list)
        for gid, cov in self.per_group.items():
            buckets[self.num_solutions.get(gid, 0)].append(cov)
        return {k: float(np.mean(v)) for k, v in sorted(buckets.items())}


def solution_coverage(predictions: np.ndarray, group_ids: Sequence[int],
                      index: SolutionIndex, out_seq_len: Optional[int] = None
                      ) -> CoverageResult:
    """Fraction of ground-truth solutions recovered per input.

    ``predictions`` has shape ``[B, N, L]``: ``N`` parallel samples per input
    (Table 1 uses ``N = 20``).
    """
    per_group: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for i, gid in enumerate(group_ids):
        gid = int(gid)
        solutions = index.get(gid)
        if solutions is None or len(solutions) == 0:
            continue
        cut = out_seq_len
        valid = {np.asarray(s, dtype=np.int32)[:cut].tobytes() for s in solutions}
        found = {
            key for key in (
                np.asarray(p, dtype=np.int32)[:cut].tobytes() for p in predictions[i]
            ) if key in valid
        }
        per_group[gid] = len(found) / len(valid)
        counts[gid] = len(valid)
    coverage = float(np.mean(list(per_group.values()))) if per_group else 0.0
    return CoverageResult(coverage, per_group, counts)


def unique_valid_fraction(predictions: np.ndarray) -> float:
    """Fraction of distinct sequences among the given predictions."""
    if len(predictions) == 0:
        return 0.0
    keys = {np.asarray(p, dtype=np.int32).tobytes() for p in predictions}
    return len(keys) / len(predictions)


# --------------------------------------------------------------------------- #
# Sudoku generation
# --------------------------------------------------------------------------- #
def sudoku_validity(samples: np.ndarray) -> Dict[str, float]:
    """Validity rate and uniqueness of unconditionally generated boards."""
    grids = [sd.tokens_to_grid(s) for s in samples]
    valid = np.asarray([sd.is_complete_valid(g) for g in grids])
    valid_keys = {g.tobytes() for g, ok in zip(grids, valid) if ok}
    return {
        "validity": float(valid.mean()) if len(valid) else 0.0,
        "unique_valid": float(len(valid_keys)) / max(1, int(valid.sum())),
        "mean_violations": float(np.mean([sd.num_violations(g) for g in grids]))
        if grids else 0.0,
    }


# --------------------------------------------------------------------------- #
# Image generation (IS / FID)
# --------------------------------------------------------------------------- #
def inception_score(probabilities: np.ndarray, splits: int = 10) -> float:
    """Inception Score from a classifier's softmax outputs."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(probabilities) == 0:
        return 0.0
    scores: List[float] = []
    chunk = max(1, len(probabilities) // splits)
    for start in range(0, len(probabilities), chunk):
        part = probabilities[start: start + chunk]
        if len(part) == 0:
            continue
        marginal = part.mean(axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-12) - np.log(marginal + 1e-12))
        scores.append(float(np.exp(kl.sum(axis=1).mean())))
    return float(np.mean(scores)) if scores else 0.0


def frechet_distance(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """Frechet distance between two Gaussians fitted to the given features."""
    a = np.asarray(features_a, dtype=np.float64)
    b = np.asarray(features_b, dtype=np.float64)
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    sigma_a = np.cov(a, rowvar=False)
    sigma_b = np.cov(b, rowvar=False)
    diff = mu_a - mu_b
    covmean = _sqrtm_psd(sigma_a @ sigma_b)
    return float(diff @ diff + np.trace(sigma_a) + np.trace(sigma_b) - 2 * np.trace(covmean))


def _sqrtm_psd(matrix: np.ndarray) -> np.ndarray:
    """Matrix square root via a symmetrised eigendecomposition."""
    matrix = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T


__all__ = [
    "CoverageResult",
    "check_predictions",
    "conflict_edges",
    "exact_match",
    "frechet_distance",
    "inception_score",
    "solution_coverage",
    "sudoku_validity",
    "token_accuracy",
    "unique_valid_fraction",
]
