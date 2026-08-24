"""Evaluation: reasoning accuracy, multi-solution coverage and the full ELBO.

``evaluate`` computes the metrics reported in the paper for whichever task the
dataset declares.  ``full_elbo`` implements the trajectory-level bound of
Eq. (13) — summing KL contributions over all ``T_total = T * N_sup``
transitions — which Appendix A.3 uses to verify that the truncated surrogate
of Eq. (14) really does improve the full variational bound.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import EvalConfig, ExperimentConfig
from .data.base import PuzzleDataset, SolutionIndex
from .inference import sample_trajectories, select
from .losses import balanced_kl, reconstruction_loss
from .metrics import (
    check_predictions,
    conflict_edges,
    exact_match,
    solution_coverage,
    sudoku_validity,
    token_accuracy,
    unique_valid_fraction,
)
from .model import GRAM


@torch.no_grad()
def evaluate(model: GRAM, dataset: PuzzleDataset, config: ExperimentConfig,
             eval_config: Optional[EvalConfig] = None) -> Dict[str, float]:
    """Run the full evaluation suite for ``dataset``'s task."""
    eval_cfg = eval_config or config.eval
    device = model.device
    model.eval()
    loader = DataLoader(dataset, batch_size=eval_cfg.batch_size, shuffle=False)
    metadata = dataset.metadata
    ignore_index = metadata.ignore_label_id

    all_predictions: List[np.ndarray] = []
    all_samples: List[np.ndarray] = []
    all_inputs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_groups: List[np.ndarray] = []
    halt_steps: List[np.ndarray] = []

    for batch in loader:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        puzzle_ids = batch["puzzle_ids"].to(device)
        output = sample_trajectories(
            model, inputs, puzzle_ids,
            num_samples=eval_cfg.num_samples,
            n_supervision=eval_cfg.n_supervision,
            use_act=eval_cfg.use_act,
            temperature=eval_cfg.temperature,
            sample_tokens=eval_cfg.sample_tokens,
        )
        prediction = select(output, eval_cfg.selection)
        all_predictions.append(prediction.cpu().numpy())
        all_samples.append(output.predictions.cpu().numpy())
        all_inputs.append(inputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_groups.append(batch["group_ids"].numpy())
        halt_steps.append(output.halt_steps.cpu().numpy())

    predictions = np.concatenate(all_predictions)
    samples = np.concatenate(all_samples)
    inputs = np.concatenate(all_inputs)
    targets = np.concatenate(all_targets)
    groups = np.concatenate(all_groups)

    # Tasks whose output is shorter than the decoded sequence (e.g. graph
    # colouring, where the input is an adjacency triangle) are compared only
    # over the real output positions.
    out_len = min(metadata.out_seq_len, predictions.shape[-1])
    pred_t = torch.from_numpy(predictions[:, :out_len])
    target_t = torch.from_numpy(targets[:, :out_len])
    metrics: Dict[str, float] = {
        "exact_match": float(exact_match(pred_t, target_t, ignore_index).mean()),
        "token_accuracy": float(token_accuracy(pred_t, target_t, ignore_index).mean()),
        "mean_halt_step": float(np.concatenate(halt_steps).mean()),
        "num_samples": float(eval_cfg.num_samples),
    }

    task = metadata.task
    extra = metadata.extra or {}
    if task in ("nqueens", "graph_coloring", "sudoku"):
        valid = check_predictions(task, predictions, inputs, extra)
        metrics["constraint_accuracy"] = float(valid.mean())
    if task == "graph_coloring":
        n = int(extra.get("num_vertices", targets.shape[-1]))
        metrics["conflict_edges"] = float(conflict_edges(predictions, inputs, n).mean())
    if task == "sudoku" and extra.get("unconditional"):
        metrics.update(
            {f"gen_{k}": v for k, v in sudoku_validity(predictions).items()}
        )

    index = SolutionIndex(dataset.directory / "solutions.npz")
    if index.available and eval_cfg.num_samples > 1:
        coverage = solution_coverage(samples, groups, index, out_len)
        metrics["coverage"] = coverage.coverage
        metrics["mean_num_solutions"] = float(
            np.mean(list(coverage.num_solutions.values())) if coverage.num_solutions else 0.0
        )
    metrics["sample_diversity"] = float(
        np.mean([unique_valid_fraction(s[:, :out_len]) for s in samples])
        if samples.size else 0.0
    )
    return metrics


@torch.no_grad()
def full_elbo(model: GRAM, dataset: PuzzleDataset, config: ExperimentConfig,
              batch_size: int = 64, max_batches: Optional[int] = None) -> Dict[str, float]:
    """Evaluate the untruncated trajectory ELBO of Eq. (13).

    Unlike the training surrogate this accumulates the KL of *every* transition
    across all ``N_sup`` supervision steps, and evaluates the reconstruction term
    only at the terminal state.
    """
    device = model.device
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    ignore_index = dataset.metadata.ignore_label_id
    kl_balance = config.train.kl_balance

    recon_total, kl_total, count = 0.0, 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        puzzle_ids = batch["puzzle_ids"].to(device)
        x_embed = model.encode_input(inputs, puzzle_ids)
        y_embed = model.encode_target(targets)
        state = model.initial_state(inputs.shape[0])

        kl_sum = 0.0
        logits = None
        for _ in range(model.config.n_supervision):
            out = model(
                state, x_embed, y_embed=y_embed,
                use_posterior=model.config.is_stochastic,
                grad_last_only=False, collect_all=True, with_heads=False,
            )
            for transition in out.transitions:
                if transition.prior is None or transition.posterior is None:
                    continue
                kl_sum += float(
                    balanced_kl(transition.posterior, transition.prior, kl_balance)
                )
            state = out.state
            logits = out.logits

        assert logits is not None
        recon_total += float(reconstruction_loss(logits, targets, ignore_index))
        kl_total += kl_sum
        count += 1

    if count == 0:
        return {"neg_elbo": float("nan"), "reconstruction": float("nan"), "kl": float("nan")}
    recon = recon_total / count
    kl = kl_total / count
    return {
        "neg_elbo": recon + config.train.beta * kl,
        "reconstruction": recon,
        "kl": kl,
        "num_transitions": float(
            model.config.n_supervision * model.config.high_level_steps
        ),
    }


@torch.no_grad()
def scaling_sweep(model: GRAM, dataset: PuzzleDataset, config: ExperimentConfig,
                  widths: List[int], depths: Optional[List[int]] = None,
                  selection: str = "majority") -> List[Dict[str, float]]:
    """Sweep the two inference-time scaling axes (Figure 4)."""
    depths = depths or [config.model.n_supervision]
    results: List[Dict[str, float]] = []
    for depth in depths:
        for width in widths:
            eval_cfg = EvalConfig(
                num_samples=width,
                n_supervision=depth,
                selection=selection if width > 1 else "first",
                use_act=config.eval.use_act,
                temperature=config.eval.temperature,
                batch_size=config.eval.batch_size,
            )
            metrics = evaluate(model, dataset, config, eval_cfg)
            results.append({"depth": depth, "width": width, **metrics})
    return results


__all__ = ["evaluate", "full_elbo", "scaling_sweep"]
