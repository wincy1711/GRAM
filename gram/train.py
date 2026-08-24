"""Training loop for GRAM (Section 2.2).

Each batch is unrolled over ``N_sup`` supervision steps.  Within a step the
first ``T-1`` latent transitions run without gradients and only the final
transition ``z_{T-1} -> z_T`` is differentiated; the loss of Eq. (14) is
backpropagated immediately and the terminal state is detached before the next
step.  Memory therefore stays constant in ``N_sup``.

Trajectories are sampled from the variational posterior ``q_phi(.|x, y)`` during
training and from the learned prior ``p_theta(.|x)`` at evaluation time.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data.base import PuzzleDataset
from .ema import ModelEMA
from .evaluate import evaluate
from .losses import act_loss, lprm_loss, supervision_step_loss
from .model import GRAM
from .utils import JsonlLogger, autocast_dtype, human_count, resolve_device, set_seed


# --------------------------------------------------------------------------- #
# Optimiser
# --------------------------------------------------------------------------- #
def build_optimizer(model: GRAM, config) -> torch.optim.Optimizer:
    """AdamW with weight decay disabled for embeddings and 1-D parameters."""
    decay, no_decay, puzzle = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "puzzle_embed" in name:
            puzzle.append(param)
        elif param.ndim < 2 or "embed" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    groups: List[Dict[str, Any]] = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if puzzle:
        groups.append({
            "params": puzzle,
            "weight_decay": 0.0,
            "lr": config.puzzle_emb_lr or config.lr,
        })
    return torch.optim.AdamW(groups, lr=config.lr, betas=(0.9, 0.95))


def lr_at(step: int, config) -> float:
    """Linear warm-up followed by cosine decay to ``lr * lr_min_ratio``."""
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    if config.lr_min_ratio >= 1.0 or config.total_steps <= 0:
        return config.lr
    total = max(1, config.total_steps - config.warmup_steps)
    progress = min(1.0, (step - config.warmup_steps) / total)
    scale = config.lr_min_ratio + (1 - config.lr_min_ratio) * 0.5 * (
        1 + math.cos(math.pi * progress)
    )
    return config.lr * scale


# --------------------------------------------------------------------------- #
# One optimisation step
# --------------------------------------------------------------------------- #
@dataclass
class BatchStats:
    loss: float = 0.0
    reconstruction: float = 0.0
    kl: float = 0.0
    act: float = 0.0
    lprm: float = 0.0
    accuracy: float = 0.0
    token_accuracy: float = 0.0
    prior_accuracy: float = 0.0
    prior_token_accuracy: float = 0.0
    grad_norm: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


def train_batch(model: GRAM, batch: Dict[str, Tensor], config: ExperimentConfig,
                optimizer: torch.optim.Optimizer, device: torch.device,
                ignore_index: int) -> BatchStats:
    """Run deep supervision over one batch and take a single optimiser step."""
    model.train()
    train_cfg = config.train
    inputs = batch["inputs"].to(device, non_blocking=True)
    targets = batch["targets"].to(device, non_blocking=True)
    puzzle_ids = batch.get("puzzle_ids")
    puzzle_ids = puzzle_ids.to(device) if puzzle_ids is not None else None

    state = model.initial_state(inputs.shape[0])

    # Without deep supervision (the Looped-Transformer baseline) the recursion
    # still runs to full depth, but the objective is applied only once, after
    # the final supervision step.
    n_sup = model.config.n_supervision
    supervise_from = 0 if model.config.deep_supervision else n_sup - 1
    n_supervised = n_sup - supervise_from
    optimizer.zero_grad(set_to_none=True)

    stats = BatchStats()
    summaries: List[Tensor] = []
    correctness: List[Tensor] = []
    accuracies: List[Tensor] = []

    for step in range(n_sup):
        supervised = step >= supervise_from
        # The embeddings are recomputed each supervision step so that every step
        # contributes its own gradient to the encoder and no autograd graph is
        # shared across the per-step backward passes.
        with torch.set_grad_enabled(supervised):
            x_embed = model.encode_input(inputs, puzzle_ids)
            y_embed = model.encode_target(targets)
            out = model(
                state,
                x_embed,
                y_embed=y_embed,
                use_posterior=model.config.is_stochastic,
                grad_last_only=True,
                with_heads=False,
            )
        if supervised:
            step_loss = supervision_step_loss(
                out.logits, targets, out.transitions,
                beta=train_cfg.beta, kl_balance=train_cfg.kl_balance,
                free_bits=train_cfg.free_bits, ignore_index=ignore_index,
            )
            (step_loss.total / n_supervised).backward()
            stats.loss += float(step_loss.total.detach()) / n_supervised
            stats.reconstruction += float(step_loss.reconstruction) / n_supervised
            stats.kl += float(step_loss.kl) / n_supervised

        with torch.no_grad():
            predictions = out.logits.argmax(-1)
            mask = targets != ignore_index
            token_acc = ((predictions == targets) & mask).sum(-1) / mask.sum(-1).clamp(min=1)
            seq_correct = (((predictions == targets) | ~mask).all(-1)).float()
            summaries.append(out.state.h[:, 0].detach())
            correctness.append(seq_correct)
            accuracies.append(token_acc)

        state = out.state.detach()

    stats.accuracy = float(correctness[-1].mean())
    stats.token_accuracy = float(accuracies[-1].mean())

    # Auxiliary heads (ACT halting and the LPRM value head) read detached
    # latents, so their gradients never reach the recursive core
    # (Appendix A.1/A.2).  By default they are fitted on a *prior* rollout,
    # because that is the distribution they must score at inference time -- the
    # posterior rollout used for the ELBO sees ``y`` and is far more accurate
    # than anything the sampler will produce.
    if model.q_head is not None or model.v_head is not None:
        if train_cfg.head_rollout == "prior":
            summaries, correctness, accuracies = rollout_for_heads(
                model, inputs, targets, puzzle_ids, n_sup, ignore_index
            )
            stats.prior_accuracy = float(correctness[-1].mean())
            stats.prior_token_accuracy = float(accuracies[-1].mean())
        train_auxiliary_heads(model, train_cfg, summaries, correctness, accuracies, stats)

    if train_cfg.grad_clip > 0:
        stats.grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        )
    optimizer.step()

    stats.loss += stats.act + stats.lprm
    return stats


@torch.no_grad()
def rollout_for_heads(model: GRAM, inputs: Tensor, targets: Tensor,
                      puzzle_ids: Optional[Tensor], n_sup: int, ignore_index: int):
    """Roll the recursion out under the prior and record per-step statistics."""
    model.eval()
    x_embed = model.encode_input(inputs, puzzle_ids)
    state = model.initial_state(inputs.shape[0])
    mask = targets != ignore_index
    denominator = mask.sum(-1).clamp(min=1)

    summaries: List[Tensor] = []
    correctness: List[Tensor] = []
    accuracies: List[Tensor] = []
    for _ in range(n_sup):
        out = model(state, x_embed, use_posterior=False, grad_last_only=False,
                    with_heads=False)
        state = out.state
        predictions = out.logits.argmax(-1)
        summaries.append(state.h[:, 0].detach())
        correctness.append((((predictions == targets) | ~mask).all(-1)).float())
        accuracies.append(((predictions == targets) & mask).sum(-1) / denominator)
    model.train()
    return summaries, correctness, accuracies


def train_auxiliary_heads(model: GRAM, train_cfg, summaries: List[Tensor],
                          correctness: List[Tensor], accuracies: List[Tensor],
                          stats: "BatchStats") -> None:
    """Accumulate gradients for the ACT halt head (Eq. 15) and LPRM (Eq. 16)."""
    head_loss = torch.zeros((), device=summaries[0].device)
    q_halt: List[Tensor] = []
    q_continue: List[Tensor] = []
    values: List[Tensor] = []
    for summary in summaries:
        if model.q_head is not None:
            q = model.q_head(summary)
            q_halt.append(q[..., 0])
            q_continue.append(q[..., 1])
        if model.v_head is not None:
            values.append(model.v_head(summary).squeeze(-1))
    if q_halt:
        loss = act_loss(q_halt, q_continue, correctness, model.config.act_mode)
        head_loss = head_loss + train_cfg.act_loss_weight * loss
        stats.act = float(loss.detach())
    if values:
        # Regression target r: the accuracy of the trajectory's final output.
        loss = lprm_loss(values, accuracies[-1])
        head_loss = head_loss + train_cfg.lprm_loss_weight * loss
        stats.lprm = float(loss.detach())
    if head_loss.requires_grad:
        head_loss.backward()


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = resolve_device(config.train.device)
        set_seed(config.train.seed)

        self.train_dataset = PuzzleDataset(config.data_dir, "train")
        self.test_dataset = PuzzleDataset(config.data_dir, "test")
        self.metadata = self.train_dataset.metadata
        self._sync_config_with_data()

        self.model = GRAM(config.model).to(self.device)
        self.optimizer = build_optimizer(self.model, config.train)
        self.ema = ModelEMA(self.model, config.train.ema_decay) if config.train.ema_decay > 0 else None

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = JsonlLogger(self.output_dir / "metrics.jsonl")
        config.save(self.output_dir / "config.json")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=config.train.num_workers,
        )
        steps_per_epoch = max(1, len(self.train_loader))
        config.train.total_steps = steps_per_epoch * config.train.epochs
        self.global_step = 0

    # ------------------------------------------------------------------ #
    def _sync_config_with_data(self) -> None:
        """Make the model config agree with the dataset that was built."""
        model_cfg = self.config.model
        auto_seq_mixer = model_cfg.seq_mixer_hidden == model_cfg.total_seq_len
        model_cfg.vocab_size = self.metadata.vocab_size
        if model_cfg.patch_encoder is not None and model_cfg.patch_encoder.enabled:
            side = model_cfg.patch_encoder.image_size // model_cfg.patch_encoder.patch_size
            model_cfg.seq_len = side * side
        else:
            model_cfg.seq_len = self.metadata.seq_len
        model_cfg.num_puzzle_identifiers = max(
            model_cfg.num_puzzle_identifiers, self.metadata.num_puzzle_identifiers
        )
        if auto_seq_mixer:
            model_cfg.seq_mixer_hidden = None  # re-derive from the new length
        model_cfg.__post_init__()
        self.ignore_index = self.metadata.ignore_label_id

    # ------------------------------------------------------------------ #
    def fit(self) -> Dict[str, float]:
        cfg = self.config.train
        print(
            f"[gram] task={self.config.task} params={human_count(self.model.num_parameters())} "
            f"device={self.device} guidance={self.config.model.guidance} "
            f"N_sup={self.config.model.n_supervision} K={self.config.model.low_level_steps} "
            f"T={self.config.model.high_level_steps}"
        )
        best: Dict[str, float] = {}
        start = time.time()
        for epoch in range(1, cfg.epochs + 1):
            epoch_stats: List[Dict[str, float]] = []
            for batch in self.train_loader:
                for group in self.optimizer.param_groups:
                    group["lr"] = lr_at(self.global_step, cfg)
                stats = train_batch(
                    self.model, batch, self.config, self.optimizer, self.device,
                    self.ignore_index,
                )
                if self.ema is not None:
                    self.ema.update(self.model)
                epoch_stats.append(stats.as_dict())
                self.global_step += 1
                if cfg.log_interval and self.global_step % cfg.log_interval == 0:
                    record = {"step": self.global_step, "epoch": epoch, "split": "train",
                              **stats.as_dict()}
                    self.logger.log(record)

            if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
                metrics = self.evaluate()
                mean = {
                    f"train_{k}": float(np.mean([s[k] for s in epoch_stats]))
                    for k in epoch_stats[0]
                }
                record = {"step": self.global_step, "epoch": epoch, "split": "eval",
                          "elapsed": time.time() - start, **mean, **metrics}
                self.logger.log(record)
                print(
                    f"  epoch {epoch:>5} | loss {mean['train_loss']:.4f} "
                    f"| rec {mean['train_reconstruction']:.4f} | kl {mean['train_kl']:.4f} "
                    f"| train_acc {mean['train_accuracy']:.3f} "
                    f"| test_acc {metrics.get('exact_match', float('nan')):.3f}"
                    + (f" | valid {metrics['constraint_accuracy']:.3f}"
                       if "constraint_accuracy" in metrics else "")
                )
                if not best or metrics.get("exact_match", 0.0) >= best.get("exact_match", 0.0):
                    best = dict(metrics)
                    self.save_checkpoint("best.pt")

            if cfg.checkpoint_interval and epoch % cfg.checkpoint_interval == 0:
                self.save_checkpoint("last.pt")

        self.save_checkpoint("last.pt")
        return best

    # ------------------------------------------------------------------ #
    def evaluate(self) -> Dict[str, float]:
        eval_cfg = self.config.eval
        if self.ema is not None:
            with self.ema.average_parameters(self.model):
                return evaluate(self.model, self.test_dataset, self.config)
        return evaluate(self.model, self.test_dataset, self.config)

    def save_checkpoint(self, name: str) -> None:
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "global_step": self.global_step,
        }
        if self.ema is not None:
            payload["ema"] = self.ema.state_dict()
        torch.save(payload, self.output_dir / name)

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        if self.ema is not None and "ema" in payload:
            self.ema.load_state_dict(payload["ema"])
        self.global_step = payload.get("global_step", 0)


def load_model(checkpoint: str | Path, device: Optional[torch.device] = None) -> GRAM:
    """Rebuild a model (with EMA weights when present) from a checkpoint."""
    payload = torch.load(checkpoint, map_location=device or "cpu", weights_only=False)
    config = ExperimentConfig.from_dict(payload["config"])
    model = GRAM(config.model)
    model.load_state_dict(payload["model"])
    if "ema" in payload:
        ema = ModelEMA(model, payload["ema"].get("decay", 0.9999))
        ema.load_state_dict(payload["ema"])
        ema.copy_to(model)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


__all__ = [
    "BatchStats",
    "Trainer",
    "build_optimizer",
    "load_model",
    "lr_at",
    "rollout_for_heads",
    "train_auxiliary_heads",
    "train_batch",
]
