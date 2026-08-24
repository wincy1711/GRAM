"""Inference-time scaling for GRAM (Section 2.3).

Two complementary axes:

* **depth** -- number of recursive supervision steps, optionally cut short per
  trajectory by adaptive computation time (ACT, Appendix A.1);
* **width** -- ``N`` latent trajectories sampled in parallel from the learned
  prior, then reduced by majority voting or best-of-N selection with the Latent
  Process Reward Model (LPRM, Appendix A.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor

from .config import EvalConfig
from .model import GRAM


@dataclass
class TrajectoryOutput:
    """A batch of sampled trajectories.

    Shapes are ``[B, N, ...]`` where ``N`` is the number of parallel samples.
    """

    predictions: Tensor  # [B, N, L] token ids
    logits: Tensor  # [B, N, L, V] logits at the halting step
    values: Optional[Tensor]  # [B, N] LPRM estimates
    halt_steps: Tensor  # [B, N] supervision step at which each trajectory halted
    trace: Optional[Tensor] = None  # [steps, B, N, L] per-step predictions


@torch.no_grad()
def sample_trajectories(model: GRAM, inputs: Tensor, puzzle_ids: Optional[Tensor] = None,
                        num_samples: int = 1, n_supervision: Optional[int] = None,
                        use_act: bool = True, temperature: float = 1.0,
                        sample_tokens: bool = False,
                        return_all_steps: bool = False) -> TrajectoryOutput:
    """Draw ``num_samples`` prior trajectories per input and decode each one.

    All samples are folded into the batch dimension so that a single forward
    pass covers the whole width; this is the "parallel trajectory sampling"
    of Section 2.3.
    """
    model.eval()
    device = model.device
    inputs = inputs.to(device)
    if puzzle_ids is not None:
        puzzle_ids = puzzle_ids.to(device)

    batch_size = inputs.shape[0]
    steps = n_supervision or model.config.n_supervision

    wide_inputs = inputs.repeat_interleave(num_samples, dim=0)
    wide_puzzle = (
        puzzle_ids.repeat_interleave(num_samples, dim=0) if puzzle_ids is not None else None
    )
    x_embed = model.encode_input(wide_inputs, wide_puzzle)
    state = model.initial_state(wide_inputs.shape[0])

    total = wide_inputs.shape[0]
    final_logits: Optional[Tensor] = None
    final_values: Optional[Tensor] = None
    halted = torch.zeros(total, dtype=torch.bool, device=device)
    halt_steps = torch.full((total,), steps, dtype=torch.long, device=device)
    trace: List[Tensor] = []

    for step in range(steps):
        out = model(
            state,
            x_embed,
            y_embed=None,
            use_posterior=False,
            temperature=temperature,
            grad_last_only=False,
        )
        state = out.state
        if final_logits is None:
            final_logits = out.logits.clone()
            if out.value is not None:
                final_values = out.value.clone()
        else:
            active = ~halted
            if active.any():
                final_logits[active] = out.logits[active]
                if out.value is not None and final_values is not None:
                    final_values[active] = out.value[active]
        if return_all_steps:
            trace.append(out.logits.argmax(-1).view(batch_size, num_samples, -1).cpu())

        if use_act and out.q_halt_logits is not None and step < steps - 1:
            should_halt = (torch.sigmoid(out.q_halt_logits) > 0.5) & ~halted
            halt_steps[should_halt] = step + 1
            halted = halted | should_halt
            if bool(halted.all()):
                break

    assert final_logits is not None
    if sample_tokens:
        probs = torch.softmax(final_logits, dim=-1)
        flat = probs.reshape(-1, probs.shape[-1])
        predictions = torch.multinomial(flat, 1).view(final_logits.shape[:-1])
    else:
        predictions = final_logits.argmax(-1)

    seq_len = predictions.shape[-1]
    result = TrajectoryOutput(
        predictions=predictions.view(batch_size, num_samples, seq_len),
        logits=final_logits.view(batch_size, num_samples, seq_len, -1),
        values=None if final_values is None else final_values.view(batch_size, num_samples),
        halt_steps=halt_steps.view(batch_size, num_samples),
    )
    if return_all_steps:
        result.trace = torch.stack(trace, dim=0)
    return result


# --------------------------------------------------------------------------- #
# Candidate selection
# --------------------------------------------------------------------------- #
def majority_vote(predictions: Tensor) -> Tensor:
    """Pick the most frequent prediction per input; ties go to the first sample."""
    batch_size, num_samples, seq_len = predictions.shape
    if num_samples == 1:
        return predictions[:, 0]
    out = predictions.new_empty(batch_size, seq_len)
    for b in range(batch_size):
        counts: dict = {}
        for n in range(num_samples):
            key = tuple(predictions[b, n].tolist())
            counts[key] = counts.get(key, 0) + 1
        best = max(counts.items(), key=lambda kv: kv[1])[0]
        out[b] = predictions.new_tensor(best)
    return out


def best_of_n(predictions: Tensor, values: Optional[Tensor]) -> Tensor:
    """Select the candidate with the highest LPRM value."""
    if values is None:
        return predictions[:, 0]
    idx = values.argmax(dim=1)
    return predictions[torch.arange(predictions.shape[0], device=predictions.device), idx]


def select(output: TrajectoryOutput, strategy: str = "majority") -> Tensor:
    if strategy == "majority":
        return majority_vote(output.predictions)
    if strategy == "lprm":
        return best_of_n(output.predictions, output.values)
    if strategy == "first":
        return output.predictions[:, 0]
    raise ValueError(f"unknown selection strategy {strategy!r}")


@torch.no_grad()
def predict(model: GRAM, inputs: Tensor, puzzle_ids: Optional[Tensor] = None,
            config: Optional[EvalConfig] = None) -> Tensor:
    """Convenience wrapper: sample, then reduce to one prediction per input."""
    config = config or EvalConfig()
    output = sample_trajectories(
        model,
        inputs,
        puzzle_ids,
        num_samples=config.num_samples,
        n_supervision=config.n_supervision,
        use_act=config.use_act,
        temperature=config.temperature,
        sample_tokens=config.sample_tokens,
    )
    return select(output, config.selection)


@torch.no_grad()
def generate(model: GRAM, num_samples: int, seq_len: int, blank_token: int = 1,
             n_supervision: Optional[int] = None, temperature: float = 1.0,
             batch_size: int = 64, sample_tokens: bool = False,
             device: Optional[torch.device] = None) -> Tensor:
    """Unconditional generation ``p_theta(x)``: run the recursion from an empty input."""
    device = device or model.device
    outputs: List[Tensor] = []
    remaining = num_samples
    while remaining > 0:
        chunk = min(batch_size, remaining)
        empty = torch.full((chunk, seq_len), blank_token, dtype=torch.long, device=device)
        result = sample_trajectories(
            model, empty, None, num_samples=1, n_supervision=n_supervision,
            use_act=False, temperature=temperature, sample_tokens=sample_tokens,
        )
        outputs.append(result.predictions[:, 0].cpu())
        remaining -= chunk
    return torch.cat(outputs, dim=0)


__all__ = [
    "TrajectoryOutput",
    "best_of_n",
    "generate",
    "majority_vote",
    "predict",
    "sample_trajectories",
    "select",
]
