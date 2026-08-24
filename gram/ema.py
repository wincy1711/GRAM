"""Exponential moving average of model parameters (Appendix B.2, decay 0.9999)."""

from __future__ import annotations

import contextlib
from typing import Dict, Iterator

import torch
from torch import nn


class ModelEMA:
    """Keeps a shadow copy of the parameters and can swap it in for evaluation."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.num_updates = 0
        self.shadow: Dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
            if param.dtype.is_floating_point
        }
        self._backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        # Warm-up so early estimates are not dominated by the initialisation.
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for name, value in model.state_dict().items():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        state = model.state_dict()
        for name, value in self.shadow.items():
            state[name].copy_(value)

    @contextlib.contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Temporarily swap the EMA weights into ``model``."""
        self._backup = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if name in self.shadow
        }
        try:
            self.copy_to(model)
            yield
        finally:
            state = model.state_dict()
            for name, value in self._backup.items():
                state[name].copy_(value)
            self._backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"decay": self.decay, "num_updates": self.num_updates, "shadow": self.shadow}

    def load_state_dict(self, state: Dict) -> None:
        self.decay = state.get("decay", self.decay)
        self.num_updates = state.get("num_updates", 0)
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}


__all__ = ["ModelEMA"]
