import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gram.config import EvalConfig, ExperimentConfig, ModelConfig, TrainConfig  # noqa: E402


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    torch.set_num_threads(1)


@pytest.fixture
def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=6, seq_len=12, hidden_size=32, num_heads=4, ffn_hidden_size=32,
        num_layers=1, low_level_steps=2, high_level_steps=2, n_supervision=3,
    )


@pytest.fixture
def tiny_experiment(tiny_model_config) -> ExperimentConfig:
    return ExperimentConfig(
        task="nqueens",
        model=tiny_model_config,
        train=TrainConfig(epochs=1, batch_size=8, eval_interval=1, log_interval=1000,
                          warmup_steps=1, ema_decay=0.9),
        eval=EvalConfig(num_samples=2, batch_size=8),
    )
