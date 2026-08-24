"""GRAM -- Generative Recursive reAsoning Models.

Reference implementation of "Generative Recursive Reasoning" (Baek, Jo, Kim,
Ren, Bengio & Ahn), which turns deterministic recursive reasoning models into
probabilistic multi-trajectory computation via learnable stochastic guidance
and amortised variational inference.
"""

from .config import EvalConfig, ExperimentConfig, ModelConfig, PatchEncoderConfig, TrainConfig
from .core import LatentState, RecursiveCore
from .guidance import Gaussian, GaussianHead, StochasticTransition
from .model import GRAM

__version__ = "0.1.0"

__all__ = [
    "EvalConfig",
    "ExperimentConfig",
    "GRAM",
    "Gaussian",
    "GaussianHead",
    "LatentState",
    "ModelConfig",
    "PatchEncoderConfig",
    "RecursiveCore",
    "StochasticTransition",
    "TrainConfig",
]
