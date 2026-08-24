"""Configuration dataclasses for GRAM (Generative Recursive Reasoning Models).

The defaults follow the hyperparameters reported in the paper (Appendix B):
``D = 512``, ``N_head = 8``, ``D_h = 512``, ``K = 4`` low-level steps (``6`` for
Sudoku), ``T = 3`` high-level steps and ``N_sup = 16`` deep-supervision steps.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class PatchEncoderConfig:
    """Convolutional patch encoder/decoder used for image tasks (Table 5)."""

    image_size: int = 28
    in_channels: int = 1
    patch_size: int = 2
    conv_kernel: int = 5
    group_norm_groups: int = 32
    enabled: bool = True


@dataclass
class ModelConfig:
    """Architecture of the encoder, recursive core and decoder."""

    # --- task shape -------------------------------------------------------- #
    vocab_size: int = 11
    seq_len: int = 81  # length of the *content* sequence (puzzle prefix excluded)

    # --- transformer backbone --------------------------------------------- #
    hidden_size: int = 512  # D
    num_heads: int = 8  # N_head
    ffn_hidden_size: int = 512  # D_h
    num_layers: int = 2  # layers inside each of f_L / f_H
    mixer: str = "attention"  # "attention" | "mlp" (SwiGLU over the sequence axis)
    seq_mixer_hidden: Optional[int] = None  # hidden width of the "mlp" mixer
    pos_encoding: str = "rope"  # "rope" | "learned" | "none"
    rope_theta: float = 10000.0

    # --- recursion --------------------------------------------------------- #
    hierarchical: bool = True  # z = (h, l) vs. a single flat latent
    low_level_steps: int = 4  # K
    high_level_steps: int = 3  # T
    n_supervision: int = 16  # N_sup
    deep_supervision: bool = True

    # --- puzzle embeddings (ARC) ------------------------------------------- #
    puzzle_emb_tokens: int = 0  # 16 for ARC, 0 elsewhere
    num_puzzle_identifiers: int = 1

    # --- stochastic guidance ----------------------------------------------- #
    guidance: str = "full"  # full | stochastic_only | guide_only | none
    min_std: float = 1e-3
    max_std: float = 1.0
    init_std: float = 0.1
    posterior_detach_input: bool = False
    # Table 4 lists no separate target encoder: the posterior conditions on a
    # plain learned embedding of y.  Enable this to add a SwiGLU projection.
    posterior_target_proj: bool = False

    # --- decoder ------------------------------------------------------------ #
    # Appendix B.1 describes a SwiGLU MLP head while Table 4 lists
    # ``Linear(D -> vocab)``; Table 4 is the default.
    lm_head: str = "linear"  # linear | swiglu

    # --- auxiliary heads ---------------------------------------------------- #
    use_act: bool = True
    act_mode: str = "halt_only"  # halt_only | q_learning
    use_lprm: bool = True

    # --- image tasks -------------------------------------------------------- #
    patch_encoder: Optional[PatchEncoderConfig] = None

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.mixer not in ("attention", "mlp"):
            raise ValueError(f"unknown mixer {self.mixer!r}")
        if self.pos_encoding not in ("rope", "learned", "none"):
            raise ValueError(f"unknown pos_encoding {self.pos_encoding!r}")
        if self.guidance not in ("full", "stochastic_only", "guide_only", "none"):
            raise ValueError(f"unknown guidance mode {self.guidance!r}")
        if self.act_mode not in ("halt_only", "q_learning"):
            raise ValueError(f"unknown act_mode {self.act_mode!r}")
        if self.lm_head not in ("linear", "swiglu"):
            raise ValueError(f"unknown lm_head {self.lm_head!r}")
        if self.mixer == "mlp" and self.pos_encoding == "rope":
            # RoPE is only meaningful inside attention.
            self.pos_encoding = "learned"
        if self.seq_mixer_hidden is None:
            self.seq_mixer_hidden = self.total_seq_len
        if isinstance(self.patch_encoder, dict):
            self.patch_encoder = PatchEncoderConfig(**self.patch_encoder)

    @property
    def total_seq_len(self) -> int:
        """Length actually processed by the core (puzzle prefix included)."""
        return self.seq_len + self.puzzle_emb_tokens

    @property
    def is_stochastic(self) -> bool:
        return self.guidance in ("full", "stochastic_only")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Optimisation hyperparameters (Appendix B.2)."""

    epochs: int = 1000
    batch_size: int = 768
    eval_batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1.0
    grad_clip: float = 1.0
    warmup_steps: int = 200
    lr_min_ratio: float = 1.0  # 1.0 -> constant LR after warmup
    beta: float = 0.1  # KL coefficient
    kl_balance: float = 0.8  # KL balancing (DreamerV2-style)
    free_bits: float = 0.0  # nats per latent dimension, 0 disables
    act_loss_weight: float = 1.0
    lprm_loss_weight: float = 1.0
    head_rollout: str = "prior"  # prior | posterior (see gram.train.train_batch)
    ema_decay: float = 0.9999
    seed: int = 0
    device: str = "auto"
    num_workers: int = 0
    eval_interval: int = 10  # epochs
    log_interval: int = 10  # optimiser steps
    checkpoint_interval: int = 100  # epochs
    eval_samples: int = 1  # trajectories drawn per test input during eval
    # Which evaluation metric selects ``best.pt``.  On multi-solution tasks
    # ``constraint_accuracy`` is the meaningful one: a prediction can satisfy
    # every constraint without matching the single stored reference solution.
    select_metric: str = "exact_match"
    eval_full_elbo: bool = False
    total_steps: int = 0  # filled in by the Trainer; used by the cosine schedule
    # Mixed precision. bf16 needs no gradient scaler, which matters here because
    # the loss is backwarded once per supervision step; fp16 is deliberately not
    # offered rather than offered without a scaler.
    amp_dtype: str = "none"  # none | bf16
    puzzle_emb_lr: Optional[float] = None  # separate LR for puzzle embeddings


@dataclass
class EvalConfig:
    """Inference-time scaling knobs (Section 2.3)."""

    num_samples: int = 1  # width, N
    n_supervision: Optional[int] = None  # depth override at test time
    selection: str = "majority"  # majority | lprm | first
    use_act: bool = True
    temperature: float = 1.0  # scales the prior std
    sample_tokens: bool = False  # argmax decoding by default
    batch_size: int = 256
    coverage_samples: int = 20


@dataclass
class ExperimentConfig:
    name: str = "gram"
    task: str = "nqueens"
    data_dir: str = "data/nqueens8"
    output_dir: str = "runs/gram"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        return _build(cls, data)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _build(cls, data: Dict[str, Any]):
    """Recursively instantiate nested dataclasses from a plain dict."""
    if data is None:
        return None
    kwargs: Dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    for key, value in data.items():
        if key not in known:
            raise TypeError(f"{cls.__name__} got an unexpected field {key!r}")
        ftype = known[key].type
        # Resolve string annotations / Optional[...] wrappers.
        target = _resolve_dataclass(ftype)
        if target is not None and isinstance(value, dict):
            kwargs[key] = _build(target, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


_DATACLASS_REGISTRY = {
    "ModelConfig": ModelConfig,
    "TrainConfig": TrainConfig,
    "EvalConfig": EvalConfig,
    "PatchEncoderConfig": PatchEncoderConfig,
    "ExperimentConfig": ExperimentConfig,
}


def _resolve_dataclass(ftype):
    if is_dataclass(ftype):
        return ftype
    name = ftype if isinstance(ftype, str) else getattr(ftype, "__name__", "")
    for key, value in _DATACLASS_REGISTRY.items():
        if key in str(name):
            return value
    return None


def apply_overrides(config: ExperimentConfig, overrides: Dict[str, Any]) -> ExperimentConfig:
    """Apply ``dotted.key=value`` style overrides in place."""
    resizes_sequence = any(
        key in ("model.seq_len", "model.puzzle_emb_tokens") for key in overrides
    )
    if resizes_sequence and "model.seq_mixer_hidden" not in overrides:
        config.model.seq_mixer_hidden = None
    for key, value in overrides.items():
        target: Any = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise KeyError(f"unknown config key {key!r}")
        current = getattr(target, leaf)
        if current is not None and not isinstance(value, type(current)):
            try:
                value = type(current)(value)
            except (TypeError, ValueError):
                pass
        setattr(target, leaf, value)
    # Re-run validation on the model config.
    config.model.__post_init__()
    return config
