"""Coverage for the unconditional generation path of Section 4.3 / Appendix D.5."""

import numpy as np
import torch

from gram.config import EvalConfig, ExperimentConfig, ModelConfig, TrainConfig
from gram.data import sudoku as sd
from gram.data.base import PuzzleDataset
from gram.inference import generate
from gram.metrics import sudoku_validity
from gram.train import Trainer


def test_validity_metric_separates_valid_from_invalid_boards():
    import random
    rng = random.Random(0)
    grids = [sd.random_complete_grid(rng) for _ in range(4)]
    valid = np.stack([sd.grid_to_tokens(g) for g in grids])
    report = sudoku_validity(valid)
    assert report["validity"] == 1.0
    assert report["mean_violations"] == 0.0
    assert report["unique_valid"] == 1.0

    broken = valid.copy()
    broken[:, 0] = broken[:, 1]
    assert sudoku_validity(broken)["validity"] == 0.0


def test_validity_metric_detects_repeated_boards():
    import random
    grid = sd.grid_to_tokens(sd.random_complete_grid(random.Random(1)))
    repeated = np.stack([grid] * 5)
    report = sudoku_validity(repeated)
    assert report["validity"] == 1.0
    assert report["unique_valid"] == 0.2  # one distinct board among five valid ones


def test_unconditional_training_and_sampling(tmp_path):
    sd.build(tmp_path / "data", num_train=4, num_test=2, seed=0, unconditional=True)
    config = ExperimentConfig(
        task="sudoku", data_dir=str(tmp_path / "data"), output_dir=str(tmp_path / "run"),
        model=ModelConfig(vocab_size=11, seq_len=81, hidden_size=32, num_heads=4,
                          ffn_hidden_size=32, num_layers=1, mixer="mlp",
                          low_level_steps=1, high_level_steps=2, n_supervision=2,
                          use_act=False),
        train=TrainConfig(epochs=1, batch_size=4, lr=1e-3, warmup_steps=1,
                          eval_interval=1, log_interval=1000, checkpoint_interval=0,
                          ema_decay=0.0),
        eval=EvalConfig(num_samples=1, batch_size=4, use_act=False),
    )
    trainer = Trainer(config)
    metrics = trainer.fit()
    # The unconditional task reports generation metrics alongside accuracy.
    assert "gen_validity" in metrics and "gen_mean_violations" in metrics

    samples = generate(trainer.model, num_samples=4, seq_len=81,
                       blank_token=sd.BLANK, batch_size=2).numpy()
    assert samples.shape == (4, 81)
    report = sudoku_validity(samples)
    assert 0.0 <= report["validity"] <= 1.0


def test_generation_is_diverse_for_a_stochastic_model():
    model = GRAM_for_sudoku()
    samples = generate(model, num_samples=16, seq_len=81, blank_token=sd.BLANK,
                       batch_size=8).numpy()
    keys = {row.tobytes() for row in samples}
    assert len(keys) > 1


def test_generation_is_degenerate_for_a_deterministic_model():
    model = GRAM_for_sudoku(guidance="none")
    samples = generate(model, num_samples=8, seq_len=81, blank_token=sd.BLANK,
                       batch_size=8).numpy()
    keys = {row.tobytes() for row in samples}
    assert len(keys) == 1  # the mode collapse the paper reports for TRM


def GRAM_for_sudoku(**overrides):
    from gram.model import GRAM
    base = dict(vocab_size=11, seq_len=81, hidden_size=32, num_heads=4,
                ffn_hidden_size=32, num_layers=1, mixer="mlp", low_level_steps=1,
                high_level_steps=2, n_supervision=2, use_act=False)
    base.update(overrides)
    return GRAM(ModelConfig(**base)).eval()
