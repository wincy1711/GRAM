"""End-to-end coverage for the image (patch-encoder) path used in Section 4.3."""

import struct

import numpy as np
import torch

from gram.config import EvalConfig, ExperimentConfig, ModelConfig, PatchEncoderConfig, TrainConfig
from gram.data import mnist
from gram.data.base import PuzzleDataset
from gram.inference import generate
from gram.metrics import frechet_distance, inception_score
from gram.train import Trainer


def write_idx(path, array):
    with open(path, "wb") as handle:
        handle.write(struct.pack(">I", 0x00000800 | array.ndim))
        handle.write(struct.pack(">" + "I" * array.ndim, *array.shape))
        handle.write(array.astype(np.uint8).tobytes())


def make_raw(tmp_path, n_train=6, n_test=2):
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(0)
    train = rng.integers(0, 256, size=(n_train, 28, 28), dtype=np.uint8)
    test = rng.integers(0, 256, size=(n_test, 28, 28), dtype=np.uint8)
    write_idx(raw / "train-images-idx3-ubyte", train)
    write_idx(raw / "t10k-images-idx3-ubyte", test)
    return raw, train, test


def test_idx_reader_round_trip(tmp_path):
    raw, train, test = make_raw(tmp_path)
    loaded_train, loaded_test = mnist.load_raw(raw)
    assert np.array_equal(loaded_train, train)
    assert np.array_equal(loaded_test, test)


def test_binarisation_thresholds_at_half():
    images = np.array([[0, 127, 128, 255]], dtype=np.uint8)
    tokens = mnist.binarize(images)
    assert tokens.tolist() == [[mnist.BLACK, mnist.BLACK, mnist.WHITE, mnist.WHITE]]


def test_build_produces_an_empty_conditioning_signal(tmp_path):
    raw, _, _ = make_raw(tmp_path)
    metadata = mnist.build(tmp_path / "out", raw, unconditional=True)
    assert metadata.seq_len == 784 and metadata.vocab_size == 3
    dataset = PuzzleDataset(tmp_path / "out", "train")
    assert bool((dataset.inputs == mnist.BLACK).all())
    assert set(dataset.targets.unique().tolist()) <= {mnist.BLACK, mnist.WHITE}


def test_conditional_build_copies_the_image(tmp_path):
    raw, _, _ = make_raw(tmp_path)
    mnist.build(tmp_path / "out", raw, unconditional=False)
    dataset = PuzzleDataset(tmp_path / "out", "train")
    assert torch.equal(dataset.inputs, dataset.targets)


def test_training_and_generation_through_the_patch_encoder(tmp_path):
    raw, _, _ = make_raw(tmp_path, n_train=8, n_test=4)
    mnist.build(tmp_path / "out", raw, unconditional=True)
    patch = PatchEncoderConfig(image_size=28, patch_size=7, group_norm_groups=4)
    config = ExperimentConfig(
        task="mnist", data_dir=str(tmp_path / "out"), output_dir=str(tmp_path / "run"),
        model=ModelConfig(vocab_size=3, seq_len=16, hidden_size=32, num_heads=4,
                          ffn_hidden_size=32, num_layers=1, low_level_steps=1,
                          high_level_steps=2, n_supervision=2, patch_encoder=patch,
                          use_act=False),
        train=TrainConfig(epochs=1, batch_size=4, lr=1e-3, warmup_steps=1,
                          eval_interval=1, log_interval=1000, checkpoint_interval=0,
                          ema_decay=0.0),
        eval=EvalConfig(num_samples=1, batch_size=4, use_act=False),
    )
    trainer = Trainer(config)
    # 28x28 pixels over 7x7 patches -> 16 patch tokens.
    assert trainer.model.config.seq_len == 16
    metrics = trainer.fit()
    assert "exact_match" in metrics

    samples = generate(trainer.model, num_samples=4, seq_len=784,
                       blank_token=mnist.BLACK, batch_size=2)
    assert samples.shape == (4, 784)
    assert set(samples.unique().tolist()) <= {0, 1, 2}


def test_is_and_fid_behave_on_known_inputs():
    balanced = np.eye(4)[np.arange(400) % 4] * 0.97 + 0.01
    collapsed = np.tile(np.eye(4)[0] * 0.97 + 0.01, (400, 1))
    assert inception_score(balanced, splits=4) > 3.0
    assert inception_score(collapsed, splits=4) < 1.05

    rng = np.random.default_rng(0)
    reference = rng.normal(size=(400, 8))
    close = rng.normal(size=(400, 8))
    far = rng.normal(loc=4.0, size=(400, 8))
    assert frechet_distance(reference, reference) < 1e-6
    assert frechet_distance(reference, close) < frechet_distance(reference, far)
