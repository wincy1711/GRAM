"""End-to-end coverage for the ARC path: puzzle embeddings and the 30x30 canvas."""

import json

import torch

from gram.config import EvalConfig, ExperimentConfig, ModelConfig, TrainConfig
from gram.data import arc
from gram.data.base import PuzzleDataset
from gram.train import Trainer, build_optimizer, train_batch


def make_tasks(tmp_path, n_tasks=3):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    for i in range(n_tasks):
        (task_dir / f"task{i}.json").write_text(json.dumps({
            "train": [
                {"input": [[i, i + 1], [i + 2, i]], "output": [[i, i + 2], [i + 1, i]]},
                {"input": [[i + 1]], "output": [[i + 3]]},
            ],
            "test": [{"input": [[i, i]], "output": [[i + 1, i + 1]]}],
        }))
    return task_dir


def arc_config(data_dir, output_dir) -> ExperimentConfig:
    return ExperimentConfig(
        task="arc", data_dir=str(data_dir), output_dir=str(output_dir),
        model=ModelConfig(vocab_size=arc.VOCAB_SIZE, seq_len=arc.SEQ_LEN,
                          hidden_size=32, num_heads=4, ffn_hidden_size=32,
                          num_layers=1, low_level_steps=1, high_level_steps=2,
                          n_supervision=2, puzzle_emb_tokens=16,
                          num_puzzle_identifiers=4),
        train=TrainConfig(epochs=1, batch_size=4, lr=1e-3, warmup_steps=1,
                          eval_interval=1, log_interval=1000, checkpoint_interval=0,
                          ema_decay=0.0, puzzle_emb_lr=1e-2),
        eval=EvalConfig(num_samples=1, batch_size=4),
    )


def test_canvas_encoding_marks_the_grid_boundary():
    tokens = arc.grid_to_canvas([[0, 1], [2, 3]]).reshape(arc.CANVAS, arc.CANVAS)
    assert tokens[0, 0] == arc.COLOR_OFFSET + 0
    assert tokens[0, 2] == arc.EOS          # end of a row
    assert bool((tokens[2] == arc.EOS).all())  # end of the grid
    assert tokens[0, 3] == arc.PAD


def test_puzzle_embedding_gets_its_own_learning_rate(tmp_path):
    task_dir = make_tasks(tmp_path)
    arc.build(tmp_path / "out", task_dir)
    config = arc_config(tmp_path / "out", tmp_path / "run")
    trainer = Trainer(config)
    puzzle_group = [
        g for g in trainer.optimizer.param_groups if g.get("lr") == 1e-2
    ]
    assert len(puzzle_group) == 1
    assert puzzle_group[0]["params"][0] is trainer.model.encoder.puzzle_embed.weight


def test_arc_training_step_updates_the_right_puzzle_embeddings(tmp_path):
    task_dir = make_tasks(tmp_path)
    arc.build(tmp_path / "out", task_dir)
    config = arc_config(tmp_path / "out", tmp_path / "run")
    trainer = Trainer(config)
    dataset = PuzzleDataset(tmp_path / "out", "train")

    # Take a batch that only references puzzle id 1.
    select = (dataset.puzzle_ids == 1).nonzero().squeeze(-1)[:2]
    batch = {"inputs": dataset.inputs[select], "targets": dataset.targets[select],
             "puzzle_ids": dataset.puzzle_ids[select]}
    before = trainer.model.encoder.puzzle_embed.weight.detach().clone()
    train_batch(trainer.model, batch, config, trainer.optimizer,
                torch.device("cpu"), 0)
    after = trainer.model.encoder.puzzle_embed.weight
    assert not torch.allclose(before[1], after[1])
    assert torch.allclose(before[2], after[2])  # an unused puzzle stays put


def test_arc_fit_runs_end_to_end(tmp_path):
    task_dir = make_tasks(tmp_path)
    arc.build(tmp_path / "out", task_dir)
    trainer = Trainer(arc_config(tmp_path / "out", tmp_path / "run"))
    metrics = trainer.fit()
    assert "exact_match" in metrics
    assert 0.0 <= metrics["token_accuracy"] <= 1.0


def test_decoder_output_length_excludes_the_puzzle_prefix(tmp_path):
    task_dir = make_tasks(tmp_path)
    arc.build(tmp_path / "out", task_dir)
    trainer = Trainer(arc_config(tmp_path / "out", tmp_path / "run"))
    model = trainer.model
    x = torch.zeros(2, arc.SEQ_LEN, dtype=torch.long)
    x_embed = model.encode_input(x, torch.tensor([1, 2]))
    assert x_embed.shape[1] == arc.SEQ_LEN + 16
    with torch.no_grad():
        out = model(model.initial_state(2), x_embed)
    assert out.logits.shape == (2, arc.SEQ_LEN, arc.VOCAB_SIZE)
