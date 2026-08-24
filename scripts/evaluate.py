#!/usr/bin/env python3
"""Evaluate a trained GRAM checkpoint.

    # single-sample accuracy
    python scripts/evaluate.py --checkpoint runs/nqueens8/best.pt

    # width scaling (Figure 4): accuracy vs. number of parallel trajectories
    python scripts/evaluate.py --checkpoint runs/nqueens8/best.pt \
        --widths 1 5 10 20 --selection lprm

    # depth x width sweep
    python scripts/evaluate.py --checkpoint runs/sudoku/best.pt \
        --widths 1 20 --depths 8 16 32

    # unconditional generation (Section 4.3 / Appendix D.5)
    python scripts/evaluate.py --checkpoint runs/sudoku_uncond/best.pt \
        --generate 1000 --steps 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gram.config import EvalConfig, ExperimentConfig  # noqa: E402
from gram.data.base import PuzzleDataset  # noqa: E402
from gram.evaluate import evaluate, full_elbo, scaling_sweep  # noqa: E402
from gram.inference import generate  # noqa: E402
from gram.metrics import sudoku_validity, unique_valid_fraction  # noqa: E402
from gram.train import load_model  # noqa: E402
from gram.utils import resolve_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None, help="override the training data dir")
    parser.add_argument("--split", default="test")
    parser.add_argument("--widths", type=int, nargs="*", default=None,
                        help="numbers of parallel trajectories to sweep")
    parser.add_argument("--depths", type=int, nargs="*", default=None,
                        help="numbers of supervision steps to sweep")
    parser.add_argument("--selection", default="majority",
                        choices=["majority", "lprm", "first"])
    parser.add_argument("--no-act", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--generate", type=int, default=0,
                        help="draw this many unconditional samples instead")
    parser.add_argument("--steps", type=int, default=None,
                        help="supervision steps used for generation")
    parser.add_argument("--elbo", action="store_true",
                        help="also report the untruncated ELBO of Eq. (13)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None, help="write results to this JSON file")
    args = parser.parse_args()

    device = resolve_device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ExperimentConfig.from_dict(payload["config"])
    model = load_model(args.checkpoint, device)

    data_dir = args.data_dir or config.data_dir
    results: dict = {"checkpoint": args.checkpoint, "task": config.task}

    if args.generate:
        blank = 1  # the "blank"/"black" token in every task vocabulary
        samples = generate(
            model, args.generate, config.model.seq_len, blank_token=blank,
            n_supervision=args.steps, temperature=args.temperature,
            batch_size=args.batch_size or 64, device=device,
        ).numpy()
        results["generation"] = {
            "num_samples": int(len(samples)),
            "steps": args.steps or config.model.n_supervision,
            "unique_fraction": unique_valid_fraction(samples),
        }
        if config.task == "sudoku":
            results["generation"].update(sudoku_validity(samples))
        np.save(Path(args.checkpoint).with_suffix(".samples.npy"), samples)
    else:
        dataset = PuzzleDataset(data_dir, args.split)
        if args.batch_size:
            config.eval.batch_size = args.batch_size
        config.eval.use_act = not args.no_act
        config.eval.temperature = args.temperature
        if args.widths or args.depths:
            results["sweep"] = scaling_sweep(
                model, dataset, config,
                widths=args.widths or [config.eval.num_samples],
                depths=args.depths, selection=args.selection,
            )
        else:
            config.eval.selection = args.selection
            results["metrics"] = evaluate(model, dataset, config)
        if args.elbo:
            results["elbo"] = full_elbo(model, dataset, config, max_batches=8)

    text = json.dumps(results, indent=2, default=float)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
