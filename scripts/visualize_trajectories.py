#!/usr/bin/env python3
"""Visualise GRAM's latent reasoning trajectories (Appendix D.6).

Samples many prior trajectories for one problem instance, projects the
high-level state to 2-D with PCA, and writes a standalone SVG plus a JSON
summary. Running it on a deterministic checkpoint (guidance="none") produces
the single collapsed path of Figure 18; on GRAM it produces the spread of
Figure 19.

    python scripts/visualize_trajectories.py --checkpoint runs/nqueens8/best.pt \
        --num-samples 50 --output runs/nqueens8/trajectories.svg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gram.analysis import collect_trajectories, pca_project, render_svg, trajectory_spread  # noqa: E402
from gram.config import ExperimentConfig  # noqa: E402
from gram.data.base import PuzzleDataset  # noqa: E402
from gram.train import load_model  # noqa: E402
from gram.utils import resolve_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0,
                        help="which test instance to analyse")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ExperimentConfig.from_dict(payload["config"])
    model = load_model(args.checkpoint, device)

    dataset = PuzzleDataset(args.data_dir or config.data_dir, args.split)
    item = dataset[args.index]
    trace = collect_trajectories(
        model, item["inputs"], item["targets"],
        puzzle_ids=item["puzzle_ids"].reshape(1),
        num_samples=args.num_samples, n_supervision=args.steps,
        temperature=args.temperature,
        ignore_index=dataset.metadata.ignore_label_id,
    )
    projected, explained = pca_project(trace.states)
    spread = trajectory_spread(projected)

    distinct = len({row.tobytes() for row in trace.predictions[:, -1]})
    summary = {
        "checkpoint": args.checkpoint,
        "guidance": config.model.guidance,
        "instance": args.index,
        "num_trajectories": trace.num_trajectories,
        "num_steps": trace.num_steps,
        "explained_variance": explained.tolist(),
        "spread_per_step": spread.tolist(),
        "distinct_final_predictions": distinct,
        "final_loss": {
            "min": float(trace.losses[:, -1].min()),
            "mean": float(trace.losses[:, -1].mean()),
            "max": float(trace.losses[:, -1].max()),
        },
        "best_final_accuracy": float(trace.accuracies[:, -1].max()),
        "mean_final_accuracy": float(trace.accuracies[:, -1].mean()),
    }
    print(json.dumps(summary, indent=2))

    output = Path(args.output) if args.output else \
        Path(args.checkpoint).with_suffix(".trajectories.svg")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_svg(projected, trace.losses))
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(output.with_suffix(".npz"), projected=projected,
                        losses=trace.losses, accuracies=trace.accuracies)
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
