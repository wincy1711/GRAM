#!/usr/bin/env python3
"""Train a GRAM model.

    python scripts/train.py --config configs/nqueens8.json
    python scripts/train.py --config configs/nqueens8.json --set train.epochs=100 model.guidance=none
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gram.config import ExperimentConfig, apply_overrides  # noqa: E402
from gram.train import Trainer  # noqa: E402


def parse_overrides(pairs) -> dict:
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"override {pair!r} must look like key.path=value")
        key, value = pair.split("=", 1)
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", dest="overrides",
                        help="dotted config overrides, e.g. train.epochs=50")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    apply_overrides(config, parse_overrides(args.overrides))

    trainer = Trainer(config)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    best = trainer.fit()
    print("\nbest evaluation metrics:")
    print(json.dumps(best, indent=2, default=float))


if __name__ == "__main__":
    main()
