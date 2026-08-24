#!/usr/bin/env python3
"""Build a GRAM dataset.

Examples
--------
    python scripts/build_dataset.py nqueens --output data/nqueens8 --n 8
    python scripts/build_dataset.py graph_coloring --output data/gc8 --n 8
    python scripts/build_dataset.py sudoku --output data/sudoku --num-train 10000
    python scripts/build_dataset.py sudoku --output data/sudoku_uncond --unconditional
    python scripts/build_dataset.py mnist --output data/mnist --raw-dir /path/to/mnist
    python scripts/build_dataset.py arc --output data/arc1 --train-dir ARC-AGI/data/training \
        --eval-dir ARC-AGI/data/evaluation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gram.data import arc, graph_coloring, mnist, nqueens, sudoku  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="task", required=True)

    p = sub.add_parser("nqueens")
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--remove", type=int, nargs="+", default=None,
                   help="how many queens to remove (default 5 6 7 for n=8, 7 8 9 for n=10)")
    p.add_argument("--num-instances", type=int, default=20000)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--min-solutions", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("graph_coloring")
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--edge-prob", type=float, default=None)
    p.add_argument("--num-instances", type=int, default=7002)
    p.add_argument("--test-fraction", type=float, default=0.035)
    p.add_argument("--min-solutions", type=int, default=2)
    p.add_argument("--max-solutions", type=int, default=None)
    p.add_argument("--max-targets", type=int, default=32,
                   help="cap on distinct training targets stored per input")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("sudoku")
    p.add_argument("--output", required=True)
    p.add_argument("--num-train", type=int, default=10000)
    p.add_argument("--num-test", type=int, default=1000)
    p.add_argument("--min-clues", type=int, default=24)
    p.add_argument("--unconditional", action="store_true")
    p.add_argument("--no-unique", action="store_true",
                   help="skip the unique-solution check (much faster)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-csv", default=None, help="Sudoku-Extreme style CSV")
    p.add_argument("--test-csv", default=None)
    p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("mnist")
    p.add_argument("--output", required=True)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--conditional", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("arc")
    p.add_argument("--output", required=True)
    p.add_argument("--train-dir", required=True)
    p.add_argument("--eval-dir", default=None)
    p.add_argument("--augmentations", type=int, default=0)

    args = parser.parse_args()

    if args.task == "nqueens":
        remove = args.remove or ((5, 6, 7) if args.n <= 8 else (7, 8, 9))
        metadata = nqueens.build(
            args.output, n=args.n, remove=tuple(remove),
            num_instances=args.num_instances, test_fraction=args.test_fraction,
            seed=args.seed, min_solutions=args.min_solutions,
        )
    elif args.task == "graph_coloring":
        metadata = graph_coloring.build(
            args.output, n=args.n, edge_prob=args.edge_prob,
            num_instances=args.num_instances, test_fraction=args.test_fraction,
            seed=args.seed, min_solutions=args.min_solutions,
            max_solutions=args.max_solutions,
            max_train_targets_per_input=args.max_targets,
        )
    elif args.task == "sudoku":
        if args.train_csv:
            if not args.test_csv:
                parser.error("--train-csv requires --test-csv")
            metadata = sudoku.load_csv(
                args.output, args.train_csv, args.test_csv,
                unconditional=args.unconditional, limit=args.limit,
            )
        else:
            metadata = sudoku.build(
                args.output, num_train=args.num_train, num_test=args.num_test,
                min_clues=args.min_clues, seed=args.seed,
                unconditional=args.unconditional,
                require_unique=not args.no_unique,
            )
    elif args.task == "mnist":
        metadata = mnist.build(
            args.output, args.raw_dir, unconditional=not args.conditional,
            threshold=args.threshold, limit=args.limit,
        )
    else:
        metadata = arc.build(
            args.output, args.train_dir, args.eval_dir,
            augmentations=args.augmentations,
        )

    print(json.dumps(metadata.__dict__, indent=2, default=str))


if __name__ == "__main__":
    main()
