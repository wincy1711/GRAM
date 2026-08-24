#!/usr/bin/env bash
# Reproduce the paper's experiments.  Each `train.py` call expects a GPU;
# Appendix B.2 reports 8x NVIDIA RTX 4090 for the original runs.
set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------- datasets --
python scripts/build_dataset.py nqueens        --output data/nqueens8  --n 8
python scripts/build_dataset.py nqueens        --output data/nqueens10 --n 10 --num-instances 40000
python scripts/build_dataset.py graph_coloring --output data/gc8       --n 8  --num-instances 7002
python scripts/build_dataset.py graph_coloring --output data/gc10      --n 10 --num-instances 13465
python scripts/build_dataset.py sudoku         --output data/sudoku    --num-train 100000 --num-test 1000
python scripts/build_dataset.py sudoku         --output data/sudoku_uncond --num-train 50000 --num-test 1000 --unconditional
# Sudoku-Extreme (used by HRM/TRM) can be substituted:
#   python scripts/build_dataset.py sudoku --output data/sudoku \
#       --train-csv sudoku-extreme/train.csv --test-csv sudoku-extreme/test.csv
# ARC-AGI and MNIST need their upstream data:
#   python scripts/build_dataset.py arc   --output data/arc1 \
#       --train-dir ARC-AGI/data/training --eval-dir ARC-AGI/data/evaluation
#   python scripts/build_dataset.py arc   --output data/arc2 \
#       --train-dir ARC-AGI-2/data/training --eval-dir ARC-AGI-2/data/evaluation
#   python scripts/build_dataset.py mnist --output data/mnist --raw-dir /path/to/mnist
# then:  python scripts/train.py --config configs/arc1.json   (likewise arc2, mnist)

# -------------------------------------------------------- main experiments --
for task in sudoku nqueens8 nqueens10 graph_coloring8 graph_coloring10; do
  python scripts/train.py --config "configs/${task}.json"
done

# ------------------------------------------------ inference-time scaling ----
python scripts/evaluate.py --checkpoint runs/sudoku/best.pt \
  --widths 1 5 10 20 50 --depths 8 16 32 128 320 --selection lprm \
  --output runs/sudoku/scaling.json

# ---------------------------------------------- multi-solution coverage -----
for task in nqueens8 nqueens10 graph_coloring8 graph_coloring10; do
  python scripts/evaluate.py --checkpoint "runs/${task}/best.pt" \
    --widths 20 --selection majority --output "runs/${task}/coverage.json"
done

# ----------------------------------------------- unconditional generation ---
python scripts/train.py --config configs/sudoku_uncond.json
python scripts/evaluate.py --checkpoint runs/sudoku_uncond/best.pt \
  --generate 100000 --steps 16 --output runs/sudoku_uncond/generation.json

# ------------------------------------------------------------- ablations ----
for variant in looped_tf hrm_trm looped_tf_sg ds_sg gram \
               stochastic_only guide_only no_guidance direct_pred; do
  python scripts/train.py --config "configs/ablations/nqueens8_${variant}.json"
  python scripts/train.py --config "configs/ablations/sudoku_${variant}.json"
done
