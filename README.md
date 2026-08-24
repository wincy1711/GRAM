# GRAM — Generative Recursive Reasoning Models

A reference implementation of **[Generative Recursive Reasoning](Generative%20Recursive%20Reasoning.pdf)**
(Baek, Jo, Kim, Ren, Bengio & Ahn), which turns deterministic Recursive
Reasoning Models (RRMs) into **probabilistic multi-trajectory computation**.

Prior RRMs such as HRM and TRM refine a latent state deterministically: given
the same input they follow one trajectory and converge to one prediction.
GRAM makes each latent transition *stochastic*, so repeated computation induces
a distribution over reasoning trajectories. The result is a latent-variable
generative model that supports conditional reasoning `p(y|x)`, unconditional
generation `p(x)`, and inference-time scaling along **two** axes — recursive
depth *and* the number of trajectories sampled in parallel.

```
                        ┌──────────────────── supervision step (T transitions) ───────────────────┐
  x ──► f_enc ──► e_x   │   l ← f_L(l, h + e_x)  ×K      u = f_H(h, l)      h = u + ε             │──► f_dec ──► ŷ
                        │   (deterministic low level)    (deterministic)    ε ~ N(μ_θ(u), σ_θ(u)²) │
                        └────────────────────────────── ×N_sup ────────────────────────────────────┘
```

## What is implemented

| Paper component | Where |
| --- | --- |
| Hierarchical recursive core `z = (h, l)`, Eq. (6)–(9) | [`gram/core.py`](gram/core.py) |
| Learnable stochastic guidance `ε ~ N(μ_θ(u), σ_θ(u)²I)`, Eq. (4)–(5) | [`gram/guidance.py`](gram/guidance.py) |
| Encoder / decoder, puzzle embeddings, image patch stem (Tables 4–5) | [`gram/model.py`](gram/model.py) |
| Truncated ELBO surrogate `L_GRAM`, Eq. (14), with KL balancing | [`gram/losses.py`](gram/losses.py) |
| Deep supervision with truncated gradients | [`gram/train.py`](gram/train.py) |
| Adaptive computation time (halt head), Appendix A.1 | [`gram/losses.py`](gram/losses.py), [`gram/inference.py`](gram/inference.py) |
| Latent Process Reward Model, Appendix A.2 | [`gram/losses.py`](gram/losses.py), [`gram/inference.py`](gram/inference.py) |
| Width scaling: parallel sampling + majority vote / best-of-N, Section 2.3 | [`gram/inference.py`](gram/inference.py) |
| Full trajectory ELBO, Eq. (13) — the Appendix A.3 diagnostic | [`gram/evaluate.py`](gram/evaluate.py) |
| Tasks: Sudoku, N-Queens, Graph Coloring, binarised MNIST, ARC-AGI | [`gram/data/`](gram/data) |
| Metrics: accuracy, coverage, conflict edges, validity, IS/FID | [`gram/metrics.py`](gram/metrics.py) |
| Latent trajectory analysis (PCA projection), Appendix D.6 | [`gram/analysis.py`](gram/analysis.py) |
| Ablations of Table 3 (architecture and mechanism) | [`configs/ablations/`](configs/ablations) |

## Install

```bash
pip install -r requirements.txt      # torch >= 2.1, numpy
pip install -e .                     # optional: install the `gram` package
```

## Quick start

```bash
# 1. build a dataset (fully self-contained — nothing is downloaded)
python scripts/build_dataset.py nqueens --output data/nqueens8 --n 8

# 2. train
python scripts/train.py --config configs/nqueens8.json

# 3. evaluate, sweeping the width axis (Figure 4)
python scripts/evaluate.py --checkpoint runs/nqueens8/best.pt \
    --widths 1 5 10 20 --selection lprm
```

There is also a CPU-sized demo showing the central claim — stochastic guidance
recovers multiple solutions where deterministic recursion collapses to one. See
[Reproduced behaviour](#reproduced-behaviour) below for the commands and what
they produce.

## Architecture

**Stochastic latent transition.** Each transition first computes a
deterministic update `u_t` from the previous state and the input embedding,
then samples a state-dependent perturbation and adds it:

```
l_{t,k} = f_L(h_{t-1}, l_{t,k-1}, e_x),  k = 1..K      # deterministic refinement
u_t     = f_H(h_{t-1}, l_t)                            # deterministic proposal
ε_t     ~ N(μ_θ(u_t), σ_θ(u_t)² I)                     # learnable stochastic guidance
h_t     = u_t + ε_t
```

`μ_θ` steers the trajectory; `σ_θ` controls exploration. Noise enters only at
the high level, where it can redirect the overall reasoning trajectory; the
low-level refinement stays deterministic (footnote 3 of the paper). The mean
head is zero-initialised, so training starts from the deterministic update and
learns to depart from it.

**Training.** GRAM is trained by amortised variational inference. Trajectories
are drawn from a target-conditioned posterior `q_φ(ε_t | u_t, y)` during
training and from the prior `p_θ(ε_t | u_t)` at inference. Gradients are
propagated only through the last transition of each supervision step, giving
the memory-constant surrogate

```
L_GRAM^(n) = E_q[ log p(y | z_T^(n), x) ] − KL( q_φ(ε_T | u_T, y) ‖ p_θ(ε_T | u_T) )
```

`gram.evaluate.full_elbo` computes the untruncated bound of Eq. (13) so you can
reproduce the Appendix A.3 check that the surrogate really does improve it.

**Inference-time scaling.** `sample_trajectories` folds the `N` parallel
trajectories into the batch dimension, so width costs one forward pass, not
`N`. Candidates are reduced by majority voting or by best-of-N with the LPRM
value head. Depth is controlled by `n_supervision` and optionally cut short per
trajectory by the ACT halt head.

## Looking at the trajectories

`scripts/visualize_trajectories.py` samples many prior trajectories for one
problem, projects the high-level state to 2-D with PCA, and writes a
standalone SVG (Appendix D.6, Figures 18–19):

```bash
python scripts/visualize_trajectories.py --checkpoint runs/nqueens8/best.pt \
    --num-samples 50 --output runs/nqueens8/trajectories.svg
```

It also reports `spread_per_step` — the mean pairwise distance between
trajectories at each recursion step — and how many *distinct* answers the
samples produced. On a deterministic checkpoint both collapse; on GRAM the
spread grows with depth:

```
guidance="none"    spread_per_step [0.000, 0.000, 0.000, 0.000]
                   1 distinct prediction from 30 trajectories
                   final loss  min 0.462  mean 0.462  max 0.462

guidance="full"    spread_per_step [0.102, 0.174, 0.205, 0.229]
                   45 distinct predictions from 50 trajectories
                   final loss  min 0.027  mean 0.789  max 2.125
                   final accuracy  best 1.000  mean 0.687
```

The spread of final losses is the point of Figure 19: some trajectories get
stuck in a bad region while others reach the solution, which is what makes
best-of-N selection worth doing. A deterministic model has one trajectory and
therefore one outcome, good or bad.

## Reproduced behaviour

The paper's headline numbers need 8× RTX 4090 (Appendix B.2). What follows was
produced by this code on a 4-core CPU, on the 6-vertex Graph Coloring demo:
a 0.4 M-parameter model (`D = 128`, `K = 2`, `T = 2`, `N_sup = 4`) trained for
40 epochs on 5.9 K examples. The *scale* is nothing like the paper's; the
*qualitative behaviour* is the thing being checked.

**Deterministic recursion gets nothing from width** — Figure 1(a) and Figure 4.
Evaluated at N = 1, 5 and 20 parallel trajectories, `guidance="none"` returns
literally the same answer every time:

| N | exact | valid | coverage | diversity |
| --- | --- | --- | --- | --- |
| 1 | 0.415 | 0.585 | – | 1.00 |
| 5 | 0.415 | 0.585 | 0.105 | 0.20 |
| 20 | 0.415 | 0.585 | 0.105 | 0.05 |

`diversity = 1/N` at every width: all 20 samples are one trajectory. Sampling
50 trajectories through `scripts/visualize_trajectories.py` gives
`spread_per_step: [0, 0, 0, 0]` and a single distinct prediction.

**GRAM scales with width.** The same backbone with `guidance="full"`:

| N | exact | valid | coverage | conflicts ↓ | diversity |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.132 | 0.321 | – | 0.943 | 1.00 |
| 5 | 0.151 | 0.302 | 0.198 | 0.981 | 0.70 |
| 20 | 0.189 | 0.377 | 0.318 | 0.868 | 0.42 |

Accuracy rises with the number of sampled trajectories, coverage rises from
0.198 to 0.318, and conflict edges fall — the second scaling axis of
Section 2.3, which the deterministic model does not have. At matched training
budget GRAM reaches **3.7× the solution coverage** of its deterministic
ablation (0.341 vs 0.091), which is the multi-solution result of Table 1 and
Figure 4 (right).

Single-sample accuracy is *lower* than the deterministic baseline at this size.
That is expected: with one target sampled per input from many valid ones, the
deterministic model converges quickly onto a single mode, while GRAM is
solving the harder problem of representing the whole solution set — and it is
doing so with a 0.4 M-parameter model where the paper uses 10 M.

**Depth extrapolation does not come for free at this size.** Sweeping
`--depths 2 4 8 16` on a model trained with `N_sup = 4` peaks at the trained
depth (exact 0.208) and degrades beyond it (0.094 at depth 16). The paper
reports the opposite on MNIST generation — quality improving monotonically from
8 to 256 steps, well past the 16 used in training (Table 2) — but that is a
converged 12.7 M-parameter model. The depth axis is wired up and sweepable
here; this particular checkpoint is simply not good enough to extrapolate along
it.

**The truncated surrogate tracks the full bound** (Appendix A.3). On the same
checkpoint, `--elbo` reports a full-trajectory KL of 0.649 summed over all 8
transitions against ~0.264 for the 4 that the surrogate actually penalises —
the gap is the cumulative KL of the earlier transitions, as the paper
describes, not an optimisation failure.

Reproduce with:

```bash
python scripts/build_dataset.py graph_coloring --output data/gc6_demo \
    --n 6 --num-instances 1500 --max-targets 5 --seed 11
python scripts/train.py --config configs/demo/gc6_gram.json
python scripts/train.py --config configs/demo/gc6_deterministic.json
python scripts/evaluate.py --checkpoint runs/gc6_demo_gram/best.pt --widths 1 5 20
python scripts/summarize.py runs/gc6_demo_gram runs/gc6_demo_deterministic
```

## Configuration

Configs are plain JSON mapping onto the dataclasses in
[`gram/config.py`](gram/config.py). Any field can be overridden on the command
line:

```bash
python scripts/train.py --config configs/nqueens8.json \
    --set train.beta=0.05 model.guidance=stochastic_only eval.num_samples=20
```

Key model knobs:

| Field | Meaning | Paper default |
| --- | --- | --- |
| `hidden_size`, `num_heads`, `ffn_hidden_size` | backbone width | 512 / 8 / 512 |
| `low_level_steps` (K) | low-level refinements per transition | 6 (Sudoku), 4 elsewhere |
| `high_level_steps` (T) | transitions per supervision step | 3 |
| `n_supervision` (N_sup) | supervision steps | 16 |
| `mixer` | `attention`, or `mlp` for the `[SwiGLU + SwiGLU]` Sudoku core | task-dependent |
| `guidance` | `full` / `stochastic_only` / `guide_only` / `none` | `full` |
| `hierarchical`, `deep_supervision` | architecture ablations of Table 3a | `true` |
| `puzzle_emb_tokens` | 16 prepended puzzle tokens (ARC only) | 16 / 0 |

Training defaults follow Appendix B.2: AdamW, lr `1e-4`, weight decay `1.0`,
gradient clipping `1.0`, global batch `768`, EMA decay `0.9999`, KL balance
`0.8`, and the per-task KL coefficient `β` listed there.

The resulting parameter counts match the paper: **10.5M** for GRAM on N-Queens
and Graph Coloring (paper: 10M), against **7.3M** for the same backbone with
`guidance="none"` (paper: 7M for Looped TF / TRM).

## Watching a run

The training log distinguishes two accuracies, and the difference matters:

* `accuracy` — the *posterior* rollout used for the ELBO. The posterior sees
  `y`, so this looks excellent even when the model is unusable.
* `prior_accuracy` — a rollout under the learned prior, which is the only thing
  that exists at inference. **This is the metric to watch.**

A large gap between them means the posterior has turned `ε` into a private
channel for `y`, and the fix is a larger `train.beta`. The KL coefficient does
not transfer between model sizes: it scales with the total latent capacity
`D × T × N_sup`. [`docs/tuning.md`](docs/tuning.md) works through this with a
measured sweep, and lists the other failure signatures and their levers.

## Ablations

`configs/ablations/` contains both halves of Table 3, for N-Queens and Sudoku:

*Architecture (Table 3a)* — `looped_tf` (flat, no deep supervision, no
guidance), `hrm_trm` (+ deep supervision + hierarchical recursion),
`looped_tf_sg` (+ stochastic guidance), `ds_sg`, and the full `gram`.

*Mechanism (Table 3b)* — `stochastic_only` (`μ = 0`), `guide_only` (`σ = 0`),
`no_guidance`, and `direct_pred` (a single-pass 8-layer transformer).

```bash
python scripts/train.py --config configs/ablations/nqueens8_stochastic_only.json
```

## Datasets

| Task | Sequence | Vocab | Source |
| --- | --- | --- | --- |
| Sudoku | 81 | 11 | generated here; `--train-csv` accepts Sudoku-Extreme |
| N-Queens | N² | 3 | generated here, with **all** valid completions per input |
| Graph Coloring | n(n−1)/2 | 6 | generated here (Erdős–Rényi, canonical 3-colourings) |
| MNIST | 196 patches | 3 | `--raw-dir` pointing at the IDX files |
| ARC-AGI | 900 | 12 | `--train-dir` pointing at the ARC JSON tasks |

Sudoku, N-Queens and Graph Coloring are generated from scratch with exact
solvers, so no download is required. N-Queens and Graph Coloring additionally
store the complete solution set for every input in `solutions.npz`, which is
what makes the exact coverage metric of Table 1 possible.

## Metrics

* `exact_match`, `token_accuracy` — structured reasoning (Section 4.1).
* `constraint_accuracy` — does the output satisfy the task constraints, whether
  or not it equals the reference solution (Table 1's "Accuracy").
* `coverage` — unique valid solutions discovered over `N` samples, divided by
  the number that exist (Table 1's "Coverage").
* `conflict_edges` — constraint-violating edges for graph colouring (lower is
  better).
* `gen_validity` / `gen_unique_valid` — unconditional Sudoku generation
  (Appendix D.5); `inception_score` / `frechet_distance` for images.

Note that `inception_score` and `frechet_distance` take *features* rather than
images, so any feature extractor can be plugged in; the paper uses a standard
Inception network on binarised MNIST.

## Tests

```bash
python -m pytest tests/ -q
```

The suite covers layer invariants (RoPE norm preservation, RMS normalisation,
permutation equivariance of attention), the KL closed form and its balancing,
each guidance mode, gradient truncation, the exact solvers behind every
dataset, ACT/LPRM behaviour, candidate selection, and an end-to-end training
run that must reduce the loss.

## Repository layout

```
gram/                 the library
  config.py           dataclass configs, JSON (de)serialisation, dotted overrides
  layers.py           RMSNorm, RoPE, attention, SwiGLU, sequence-mixing SwiGLU
  guidance.py         Gaussian heads, prior/posterior, stochastic transition
  core.py             latent state, f_L / f_H, transitions, supervision steps
  model.py            encoder, decoder, patch encoder, halt and value heads
  losses.py           ELBO surrogate, KL balancing, ACT and LPRM objectives
  train.py            deep-supervision loop, optimiser, Trainer
  evaluate.py         metric suite, full ELBO, scaling sweeps
  inference.py        prior sampling, ACT halting, majority vote / best-of-N
  metrics.py          accuracy, coverage, conflicts, validity, IS/FID
  ema.py, utils.py    EMA weights, seeding, device selection, JSONL logging
  data/               per-task generators, loaders and tokenisation
configs/              per-task configs, ablations/, demo/
  analysis.py         trajectory collection, PCA projection, SVG rendering
scripts/              build_dataset, train, evaluate, summarize,
                      visualize_trajectories, reproduce.sh
tests/                104 unit and integration tests
docs/tuning.md        choosing beta, diagnosing a run
```

## Scope

This repository implements GRAM itself, plus the deterministic recursive
models it extends — Looped Transformer, HRM/TRM and single-pass direct
prediction are reached by config, since the paper frames them as GRAM with
components removed (Table 3a).

The *external* baselines the paper compares against — the autoregressive
Transformer and MDLM in Table 1, and D3PM and the VAE in Table 2 and Figure 5 —
are separate model families and are not reimplemented here. The paper lists
their upstream repositories in Table 10.

## Deviations from the paper

The paper leaves a few implementation details unspecified; the choices made
here are listed so they are easy to change:

1. **Auxiliary-head rollouts.** The ACT and LPRM heads are fitted on a
   *prior* rollout rather than the posterior rollout used for the ELBO. The
   posterior sees `y`, so its predictions are near-perfect and a halt head
   trained on them halts at step 1 at inference. Set
   `train.head_rollout="posterior"` to use the (cheaper) posterior rollout.
2. **ACT variant.** `act_mode="halt_only"` is the default, matching the
   paper's note that its released code halts on `σ(q_halt) > 0.5` without the
   continue branch. `act_mode="q_learning"` implements the full Eq. (15) with
   the bootstrapped target.
3. **LPRM target.** The regression target `r` is the token accuracy of the
   trajectory's final prediction, applied to every supervision step.
4. **Posterior conditioning.** `q_φ` conditions on `u_t + embed(y)`, matching
   Table 4's "SwiGLU MLP for each parameter" with no separate target encoder.
   `model.posterior_target_proj=true` adds one.
5. **`guide_only` KL.** With `σ = 0` both distributions are Dirac deltas and
   the KL is undefined; the squared distance between the means is used, which
   is its `σ → 1` limit. (The paper reports this variant fails outright.)
6. **Halt-head loss.** Binary cross-entropy on the Q-logits, as in HRM, rather
   than the squared error literally written in Eq. (15) — `σ(q_halt)` at
   inference implies the head outputs logits.

## Citation

```bibtex
@article{baek2026gram,
  title  = {Generative Recursive Reasoning},
  author = {Baek, Junyeob and Jo, Mingyu and Kim, Minsu and Ren, Mengye and
            Bengio, Yoshua and Ahn, Sungjin},
  journal = {arXiv preprint arXiv:2605.19376},
  year   = {2026}
}
```

Project page: <https://ahn-ml.github.io/gram-website>
