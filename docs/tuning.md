# Practical notes on training GRAM

## Choosing the KL coefficient β

The paper reports per-task β values (Appendix B.2): 0.1 for Sudoku, 0.04/0.1 for
ARC-AGI-1/2, 0.07/0.045 for N-Queens 8×8/10×10, 0.5/0.45 for Graph Coloring with
8/10 nodes, and 0.07 for MNIST. Those are tied to the paper's architecture:
`D = 512`, `T = 3`, `N_sup = 16`, i.e. 48 latent transitions each carrying a
512-dimensional Gaussian.

**β does not transfer across scales.** The objective is

```
CE(mean over tokens)  +  β · KL(sum over latent dims, mean over positions)
```

so the KL term grows with the *total latent capacity* — roughly
`D × T × N_sup`. Shrink the model and the same β applies proportionally less
pressure. As a rule of thumb, when moving from the paper's configuration to
another one, scale β by

```
β_new ≈ β_paper × (D_paper × T_paper × N_sup_paper) / (D_new × T_new × N_sup_new)
```

### Why this matters: the private-channel failure

GRAM is trained with the posterior `q_φ(ε|u, y)` in the loop, so `ε` can become
a private channel that smuggles `y` to the decoder. When that happens you see a
very characteristic signature in the training log:

```
rec=0.0020  kl=0.0524  post_acc=0.999  prior_acc=0.007
```

Reconstruction is near-perfect, the KL looks reassuringly small, and yet the
*prior* rollout — the only thing that exists at inference — is at chance. The
KL is small because the posterior only needs to shift `μ_φ` by a fraction of
`σ` in each of `D` dimensions at each of `T × N_sup` transitions; integrated
over thousands of elements that is a high-SNR signal for the decoder, but a
prior sample never lands on it.

This is why the training loop logs `prior_accuracy` alongside `accuracy`.
**`prior_accuracy` is the metric to watch** — `accuracy` measures the posterior
rollout and will look excellent even when the model is unusable.

A measured sweep on the 6-vertex Graph Coloring demo (`D = 128`, `T = 2`,
`N_sup = 4`, i.e. ~24× less latent capacity than the paper's setting) shows the
transition clearly:

| β | reconstruction | KL | posterior acc. | prior acc. |
| --- | --- | --- | --- | --- |
| 0.5 (the paper's value for this task) | 0.002 | 0.052 | 0.999 | 0.007 |
| 2.0 | 0.635 | 0.037 | 0.037 | 0.037 |
| 8.0 | 0.664 | 0.003 | 0.037 | 0.074 |

At β = 0.5 the posterior and prior rollouts differ by 140×. At β ≥ 2 they
coincide, which is what a healthy variational model looks like. Note that
`0.5 × 24 ≈ 12` — the same order as where the transition actually happens.

At the other extreme (β = 8 here) the KL collapses to ~0 and GRAM degenerates
into its deterministic ablation: the stochastic guidance stops carrying
anything. Useful β values sit in the band where `prior_accuracy ≈ accuracy`
*and* the KL is still clearly non-zero.

### The gap grows over training

Raising β delays the private channel rather than forbidding it. On the same
demo at β = 2, the posterior/prior ratio widens steadily as training proceeds:

| epoch | KL | posterior acc. | prior acc. | ratio |
| --- | --- | --- | --- | --- |
| 4 | 0.038 | 0.068 | 0.041 | 1.6× |
| 8 | 0.033 | 0.108 | 0.059 | 1.8× |
| 12 | 0.057 | 0.215 | 0.082 | 2.6× |
| 16 | 0.062 | 0.722 | 0.069 | 10.4× |
| 20 | 0.038 | 0.943 | 0.050 | 19.0× |
| 24 | 0.021 | 0.984 | 0.041 | 23.9× |

So `prior_accuracy` is worth watching for the whole run, not just at the end,
and the checkpoint should be selected on a metric that reflects the prior
rollout. `train.select_metric` exists for this: the multi-solution configs set
it to `constraint_accuracy`, which on this run picks epoch 16 — the point
where validity peaks (0.528) and before the gap opens up.

Two things make this less likely at the paper's scale: much larger latent
capacity (so a given amount of smuggled information costs proportionally more
KL), and 48 transitions rather than 8 (so the deterministic recursion is
strong enough that the shortcut is not the cheaper option). The paper's own
Table 3b is consistent with this — its `stochasticity only` ablation, whose
posterior can only modulate `σ` and therefore has a far weaker channel,
performs on par with full GRAM on Sudoku (94.88 vs 93.96).

## Diagnosing a run

| Symptom | Likely cause | Lever |
| --- | --- | --- |
| `accuracy` ≫ `prior_accuracy` | posterior is a private channel | raise `train.beta` |
| `kl` ≈ 0 and `sample_diversity` ≈ 0 | posterior collapse; GRAM ≡ deterministic RRM | lower `train.beta`, raise `train.kl_balance`, or set `train.free_bits` > 0 |
| `mean_halt_step` = 1 with poor accuracy | ACT head is over-confident | check `train.head_rollout` is `"prior"` |
| `sample_diversity` high, `constraint_accuracy` low | prior samples land between solution modes | more recursion (`n_supervision`, `high_level_steps`) or a longer run |

`gram.evaluate.full_elbo` computes the untruncated bound of Eq. (13). Tracking
it next to the training surrogate reproduces the Appendix A.3 check that
optimising `L_GRAM` really does improve the full variational bound.

## KL balancing

`train.kl_balance = 0.8` (the paper's value) puts 80 % of the gradient on
`KL(sg(q) ‖ p)`, which trains the *prior* towards the posterior, and 20 % on
`KL(q ‖ sg(p))`, which pulls the posterior back. Raising it further makes the
prior chase the posterior harder, which helps when `prior_accuracy` lags but
does not by itself close a large gap — β is the stronger lever.

## Compute

The paper trains on 8× RTX 4090 (Appendix B.2, Table 7): 2 h for Sudoku, 1–6 h
for the multi-solution tasks, 16 h for MNIST and 5 days for ARC-AGI. The demo
configs in `configs/demo/` are sized for a laptop CPU and are meant for
verifying the pipeline and reproducing the qualitative multi-solution effect,
not the paper's numbers.
