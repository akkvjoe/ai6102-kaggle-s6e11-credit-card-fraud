# Model choices and why

Decision record for S6E11. Results are measured, 5-fold **stratified** CV,
ROC AUC. Reproduce any row with `python -m scripts.run_experiment --model <name>`;
the full log is `results/experiments.csv`, and `notebooks/02_models.ipynb`
computes the same numbers through the same `src/` code.

## The data in one paragraph

699,960 train rows, 299,984 test rows, 24 features (13 numeric, 5 categorical,
6 boolean), **no missing values and no duplicate feature rows**. The target
`is_fraud` is positive in 11,689 rows — a rate of **0.01670, about 1 in 60**.
`sample_submission.csv` ships the constant 0.0166995, which matches the train
positive rate to 2e-8: the organisers seeded it with the base rate.

The imbalance is the defining property. Always predicting "not fraud" scores
98.33% accuracy and is worthless, which is why the metric is ROC AUC, and why
every fold below is stratified.

## Results

| Model | Rows | Mean fold AUC | Std | Fit (s) | Role |
|---|---|---|---|---|---|
| `hgb` (sklearn HistGradientBoosting) | 699,960 | **0.94526** | 0.00391 | 36 | best single model |
| `lgbm` (LightGBM, untuned) | 699,960 | 0.94277 | 0.00353 | 294 | tuning vehicle |
| `linear_main` (logistic, main effects) | 699,960 | 0.94139 | 0.00390 | 11 | control |
| `nb_binned` (CategoricalNB, 10 quantile bins) | 699,960 | 0.85080 | 0.00503 | 6 | independence test |
| `tree_d4` (depth-4 tree) | 699,960 | 0.79944 | 0.00491 | 11 | legibility floor |
| `linear_main` @150k | 150,000 | 0.94168 | 0.00611 | 6 | matched control |
| `linear_inter` @150k (all pairwise) | 150,000 | 0.92573 | 0.00746 | 168 | interaction test |

## Finding 1 — a plain logistic regression is within 0.004 of gradient boosting

`hgb` 0.94526 against `linear_main` 0.94139: a gap of **0.00387**, against a
fold-to-fold standard deviation of roughly 0.0039. Gradient boosting wins, but
by about one fold's worth of noise, in 3x the time.

This mirrors S6E9, where the same two models finished 0.0035 apart. Two
competitions in a row, a generalised linear model has come within half a
percent of tuned-ish boosting. The headline is not "trees beat linear models";
it is that these synthetic Playground datasets are close to generalised linear
in structure.

## Finding 2 — all pairwise interactions actively hurt

Expanding to all 1,225 pairwise polynomial terms moved AUC from 0.94168 to
0.92573 on the same 150,000-row subsample: **−0.01632**, against a fold std of
0.00746. More than two standard deviations in the wrong direction.

Three things make this a result rather than an artefact:

1. **It was run against a matched control.** Both arms use the identical
   150,000-row subsample and the same folds. Comparing a subsampled
   interaction model against full-data main effects would have measured the
   subsample, not the interactions.
2. **It is not a convergence failure.** Re-run with warnings surfaced, the
   only warning emitted is a benign sklearn `n_jobs` FutureWarning. No
   `ConvergenceWarning` appears.
3. **The mechanism is legible.** At 150,000 rows and a 1.67% positive rate
   there are roughly 2,500 positives against 1,225 free parameters — about two
   positive examples per parameter. The interaction model is fitting noise in
   the minority class. This is the imbalance biting the variance term, which is
   exactly the behaviour the small positive class predicts.

**Why the experiment is subsampled at all:** 24 features expand to 1,225
polynomial terms, which at 560,000 training rows is a 5.11 GiB dense float64
array. It raised `numpy._core._exceptions._ArrayMemoryError` on the first
attempt. Subsampling both arms and saying so is the honest response; dropping
the experiment would have left the motivation untested.

## Finding 3 — the signal is concentrated and mostly marginal

Univariate ROC AUC (descriptive only; the categorical encodings leak and are
not model features):

| Feature | Univariate AUC |
|---|---|
| `merchant_risk_score` | 0.74894 |
| `velocity_score` | 0.71079 |
| `cvv_retry_count` | 0.70808 |
| `auth_method` | 0.68657 |
| `is_new_merchant` | 0.64224 |

and the strongest level effects, against the 1.67% base rate:

| Feature | Level | n | Fraud rate | Lift |
|---|---|---|---|---|
| `cvv_retry_count` | 2 | 1,905 | 0.41732 | 25.0x |
| `ip_country_mismatch` | True | 11,312 | 0.16496 | 9.9x |
| `is_ai_generated_scam_attempt` | True | 2,299 | 0.15224 | 9.1x |
| `billing_shipping_mismatch` | True | 9,054 | 0.12989 | 7.8x |
| `is_foreign_transaction` | True | 12,098 | 0.12572 | 7.5x |

`nb_binned` reaching 0.85080 while assuming full conditional independence, in
six seconds, quantifies how much of the problem is independent marginal
evidence. The 0.09 it leaves on the table against `hgb` is what joint
structure is worth here.

## Deliberately excluded

**Kernel SVM** — roughly O(n^2); 699,960 rows is far outside feasible. To be
fitted on a stratified subsample and reported as a compute limitation rather
than omitted silently.

**k-NN** — 24 features, distance poorly defined over one-hot categoricals,
query cost linear in training size.

**Neural network / GPU** — permitted by the lecturer, but see Finding 1: a
linear model is within 0.004 of boosting on 24 clean tabular features. There is
no representational gap for a network to exploit, and nothing here is
GPU-bound. The binding constraints encountered so far were **RAM** (the 5.11
GiB interaction array) and **single-core CPU time**, neither of which a GPU
addresses.

## Finding 4 — cross-validation tracks the leaderboard

| | ROC AUC |
|---|---|
| `hgb` default, 5-fold CV on 699,960 rows | 0.94526 |
| Same predictions, public leaderboard | **0.94629** |
| Gap | **+0.00103** |

The leaderboard came in *above* CV, by about a quarter of the fold-to-fold
standard deviation. Validation is slightly conservative and well calibrated.

This licenses the whole tuning strategy: **select against CV, not the
leaderboard.** CV is computed on 699,960 rows; the public leaderboard is
~20% of the test set, 59,997 rows holding roughly 1,002 positives. By
Hanley-McNeil the standard error of an AUC near 0.946 on that sample is
**0.00497**, so the entire gap from rank 43 to rank 1 (0.00329) is 0.66
standard errors. The public leaderboard cannot distinguish most of the field;
CV can.

## Finding 5 — the optimal gradient-boosting model is a stump ensemble

Randomised search, three rounds, each against a matched control on the same
200,000-row subsample and folds:

| Round | `max_leaf_nodes` searched | Best | AUC @200k/3-fold |
|---|---|---|---|
| control | — (default 31) | 31 | 0.94047 |
| 1 | 15, 31, 63, 127 | **15** | 0.94215 |
| 2 | 4, 6, 8, 12, 15 | **4** | 0.94338 |
| 3 | 2, 3, 4, 5 | **2** | **0.94443** |

Every round's winner sat on the *lower* boundary of its grid, so every round's
grid was wrong and had to be extended downward. The search terminated at
`max_leaf_nodes = 2` — decision stumps, each tree splitting on one feature
once. **A stump ensemble cannot represent feature interactions at all.**

Best configuration: `learning_rate=0.05, max_leaf_nodes=2,
min_samples_leaf=20, l2_regularization=0.5, max_iter=2000, max_features=0.7`.

Validated at full data, 5 folds: **0.94612**, against the default's 0.94526 —
a gain of **+0.00086**. Note that this is far smaller than the +0.00396 the
subsample promised: 699,960 rows support model capacity that 200,000 do not,
so the default's excess complexity costs less at full scale. **A tuning gain
measured on a subsample overstates the gain at full data**, and the honest
number is the full-data one.

## Finding 6 — blending with the linear model adds exactly nothing

Rank-averaging the stump ensemble with `linear_main`, weight swept 0 to 1:

| w(stumps) | 0.0 | 0.3 | 0.5 | 0.7 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|
| Blend AUC | 0.94136 | 0.94388 | 0.94496 | 0.94567 | 0.94604 | **0.94610** |

Monotonic to w = 1. The optimum is pure stumps; the linear model contributes
nothing at any weight. Their OOF rank correlation is 0.9466.

The reason is structural, and it closes the argument. A stump ensemble is a
**generalised additive model**: a sum of univariate step functions, one
feature at a time. Logistic regression is also additive, with linear instead of
step-shaped terms. The stumps therefore *subsume* the linear model — they span
the same additive space and fit richer per-feature shapes within it. Blending a
model with a strictly weaker model of the same family cannot help.

## The converging argument

Three independent experiments point the same way:

1. `linear_main`, purely additive, lands within 0.0039 of default boosting.
2. All pairwise interactions **hurt** by 0.0163 against a matched control.
3. Boosting tuned freely drives itself down to stumps — **zero interactions** —
   and gains 0.0040 at 200k doing it.

**This dataset is additive in log-odds.** Not approximately, not as a
simplifying assumption: three different model families, asked independently how
much interaction structure they want, all answer none. Any feature engineering
built on crossing features is predicted to fail, and that prediction is now
worth stating in advance and testing.

## Where the headroom is

Worst to best across the full-data models spans 0.146 AUC, but that range is
carried almost entirely by `tree_d4` and `nb_binned`, which are diagnostics
rather than contenders. Among the three serious models the spread is 0.0039 —
one fold's noise. As in S6E9, leaderboard position will come from tuning and
blending in the fourth decimal, not from a better model class.

## Reproducibility note

`hgb` scores 0.94526 under scikit-learn 1.7.2 and 0.94535 under 1.9.1 on
identical data, folds and seed — a 9e-5 difference from the library version
alone. Report the version alongside any number quoted to five decimals.
