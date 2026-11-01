# Kaggle Playground S6E11 — Predicting Credit Card Fraud

AI6102 team project (NTU, Machine Learning Methodologies & Applications).

| | |
|---|---|
| Competition | [Playground Series S6E11](https://www.kaggle.com/competitions/playground-series-s6e11) |
| Task | Binary classification — probability that `is_fraud` = 1 |
| Metric | ROC AUC |
| Competition opened | 1 November 2026 |
| Final submission | **30 November 2026**, 11:59 PM UTC |
| Course report deadline | **20 November 2026** — PDF, emailed to the lecturer's postdoc |
| Team | Confirm size before adding a third member — see "Open questions" |

Because the competition is still running on 20 November, this counts as an
**ongoing** competition for the course, which is what makes the **+20% top-10%
bonus** reachable. S6E9, used earlier, closed on 30 September and does not.

## Rules this repo is built around

Kaggle's competition rules, clause 6a: privately sharing competition code
outside your team during the competition is a breach and may cause
disqualification. Clause 6b permits public sharing **only** on Kaggle's own
forums or notebooks.

**Keep this repository private until 30 November 2026**, and share it only with
registered teammates. Competition data is gitignored for the same reason — do
not commit `data/`.

## Layout

```
src/
  config.py        paths, column names, seed, fold count
  data.py          loading, category alignment across train/test, X/y split
  models.py        the model registry
  cv.py            stratified K-fold, out-of-fold predictions, ROC AUC
  submit.py        submission writer with format checks
  run_baseline.py  entry point
scripts/fetch_data.sh      download the competition data into data/
scripts/profile_data.py    profile the data, writing results to eda/
scripts/run_experiment.py  run one registered model, log to results/
scripts/make_synthetic.py  stand-in data for running the code without the real files
tests/test_pipeline.py     end-to-end smoke test
data/          gitignored — train.csv, test.csv, sample_submission.csv
submissions/   gitignored — generated submission CSVs
eda/           profiling output
results/       experiments.csv — the measured CV log
reports/       the LaTeX report
```

## Provenance

Ported from the S6E9 repository (`../kaggle-s6e9-ev-purchases`), which was
built deliberately schema-agnostic: the pipeline drops `id` and the target,
treats every remaining non-numeric column as categorical and passes the rest
through as numeric, so it does not depend on feature names. The port changed
only the target column, the competition slug and the documentation. The smoke
test passes on synthetic data and emits the correct `id,is_fraud` submission
header.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Get the data

Join the competition first — the rules must be accepted on the account the API
token belongs to — then:

```bash
./scripts/fetch_data.sh          # no-op if the files are already there
./scripts/fetch_data.sh --force  # re-download and overwrite
```

Needs the Kaggle CLI (`pip install kaggle`) and credentials — `kaggle auth
login` (OAuth), `KAGGLE_API_TOKEN`, or a token file at
`~/.kaggle/access_token` (`~/.kaggle/kaggle.json` is the legacy format).
Failing that, download the three files by hand from the
competition Data tab into `data/`.

## Run

```bash
python -m src.run_baseline               # 5-fold CV, writes submissions/baseline_<utc>.csv
python -m scripts.run_experiment --model hgb
python -m scripts.profile_data           # refresh eda/
python -m tests.test_pipeline            # smoke test, no real data needed
```

## What the report needs

Per the team project deck, slide 4: problem statement in your own words,
challenges, proposed solution with motivation and detailed steps, experiments
that **verify the motivations**, experiments showing performance with **both
cross-validation results and a leaderboard screenshot**, and what the team
learned. Include every member's name and matriculation number. LaTeX is
recommended; source code is not submitted.

Slide 8 grades **writing** (logical organisation, clarity, typos) and
**experiment extensiveness**. Leaderboard position is not itself graded — it
only triggers the bonus.

## Expected first move: the imbalance

Fraud datasets are heavily imbalanced, and that shapes the experimental plan
before any data is seen. Resampling, class weighting, threshold selection and
ROC-AUC versus PR-AUC behaviour each give a natural "verify your motivations"
experiment. Confirm the actual positive rate with `scripts/profile_data.py`
before committing to any of it.

## Open questions

- **Team size.** Lecture 1's schedule says `<=3 members/team`; the Lecture 3
  recording says "less than three", twice. Confirm with the postdoc before
  committing a third member.
- **Bonus timing.** The deck does not say how the top-10% bonus is judged when
  the competition is still running on the report deadline — presumably the
  leaderboard screenshot taken on 20 November, ten days before close. Confirm.

## Status

- [x] Repo scaffold ported from S6E9, smoke test passing
- [ ] Joined the competition on Kaggle
- [ ] Real data downloaded
- [ ] Data profiled -> eda/profile.md
- [ ] First leaderboard submission
- [ ] Model comparison across families
- [ ] Report
