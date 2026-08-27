# Portfolio Loss Modeling — LendingClub

**A risk analyst's loss study on 2.26 million real consumer loans: vintage curves, a
default-probability model, and a portfolio tail view.**

Credit risk and catastrophe risk are the same intellectual move performed on
different exposures. You take a portfolio of correlated risks, model the
distribution of loss, and reason about the tail. The vocabulary diverges — a credit
team says *vintage curve* where a reinsurance team says *loss development triangle*,
*expected loss* where an ILS investor says *AAL*, and *loss distribution* where a
cat modeler says *EP curve* — but the machinery underneath is one thing.

This study is built on consumer credit data and framed around portfolio loss, so it
reads as relevant on either side of that bridge.

| Credit concept | Reinsurance / ILS analog | Where it appears |
|---|---|---|
| Vintage curve | Loss development triangle | [Notebook 02](notebooks/02_vintage_curves.ipynb) |
| Cumulative charge-off | Ultimate loss ratio | [Notebook 02](notebooks/02_vintage_curves.ipynb) |
| Segmentation by FICO / grade | Risk-based pricing, exposure segmentation | [Notebook 03](notebooks/03_segmentation.ipynb) |
| Frequency × severity | Frequency × severity | [Notebook 03](notebooks/03_segmentation.ipynb) |
| Portfolio expected loss | AAL (average annual loss) | [Notebook 05](notebooks/05_portfolio_tail.ipynb) |
| Loss distribution / tail | EP curve (exceedance probability) | [Notebook 05](notebooks/05_portfolio_tail.ipynb) |
| Unexpected loss | Capital requirement above the technical rate | [Notebook 05](notebooks/05_portfolio_tail.ipynb) |

**📄 [Read the report](REPORT.md)** — the plain-English narrative, no code required.

---

## The four findings

**1. LendingClub's credit quality deteriorated steadily from 2013 to 2016, and the
aggregate loss rate hid it.**

Loss at 12 months on book rose from 2.9% (2013 vintage) to 4.7% (2016) on identical
36-month product. The whole-book loss rate — a single 8% number — blends a 5.9%
cohort with one tracking toward 9% and shows no trend at all.

![Vintage curves](figures/02_vintage_curves.png)

**2. The 2017 "improvement" was mostly a mix shift, not tighter underwriting.**

Decomposing the 0.32-point improvement: 0.19 points came from reweighting the book
toward better FICO bands, only 0.13 from tighter standards within bands. At the
bottom of the score range 2017 was marginally *worse* than 2016. A mix shift is a
volume decision and reverses with the next growth target; tighter underwriting
persists. The aggregate number cannot tell them apart.

**3. Better information beat a better model.** A logistic regression on borrower
attributes reaches 0.657 out-of-time AUC. LightGBM on the same features reaches
0.675. Adding LendingClub's own loan grade to the *logistic* model reaches 0.684 —
better than the boosted model. The interpretable model is the right production
choice, and that is a modeling judgement rather than a limitation.

**4. The portfolio's tail is made entirely of correlation.**

The tail simulation runs on the **2015–2016 holdout book** — 333,721 loans, $4.32B —
rather than the full 732,629. That is deliberate: the loan-level probabilities come
from a model that never saw these vintages, and a loss distribution built on
in-sample fitted probabilities understates its own dispersion, because the model has
already been shown the answers. (See [the reconciliation](#how-the-loan-counts-reconcile) below.)

Simulate those loans defaulting independently and the distribution collapses to a
spike — standard deviation $1.5M on a $322M expected loss. Add a single systematic
factor at the Basel retail correlation and the standard deviation becomes $86.4M,
with a 1-in-100 loss of $558M. Same loans, same expected loss, entirely different
risk.

![Loss distribution](figures/05_loss_distribution.png)

Diversification does not protect against a common cause. That sentence belongs in a
credit committee and a reinsurance underwriting meeting equally.

---

## The analysis

| Notebook | What it does |
|---|---|
| [`01_data_prep`](notebooks/01_data_prep.ipynb) | 2.26M raw loans → 732,629 resolved, fully seasoned loans. Binary target, explicit leakage screen, dollar loss measurement. |
| [`02_vintage_curves`](notebooks/02_vintage_curves.ipynb) | **The centerpiece.** Cumulative charge-off by months on book, per origination cohort. A loss development triangle in credit clothing. |
| [`03_segmentation`](notebooks/03_segmentation.ipynb) | Loss by FICO band and loan grade, split into frequency and severity. Pricing adequacy. Mix-vs-quality decomposition. |
| [`04_default_model`](notebooks/04_default_model.ipynb) | Logistic regression on origination-time features, interpreted coefficient by coefficient. Out-of-time validation, calibration, decile lift. LightGBM as a benchmark. |
| [`05_portfolio_tail`](notebooks/05_portfolio_tail.ipynb) | Portfolio expected loss, Vasicek single-factor simulation, EP curve, return-period table, correlation sensitivity. |

### How the loan counts reconcile

Different sections quote different loan counts. Every step is a deliberate filter,
and they add up exactly:

| Stage | Loans | Why the count changes |
|---|---:|---|
| Raw file | 2,260,701 | — |
| Resolved outcome only | 1,345,350 | Current / Late / In Grace Period have no outcome, so no label |
| Fully seasoned | **732,629** | Full contractual term elapsed before the March 2019 snapshot |
| ↳ minus sub-660 FICO | 732,627 | 2 policy exceptions; a segment built on two loans is noise |
| ↳ **train** — vintages ≤ 2014 | 398,906 | Where the default model is *fit* |
| ↳ **holdout** — vintages 2015–2016 | **333,721** | Where the model is *scored* — and the book notebook 05 simulates |

`398,906 + 333,721 = 732,627`. The split is by origination year rather than at
random, because the job is to underwrite loans that have not been written yet — see
notebook 04.

### On leakage

The single most common error in credit-modeling writeups is predicting default from
evidence of default. This dataset makes it easy to commit: `debt_settlement_flag` is
`Y` for borrowers who negotiated a settlement — which happens *after* they stop
paying — and implies charge-off 100% of the time. Feed it to a model and you get a
0.99 AUC that means nothing.

Notebook 01 excludes 38 post-origination columns in three named groups (payment
performance, post-funding credit refresh, distress flags) and says why for each. The
model sees only what an underwriter could have seen on the day the loan was funded.

---

## Running it

```bash
git clone <this repo>
cd portfolio-loss-modeling
pip install -r requirements.txt

# Download the raw data (~1.7 GB) -- see data/README.md for details
mkdir -p data/raw
curl -L -o data/raw/accepted_2007_to_2018Q4.csv \
  "https://huggingface.co/datasets/codesignal/lending-club-loan-accepted/resolve/main/accepted_2007_to_2018Q4.csv"

jupyter lab notebooks/
```

Run the notebooks in order. `01` does the one slow pass over the raw CSV (~40
seconds) and writes two Parquet tables that everything downstream reads. Notebooks
02–05 each run in well under a minute. Every notebook runs top to bottom without
manual intervention, and the figures in this README are regenerated by running them.

**Stack:** pandas, scikit-learn, statsmodels, matplotlib, LightGBM. Nothing exotic —
every technique here can be explained in a sentence.

```
portfolio-loss-modeling/
├── README.md              # you are here
├── REPORT.md              # the written narrative
├── requirements.txt
├── data/
│   └── README.md          # where to download the dataset (raw data is not committed)
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_vintage_curves.ipynb
│   ├── 03_segmentation.ipynb
│   ├── 04_default_model.ipynb
│   └── 05_portfolio_tail.ipynb
├── src/
│   ├── data_prep.py       # cleaning, target definition, leakage lists, triangle builder
│   └── plotting.py        # palette and chart helpers
└── figures/               # exported charts
```

`.py` copies of each notebook sit alongside the `.ipynb` files (paired via
[jupytext](https://jupytext.readthedocs.io/)) so the diffs are readable in review.

---

## Data

**LendingClub accepted loans, 2007–2018Q4** — 2,260,701 loans, 151 columns,
performance observed through March 2019. Public dataset; the raw file is not
committed. See [`data/README.md`](data/README.md) for the download link and the
directory layout.

## Scope

Deliberately excluded, and worth saying so: no deep learning, no stacked ensembles,
no feature that cannot be explained in one sentence. The point of this study is a
loss model whose every choice is defensible out loud — not a leaderboard score.

## What I'd do next

The four things this study does not do, in the order I'd tackle them:

1. **Stress the tail against a downturn.** The correlation is calibrated on
   2012–2015 — a window with no recession in it. Overlaying the 2007–2009 experience
   would replace an extrapolation with a scenario.
2. **Model LGD instead of holding it at 52%.** Severity is stable enough to fix
   across the whole book, but it runs 45% to 66% across the grade scale — so a
   grade- and term-conditional LGD would sharpen expected loss where it is worst.
3. **Make PD macro-conditional.** The model under-predicts out of time by 12%; a
   time-varying intercept would explain that structurally rather than patching it
   with a recalibration scalar.
4. **Give the 60-month book its own triangle.** It is excluded here to keep the
   vintage curves comparable, and it is the structurally riskier half.
