# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 — Data prep: building an honest modeling table
#
# **What this notebook does:** turns 2.26 million raw LendingClub loan records into
# a clean table where every row is a loan whose outcome we actually know, described
# only by information that existed on the day it was funded.
#
# Three decisions do all the work here, and each one is a place where a loss study
# quietly goes wrong:
#
# | Decision | The mistake it avoids |
# |---|---|
# | **Resolution** — keep only loans that finished (paid off or charged off) | Scoring a loan that is still paying as "did not default" |
# | **Seasoning** — keep only loans whose full term elapsed before the data snapshot | Treating an immature cohort as fully developed — the credit version of using an unclosed accident year at ultimate |
# | **Leakage** — forbid any field recorded after funding | Predicting default from evidence of default, and reporting a 0.99 AUC that means nothing |
#
# The leakage screen is the one worth reading closely. It is the most common error
# in credit-modeling writeups and the fastest way to lose a credit interviewer.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make src/ importable regardless of where the notebook is launched from.
REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from src import data_prep as dp  # noqa: E402
from src import plotting as pl  # noqa: E402

pl.set_style()
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 40)

print(f"Repo root: {REPO_ROOT}")
print(f"Raw file:  {dp.RAW_CSV.name}")
print(f"Snapshot:  {dp.SNAPSHOT:%Y-%m}  (latest month with recorded payments)")

# %% [markdown]
# ## 1. Load
#
# The raw file is ~1.7 GB across 151 columns. We read it in chunks and keep only
# the 30 columns the study needs — the other 121 are either leakage, free text,
# identifiers, or secondary-applicant fields that are >98% null.
#
# Reading the whole file at once will exhaust memory on a normal laptop; the
# chunked reader in `src/data_prep.py` is there so this notebook runs anywhere.

# %%
raw = dp.load_raw()
print(f"Raw rows loaded: {len(raw):,}")
print(f"Columns kept:    {raw.shape[1]} of 151")

# %% [markdown]
# ## 2. The leakage screen — what we are refusing to use, and why
#
# "Leakage" means a predictor that would not have existed at the moment of the
# decision. In credit data it hides in plain sight, because the file is a *record
# of the loan's whole life*, not a snapshot of underwriting.
#
# The excluded columns fall into three groups, and the reason differs by group:

# %%
leakage_groups = pd.DataFrame(
    [
        (
            "Payment performance",
            len(dp.LEAKAGE_PAYMENT_PERFORMANCE),
            "How the loan actually performed after funding — total_pymnt, "
            "recoveries, last_pymnt_amnt. This is the answer, not a predictor.",
        ),
        (
            "Post-origination credit refresh",
            len(dp.LEAKAGE_POST_ORIGINATION_CREDIT),
            "last_fico_range_* is the borrower's score TODAY, not at "
            "underwriting. A score that has collapsed 200 points usually means "
            "the default already happened.",
        ),
        (
            "Distress flags",
            len(dp.LEAKAGE_DISTRESS_FLAGS),
            "hardship_*, settlement_*, debt_settlement_flag. These fields only "
            "get populated BECAUSE the loan went bad — consequences of default "
            "dressed up as features.",
        ),
    ],
    columns=["Group", "# columns", "Why it is excluded"],
)
leakage_groups

# %% [markdown]
# It is worth being concrete about how badly this bites. `debt_settlement_flag`
# is `Y` for essentially only those borrowers who negotiated a settlement — which
# happens after they stopped paying. A model handed that column reports a near
# perfect AUC and has learned nothing about credit.

# %%
# Sanity check on the raw file: how strongly does one distress flag give the
# answer away? (We read it here purely to demonstrate the problem, then discard it.)
flag_check = pd.read_csv(
    dp.RAW_CSV, usecols=["debt_settlement_flag", "loan_status"], nrows=400_000
)
flag_check = flag_check[flag_check["loan_status"].isin(["Fully Paid", "Charged Off"])]
xtab = pd.crosstab(
    flag_check["debt_settlement_flag"],
    flag_check["loan_status"],
    normalize="index",
).round(3)
print("P(loan_status | debt_settlement_flag), first 400k rows:\n")
print(xtab)
print(
    "\nA settlement flag of 'Y' implies charge-off "
    f"{xtab.loc['Y', 'Charged Off']:.1%} of the time. That is not a feature — "
    "that is the outcome."
)
del flag_check

# %% [markdown]
# ### What survives
#
# What we keep is the origination-time credit file: what an underwriter could
# actually have seen when the decision was made.
#
# One further judgement call. `grade`, `sub_grade` and `int_rate` **are** known at
# origination, so using them would not be leakage. But they are LendingClub's own
# risk assessment — a model built on them mostly re-learns LC's pricing engine
# rather than the underlying credit risk. The headline model in notebook 04 is
# therefore borrower-only, with grade added afterwards as a clearly-labeled
# comparison so we can size how much signal the lender had already priced in.

# %%
feature_inventory = pd.DataFrame(
    {
        "Feature": dp.NUMERIC_FEATURES + dp.CATEGORICAL_FEATURES,
        "Type": ["numeric"] * len(dp.NUMERIC_FEATURES)
        + ["categorical"] * len(dp.CATEGORICAL_FEATURES),
    }
)
print(f"{len(feature_inventory)} origination-time features retained:\n")
feature_inventory

# %% [markdown]
# ## 3. Parse, label, and season
#
# Now the sequence that turns raw records into a modeling table. Each step is a
# named function in `src/data_prep.py`; the counts below show what each one costs.

# %%
funnel = []


def record(stage: str, frame: pd.DataFrame) -> None:
    funnel.append({"Stage": stage, "Loans": len(frame)})


record("Raw records", raw)

parsed = dp.parse_dates(raw)
parsed = dp.parse_terms(parsed)
df = parsed
record("Parsed dates & terms", df)

# Keep only loans with a known outcome. Current / Late / In Grace Period are
# genuinely unresolved — we do not know how they end, so they cannot be labeled.
df = dp.define_target(df)
record("Resolved outcome only", df)

# Keep only loans whose full contractual term elapsed before the snapshot.
df = dp.apply_seasoning_filter(df)
record("Fully seasoned", df)

funnel_df = pd.DataFrame(funnel)
funnel_df["Dropped"] = -funnel_df["Loans"].diff().fillna(0).astype(int)
funnel_df["% of raw"] = (funnel_df["Loans"] / len(raw) * 100).round(1)
funnel_df

# %% [markdown]
# The seasoning filter is the expensive one, and it should be. With a March 2019
# snapshot it keeps **36-month loans issued through 2016-03** and **60-month loans
# issued through 2014-03**.
#
# Why refuse the rest? A 36-month loan issued in mid-2017 that currently reads
# "Fully Paid" is a real observation — but its *cohort* is still developing, and
# that cohort's eventual defaults are disproportionately still sitting in
# "Current" and "Late." Keeping the resolved ones and dropping the rest is a
# survivorship filter that biases the observed default rate **down**. The same
# discipline as a reserving cut-off: an immature period does not get treated as
# though it were at ultimate.

# %%
seasoning_view = (
    df.groupby([df["issue_d"].dt.year, "term_months"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={36.0: "36-month", 60.0: "60-month"})
)
seasoning_view.index.name = "Issue year"
print("Seasoned loans retained, by issue year and term:\n")
seasoning_view

# %% [markdown]
# ## 4. From a binary flag to dollars of loss
#
# A default indicator is not a loss study. A charge-off on a \\$35,000 loan that
# stopped paying in month 4 is a different event from one on a \\$2,000 loan that
# made 33 of 36 payments, and a portfolio view has to know the difference.
#
# So we compute realized dollar loss directly:
#
# ```
# gross charge-off = funded principal − principal actually repaid
# net charge-off   = gross charge-off − post-charge-off recoveries
# LGD              = net charge-off / funded principal   (bad loans only)
# ```
#
# Interest is deliberately excluded. Writing off unearned interest would inflate
# loss relative to how a lender's book actually reports it — the loss that matters
# is principal handed over and not returned.
#
# Note the asymmetry this creates with the leakage rule: `total_rec_prncp` and
# `recoveries` are **forbidden as predictors** and **required to measure the
# outcome**. A column can be both.

# %%
df = dp.add_loss_dollars(df)
df = dp.add_vintage_fields(df)
df = dp.add_fico_band(df)

bad = df[df["charged_off"] == 1]
print(f"Charged-off loans: {len(bad):,} ({df['charged_off'].mean():.2%} of book)")
print(f"Mean LGD (severity given default): {bad['lgd'].mean():.1%}")
print(f"Median LGD:                        {bad['lgd'].median():.1%}")
print()
print(f"Total funded principal:   ${df['funded_amnt'].sum()/1e9:,.2f}B")
print(f"Total gross charge-off:   ${df['gross_chargeoff'].sum()/1e9:,.2f}B")
print(f"Total recoveries:         ${bad['recoveries'].sum()/1e6:,.1f}M")
print(f"Total net charge-off:     ${df['net_chargeoff'].sum()/1e9:,.2f}B")
print()
print(
    f"Portfolio net loss rate:  {df['net_chargeoff'].sum()/df['funded_amnt'].sum():.2%} "
    "of funded principal"
)

# %% [markdown]
# Three numbers worth carrying into an interview:
#
# * **Frequency and severity are separate levers.** About 15% of loans charge off,
#   but a charged-off loan does not lose 100% of principal — the borrower usually
#   made payments before failing. Mean LGD is roughly 52%. Multiply the two and
#   you land near the ~8% portfolio net loss rate. Frequency × severity is the
#   decomposition, and the two move for different reasons.
# * **Severity is remarkably stable.** Mean and median LGD sit within two points of
#   each other, so severity behaves close to a constant. That is why nearly all the
#   analytical effort in unsecured consumer credit goes into modeling *frequency* —
#   it is the term that actually varies. (In reinsurance terms: this is a
#   high-frequency, low-variance-severity book, the opposite of a cat portfolio.)
# * **Recoveries are real but thin.** Post-charge-off recoveries claw back roughly
#   an eighth of gross written-off principal — enough to matter in pricing, nowhere
#   near enough to change the shape of the loss distribution.

# %% [markdown]
# ## 5. Missingness check
#
# Before writing the table out: which retained features have gaps, and are the
# gaps small enough to handle with a median fill rather than a modeling decision?

# %%
model_cols = dp.NUMERIC_FEATURES + dp.CATEGORICAL_FEATURES
missing = (
    df[model_cols].isna().mean().sort_values(ascending=False).to_frame("% missing")
)
missing["% missing"] = (missing["% missing"] * 100).round(2)
missing[missing["% missing"] > 0]

# %% [markdown]
# All small. `emp_length_years` is the only meaningful gap — a blank employment
# length is typically a borrower who did not supply one, which is itself mildly
# informative, but not enough to justify a separate indicator here. Median
# imputation inside the model pipeline is sufficient and easy to defend.

# %% [markdown]
# ## 6. A second table: the vintage development panel
#
# The table above is deliberately narrow — resolved, fully seasoned — because
# that is the right universe for **fitting a model**.
#
# It is the wrong universe for a **development triangle**. The entire point of a
# triangle is to put immature cohorts next to mature ones and ask whether the
# young ones are running hot *at the same age*. Restricting to seasoned loans
# would square the triangle off and delete the rows a credit officer actually
# worries about — the 2017 and 2018 vintages.
#
# So we build a second panel that keeps **every** loan and simply refuses to
# report a cohort past the age it has actually been observed to. A still-current
# 2018 loan contributes denominator, not numerator, which is correct: at 14
# months on book it genuinely has not lost anything yet.
#
# This is the same object as a paid-loss triangle in reserving: exposure down the
# side, development age across the top, and a blank lower-right where the future
# has not happened yet.

# %%
panel = dp.build_vintage_panel(parsed)

print(f"Vintage panel: {len(panel):,} loans (vs {len(df):,} in the modeling table)")
print(f"Issue dates:   {panel['issue_d'].min():%Y-%m} to {panel['issue_d'].max():%Y-%m}")
print(f"Exposure:      ${panel['funded_amnt'].sum()/1e9:,.1f}B funded principal")
print()
print("Months of development observed, by vintage year:")
print(dp.observable_mob(panel, "vintage_year").to_string())

# %% [markdown]
# ## 7. Write both tables
#
# Everything downstream reads these files, so the slow pass over the raw CSV
# happens exactly once. Parquet rather than CSV: types survive the round trip,
# and it is an order of magnitude smaller and faster.

# %%
dp.PROCESSED.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(dp.PROCESSED, index=False)
panel.to_parquet(dp.VINTAGE_PANEL, index=False)

print(f"Wrote {len(df):,} loans × {df.shape[1]} columns")
print(f"  -> {dp.PROCESSED.relative_to(REPO_ROOT)}  ({dp.PROCESSED.stat().st_size / 1e6:.1f} MB)")
print(f"Wrote {len(panel):,} loans × {panel.shape[1]} columns")
print(f"  -> {dp.VINTAGE_PANEL.relative_to(REPO_ROOT)}  ({dp.VINTAGE_PANEL.stat().st_size / 1e6:.1f} MB)")
print()
print("Modeling universe summary (seasoned + resolved)")
print("-" * 46)
print(f"  Issue dates      {df['issue_d'].min():%Y-%m} to {df['issue_d'].max():%Y-%m}")
print(f"  Vintages         {df['vintage_year'].nunique()} origination years")
print(f"  Funded principal ${df['funded_amnt'].sum()/1e9:,.2f}B")
print(f"  Charge-off rate  {df['charged_off'].mean():.2%} of loans")
print(f"  Net loss rate    {df['net_chargeoff'].sum()/df['funded_amnt'].sum():.2%} of principal")

# %% [markdown]
# ---
# **Next:** `02_vintage_curves.ipynb` — cumulative loss by months-on-book for each
# origination cohort. That is the credit analog of a loss development triangle,
# and it is the centerpiece of the study.
