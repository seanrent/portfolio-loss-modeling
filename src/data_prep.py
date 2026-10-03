"""
Reusable cleaning functions for the LendingClub portfolio loss study.

Everything that decides *what counts as a loan we can learn from* lives here,
so the notebooks stay narrative and the rules stay in one auditable place.

The three decisions that matter, in order of how much they change the answer:

1. **Seasoning.** A loan is only usable once its full term has elapsed as of the
   data snapshot. Otherwise "hasn't defaulted yet" gets silently scored as
   "didn't default," which biases loss down. This is the same discipline as a
   reserving cut-off: you don't treat an immature period as fully developed.

2. **Leakage.** Anything recorded *after* the loan was funded is forbidden as a
   predictor. Some of these columns (hardship plans, settlement flags, recovery
   amounts) only exist *because* the loan went bad -- feed them to a model and it
   will score AUC 0.99 by reading the answer off the back of the card.

3. **Feature availability.** What survives is the origination-time credit file:
   what an underwriter could actually have seen at the moment of the decision.

Note the distinction the LEAKAGE list encodes: a column can be forbidden as a
*predictor* and still be required to *measure the outcome*. `total_rec_prncp`
and `recoveries` are how we compute realized dollar loss. They are never
features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = REPO_ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv"
PROCESSED = REPO_ROOT / "data" / "processed" / "loans_clean.parquet"

# The dataset's performance snapshot: the latest month in which any payment is
# recorded. Everything about seasoning hangs off this one date.
SNAPSHOT = pd.Timestamp("2019-03-01")


# --------------------------------------------------------------------------- #
# Target definition
# --------------------------------------------------------------------------- #

# A loan is "bad" if the lender wrote it off.
CHARGED_OFF_STATUSES = ["Charged Off", "Default"]

# A loan is "good" if the borrower repaid it in full.
FULLY_PAID_STATUSES = ["Fully Paid"]

# Everything else -- Current, In Grace Period, Late (16-30), Late (31-120) --
# is UNRESOLVED. We do not know the outcome, so it cannot be a training label.
#
# The "Does not meet the credit policy" statuses are also dropped: those are
# legacy 2007-2010 loans booked under a different (looser) underwriting standard
# than everything after. Mixing two credit policies into one target muddies both.


# --------------------------------------------------------------------------- #
# Leakage -- the columns that must never be features
# --------------------------------------------------------------------------- #
#
# Grouped by *why*, because the reason is the point. Using post-origination data
# to predict default is the single most common mistake in credit modeling
# writeups, and the fastest way to lose a credit interviewer's confidence.

LEAKAGE_PAYMENT_PERFORMANCE = [
    # How the loan actually performed after funding. Directly encodes the answer.
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
]

LEAKAGE_POST_ORIGINATION_CREDIT = [
    # A credit-bureau refresh pulled *after* funding. `last_fico_range_*` is the
    # borrower's score today, not at underwriting -- a borrower whose score has
    # collapsed 200 points has usually already defaulted.
    "last_credit_pull_d",
    "last_fico_range_high",
    "last_fico_range_low",
]

LEAKAGE_DISTRESS_FLAGS = [
    # These fields only get populated *because* the loan went bad. They are
    # consequences of default, dressed up as features. Include them and the
    # model scores ~0.99 AUC and predicts nothing.
    "hardship_flag",
    "hardship_type",
    "hardship_reason",
    "hardship_status",
    "hardship_start_date",
    "hardship_end_date",
    "hardship_amount",
    "hardship_length",
    "hardship_dpd",
    "hardship_loan_status",
    "hardship_payoff_balance_amount",
    "hardship_last_payment_amount",
    "orig_projected_additional_accrued_interest",
    "deferral_term",
    "payment_plan_start_date",
    "pymnt_plan",
    "debt_settlement_flag",
    "debt_settlement_flag_date",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
]

LEAKAGE = (
    LEAKAGE_PAYMENT_PERFORMANCE
    + LEAKAGE_POST_ORIGINATION_CREDIT
    + LEAKAGE_DISTRESS_FLAGS
)

# Dropped for reasons other than leakage -- identifiers, free text, and columns
# that are effectively constant or almost entirely null. Kept separate from
# LEAKAGE so the leakage list stays a clean statement about timing.
DROP_IDENTIFIERS_AND_TEXT = [
    "id",
    "member_id",
    "url",
    "desc",
    "emp_title",  # free text, ~500k distinct values
    "title",  # free text restatement of `purpose`
    "zip_code",  # partial ZIP; geographic risk is out of scope here
    "policy_code",  # constant
]


# --------------------------------------------------------------------------- #
# What we keep: origination-time features only
# --------------------------------------------------------------------------- #

# Raw columns pulled off the CSV. Deliberately small -- 151 columns is a
# haystack, and most of the extras are sparse joint-applicant or secondary-
# applicant fields that are >98% null.
RAW_COLUMNS = [
    # --- identity of the loan / timing ------------------------------------- #
    "issue_d",
    "term",
    "loan_status",
    # --- loan terms known at origination ----------------------------------- #
    "loan_amnt",
    "funded_amnt",
    "int_rate",
    "grade",
    "sub_grade",
    # --- borrower credit file at origination ------------------------------- #
    "fico_range_low",
    "fico_range_high",
    "dti",
    "annual_inc",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",
    "addr_state",
    "earliest_cr_line",
    "open_acc",
    "total_acc",
    "revol_bal",
    "revol_util",
    "delinq_2yrs",
    "inq_last_6mths",
    "pub_rec",
    "pub_rec_bankruptcies",
    "mort_acc",
    # --- outcome measurement (NOT features) -------------------------------- #
    "total_rec_prncp",  # principal actually repaid -> used for loss dollars
    "recoveries",  # post-charge-off recoveries -> used for NET loss
    "last_pymnt_d",  # dates the default event -> used for months-on-book
]

# The feature set handed to the models. Note what is absent: `grade`,
# `sub_grade` and `int_rate` are all known at origination and would be perfectly
# legitimate inputs -- but they are LendingClub's *own* risk assessment. A model
# built on them largely re-learns LC's pricing model rather than the underlying
# credit risk. Notebook 04 fits the borrower-only model first and then adds
# grade as a separate, clearly-labeled comparison.
NUMERIC_FEATURES = [
    "loan_amnt",
    "term_months",
    "fico_score",
    "dti",
    "log_annual_inc",
    "emp_length_years",
    "credit_history_years",
    "open_acc",
    "total_acc",
    "revol_bal",
    "revol_util",
    "delinq_2yrs",
    "inq_last_6mths",
    "pub_rec",
    "mort_acc",
]

CATEGORICAL_FEATURES = [
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_raw(
    path: Path = RAW_CSV, chunksize: int = 250_000, columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Read the raw CSV in chunks, keeping only RAW_COLUMNS (or `columns`, if given).

    Chunked because the file is ~1.7 GB across 151 columns and pandas' type
    inference will happily exhaust memory trying to read it in one go. Reading
    in chunks and concatenating the narrow slice keeps peak memory modest.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}.\n"
            "See data/README.md for the download link and expected location."
        )

    chunks = [
        chunk
        for chunk in pd.read_csv(
            path,
            usecols=columns or RAW_COLUMNS,
            chunksize=chunksize,
            low_memory=True,
        )
    ]
    return pd.concat(chunks, ignore_index=True)


# --------------------------------------------------------------------------- #
# Cleaning steps
# --------------------------------------------------------------------------- #


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert LendingClub's 'Mon-YYYY' date strings to real timestamps."""
    df = df.copy()
    for col in ["issue_d", "earliest_cr_line", "last_pymnt_d"]:
        df[col] = pd.to_datetime(df[col], format="%b-%Y", errors="coerce")
    return df


def parse_terms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the messy origination-time fields into numbers.

    `term` arrives as ' 36 months'; `emp_length` as '10+ years' / '< 1 year'.
    Both are genuinely ordinal, so they become integers rather than dummies.
    """
    df = df.copy()

    df["term_months"] = (
        df["term"].astype(str).str.extract(r"(\d+)").astype(float)
    )

    # '10+ years' -> 10, '< 1 year' -> 0, '3 years' -> 3, missing stays missing.
    emp = df["emp_length"].astype(str)
    df["emp_length_years"] = (
        emp.str.extract(r"(\d+)").astype(float).where(emp.ne("nan"))
    )
    df.loc[emp.str.contains("< 1", na=False), "emp_length_years"] = 0.0

    # FICO arrives as a 5-point band; the midpoint is the conventional summary.
    df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

    # Income is right-skewed by orders of magnitude (a $30k and a $3M borrower
    # in the same linear term is meaningless). log1p is the standard fix and
    # makes the coefficient read as "per proportional change in income."
    df["log_annual_inc"] = np.log1p(df["annual_inc"].clip(lower=0))

    # Length of credit file, in years, as of origination. A thin file is a
    # different kind of risk from a long clean one.
    df["credit_history_years"] = (
        (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25
    )

    return df


def define_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the binary target and drop loans whose outcome is unknown.

    charged_off = 1 if the lender wrote the loan off, 0 if repaid in full.
    Anything still paying is removed -- an unresolved loan has no label.
    """
    df = df.copy()

    resolved = df["loan_status"].isin(CHARGED_OFF_STATUSES + FULLY_PAID_STATUSES)
    df = df.loc[resolved].copy()

    df["charged_off"] = df["loan_status"].isin(CHARGED_OFF_STATUSES).astype(int)
    return df


def apply_seasoning_filter(
    df: pd.DataFrame, snapshot: pd.Timestamp = SNAPSHOT
) -> pd.DataFrame:
    """
    Keep only loans whose full contractual term had elapsed by the snapshot.

    Why this matters more than it looks: a 36-month loan issued in June 2016 has
    only had ~33 months to fail as of the March 2019 snapshot. If it shows
    "Fully Paid" it is genuinely resolved -- but the *cohort* it belongs to is
    still developing, and its charged-off siblings are disproportionately still
    sitting in "Current" or "Late." Including immature cohorts therefore biases
    the observed default rate DOWN.

    Concretely, with a 2019-03 snapshot this keeps 36-month loans issued through
    2016-02 and 60-month loans issued through 2014-03.
    """
    df = df.copy()
    maturity = df["issue_d"] + pd.to_timedelta(df["term_months"] * 30.44, unit="D")
    df["fully_seasoned"] = maturity <= snapshot
    return df.loc[df["fully_seasoned"]].drop(columns="fully_seasoned")


def add_loss_dollars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the binary outcome into realized dollar loss.

    This is the step that turns a classification exercise into a loss study.
    A default on a $35,000 loan that stopped paying in month 4 is not the same
    event as a default on a $2,000 loan that paid 33 of 36 installments, and a
    portfolio view has to know the difference.

        gross_chargeoff = funded principal - principal actually repaid
        net_chargeoff   = gross charge-off - post-charge-off recoveries

    Interest is deliberately excluded. Charging off unearned interest would
    inflate loss relative to how a lender's book actually reports it; the loss
    that matters is principal you handed over and did not get back.
    """
    df = df.copy()

    unpaid_principal = (df["funded_amnt"] - df["total_rec_prncp"]).clip(lower=0)
    df["gross_chargeoff"] = np.where(df["charged_off"] == 1, unpaid_principal, 0.0)
    df["net_chargeoff"] = np.maximum(
        df["gross_chargeoff"] - df["recoveries"].fillna(0.0), 0.0
    )

    # Loss given default: the share of exposure actually lost on a bad loan.
    # Reinsurance analog: severity, conditional on the event occurring.
    df["lgd"] = np.where(
        df["charged_off"] == 1,
        (df["net_chargeoff"] / df["funded_amnt"]).clip(0, 1),
        np.nan,
    )
    return df


def add_vintage_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the cohort and development-age fields the vintage curves need.

    `mob_last_payment` is months-on-book at the borrower's final payment. For a
    charged-off loan this dates the failure. It is a proxy, not the charge-off
    date itself -- LendingClub charges off at 120 days past due, so the
    accounting event lands roughly four months after the last payment. Using the
    last payment keeps the timing observable and consistent across every loan,
    and the four-month constant shifts all cohorts equally, so the *comparison
    between vintages* -- which is the whole point of the chart -- is unaffected.
    """
    df = df.copy()
    df["vintage_year"] = df["issue_d"].dt.year
    df["vintage_quarter"] = df["issue_d"].dt.to_period("Q").astype(str)

    months = (
        (df["last_pymnt_d"].dt.year - df["issue_d"].dt.year) * 12
        + (df["last_pymnt_d"].dt.month - df["issue_d"].dt.month)
    )
    df["mob_last_payment"] = months.clip(lower=0)
    return df


def add_fico_band(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket FICO into the conventional lending bands.

    Bands rather than raw score because this is how the risk is actually
    managed: cutoffs, pricing tiers and policy rules are written on bands, and
    a band-level loss table is what a pricing conversation is held over.
    """
    df = df.copy()
    bins = [0, 660, 680, 700, 720, 740, 780, 900]
    labels = ["<660", "660-679", "680-699", "700-719", "720-739", "740-779", "780+"]
    df["fico_band"] = pd.cut(
        df["fico_score"], bins=bins, labels=labels, right=False
    )
    return df


# --------------------------------------------------------------------------- #
# Vintage development panel
# --------------------------------------------------------------------------- #
#
# The modeling table above is deliberately narrow: resolved outcomes, fully
# seasoned. That is the right universe for FITTING a model.
#
# It is the wrong universe for a DEVELOPMENT TRIANGLE. A triangle's whole purpose
# is to show immature cohorts alongside mature ones, so you can see whether the
# young ones are running hot at the same age. Restricting to seasoned loans would
# square off the triangle and throw away the most interesting rows -- the recent
# vintages a credit officer actually worries about.
#
# So the vintage panel keeps every loan and simply refuses to report a cohort
# beyond the age at which it has actually been observed. That is exactly what a
# reserving triangle does.

VINTAGE_PANEL = REPO_ROOT / "data" / "processed" / "vintage_panel.parquet"


def build_vintage_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the loan-level panel used for vintage / development analysis.

    Takes the parsed frame BEFORE the resolution and seasoning filters, so
    still-performing loans stay in as exposure. A loan that has not charged off
    contributes denominator, not numerator -- which is correct: at month 18 a
    2017 loan that is still current genuinely has not lost anything yet.
    """
    df = df.copy()
    df = df[df["issue_d"].notna() & df["funded_amnt"].notna()]

    df["charged_off"] = df["loan_status"].isin(CHARGED_OFF_STATUSES).astype(int)

    unpaid = (df["funded_amnt"] - df["total_rec_prncp"]).clip(lower=0)
    df["gross_chargeoff"] = np.where(df["charged_off"] == 1, unpaid, 0.0)
    df["net_chargeoff"] = np.maximum(
        df["gross_chargeoff"] - df["recoveries"].fillna(0.0), 0.0
    )

    df = add_vintage_fields(df)
    df = add_fico_band(df)

    keep = [
        "issue_d",
        "vintage_year",
        "vintage_quarter",
        "term_months",
        "funded_amnt",
        "grade",
        "fico_band",
        "loan_status",
        "charged_off",
        "gross_chargeoff",
        "net_chargeoff",
        "mob_last_payment",
    ]
    return df[keep].reset_index(drop=True)


def observable_mob(panel: pd.DataFrame, cohort_col: str, snapshot: pd.Timestamp = SNAPSHOT) -> pd.Series:
    """
    How many months of development each cohort has actually been observed for.

    Uses the cohort's LATEST issue month, so the reported age is one that every
    loan in the cohort has genuinely reached. Reporting an annual cohort at the
    age of its oldest member would overstate the development of the January-heavy
    part of the year and quietly flatter the curve.
    """
    latest_issue = panel.groupby(cohort_col)["issue_d"].max()
    return (
        (snapshot.year - latest_issue.dt.year) * 12
        + (snapshot.month - latest_issue.dt.month)
    )


def vintage_triangle(
    panel: pd.DataFrame,
    cohort_col: str = "vintage_year",
    max_mob: int = 36,
    loss_col: str = "net_chargeoff",
    snapshot: pd.Timestamp = SNAPSHOT,
) -> pd.DataFrame:
    """
    Cumulative loss rate by cohort (rows) and months-on-book (columns).

    Cell (v, m) = cumulative loss dollars from cohort v recognised by month m,
    divided by cohort v's total funded principal. Dollar-weighted, not loan-count
    weighted, because a portfolio loses money, not loans.

    Cells beyond a cohort's observed age are left NaN -- which is what gives the
    output its triangular shape, and is the honest thing to do: we have not seen
    the 2018 vintage at 30 months on book, so we do not draw it.
    """
    p = panel.copy()
    exposure = p.groupby(cohort_col)["funded_amnt"].sum()

    bad = p[(p["charged_off"] == 1) & p["mob_last_payment"].notna()].copy()
    bad["mob"] = bad["mob_last_payment"].astype(int)

    # Charge-offs dated beyond the requested window are EXCLUDED, not clipped
    # into the final column. Clipping would silently pull a month-30 default
    # into a "loss by month 12" figure and inflate every early cell -- the
    # cumulative curve has to mean "recognised by month m," nothing else.
    bad = bad[bad["mob"] <= max_mob]

    incremental = (
        bad.groupby([cohort_col, "mob"])[loss_col]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(columns=range(0, max_mob + 1), fill_value=0.0)
    )

    cumulative = incremental.cumsum(axis=1).div(exposure, axis=0)

    # Mask the cells we have not lived through yet -> the triangle.
    ages = observable_mob(p, cohort_col, snapshot)
    for cohort, age in ages.items():
        if cohort in cumulative.index:
            cumulative.loc[cohort, cumulative.columns > age] = np.nan

    cumulative.columns.name = "months_on_book"
    return cumulative


def build_clean_table(path: Path = RAW_CSV, columns: list[str] | None = None) -> pd.DataFrame:
    """
    End-to-end pipeline: raw CSV in, modeling table out.

    Deliberately a flat sequence of named steps rather than a class hierarchy or
    an sklearn Pipeline. Someone reading this should be able to see the whole
    cleaning story in eight lines without chasing an abstraction.

    `columns` lets a later notebook carry extra outcome-measurement fields (e.g.
    interest received, for notebook 06) through the identical cleaning steps.
    """
    df = load_raw(path, columns=columns)
    df = parse_dates(df)
    df = parse_terms(df)
    df = define_target(df)
    df = apply_seasoning_filter(df)
    df = add_loss_dollars(df)
    df = add_vintage_fields(df)
    df = add_fico_band(df)
    return df


# Categorical levels rarer than this are folded into a single "OTHER" bucket
# before encoding. LendingClub's `home_ownership` has ANY / NONE levels with a
# few dozen loans between them across 730k records; a dummy for those estimates a
# coefficient off statistical noise and, because the surviving levels then sum to
# one, makes the design matrix collinear with the intercept.
RARE_LEVEL_THRESHOLD = 0.005


def _collapse_rare(series: pd.Series, threshold: float = RARE_LEVEL_THRESHOLD) -> pd.Series:
    """Fold levels below `threshold` share into a single OTHER bucket."""
    s = series.astype("object").fillna("MISSING")
    share = s.value_counts(normalize=True)
    rare = share[share < threshold].index
    return s.where(~s.isin(rare), "OTHER")


def feature_matrix(
    df: pd.DataFrame,
    include_grade: bool = False,
    return_reference_levels: bool = False,
):
    """
    Assemble the model design matrix: numeric features + one-hot categoricals.

    Two encoding choices, both about interpretability rather than accuracy:

    * **Rare levels are collapsed** (see RARE_LEVEL_THRESHOLD). A dummy fired by
      forty loans is not a finding.
    * **The reference level is the MODAL level, not the alphabetically first
      one.** pandas' `drop_first=True` would make "ANY home ownership" the
      baseline that every other coefficient is measured against, which is
      meaningless. Dropping the most common level instead means each coefficient
      reads as "relative to the typical borrower" -- which is how the result
      actually gets explained to a credit committee.

    `include_grade=True` adds LendingClub's own letter grade. Used in notebook 04
    purely to size how much of the signal is already priced in by the lender --
    it is not the headline model, because a model that leans on the lender's
    grade is measuring the lender, not the borrower.
    """
    numeric = df[NUMERIC_FEATURES].copy()

    cats = CATEGORICAL_FEATURES + (["grade"] if include_grade else [])
    blocks, references = [], {}

    for col in cats:
        collapsed = _collapse_rare(df[col])
        reference = collapsed.value_counts().idxmax()
        references[col] = reference

        dummies = pd.get_dummies(collapsed, prefix=col, dtype=float)
        blocks.append(dummies.drop(columns=f"{col}_{reference}"))

    X = pd.concat([numeric] + blocks, axis=1)
    return (X, references) if return_reference_levels else X
