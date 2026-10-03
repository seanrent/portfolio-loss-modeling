"""
Credit policy simulator: what a score cutoff and a risk-based limit schedule
would have done to the 2015-2016 book.

It sits on top of notebook 04's out-of-time scores. The logistic model was fit
on 2007-2014 vintages and never saw these loans, so replaying the policy on them
is a fair test of the score as an underwriting tool.

Two rules shape every number this module produces:

* **Only booked loans have outcomes.** LendingClub's declined applicants never
  got a loan, so there is no way to know how they would have performed (no
  reject inference). The simulator can size TIGHTENING the policy -- declining
  or shrinking loans that were actually made -- but says nothing about loosening it.

* **A limit changes the loan size, not the borrower.** A capped loan is assumed
  to default exactly when the full-size loan did, with its loss and interest
  scaled down in proportion to the smaller balance. In reality a smaller payment
  may lower default risk, and some borrowers would walk away from a smaller
  offer. Neither effect is modeled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import data_prep as dp

# Notebook 04's out-of-time split: fit on vintages through 2014, score 2015 on.
TRAIN_THROUGH = 2014

SCORED_TEST = dp.PROCESSED.parent / "scored_test.parquet"
POLICY_BOOK = dp.PROCESSED.parent / "policy_book.parquet"

# Interest received is an OUTCOME, measured after funding -- it is on the
# leakage list in data_prep and is never a model input. It is carried here only
# to value the loans a policy would have turned away.
OUTCOME_COLUMNS = ["total_rec_int"]

# The bad definition. "Does not meet the credit policy" loans are the legacy
# 2007-2010 book; listing the status keeps the rule complete, although none of
# those loans fall in the 2015-2016 holdout.
BAD_STATUSES = [
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
]


# --------------------------------------------------------------------------- #
# The book
# --------------------------------------------------------------------------- #


def build_policy_book() -> pd.DataFrame:
    """
    The holdout book with everything a policy needs, one row per loan.

    Runs the same cleaning steps as notebook 01 with interest received added,
    keeps notebook 04's test universe (vintages after TRAIN_THROUGH, FICO 660+),
    and attaches its out-of-time PD. The two tables are lined up row by row, and
    the function refuses to continue if they do not match exactly.
    """
    df = dp.build_clean_table(columns=dp.RAW_COLUMNS + OUTCOME_COLUMNS).reset_index(drop=True)
    book = df[(df["fico_band"] != "<660") & (df["vintage_year"] > TRAIN_THROUGH)].reset_index(drop=True)

    scored = pd.read_parquet(SCORED_TEST)
    for col in ["issue_d", "funded_amnt", "charged_off"]:
        if len(scored) != len(book) or not (scored[col].values == book[col].values).all():
            raise ValueError(f"Holdout rows do not line up with scored_test.parquet on '{col}'")
    book["pd_pred"] = scored["pd_pred"].values

    book["bad"] = book["loan_status"].isin(BAD_STATUSES).astype(int)
    book["exposure"] = book["funded_amnt"]
    book["realized_loss"] = (
        book["funded_amnt"] - book["total_rec_prncp"] - book["recoveries"].fillna(0.0)
    ).clip(lower=0.0)
    book["interest_received"] = book["total_rec_int"].fillna(0.0)

    keep = ["issue_d", "vintage_year", "term_months", "loan_status", "grade", "fico_band",
            "int_rate", "pd_pred", "bad", "charged_off", "exposure", "realized_loss",
            "net_chargeoff", "interest_received"]
    return book[keep]


def load_policy_book(rebuild: bool = False) -> pd.DataFrame:
    """Load the cached policy book, building it from the raw CSV on first use."""
    if rebuild or not POLICY_BOOK.exists():
        build_policy_book().to_parquet(POLICY_BOOK, index=False)
    return pd.read_parquet(POLICY_BOOK)


def _totals(book: pd.DataFrame, weight: pd.Series | float = 1.0) -> dict:
    """Exposure, loss and interest for a set of loans, optionally scaled by a limit."""
    exposure = (book["exposure"] * weight).sum()
    loss = (book["realized_loss"] * weight).sum()
    interest = (book["interest_received"] * weight).sum()
    return {
        "Loans": len(book),
        "Bad rate": book["bad"].mean(),
        "Exposure ($M)": exposure / 1e6,
        "Loss ($M)": loss / 1e6,
        "Loss rate (bps)": loss / exposure * 1e4,
        "Interest ($M)": interest / 1e6,
        "Net of loss ($M)": (interest - loss) / 1e6,
        "Net of loss (bps)": (interest - loss) / exposure * 1e4,
    }


# --------------------------------------------------------------------------- #
# Part A -- score cutoffs
# --------------------------------------------------------------------------- #


def approve(book: pd.DataFrame, approval_rate: float) -> pd.DataFrame:
    """The lowest-PD `approval_rate` share of loans: the ones a cutoff would keep."""
    n = int(round(len(book) * approval_rate))
    return book.nsmallest(n, "pd_pred")


def cutoff_sweep(book: pd.DataFrame, approval_rates=None) -> pd.DataFrame:
    """
    Approve-all down to approve-50%, in 5-point steps, with changes vs approve-all.

    "PD cutoff" is the model PD of the riskiest loan still approved. It is on the
    model's own scale, which under-predicts this book's level by ~12% (notebook
    04), so the approval rate is the more portable way to state the policy.
    """
    if approval_rates is None:
        approval_rates = np.round(np.arange(1.0, 0.499, -0.05), 2)
    base = _totals(book)
    rows = []
    for rate in approval_rates:
        kept = approve(book, rate)
        t = _totals(kept)
        rows.append({
            "Approval rate": rate,
            "PD cutoff": kept["pd_pred"].max(),
            **t,
            "Δ approvals": t["Loans"] / base["Loans"] - 1,
            "Δ bad rate (pts)": (t["Bad rate"] - base["Bad rate"]) * 100,
            "Δ loss rate (bps)": t["Loss rate (bps)"] - base["Loss rate (bps)"],
            "Δ exposure": t["Exposure ($M)"] / base["Exposure ($M)"] - 1,
            "Δ loss $": t["Loss ($M)"] / base["Loss ($M)"] - 1,
        })
    return pd.DataFrame(rows).set_index("Approval rate")


def marginal_slices(book: pd.DataFrame, step: float = 0.05, floor: float = 0.5) -> pd.DataFrame:
    """
    The loans each 5-point tightening removes, valued on their own.

    This is the question a cutoff actually answers: not "what is the loss rate of
    what's left" but "were the loans being turned away worth keeping?"
    """
    ranked = book.sort_values("pd_pred").reset_index(drop=True)
    n = len(ranked)
    rows = []
    for upper in np.round(np.arange(1.0, floor - 1e-9, -step), 2)[:-1]:
        lower = round(upper - step, 2)
        slice_ = ranked.iloc[int(round(n * lower)): int(round(n * upper))]
        t = _totals(slice_)
        rows.append({"Declined slice": f"{lower:.0%}–{upper:.0%}",
                     "Avg coupon": slice_["int_rate"].mean() / 100, **t})
    return pd.DataFrame(rows).set_index("Declined slice")


# --------------------------------------------------------------------------- #
# Part B -- risk-based limits
# --------------------------------------------------------------------------- #


def pd_bands(book: pd.DataFrame, n_bands: int = 5) -> pd.Series:
    """PD quintiles on the full holdout, 1 = lowest risk. Fixed before any cutoff applies."""
    return pd.qcut(book["pd_pred"], n_bands, labels=range(1, n_bands + 1)).astype(int)


def limit_weight(bands: pd.Series, caps: list[float]) -> pd.Series:
    """Share of the requested amount funded under a cap schedule (caps[0] = best band)."""
    return bands.map(dict(zip(range(1, len(caps) + 1), caps)))


def limit_comparison(book: pd.DataFrame, schedules: dict[str, list[float]]) -> pd.DataFrame:
    """Exposure, loss and worst-band concentration under each cap schedule, vs uncapped."""
    bands = pd_bands(book)
    worst = bands == bands.max()
    base = _totals(book)
    rows = []
    for name, caps in schedules.items():
        w = limit_weight(bands, caps)
        t = _totals(book, w)
        rows.append({
            "Schedule": name,
            "Caps, best → worst band": " / ".join(f"{c:.0%}" for c in caps),
            "Exposure ($M)": t["Exposure ($M)"],
            "Δ exposure": t["Exposure ($M)"] / base["Exposure ($M)"] - 1,
            "Δ loss $": t["Loss ($M)"] / base["Loss ($M)"] - 1,
            "Loss rate (bps)": t["Loss rate (bps)"],
            "Δ loss rate (bps)": t["Loss rate (bps)"] - base["Loss rate (bps)"],
            "Worst-band share before": book.loc[worst, "exposure"].sum() / book["exposure"].sum(),
            "Worst-band share after": (book["exposure"] * w)[worst].sum() / (book["exposure"] * w).sum(),
            "Δ interest": t["Interest ($M)"] / base["Interest ($M)"] - 1,
        })
    return pd.DataFrame(rows).set_index("Schedule")


# --------------------------------------------------------------------------- #
# Part C -- cutoff and limits together
# --------------------------------------------------------------------------- #


def evaluate_policy(book: pd.DataFrame, approval_rate: float, caps: list[float]) -> dict:
    """A cutoff and a cap schedule applied together, with changes vs approve-all, uncapped."""
    bands = pd_bands(book)
    kept = approve(book, approval_rate)
    t = _totals(kept, limit_weight(bands.loc[kept.index], caps))
    base = _totals(book)
    return {
        "Approval rate": approval_rate,
        "Caps, best → worst band": " / ".join(f"{c:.0%}" for c in caps),
        **{k: t[k] for k in ["Exposure ($M)", "Loss ($M)", "Loss rate (bps)", "Net of loss ($M)"]},
        "Δ approvals": t["Loans"] / base["Loans"] - 1,
        "Δ exposure": t["Exposure ($M)"] / base["Exposure ($M)"] - 1,
        "Δ loss rate (bps)": t["Loss rate (bps)"] - base["Loss rate (bps)"],
        "Δ loss $": t["Loss ($M)"] / base["Loss ($M)"] - 1,
        "Δ interest": t["Interest ($M)"] / base["Interest ($M)"] - 1,
        "Δ net of loss $": t["Net of loss ($M)"] / base["Net of loss ($M)"] - 1,
    }
