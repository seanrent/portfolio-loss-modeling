# Data

The raw dataset is **not committed to this repo** (~1.7 GB, and redistribution is
the mirror's business rather than ours). Download it once and drop it here.

## What you need

**LendingClub accepted loans, 2007 – 2018Q4** — loan-level, 2,260,701 rows,
151 columns. This is the standard public LendingClub extract.

| | |
|---|---|
| File name | `accepted_2007_to_2018Q4.csv` |
| Size | ~1.7 GB |
| Rows | 2,260,701 |
| Coverage | issue dates 2007-06 through 2018-12 |
| Snapshot date | performance observed through **2019-03** (latest `last_pymnt_d`) |

## Where to get it

Either source gives the identical file:

1. **Hugging Face mirror** (no login required):
   ```bash
   curl -L -o data/raw/accepted_2007_to_2018Q4.csv \
     "https://huggingface.co/datasets/codesignal/lending-club-loan-accepted/resolve/main/accepted_2007_to_2018Q4.csv"
   ```

2. **Kaggle** — [`wordsforthewise/lending-club`](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
   (requires a Kaggle account). Use the `accepted_2007_to_2018Q4.csv` file.

## Where to put it

```
data/
├── README.md                          <- this file
├── raw/
│   └── accepted_2007_to_2018Q4.csv    <- put it here
└── processed/
    └── loans_clean.parquet            <- written by notebooks/01_data_prep.ipynb
```

`notebooks/01_data_prep.ipynb` reads from `data/raw/` and writes the cleaned
modeling table to `data/processed/`. Every later notebook reads only the
processed file, so the slow pass over the raw CSV happens exactly once.

## A note on the snapshot date

Performance is observed through **March 2019**. That single fact drives the
seasoning filter in notebook 01: a loan only enters the modeling universe if
its full term had elapsed by the snapshot. A 36-month loan issued in June 2016
has not finished paying, so counting it as "did not default" would understate
loss. This is the credit-risk version of a **reserving cut-off** — you only
treat a period as fully developed once it has had time to develop.
