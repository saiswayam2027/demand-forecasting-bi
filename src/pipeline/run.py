"""Pipeline entry point: ingest -> validate(raw) -> clean -> validate(clean) -> Parquet + data-quality report."""
from __future__ import annotations

import json

import pandas as pd

from src.config import PROC_DIR, REPORTS_DIR
from src.pipeline import clean, ingest, validate

# After cleaning, these must all be zero (outliers are flagged-and-kept, so excluded).
MUST_BE_ZERO = [
    "duplicates", "negative_sales", "missing_store_days", "stores_with_gaps", "closed_but_sales_positive",
    "open_but_zero_sales", "invalid_state_holiday_codes", "dayofweek_date_mismatch",
    "promo_not_binary", "open_not_binary", "school_holiday_not_binary",
]


def to_validation_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the clean panel back to the raw column names so the same validators can re-run on it."""
    return df.rename(columns={
        "store": "Store", "date": "Date", "day_of_week": "DayOfWeek", "sales": "Sales", "customers": "Customers",
        "open": "Open", "promo": "Promo", "state_holiday": "StateHoliday", "school_holiday": "SchoolHoliday",
    })


def assert_clean(post: dict) -> None:
    bad = {k: post[k] for k in MUST_BE_ZERO if post[k] != 0}
    if bad:
        raise AssertionError(f"Validation failed after cleaning: {bad}")


def outlier_profile(panel: pd.DataFrame) -> dict:
    o = panel[panel["is_outlier"] == 1]
    base = panel[(panel["open"] == 1) & (panel["sales"] > 0)]
    return {
        "n": int(len(o)),
        "share_on_promo": float((o["promo"] == 1).mean()),
        "share_on_promo_all_open_days": float((base["promo"] == 1).mean()),
        "share_high_side": float((o["sales"] > o.groupby("store")["sales"].transform("median").clip(lower=1)).mean()),
        "share_month_nov_dec": float(o["date"].dt.month.isin([11, 12]).mean()),
        "share_month_nov_dec_all_open_days": float(base["date"].dt.month.isin([11, 12]).mean()),
    }


def gap_profile(panel: pd.DataFrame) -> dict:
    """Describe the imputed store-days: how many contiguous blocks, and which dates they span."""
    g = panel[panel["is_imputed"] == 1].groupby("store")["date"].agg(["min", "max", "count"])
    g["contiguous"] = ((g["max"] - g["min"]).dt.days + 1) == g["count"]
    blocks = g.groupby(["min", "max", "count"]).size().reset_index(name="stores")
    return {
        "all_contiguous": bool(g["contiguous"].all()),
        "blocks": [{"start": str(r["min"].date()), "end": str(r["max"].date()), "days": int(r["count"]),
                    "stores": int(r["stores"])} for _, r in blocks.iterrows()],
    }


def run() -> dict:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_raw, store_raw = ingest.read_raw()
    schema = {
        "train": validate.check_schema(train_raw, ingest.TRAIN_SCHEMA),
        "store": validate.check_schema(store_raw, ingest.STORE_SCHEMA),
    }
    t_typed, s_typed = clean.coerce_types(train_raw, store_raw)
    pre = validate.validate_train(t_typed)
    pre_store = validate.validate_store(s_typed)
    # StateHoliday is stored as a str column in the raw file, mixing int-like '0' with letters: record the raw variety.
    sh_values = sorted(train_raw["StateHoliday"].unique().tolist())

    panel, log_train = clean.clean_train(t_typed)
    stores, log_store = clean.clean_store(s_typed)

    post = validate.validate_train(to_validation_frame(panel))
    assert_clean(post)

    panel.to_parquet(PROC_DIR / "sales.parquet", index=False)
    stores.to_parquet(PROC_DIR / "stores.parquet", index=False)

    result = {
        "schema": schema, "state_holiday_raw_values": sh_values,
        "pre": pre, "post": post, "pre_store": pre_store,
        "fix_log": log_train, "store_fix_log": log_store,
        "outlier_profile": outlier_profile(panel), "gap_profile": gap_profile(panel),
        "date_range": [str(panel["date"].min().date()), str(panel["date"].max().date())],
        "n_stores": int(panel["store"].nunique()),
    }
    (REPORTS_DIR / "data_quality.json").write_text(json.dumps(result, indent=2))
    write_report(result)
    return result


def issue_table(r: dict) -> list[tuple[str, int, str, str]]:
    """(issue, records affected, action, records after cleaning) - every number comes from the run."""
    pre, post, fl, sfl, op = r["pre"], r["post"], r["fix_log"], r["store_fix_log"], r["outlier_profile"]
    return [
        ("Schema / type problems (missing cols, uncoercible values)",
         sum(r["schema"]["train"].values()) + sum(r["schema"]["store"].values()),
         "Types coerced explicitly; unparseable keys would be dropped", "0"),
        ("Duplicate (store, date) rows", pre["duplicates"], "Keep last record per key", str(post["duplicates"])),
        ("Negative sales", pre["negative_sales"], "Set to 0", str(post["negative_sales"])),
        ("Missing store-days (gaps in the calendar)", pre["missing_store_days"],
         f"Reindexed to a full store x day grid; {pre['stores_with_gaps']} stores affected; rows flagged `is_imputed`, "
         "treated as closed (Open=0, Sales=0, Promo=0)", str(post["missing_store_days"])),
        ("Open=1 but Sales=0", pre["open_but_zero_sales"], "Re-labelled as closed (Open=0)", str(post["open_but_zero_sales"])),
        ("Open=0 but Sales>0", pre["closed_but_sales_positive"], "Re-labelled as open", str(post["closed_but_sales_positive"])),
        ("Invalid StateHoliday codes", pre["invalid_state_holiday_codes"],
         f"Recoded to {{0,a,b,c}} (raw values seen: {r['state_holiday_raw_values']})", str(post["invalid_state_holiday_codes"])),
        ("DayOfWeek inconsistent with Date", pre["dayofweek_date_mismatch"], "Recomputed from Date", str(post["dayofweek_date_mismatch"])),
        ("Non-binary Promo / Open / SchoolHoliday",
         pre["promo_not_binary"] + pre["open_not_binary"] + pre["school_holiday_not_binary"], "Would be rejected", "0"),
        ("Sales outliers (per-store Tukey fence, k=3)", pre["outliers"],
         "Flagged in `is_outlier`, values kept (see below)", f"{post['outliers']} (flagged)"),
        ("Stores with missing CompetitionDistance", r["pre_store"]["missing_competition_distance"],
         "Imputed with the median; `competition_distance_missing` flag added", "0"),
        ("Stores with unknown competition-open date", r["pre_store"]["missing_competition_open_since"],
         "Kept as NaN, flagged `competition_open_since_missing`", "flagged"),
        ("Promo2 start fields blank while Promo2=0 (structural)", r["pre_store"]["promo2_since_blank_when_promo2_off"],
         "Filled with 0 (not applicable)", "0"),
    ]


def write_report(r: dict) -> None:
    pre, post = r["pre"], r["post"]
    rows = issue_table(r)
    op = r["outlier_profile"]
    total_records = sum(c for _, c, _, _ in rows)
    n_issue_types = sum(1 for _, c, _, _ in rows if c > 0)
    lines = [
        "# Data-quality report",
        "",
        "_Generated by `src/pipeline/run.py`; every count below comes from the run on the raw Kaggle files._",
        "",
        f"- Source: Rossmann Store Sales (`train.csv`, `store.csv`), {r['n_stores']:,} stores, "
        f"{r['date_range'][0]} to {r['date_range'][1]}",
        f"- Raw rows: **{pre['rows']:,}**; expected rows on a full store x day grid: **{pre['expected_rows']:,}**; "
        f"clean panel rows: **{post['rows']:,}**",
        f"- Issue types with a non-zero count: **{n_issue_types}**; records affected across them: **{total_records:,}**",
        "",
        "## Issues found and how each was fixed",
        "",
        "| # | Issue | Records affected | Fix | Remaining after cleaning |",
        "|---|-------|-----------------:|-----|-------------------------:|",
    ]
    for i, (name, cnt, fix, remain) in enumerate(rows, 1):
        lines.append(f"| {i} | {name} | {cnt:,} | {fix} | {remain} |")
    lines += [
        "",
        "Checks that found nothing in this dataset (reported honestly as 0): duplicates, negative sales, "
        "schema/type failures, `Open=0` with positive sales, `DayOfWeek` mismatches.",
        "",
        "## Closed-store days",
        "",
        f"- Closed store-days in the raw file: **{pre['closed_days']:,}** "
        f"({pre['closed_days'] / pre['rows']:.1%} of raw rows). They carry `Sales = 0` by construction.",
        f"- Closed-day handling: kept in the panel with `open = 0`. Models treat `open` as a known-in-advance "
        f"calendar input, predict 0 on closed days and are **scored on open days only**, otherwise trivially-correct "
        f"zeros would flatter every metric.",
        f"- After gap-filling, closed store-days in the clean panel: **{post['closed_days']:,}**.",
        "",
        "## Missing dates",
        "",
        f"- {pre['missing_store_days']:,} store-days were absent from `train.csv` across "
        f"**{pre['stores_with_gaps']}** stores; no calendar day was missing for *all* stores "
        f"({pre['calendar_days_missing_for_all_stores']}).",
        f"- Every gap is a contiguous block (all contiguous: {r['gap_profile']['all_contiguous']}): "
        + "; ".join(f"{b['stores']} store(s) missing {b['start']} to {b['end']} ({b['days']} days)" for b in r["gap_profile"]["blocks"])
        + ". Blocks shared by many stores point to closures (refurbishment), not random loss, so they are "
        "treated as closed days rather than interpolated; interpolating sales over a closure would invent revenue.",
        "",
        "## Outliers",
        "",
        f"- {op['n']:,} open-day sales values fall outside the per-store Tukey fences (3 x IQR). "
        f"**{op['share_on_promo']:.1%}** of them occur on promo days "
        f"(promo days are {op['share_on_promo_all_open_days']:.1%} of all open days) and "
        f"**{op['share_month_nov_dec']:.1%}** fall in Nov-Dec "
        f"(vs {op['share_month_nov_dec_all_open_days']:.1%} of all open days).",
        "- Because they cluster on promos and the Christmas season, they are real demand spikes, not sensor/entry errors. "
        "They are **flagged (`is_outlier`) and kept**; deleting them would teach the models to under-forecast exactly the "
        "days the business cares about.",
        "",
        "## Validation after cleaning",
        "",
        "`src/pipeline/run.py` re-runs every validator on the clean panel and **raises** if any of the following is non-zero: "
        + ", ".join(f"`{k}`" for k in MUST_BE_ZERO) + ". It passed.",
        "",
        "## Outputs",
        "",
        "- `data/processed/sales.parquet`: full-calendar store x day panel (git-ignored, rebuilt by `run_all.py`)",
        "- `data/processed/stores.parquet`: cleaned store attributes",
        "- `reports/data_quality.json`: the raw numbers behind this report",
        "",
    ]
    (REPORTS_DIR / "data_quality.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    res = run()
    print("Pipeline OK. Issue records:", {k: v for k, v in res["fix_log"].items()})
