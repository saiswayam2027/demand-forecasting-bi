"""Data-quality checks. Pure functions: each takes a DataFrame and returns counts (no mutation).

`validate_train` expects a *typed* frame (see `clean.coerce_types`) so the same checks can be
run before cleaning (to report issues) and after cleaning (to prove they are fixed).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

VALID_STATE_HOLIDAY = {"0", "a", "b", "c"}


@dataclass
class Issue:
    check: str
    count: int
    detail: str = ""
    extra: dict = field(default_factory=dict)


def check_schema(df: pd.DataFrame, schema: dict[str, str]) -> dict:
    """Missing / unexpected columns and values that cannot be coerced to the expected type."""
    missing = sorted(set(schema) - set(df.columns))
    unexpected = sorted(set(df.columns) - set(schema))
    bad_total = 0
    for col, kind in schema.items():
        if col not in df.columns:
            continue
        s = df[col].astype(str)
        if kind in ("int", "float"):
            nonblank = s.str.strip() != ""
            bad = int((pd.to_numeric(s.where(nonblank), errors="coerce").isna() & nonblank).sum())
        elif kind == "date":
            bad = int(pd.to_datetime(s, format="%Y-%m-%d", errors="coerce").isna().sum())
        else:
            bad = 0
        bad_total += bad
    return {
        "schema_missing_columns": len(missing),
        "schema_unexpected_columns": len(unexpected),
        "schema_uncoercible_values": bad_total,
    }


def count_duplicates(df: pd.DataFrame, keys=("Store", "Date")) -> int:
    return int(df.duplicated(list(keys), keep="first").sum())


def missing_dates(df: pd.DataFrame) -> dict:
    """Store-days absent from the table, relative to a full (store x calendar-day) grid."""
    dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    expected = df["Store"].nunique() * len(dates)
    uniq = df.drop_duplicates(["Store", "Date"])
    per_store = uniq.groupby("Store").size()
    return {
        "missing_store_days": int(expected - len(uniq)),
        "stores_with_gaps": int((per_store < len(dates)).sum()),
        "calendar_days_missing_for_all_stores": int(len(dates) - df["Date"].nunique()),
        "expected_rows": int(expected),
    }


def count_negative_sales(df: pd.DataFrame) -> int:
    return int((df["Sales"] < 0).sum())


def closed_day_stats(df: pd.DataFrame) -> dict:
    return {
        "closed_days": int((df["Open"] == 0).sum()),
        "closed_but_sales_positive": int(((df["Open"] == 0) & (df["Sales"] > 0)).sum()),
        "open_but_zero_sales": int(((df["Open"] == 1) & (df["Sales"] == 0)).sum()),
    }


def flag_outliers(df: pd.DataFrame, k: float = 3.0) -> pd.Series:
    """Per-store Tukey fences (Q1 - k*IQR, Q3 + k*IQR) on open-day sales. k=3 means far-out outliers."""
    s = df["Sales"].where((df["Open"] == 1) & (df["Sales"] > 0))
    g = s.groupby(df["Store"])
    q1, q3 = g.transform(lambda x: x.quantile(0.25)), g.transform(lambda x: x.quantile(0.75))
    iqr = q3 - q1
    return ((s > q3 + k * iqr) | (s < q1 - k * iqr)).fillna(False)


def count_invalid_categories(df: pd.DataFrame) -> dict:
    sh = df["StateHoliday"].astype(str)
    return {
        "invalid_state_holiday_codes": int((~sh.isin(VALID_STATE_HOLIDAY)).sum()),
        "dayofweek_date_mismatch": int((df["Date"].dt.day_of_week + 1 != df["DayOfWeek"]).sum()),
        "promo_not_binary": int((~df["Promo"].isin([0, 1])).sum()),
        "open_not_binary": int((~df["Open"].isin([0, 1])).sum()),
        "school_holiday_not_binary": int((~df["SchoolHoliday"].isin([0, 1])).sum()),
    }


def validate_train(df: pd.DataFrame) -> dict:
    """Run every row-level check on a typed training frame."""
    return {
        "rows": int(len(df)),
        "duplicates": count_duplicates(df),
        "negative_sales": count_negative_sales(df),
        **missing_dates(df),
        **closed_day_stats(df),
        "outliers": int(flag_outliers(df).sum()),
        **count_invalid_categories(df),
    }


def validate_store(store: pd.DataFrame) -> dict:
    promo2_off = store["Promo2"] == 0
    return {
        "stores": int(len(store)),
        "duplicate_stores": int(store["Store"].duplicated().sum()),
        "missing_competition_distance": int(store["CompetitionDistance"].isna().sum()),
        "missing_competition_open_since": int(store["CompetitionOpenSinceYear"].isna().sum()),
        "promo2_since_missing_when_promo2_active": int(((~promo2_off) & store["Promo2SinceYear"].isna()).sum()),
        "promo2_since_blank_when_promo2_off": int((promo2_off & store["Promo2SinceYear"].isna()).sum()),
    }
