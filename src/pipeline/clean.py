"""Cleaning: type coercion, a fix for every issue the validators count, and a full-calendar panel."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.validate import flag_outliers

TRAIN_INT_COLS = ["Store", "DayOfWeek", "Sales", "Customers", "Open", "Promo", "SchoolHoliday"]


def coerce_types(train_raw: pd.DataFrame, store_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """String -> typed frames. Unparseable values become NaN (they are counted by check_schema)."""
    t = train_raw.copy()
    t["Date"] = pd.to_datetime(t["Date"], format="%Y-%m-%d", errors="coerce")
    for c in TRAIN_INT_COLS:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t["StateHoliday"] = t["StateHoliday"].astype(str).str.strip()
    s = store_raw.copy()
    for c in ["Store", "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
              "Promo2", "Promo2SinceWeek", "Promo2SinceYear"]:
        s[c] = pd.to_numeric(s[c].replace("", np.nan), errors="coerce")
    return t, s


def clean_train(t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return the clean *full-calendar* panel and a log of how many rows each fix touched."""
    log: dict[str, int] = {}
    t = t.copy()

    # 1. rows with unusable keys cannot be repaired
    bad_key = t[["Store", "Date"]].isna().any(axis=1)
    log["dropped_rows_unparseable_key"] = int(bad_key.sum())
    t = t[~bad_key]

    # 2. duplicates on (Store, Date): keep the last record
    dup = t.duplicated(["Store", "Date"], keep="last")
    log["duplicates_removed"] = int(dup.sum())
    t = t[~dup]

    # 3. negative sales -> 0
    neg = t["Sales"] < 0
    log["negative_sales_set_to_zero"] = int(neg.sum())
    t.loc[neg, "Sales"] = 0

    # 4. StateHoliday mixed type / invalid codes -> clean category
    sh = t["StateHoliday"].where(t["StateHoliday"].isin(["0", "a", "b", "c"]), "0")
    log["invalid_state_holiday_recoded"] = int((sh != t["StateHoliday"]).sum())
    t["StateHoliday"] = sh

    # 5. Closed-day consistency
    closed_pos = (t["Open"] == 0) & (t["Sales"] > 0)
    log["closed_but_sales_positive_set_open"] = int(closed_pos.sum())
    t.loc[closed_pos, "Open"] = 1
    open_zero = (t["Open"] == 1) & (t["Sales"] == 0)
    log["open_but_zero_sales_set_closed"] = int(open_zero.sum())
    t.loc[open_zero, "Open"] = 0

    # 6. Missing dates -> reindex to the full store x day grid
    dates = pd.date_range(t["Date"].min(), t["Date"].max(), freq="D")
    grid = pd.MultiIndex.from_product([sorted(t["Store"].unique()), dates], names=["Store", "Date"])
    t = t.set_index(["Store", "Date"]).reindex(grid)
    t = t.reset_index()
    imputed = t["Sales"].isna()
    log["missing_store_days_filled"] = int(imputed.sum())
    t["is_imputed"] = imputed.astype(int)
    # Absent store-days are the known Rossmann refurbishment closures: closed, no sales, no promo.
    t.loc[imputed, ["Sales", "Customers", "Open", "Promo"]] = 0
    # State holiday for new rows comes from the other stores on that date; school holiday defaults to 0.
    cal_mode = t.loc[~imputed].groupby("Date")["StateHoliday"].agg(lambda x: x.mode().iat[0])
    t.loc[imputed, "StateHoliday"] = t.loc[imputed, "Date"].map(cal_mode).fillna("0")
    t.loc[imputed, "SchoolHoliday"] = 0

    # 7. DayOfWeek is derived from Date (repairs mismatches, fills new rows)
    derived = t["Date"].dt.day_of_week + 1
    log["dayofweek_recomputed"] = int(((t["DayOfWeek"] != derived) & ~imputed).sum())
    t["DayOfWeek"] = derived

    # 8. Outliers: flagged, not removed (see data_quality report for why)
    t["is_outlier"] = flag_outliers(t).astype(int)
    log["outliers_flagged_kept"] = int(t["is_outlier"].sum())

    for c in TRAIN_INT_COLS:
        t[c] = t[c].astype(int)
    out = t.rename(columns={
        "Store": "store", "Date": "date", "DayOfWeek": "day_of_week", "Sales": "sales", "Customers": "customers",
        "Open": "open", "Promo": "promo", "StateHoliday": "state_holiday", "SchoolHoliday": "school_holiday",
    })
    out["state_holiday_flag"] = (out["state_holiday"] != "0").astype(int)
    return out.sort_values(["store", "date"]).reset_index(drop=True), log


def clean_store(s: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    s = s.drop_duplicates("Store", keep="last").copy()
    log = {}
    miss = s["CompetitionDistance"].isna()
    log["competition_distance_median_imputed"] = int(miss.sum())
    s["competition_distance_missing"] = miss.astype(int)
    s["CompetitionDistance"] = s["CompetitionDistance"].fillna(s["CompetitionDistance"].median())
    no_comp = s["CompetitionOpenSinceYear"].isna()
    log["competition_open_since_unknown_flagged"] = int(no_comp.sum())
    s["competition_open_since_missing"] = no_comp.astype(int)
    s["competition_open_since"] = pd.to_datetime(
        {"year": s["CompetitionOpenSinceYear"], "month": s["CompetitionOpenSinceMonth"].fillna(1), "day": 1},
        errors="coerce",
    )
    off = s["Promo2"] == 0
    log["promo2_fields_structurally_blank_filled"] = int((off & s["Promo2SinceYear"].isna()).sum())
    s["PromoInterval"] = s["PromoInterval"].replace("", "None").fillna("None")
    s.loc[off, ["Promo2SinceWeek", "Promo2SinceYear"]] = 0
    s = s.rename(columns={
        "Store": "store", "StoreType": "store_type", "Assortment": "assortment",
        "CompetitionDistance": "competition_distance", "Promo2": "promo2",
        "Promo2SinceWeek": "promo2_since_week", "Promo2SinceYear": "promo2_since_year",
        "PromoInterval": "promo_interval",
    }).drop(columns=["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"])
    return s.reset_index(drop=True), log
