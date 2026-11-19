"""Leakage-safe feature engineering for a 42-day-ahead *direct* forecasting model.

Leakage rule
------------
The model is trained once per forecast origin T and then asked about every date d in (T, T + H].
A feature built from sales therefore must only use sales observed on or before d - H, because
d - H <= T for every d in the forecast window. Hence every lag is >= H and every rolling window is
computed on the series shifted by H. Everything else is either a deterministic calendar fact or a
*planned/known-in-advance* input (store open flag, promo schedule, public/school holidays) - the same
inputs Rossmann provides for the future in the original competition.
`tests/test_features.py` proves this by perturbing sales after T and checking nothing in the window moves.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.config import HORIZON

MONTH_ABBR = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
STORE_TYPE_CODE = {"a": 0, "b": 1, "c": 2, "d": 3}
ASSORT_CODE = {"a": 0, "b": 1, "c": 2}
ROLL_WINDOWS = (7, 28, 91)
YEARLY_LAG = 364  # same weekday one year earlier


def lag_days(horizon: int = HORIZON) -> list[int]:
    """Sales lags used (days). All >= horizon by construction."""
    return [horizon, horizon + 7, horizon + 14, horizon + 21, YEARLY_LAG]


def sales_feature_names(horizon: int = HORIZON) -> list[str]:
    names = [f"lag_{l}" for l in lag_days(horizon)] + ["lag_same_dow_mean4"]
    names += [f"roll_mean_{w}" for w in ROLL_WINDOWS] + ["roll_std_28", "open_rate_28"]
    return names


KNOWN_FEATURES = [
    "day_of_week", "day_of_month", "month", "week_of_year", "day_of_year", "is_month_start", "is_month_end",
    "open", "promo", "promo_prev1", "promo_next1", "state_holiday_flag", "school_holiday",
    "state_holiday_type", "days_to_next_holiday", "days_since_last_holiday", "promo2_active",
    "store_type", "assortment", "competition_distance", "competition_open_months",
]


def feature_columns(horizon: int = HORIZON) -> list[str]:
    return KNOWN_FEATURES + sales_feature_names(horizon)


def _calendar_holiday_distance(panel: pd.DataFrame) -> pd.DataFrame:
    """Days to the next / since the previous state holiday, at calendar level (a holiday for any store)."""
    cal = panel.groupby("date")["state_holiday_flag"].max()
    idx = cal.index
    pos = np.arange(len(idx))
    hol = np.where(cal.to_numpy() == 1, pos, np.nan)
    last = pd.Series(hol, index=idx).ffill()
    nxt = pd.Series(hol, index=idx).bfill()
    return pd.DataFrame({
        "days_since_last_holiday": (pos - last.to_numpy()).clip(max=60),
        "days_to_next_holiday": (nxt.to_numpy() - pos).clip(max=60),
    }, index=idx).fillna(60.0)


def _promo2_active(dates: pd.Series, stores: pd.DataFrame, store_col: pd.Series) -> np.ndarray:
    s = stores.set_index("store")
    on = s["promo2"] == 1
    start = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    yw = s.loc[on]
    start.loc[on] = pd.to_datetime(
        yw["promo2_since_year"].astype(int).astype(str) + "-" + yw["promo2_since_week"].astype(int).clip(1, 53).astype(str).str.zfill(2) + "-1",
        format="%G-%V-%u", errors="coerce")
    # promo_interval uses 3-4 letter month names ('Jan,Apr,Jul,Oct', 'Mar,Jun,Sept,Dec'): match on the first 3 letters
    month_mask = np.zeros((s.index.max() + 1, 13), dtype=bool)
    for st_id, interval in s["promo_interval"].items():
        if interval not in ("None", ""):
            for m in interval.split(","):
                month_mask[st_id, MONTH_ABBR[m.strip()[:3]]] = True
    started = dates.to_numpy() >= store_col.map(start).to_numpy()
    in_month = month_mask[store_col.to_numpy(), dates.dt.month.to_numpy()]
    return (store_col.map(on).to_numpy() & started & in_month).astype(int)


def build_features(panel: pd.DataFrame, stores: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Return the panel plus features, sorted by (date, store). Requires a complete store x day grid."""
    p = panel.sort_values(["date", "store"]).reset_index(drop=True).copy()
    n_dates, n_stores = p["date"].nunique(), p["store"].nunique()
    if len(p) != n_dates * n_stores:
        raise ValueError("panel must be a complete store x date grid (run the cleaning pipeline first)")
    if horizon > YEARLY_LAG:
        raise ValueError("horizon must not exceed the yearly lag")

    # ---- known / calendar features ----
    d = p["date"]
    p["day_of_week"] = d.dt.day_of_week
    p["day_of_month"] = d.dt.day
    p["month"] = d.dt.month
    p["week_of_year"] = d.dt.isocalendar().week.astype(int)
    p["day_of_year"] = d.dt.day_of_year
    p["is_month_start"] = d.dt.is_month_start.astype(int)
    p["is_month_end"] = d.dt.is_month_end.astype(int)
    p["state_holiday_type"] = p["state_holiday"].map({"0": 0, "a": 1, "b": 2, "c": 3}).astype(int)
    pr = p.pivot(index="date", columns="store", values="promo")
    p["promo_prev1"] = pr.shift(1).fillna(0).to_numpy().ravel().astype(int)
    p["promo_next1"] = pr.shift(-1).fillna(0).to_numpy().ravel().astype(int)
    hd = _calendar_holiday_distance(p)
    p = p.join(hd, on="date")
    p["promo2_active"] = _promo2_active(p["date"], stores, p["store"])

    st = stores.set_index("store")
    p["store_type"] = p["store"].map(st["store_type"].map(STORE_TYPE_CODE))
    p["assortment"] = p["store"].map(st["assortment"].map(ASSORT_CODE))
    p["competition_distance"] = p["store"].map(st["competition_distance"])
    comp_start = p["store"].map(st["competition_open_since"])
    months = (p["date"] - comp_start).dt.days / 30.44
    p["competition_open_months"] = months.clip(lower=0)  # NaN when unknown; trees handle NaN natively

    # ---- sales-derived features (all shifted >= horizon) ----
    wide = p.pivot(index="date", columns="store", values="sales").where(
        p.pivot(index="date", columns="store", values="open") == 1)
    cols_sorted = wide.columns

    def flat(w: pd.DataFrame) -> np.ndarray:
        return w[cols_sorted].to_numpy().ravel()

    lags = {l: wide.shift(l) for l in lag_days(horizon)}
    for l, w in lags.items():
        p[f"lag_{l}"] = flat(w)
    dow_lags = [lags[horizon + 7 * k] if k else lags[horizon] for k in range(4)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices legitimately give NaN
        p["lag_same_dow_mean4"] = np.nanmean(np.stack([flat(w) for w in dow_lags]), axis=0)
    base = wide.shift(horizon)
    for w in ROLL_WINDOWS:
        p[f"roll_mean_{w}"] = flat(base.rolling(w, min_periods=1).mean())
    p["roll_std_28"] = flat(base.rolling(28, min_periods=2).std())
    p["open_rate_28"] = flat(base.notna().astype(float).rolling(28, min_periods=1).mean())
    return p
