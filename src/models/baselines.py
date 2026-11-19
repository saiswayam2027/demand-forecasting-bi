"""Baseline forecasters. Both see sales only up to the forecast origin T.

Every model in this project treats `open` as a known-in-advance input: forecasts on closed days are 0.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.config import HORIZON


def _wide_sales(panel: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """date x store matrix of open-day sales observed up to and including `origin` (NaN = closed)."""
    h = panel[panel["date"] <= origin]
    sales = h.pivot(index="date", columns="store", values="sales")
    opened = h.pivot(index="date", columns="store", values="open")
    return sales.where(opened == 1)


def _target_frame(panel: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
    end = origin + pd.Timedelta(days=horizon)
    return panel[(panel["date"] > origin) & (panel["date"] <= end)][["store", "date", "open", "sales"]].copy()


def naive_forecast(panel: pd.DataFrame, origin: pd.Timestamp, horizon: int = HORIZON) -> pd.DataFrame:
    """Naive: every future open day = the last observed open-day sales value of that store."""
    wide = _wide_sales(panel, origin)
    last = wide.ffill().iloc[-1].fillna(0.0)
    tgt = _target_frame(panel, origin, horizon)
    tgt["forecast"] = tgt["store"].map(last).to_numpy() * tgt["open"].to_numpy()
    return tgt


def seasonal_naive_forecast(panel: pd.DataFrame, origin: pd.Timestamp, horizon: int = HORIZON) -> pd.DataFrame:
    """Seasonal naive (period 7): value of the same weekday in the most recent observed week.

    For date d the reference is d - 7*ceil((d - T)/7), which always lies in (T-7, T]. If that day was
    closed (NaN, e.g. a holiday) fall back to the mean of the previous 3 same-weekdays, and finally to the
    store's last observed value, so the baseline is never undefined for an open day.
    """
    wide = _wide_sales(panel, origin)
    last = wide.ffill().iloc[-1].fillna(0.0)
    tgt = _target_frame(panel, origin, horizon)
    ref_day_cache: dict[pd.Timestamp, pd.Series] = {}
    out = np.zeros(len(tgt))
    for i, (d, grp) in enumerate(tgt.groupby("date")):
        k = int(np.ceil((d - origin).days / 7))
        refs = [d - pd.Timedelta(days=7 * (k + j)) for j in range(4)]
        mat = np.stack([wide.loc[r].to_numpy() if r in wide.index else np.full(wide.shape[1], np.nan) for r in refs])
        first = mat[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fb = np.nanmean(mat[1:], axis=0)
        val = pd.Series(np.where(np.isnan(first), fb, first), index=wide.columns)
        val = val.fillna(last)
        ref_day_cache[d] = val
    tgt["forecast"] = [ref_day_cache[d][s] for d, s in zip(tgt["date"], tgt["store"])]
    tgt["forecast"] = tgt["forecast"].to_numpy() * tgt["open"].to_numpy()
    return tgt
