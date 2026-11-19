"""Per-store SARIMAX with weekly seasonality and known-future regressors.

Model: SARIMAX(1,0,1)(1,1,1)_7 on daily sales (closed days = 0) with exogenous [open, promo,
state_holiday_flag, school_holiday]. The open flag lets the model treat closed days as a level shift
instead of noise. Fitted on the last `max_history` days before the origin for speed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.config import HORIZON

EXOG = ["open", "promo", "state_holiday_flag", "school_holiday"]


def _one_store(store: int, hist: pd.DataFrame, fut: pd.DataFrame) -> pd.DataFrame:
    y = hist["sales"].astype(float).to_numpy()
    X = hist[EXOG].astype(float).to_numpy()
    Xf = fut[EXOG].astype(float).to_numpy()
    fallback = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = SARIMAX(y, exog=X, order=(1, 0, 1), seasonal_order=(1, 1, 1, 7),
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=60)
            fc = res.forecast(steps=len(fut), exog=Xf)
        except Exception:  # fall back to the seasonal-naive-like mean of open days so one bad fit cannot kill a fold
            fallback = True
            fc = np.full(len(fut), y[X[:, 0] == 1][-28:].mean() if (X[:, 0] == 1).any() else 0.0)
    out = fut[["store", "date", "open", "sales"]].copy()
    out["forecast"] = np.clip(np.asarray(fc), 0, None) * fut["open"].to_numpy()
    out["fallback"] = fallback  # surfaced in the backtest log so a failed fit is never silent
    return out


def sarima_forecast(panel: pd.DataFrame, stores: list[int], origin: pd.Timestamp, horizon: int = HORIZON,
                    max_history: int = 730, n_jobs: int = -1) -> pd.DataFrame:
    start = origin - pd.Timedelta(days=max_history - 1)
    end = origin + pd.Timedelta(days=horizon)
    sub = panel[panel["store"].isin(stores) & (panel["date"] >= start) & (panel["date"] <= end)]
    jobs = []
    for s, g in sub.groupby("store"):
        g = g.sort_values("date")
        jobs.append(delayed(_one_store)(s, g[g["date"] <= origin], g[g["date"] > origin]))
    return pd.concat(Parallel(n_jobs=n_jobs)(jobs), ignore_index=True)
