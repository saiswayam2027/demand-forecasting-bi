"""Forecast accuracy metrics. All are computed on *open* store-days only (actual > 0)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def wape(actual, forecast) -> float:
    """Weighted absolute percentage error: sum|e| / sum|y|, in %. Robust to small actuals."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    return float(np.abs(a - f).sum() / np.abs(a).sum() * 100)


def mape(actual, forecast) -> float:
    """Mean absolute percentage error in %, over rows with actual > 0 (undefined otherwise)."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    m = a > 0
    return float((np.abs(a[m] - f[m]) / a[m]).mean() * 100)


def rmse(actual, forecast) -> float:
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    return float(np.sqrt(((a - f) ** 2).mean()))


def bias(actual, forecast) -> float:
    """Signed error in % of total actual (positive = over-forecast)."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    return float((f - a).sum() / a.sum() * 100)


def score(df: pd.DataFrame, actual="actual", forecast="forecast") -> dict:
    d = df[df[actual] > 0]
    return {
        "WAPE": wape(d[actual], d[forecast]), "MAPE": mape(d[actual], d[forecast]),
        "RMSE": rmse(d[actual], d[forecast]), "Bias": bias(d[actual], d[forecast]), "n_rows": int(len(d)),
    }


def interval_report(df: pd.DataFrame, nominal: float) -> dict:
    """Empirical coverage and mean width of [lower, upper] on open days."""
    d = df[df["actual"] > 0]
    inside = (d["actual"] >= d["lower"]) & (d["actual"] <= d["upper"])
    return {
        "nominal_coverage": nominal, "empirical_coverage": float(inside.mean()),
        "mean_width": float((d["upper"] - d["lower"]).mean()),
        "mean_width_pct_of_actual": float(((d["upper"] - d["lower"]) / d["actual"]).mean() * 100),
    }
