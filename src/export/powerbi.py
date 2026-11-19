"""Export a star schema for Power BI Desktop (CSV, UTF-8, ISO dates, '.' decimal).

Tables
  dim_date            1 row per calendar day
  dim_store           1 row per store
  dim_model           1 row per forecasting model (+ best-model flag)
  fact_sales_actuals  1 row per OPEN store-day (closed days are absent; see DASHBOARD_GUIDE.md)
  fact_forecasts      1 row per (model, fold, open store-day) with lower/upper where the model provides them
  model_metrics       1 row per (model, fold, scope); fold 0 = all folds pooled
  feature_importance  LightGBM mean |SHAP| and gain share (for the Model Performance page)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import POWERBI_DIR, PROC_DIR, REPORTS_DIR, SARIMA_N_STORES, SEED
from src.models import metrics
from src.models.backtest import LOWER_Q, UPPER_Q

MODEL_INFO = {
    "naive": ("Baseline", "Last observed open-day sales repeated for the horizon"),
    "seasonal_naive": ("Baseline", "Same weekday in the most recent observed week"),
    "sarima": ("Statistical", "Per-store SARIMAX(1,0,1)(1,1,1)7 with open/promo/holiday regressors (store sample only)"),
    "lightgbm": ("Machine learning", "Global LightGBM on lag/rolling/calendar/promo features; quantile models give the 80% interval"),
}


def dim_date(start: pd.Timestamp, end: pd.Timestamp, holidays: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    d["date_key"] = d["date"].dt.strftime("%Y%m%d").astype(int)
    d["year"], d["quarter"], d["month"] = d["date"].dt.year, d["date"].dt.quarter, d["date"].dt.month
    d["month_name"] = d["date"].dt.strftime("%b")
    d["year_month"] = d["date"].dt.strftime("%Y-%m")
    d["week_of_year"] = d["date"].dt.isocalendar().week.astype(int)
    d["day_of_week"] = d["date"].dt.day_of_week + 1          # 1 = Monday
    d["day_name"] = d["date"].dt.strftime("%a")
    d["is_weekend"] = (d["day_of_week"] >= 6).astype(int)
    d["day_of_month"] = d["date"].dt.day
    d["state_holiday_any_store"] = d["date"].map(holidays).fillna(0).astype(int)
    d["date"] = d["date"].dt.strftime("%Y-%m-%d")
    return d


def export() -> dict:
    POWERBI_DIR.mkdir(exist_ok=True)
    panel = pd.read_parquet(PROC_DIR / "sales.parquet")
    stores = pd.read_parquet(PROC_DIR / "stores.parquet")
    P = pd.read_parquet(PROC_DIR / "predictions.parquet")
    summary = json.loads((REPORTS_DIR / "metrics.json").read_text())
    best = summary["best_model"]

    rng = np.random.default_rng(SEED)
    sarima_stores = set(rng.choice(sorted(panel["store"].unique()), SARIMA_N_STORES, replace=False).tolist())
    assert sarima_stores == set(P.loc[P["model"] == "sarima", "store"].unique()), "SARIMA store sample drifted"

    hol = panel.groupby("date")["state_holiday_flag"].max()
    dd = dim_date(panel["date"].min(), panel["date"].max(), hol)

    ds = stores.copy()
    ds["competition_open_since"] = ds["competition_open_since"].dt.strftime("%Y-%m-%d")
    ds["in_sarima_sample"] = ds["store"].isin(sarima_stores).astype(int)
    ds["store_label"] = "Store " + ds["store"].astype(str)
    ds = ds.rename(columns={"store": "store_key"})

    op = panel[panel["open"] == 1]
    fs = pd.DataFrame({
        "store_key": op["store"], "date_key": op["date"].dt.strftime("%Y%m%d").astype(int),
        "sales": op["sales"], "customers": op["customers"], "promo": op["promo"],
        "state_holiday": op["state_holiday"], "school_holiday": op["school_holiday"], "is_outlier": op["is_outlier"],
    })

    f = P[P["open"] == 1].copy()
    ff = pd.DataFrame({
        "model": f["model"], "fold": f["fold"], "origin_date_key": f["origin"].dt.strftime("%Y%m%d").astype(int),
        "store_key": f["store"], "date_key": f["date"].dt.strftime("%Y%m%d").astype(int),
        "horizon_day": f["horizon_day"], "actual_sales": f["actual"],
        "forecast": f["forecast"].round(1), "lower_80": f["lower"].round(1), "upper_80": f["upper"].round(1),
    })
    ff["abs_error"] = (ff["forecast"] - ff["actual_sales"]).abs().round(1)
    ff = ff.sort_values(["model", "fold", "store_key", "date_key"])

    rows = []
    for scope, frame in (("all_stores", P), ("sarima_sample", P[P["store"].isin(sarima_stores)])):
        for (m, fold), g in list(frame.groupby(["model", "fold"])) + [((m, 0), g) for m, g in frame.groupby("model")]:
            if scope == "all_stores" and m == "sarima":
                continue
            r = {"model": m, "fold": fold, "scope": scope, **metrics.score(g)}
            r["origin_date"] = str(g["origin"].iloc[0].date()) if fold else None
            if m == "lightgbm":
                ir = metrics.interval_report(g, UPPER_Q - LOWER_Q)
                r["coverage_80"], r["interval_width_pct_of_actual"] = ir["empirical_coverage"] * 100, ir["mean_width_pct_of_actual"]
            rows.append(r)
    mm = pd.DataFrame(rows)
    keep = ["model", "fold", "scope", "origin_date", "WAPE", "MAPE", "RMSE", "Bias", "n_rows", "coverage_80", "interval_width_pct_of_actual"]
    mm = mm.reindex(columns=keep).round(3)
    mm = mm.rename(columns={"WAPE": "wape_pct", "MAPE": "mape_pct", "RMSE": "rmse", "Bias": "bias_pct"})

    dm = pd.DataFrame([
        {"model": m, "model_type": t, "description": d, "is_best_model": int(m == best),
         "has_interval": int(m == "lightgbm"), "scope": "sarima_sample" if m == "sarima" else "all_stores"}
        for m, (t, d) in MODEL_INFO.items()
    ])
    imp = pd.read_csv(REPORTS_DIR / "feature_importance.csv").round(5)

    out = {"dim_date": dd, "dim_store": ds, "dim_model": dm, "fact_sales_actuals": fs, "fact_forecasts": ff,
           "model_metrics": mm, "feature_importance": imp}
    for name, df in out.items():
        df.to_csv(POWERBI_DIR / f"{name}.csv", index=False)
    return {k: len(v) for k, v in out.items()}


if __name__ == "__main__":
    print(export())
