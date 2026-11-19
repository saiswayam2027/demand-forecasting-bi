"""Rolling-origin (walk-forward) backtest of every model.

For each fold the origin T is fixed, every model is (re)fitted using only data dated <= T, and it forecasts
the next 42 days for every store. Folds are consecutive, non-overlapping 42-day windows ending at the last
date of the dataset, so the test windows tile the final N_FOLDS x 6 weeks.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.config import HORIZON, N_FOLDS, PROC_DIR, REPORTS_DIR, SARIMA_N_STORES, SEED
from src.models import baselines, lgbm, metrics, sarima
from src.models.explain import explain_lgbm
from src.models.features import build_features, feature_columns

LOWER_Q, UPPER_Q = 0.1, 0.9  # 80% prediction interval
MODELS_ALL_STORES = ["naive", "seasonal_naive", "lightgbm"]


def make_origins(last_date: pd.Timestamp, n_folds: int = N_FOLDS, horizon: int = HORIZON) -> list[pd.Timestamp]:
    """Chronological forecast origins; fold i tests (origin_i, origin_i + horizon]. Last window ends at last_date."""
    return [last_date - pd.Timedelta(days=horizon * (n_folds - i + 1)) for i in range(1, n_folds + 1)]


def _tag(df: pd.DataFrame, model: str, fold: int, origin: pd.Timestamp) -> pd.DataFrame:
    out = df.rename(columns={"sales": "actual"}).copy()
    out["model"], out["fold"], out["origin"] = model, fold, origin
    out["horizon_day"] = (out["date"] - origin).dt.days
    for c in ("lower", "upper"):
        if c not in out:
            out[c] = np.nan
    return out[["model", "fold", "origin", "store", "date", "horizon_day", "open", "actual", "forecast", "lower", "upper"]]


def run_backtest(verbose: bool = True) -> dict:
    panel = pd.read_parquet(PROC_DIR / "sales.parquet")
    stores = pd.read_parquet(PROC_DIR / "stores.parquet")
    feat = build_features(panel, stores, HORIZON)
    cols = feature_columns(HORIZON)
    origins = make_origins(panel["date"].max())
    rng = np.random.default_rng(SEED)
    sarima_stores = sorted(rng.choice(sorted(panel["store"].unique()), SARIMA_N_STORES, replace=False).tolist())

    preds, sarima_fallbacks, last_model, last_X = [], 0, None, None
    for fold, origin in enumerate(origins, 1):
        t0 = time.time()
        test_end = origin + pd.Timedelta(days=HORIZON)
        assert test_end <= panel["date"].max()
        preds.append(_tag(baselines.naive_forecast(panel, origin), "naive", fold, origin))
        preds.append(_tag(baselines.seasonal_naive_forecast(panel, origin), "seasonal_naive", fold, origin))

        train = lgbm.training_rows(feat, origin)
        test = feat[(feat["date"] > origin) & (feat["date"] <= test_end)]
        assert train["date"].max() <= origin < test["date"].min()
        point = lgbm.fit(train, cols)
        lo = lgbm.fit(train, cols, alpha=LOWER_Q)
        hi = lgbm.fit(train, cols, alpha=UPPER_Q)
        out = test[["store", "date", "open", "sales"]].copy()
        out["forecast"] = lgbm.predict(point, test, cols)
        l, u = lgbm.predict(lo, test, cols), lgbm.predict(hi, test, cols)
        out["lower"], out["upper"] = np.minimum(l, u), np.maximum(l, u)  # guard against quantile crossing
        preds.append(_tag(out, "lightgbm", fold, origin))
        last_model, last_X = point, test.loc[test["open"] == 1, cols]

        sar = sarima.sarima_forecast(panel, sarima_stores, origin)
        sarima_fallbacks += int(sar.groupby("store")["fallback"].first().sum())
        preds.append(_tag(sar.drop(columns="fallback"), "sarima", fold, origin))
        if verbose:
            print(f"fold {fold}/{len(origins)} origin={origin.date()} done in {time.time() - t0:.0f}s", flush=True)

    P = pd.concat(preds, ignore_index=True)
    P.to_parquet(PROC_DIR / "predictions.parquet", index=False)
    imp = explain_lgbm(last_model, last_X)
    return summarize(P, origins, sarima_stores, sarima_fallbacks, imp)


def summarize(P: pd.DataFrame, origins, sarima_stores: list[int], sarima_fallbacks: int, imp: pd.DataFrame) -> dict:
    by_fold = pd.DataFrame([
        {"model": m, "fold": f, "origin": str(g["origin"].iloc[0].date()), **metrics.score(g)}
        for (m, f), g in P.groupby(["model", "fold"])
    ])
    overall = pd.DataFrame([{"model": m, **metrics.score(g)} for m, g in P.groupby("model")])
    fold_stats = by_fold.groupby("model")["WAPE"].agg(["mean", "std"]).rename(columns={"mean": "WAPE_fold_mean", "std": "WAPE_fold_std"})
    overall = overall.merge(fold_stats, on="model")

    sub = P[P["store"].isin(sarima_stores)]
    overall_subset = pd.DataFrame([{"model": m, **metrics.score(g)} for m, g in sub.groupby("model")])

    all_store = overall[overall["model"].isin(MODELS_ALL_STORES)].sort_values("WAPE")
    best = all_store.iloc[0]["model"]
    base = overall.set_index("model").loc["seasonal_naive"]
    bm = overall.set_index("model").loc[best]
    improvement = {k: float((base[k] - bm[k]) / base[k] * 100) for k in ("WAPE", "MAPE", "RMSE")}

    lg = P[P["model"] == "lightgbm"]
    interval = metrics.interval_report(lg, UPPER_Q - LOWER_Q)
    interval_by_fold = {
        int(f): metrics.interval_report(g, UPPER_Q - LOWER_Q)["empirical_coverage"] for f, g in lg.groupby("fold")
    }
    lg_h = lg.assign(week=(lg["horizon_day"] - 1) // 7 + 1)
    by_week = pd.DataFrame([{"week": int(w), **metrics.score(g)} for w, g in lg_h.groupby("week")])
    sn = P[P["model"] == "seasonal_naive"].assign(week=lambda d: (d["horizon_day"] - 1) // 7 + 1)
    by_week["seasonal_naive_WAPE"] = [metrics.score(g)["WAPE"] for _, g in sn.groupby("week")]

    REPORTS_DIR.mkdir(exist_ok=True)
    by_fold.to_csv(REPORTS_DIR / "metrics_by_fold.csv", index=False)
    overall.to_csv(REPORTS_DIR / "metrics_overall.csv", index=False)
    overall_subset.to_csv(REPORTS_DIR / "metrics_sarima_subset.csv", index=False)
    by_week.to_csv(REPORTS_DIR / "metrics_by_horizon_week.csv", index=False)
    imp.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    result = {
        "origins": [str(o.date()) for o in origins], "horizon_days": HORIZON, "n_folds": len(origins),
        "test_period": [str((origins[0] + pd.Timedelta(days=1)).date()), str((origins[-1] + pd.Timedelta(days=HORIZON)).date())],
        "n_stores": int(P["store"].nunique()), "sarima_stores": len(sarima_stores), "sarima_fallback_fits": sarima_fallbacks,
        "best_model": best, "improvement_vs_seasonal_naive_pct": improvement,
        "overall": overall.to_dict("records"), "overall_sarima_subset": overall_subset.to_dict("records"),
        "by_fold": by_fold.to_dict("records"), "by_horizon_week": by_week.to_dict("records"),
        "interval_80": interval, "interval_80_coverage_by_fold": interval_by_fold,
        "top_features": imp.head(10)[["feature", "mean_abs_shap", "gain_share"]].to_dict("records"),
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    r = run_backtest()
    print(pd.DataFrame(r["overall"]).round(2).to_string())
    print("best:", r["best_model"], r["improvement_vs_seasonal_naive_pct"])
