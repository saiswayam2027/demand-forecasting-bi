"""LightGBM point + quantile forecasters (one global model across all stores, direct 42-day horizon)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import SEED

BASE_PARAMS = dict(
    n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=SEED, n_jobs=-1, verbose=-1,
)
CATEGORICAL = ["state_holiday_type", "store_type", "assortment"]


def training_rows(feat: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Rows a model fitted at `origin` may learn from: observed, open days on or before the origin."""
    return feat[(feat["date"] <= origin) & (feat["open"] == 1)]


def fit(train: pd.DataFrame, features: list[str], alpha: float | None = None, params: dict | None = None) -> lgb.LGBMRegressor:
    """Fit on log1p(sales). alpha=None -> squared error; otherwise a quantile model.

    A monotone transform preserves quantiles, so the quantile models in log space map back with expm1.
    """
    p = {**BASE_PARAMS, **(params or {})}
    if alpha is None:
        p["objective"] = "regression"
    else:
        p.update(objective="quantile", alpha=alpha)
    model = lgb.LGBMRegressor(**p)
    cats = [c for c in CATEGORICAL if c in features]
    model.fit(train[features], np.log1p(train["sales"]), categorical_feature=cats)
    return model


def predict(model: lgb.LGBMRegressor, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.clip(np.expm1(model.predict(df[features])), 0, None) * df["open"].to_numpy()
