"""SHAP + gain feature-importance plots for the LightGBM model."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import FIG_DIR, SEED


def explain_lgbm(model, X: pd.DataFrame, n_sample: int = 5000) -> pd.DataFrame:
    """Save SHAP summary + gain-importance figures; return mean |SHAP| per feature (log1p-sales units)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    Xs = X.sample(min(n_sample, len(X)), random_state=SEED)
    sv = shap.TreeExplainer(model).shap_values(Xs)

    plt.figure()
    shap.summary_plot(sv, Xs, max_display=20, show=False)
    plt.title("SHAP summary - LightGBM (impact on log1p(sales))")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary.png", dpi=130)
    plt.close()

    imp = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(sv).mean(axis=0),
        "gain": model.booster_.feature_importance("gain"),
    }).sort_values("mean_abs_shap", ascending=False)
    imp["gain_share"] = imp["gain"] / imp["gain"].sum()

    top = imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(11, 6), sharey=False)
    ax[0].barh(top["feature"], top["mean_abs_shap"], color="#2a6f97")
    ax[0].set_title("Mean |SHAP|")
    top_g = imp.sort_values("gain", ascending=False).head(20).iloc[::-1]
    ax[1].barh(top_g["feature"], top_g["gain_share"], color="#6a994e")
    ax[1].set_title("Share of total split gain")
    fig.suptitle("LightGBM feature importance (latest fold)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=130)
    plt.close(fig)
    return imp
