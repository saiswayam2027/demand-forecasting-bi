import numpy as np
import pandas as pd
import pytest

from src.models import metrics


def test_wape_mape_rmse_known_values():
    a, f = np.array([100.0, 200.0, 300.0]), np.array([110.0, 180.0, 330.0])
    assert metrics.wape(a, f) == pytest.approx(60 / 600 * 100)
    assert metrics.mape(a, f) == pytest.approx((0.1 + 0.1 + 0.1) / 3 * 100)
    assert metrics.rmse(a, f) == pytest.approx(np.sqrt((100 + 400 + 900) / 3))
    assert metrics.bias(a, f) == pytest.approx(20 / 600 * 100)


def test_perfect_forecast_is_zero():
    a = np.array([5.0, 10.0])
    assert metrics.wape(a, a) == 0 and metrics.mape(a, a) == 0 and metrics.rmse(a, a) == 0


def test_score_ignores_closed_days():
    df = pd.DataFrame({"actual": [100.0, 0.0, 200.0], "forecast": [100.0, 0.0, 200.0]})
    assert metrics.score(df)["n_rows"] == 2


def test_interval_coverage():
    df = pd.DataFrame({"actual": [10.0, 20.0, 30.0, 40.0], "lower": [5, 25, 25, 35], "upper": [15, 30, 35, 45]})
    r = metrics.interval_report(df, 0.8)
    assert r["empirical_coverage"] == pytest.approx(0.75)
    assert r["mean_width"] == pytest.approx(8.75)
