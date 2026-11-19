"""Feature-engineering tests, including the no-leakage guarantee."""
import numpy as np
import pandas as pd
import pytest

from src.models import baselines
from src.models.backtest import make_origins
from src.models.features import (YEARLY_LAG, build_features, feature_columns, lag_days, sales_feature_names)

H = 42


@pytest.fixture(scope="module")
def synth():
    """3 stores x 500 days with weekly seasonality, a promo effect, Sundays closed."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2013-01-01", periods=500)
    rows = []
    for s in (1, 2, 3):
        for d in dates:
            is_open = int(d.day_of_week != 6)
            promo = int(d.day_of_week in (0, 1, 2) and (d.isocalendar().week % 2 == 0))
            sales = int(is_open * (1000 * s + 100 * d.day_of_week + 300 * promo + rng.integers(0, 50)))
            rows.append(dict(store=s, date=d, day_of_week=d.day_of_week + 1, sales=sales, customers=1, open=is_open,
                             promo=promo, state_holiday="0", school_holiday=0, is_imputed=0, is_outlier=0,
                             state_holiday_flag=0))
    panel = pd.DataFrame(rows)
    stores = pd.DataFrame({
        "store": [1, 2, 3], "store_type": ["a", "b", "c"], "assortment": ["a", "b", "c"],
        "competition_distance": [100.0, 200.0, 300.0], "promo2": [0, 1, 1], "promo2_since_week": [0, 5, 10],
        "promo2_since_year": [0, 2013, 2013], "promo_interval": ["None", "Jan,Apr,Jul,Oct", "Mar,Jun,Sept,Dec"],
        "competition_open_since": pd.to_datetime(["2012-01-01", None, "2013-03-01"]),
    })
    return panel, stores


def test_all_sales_lags_at_least_horizon():
    assert min(lag_days(H)) >= H
    assert YEARLY_LAG >= H


def test_no_leakage_future_sales_do_not_change_features_in_forecast_window(synth):
    """For origin T, features of every date in (T, T+H] must be identical no matter what sales happen after T."""
    panel, stores = synth
    origin = pd.Timestamp("2014-03-01")
    base = build_features(panel, stores, H)

    tampered = panel.copy()
    future = tampered["date"] > origin
    tampered.loc[future, "sales"] = tampered.loc[future, "sales"] * 7 + 12345   # wildly different future
    alt = build_features(tampered, stores, H)

    win = (base["date"] > origin) & (base["date"] <= origin + pd.Timedelta(days=H))
    cols = feature_columns(H)
    pd.testing.assert_frame_equal(base.loc[win, cols].reset_index(drop=True), alt.loc[win, cols].reset_index(drop=True))


def test_leakage_test_is_sensitive(synth):
    """Sanity check on the test above: features *beyond* T+H do depend on sales after T, so tampering would be detected."""
    panel, stores = synth
    origin = pd.Timestamp("2014-03-01")
    base = build_features(panel, stores, H)
    tampered = panel.copy()
    future = tampered["date"] > origin
    tampered.loc[future, "sales"] = tampered.loc[future, "sales"] + 5000
    alt = build_features(tampered, stores, H)
    beyond = base["date"] > origin + pd.Timedelta(days=H + 7)
    sales_cols = sales_feature_names(H)
    assert not base.loc[beyond, sales_cols].equals(alt.loc[beyond, sales_cols])


def test_lag_alignment(synth):
    panel, stores = synth
    f = build_features(panel, stores, H).set_index(["store", "date"])
    d = pd.Timestamp("2014-02-10")
    for lag in lag_days(H):
        got = f.loc[(2, d), f"lag_{lag}"]
        src = panel[(panel.store == 2) & (panel.date == d - pd.Timedelta(days=lag))].iloc[0]
        expect = src["sales"] if src["open"] == 1 else np.nan
        assert (np.isnan(got) and np.isnan(expect)) or got == expect


def test_rolling_mean_uses_only_shifted_window(synth):
    panel, stores = synth
    f = build_features(panel, stores, H).set_index(["store", "date"])
    d = pd.Timestamp("2014-02-10")
    window = panel[(panel.store == 1) & (panel.date <= d - pd.Timedelta(days=H)) &
                   (panel.date > d - pd.Timedelta(days=H + 28)) & (panel.open == 1)]["sales"]
    assert f.loc[(1, d), "roll_mean_28"] == pytest.approx(window.mean())


def test_requires_complete_grid(synth):
    panel, stores = synth
    with pytest.raises(ValueError):
        build_features(panel.iloc[1:], stores, H)


def test_promo2_active_respects_interval_and_start(synth):
    panel, stores = synth
    f = build_features(panel, stores, H)
    assert f.loc[f.store == 1, "promo2_active"].sum() == 0
    s2 = f[f.store == 2]
    assert s2.loc[s2.month.isin([1, 4, 7, 10]) & (s2.date >= "2013-02-01"), "promo2_active"].eq(1).all()
    assert s2.loc[s2.month.isin([2, 3, 5]), "promo2_active"].eq(0).all()
    assert f.loc[(f.store == 3) & (f.month == 9), "promo2_active"].max() == 1  # "Sept" spelling handled


def test_baselines_ignore_future(synth):
    panel, _ = synth
    origin = pd.Timestamp("2014-03-01")
    tampered = panel.copy()
    tampered.loc[tampered["date"] > origin, "sales"] = 999999
    for fn in (baselines.naive_forecast, baselines.seasonal_naive_forecast):
        a = fn(panel, origin).sort_values(["store", "date"])["forecast"].to_numpy()
        b = fn(tampered, origin).sort_values(["store", "date"])["forecast"].to_numpy()
        np.testing.assert_array_equal(a, b)


def test_seasonal_naive_repeats_last_week(synth):
    panel, _ = synth
    origin = pd.Timestamp("2014-03-02")  # a Sunday
    fc = baselines.seasonal_naive_forecast(panel, origin).set_index(["store", "date"])["forecast"]
    d = pd.Timestamp("2014-03-12")  # horizon 10 -> reference = d - 14 = 2014-02-26
    ref = panel[(panel.store == 1) & (panel.date == d - pd.Timedelta(days=14))]["sales"].item()
    assert fc.loc[(1, d)] == ref


def test_origins_are_chronological_non_overlapping_and_end_at_last_date():
    last = pd.Timestamp("2015-07-31")
    o = make_origins(last, n_folds=6, horizon=H)
    assert len(o) == 6 and o == sorted(o)
    assert o[-1] + pd.Timedelta(days=H) == last
    assert all((b - a).days == H for a, b in zip(o, o[1:]))
