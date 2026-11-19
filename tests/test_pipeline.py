"""Tests for pipeline validation + cleaning, using small hand-built frames with known injected defects."""
import numpy as np
import pandas as pd
import pytest

from src.pipeline import clean, validate
from src.pipeline.ingest import TRAIN_SCHEMA
from src.pipeline.run import MUST_BE_ZERO, assert_clean, to_validation_frame


def make_raw(rows):
    cols = ["Store", "DayOfWeek", "Date", "Sales", "Customers", "Open", "Promo", "StateHoliday", "SchoolHoliday"]
    return pd.DataFrame(rows, columns=cols).astype(str)


def base_rows():
    # store 1: 5 consecutive days (2015-01-05 is a Monday), store 2: same 5 days
    rows = []
    for store in (1, 2):
        for i, d in enumerate(pd.date_range("2015-01-05", periods=5)):
            rows.append([store, d.day_of_week + 1, d.strftime("%Y-%m-%d"), 100 + i, 10, 1, 0, "0", 0])
    return rows


def typed(rows, store_rows=None):
    store_raw = pd.DataFrame({
        "Store": ["1", "2"], "StoreType": ["a", "b"], "Assortment": ["a", "a"], "CompetitionDistance": ["100", ""],
        "CompetitionOpenSinceMonth": ["3", ""], "CompetitionOpenSinceYear": ["2010", ""], "Promo2": ["0", "1"],
        "Promo2SinceWeek": ["", "5"], "Promo2SinceYear": ["", "2011"], "PromoInterval": ["", "Jan,Apr"],
    })
    return clean.coerce_types(make_raw(rows), store_raw)


def test_clean_input_has_no_issues():
    t, _ = typed(base_rows())
    v = validate.validate_train(t)
    assert all(v[k] == 0 for k in MUST_BE_ZERO)


def test_duplicates_detected_and_removed():
    rows = base_rows() + [base_rows()[0]]
    t, _ = typed(rows)
    assert validate.count_duplicates(t) == 1
    panel, log = clean.clean_train(t)
    assert log["duplicates_removed"] == 1
    assert not panel.duplicated(["store", "date"]).any()


def test_negative_sales_detected_and_fixed():
    rows = base_rows()
    rows[2][3] = -50
    t, _ = typed(rows)
    assert validate.count_negative_sales(t) == 1
    panel, log = clean.clean_train(t)
    assert log["negative_sales_set_to_zero"] == 1
    assert (panel["sales"] >= 0).all()


def test_missing_dates_detected_and_filled_as_closed():
    rows = [r for r in base_rows() if not (r[0] == 2 and r[2] in ("2015-01-07", "2015-01-08"))]
    t, _ = typed(rows)
    md = validate.missing_dates(t)
    assert md["missing_store_days"] == 2 and md["stores_with_gaps"] == 1
    panel, log = clean.clean_train(t)
    assert len(panel) == 10 and log["missing_store_days_filled"] == 2
    filled = panel[panel["is_imputed"] == 1]
    assert (filled["open"] == 0).all() and (filled["sales"] == 0).all()


def test_closed_day_inconsistencies():
    rows = base_rows()
    rows[0][5] = 0          # closed but sales 100 > 0
    rows[1][3] = 0          # open but sales 0
    t, _ = typed(rows)
    stats = validate.closed_day_stats(t)
    assert stats["closed_but_sales_positive"] == 1 and stats["open_but_zero_sales"] == 1
    panel, _ = clean.clean_train(t)
    post = validate.closed_day_stats(to_validation_frame(panel))
    assert post["closed_but_sales_positive"] == 0 and post["open_but_zero_sales"] == 0


def test_outlier_flagging_catches_spike_only():
    rows = []
    for i, d in enumerate(pd.date_range("2015-01-01", periods=60)):
        rows.append([1, d.day_of_week + 1, d.strftime("%Y-%m-%d"), 100 + (i % 5), 10, 1, 0, "0", 0])
    rows[30][3] = 5000
    t, _ = typed(rows)
    flags = validate.flag_outliers(t)
    assert flags.sum() == 1 and flags.iloc[30]


def test_schema_checks_catch_bad_types_and_columns():
    raw = make_raw(base_rows())
    raw.loc[0, "Sales"] = "abc"
    raw.loc[1, "Date"] = "31/12/2015"
    res = validate.check_schema(raw, TRAIN_SCHEMA)
    assert res["schema_uncoercible_values"] == 2
    assert validate.check_schema(raw.drop(columns=["Promo"]), TRAIN_SCHEMA)["schema_missing_columns"] == 1


def test_invalid_state_holiday_and_dow_mismatch():
    rows = base_rows()
    rows[0][7] = "x"
    rows[1][1] = 7           # wrong DayOfWeek
    t, _ = typed(rows)
    cats = validate.count_invalid_categories(t)
    assert cats["invalid_state_holiday_codes"] == 1 and cats["dayofweek_date_mismatch"] == 1
    panel, log = clean.clean_train(t)
    assert log["invalid_state_holiday_recoded"] == 1 and log["dayofweek_recomputed"] == 1
    assert set(panel["state_holiday"]) <= {"0", "a", "b", "c"}


def test_assert_clean_raises_on_remaining_issue():
    t, _ = typed(base_rows())
    post = validate.validate_train(t)
    post["negative_sales"] = 3
    with pytest.raises(AssertionError):
        assert_clean(post)


def test_store_cleaning_imputes_and_flags():
    _, s = typed(base_rows())
    assert validate.validate_store(s)["missing_competition_distance"] == 1
    cleaned, log = clean.clean_store(s)
    assert log["competition_distance_median_imputed"] == 1
    assert cleaned["competition_distance"].notna().all()
    assert cleaned.loc[cleaned["store"] == 2, "competition_distance_missing"].item() == 1
    assert (cleaned["promo_interval"] != "").all()
