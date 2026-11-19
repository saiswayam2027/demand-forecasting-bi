"""Ingestion: read raw CSVs exactly as delivered (no cleaning)."""
from __future__ import annotations

import pandas as pd

from src.config import RAW_DIR

TRAIN_SCHEMA = {
    "Store": "int", "DayOfWeek": "int", "Date": "date", "Sales": "int", "Customers": "int",
    "Open": "int", "Promo": "int", "StateHoliday": "str", "SchoolHoliday": "int",
}
STORE_SCHEMA = {
    "Store": "int", "StoreType": "str", "Assortment": "str", "CompetitionDistance": "float",
    "CompetitionOpenSinceMonth": "float", "CompetitionOpenSinceYear": "float", "Promo2": "int",
    "Promo2SinceWeek": "float", "Promo2SinceYear": "float", "PromoInterval": "str",
}


def read_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Everything is read as string first so schema/type problems can be *counted*, not hidden."""
    train = pd.read_csv(RAW_DIR / "train.csv", dtype=str, keep_default_na=False)
    store = pd.read_csv(RAW_DIR / "store.csv", dtype=str, keep_default_na=False)
    return train, store
