"""Central configuration: paths and modelling constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
POWERBI_DIR = ROOT / "powerbi"

COMPETITION = "rossmann-store-sales"

# Forecasting setup
HORIZON = 42          # 6 weeks, daily
N_FOLDS = 6           # rolling-origin folds, stepped back by HORIZON days
SEED = 42
SARIMA_N_STORES = 60  # SARIMA is fitted per store, so it runs on a seeded sample
