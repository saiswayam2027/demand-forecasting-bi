"""Download the Rossmann Store Sales data into data/raw/.

Order of attempts:
  1. files already present in data/raw/ (extracted or the competition zip)
  2. `kaggle competitions download` (needs ~/.kaggle/access_token or kaggle.json
     and accepted competition rules at kaggle.com/c/rossmann-store-sales)
"""
from __future__ import annotations

import subprocess
import zipfile

from src.config import COMPETITION, RAW_DIR

NEEDED = ["train.csv", "store.csv"]


def _have_all() -> bool:
    return all((RAW_DIR / f).exists() for f in NEEDED)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if _have_all():
        print("Raw data already present - nothing to download.")
        return
    zips = list(RAW_DIR.glob("*.zip"))
    if not zips:
        print(f"Downloading '{COMPETITION}' with the kaggle CLI ...")
        res = subprocess.run(
            ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(RAW_DIR)],
        )
        if res.returncode != 0:
            raise SystemExit(
                "Kaggle download failed. Configure credentials (https://www.kaggle.com/settings -> API token), "
                "accept the competition rules, or manually place rossmann-store-sales.zip in data/raw/."
            )
        zips = list(RAW_DIR.glob("*.zip"))
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW_DIR)
    if not _have_all():
        raise SystemExit(f"Expected files {NEEDED} not found in {RAW_DIR}.")
    print("Raw data ready in", RAW_DIR)


if __name__ == "__main__":
    main()
