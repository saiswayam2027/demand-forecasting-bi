"""Reproduce everything end to end:  python run_all.py   (add --skip-download / --skip-notebook / --skip-tests)

download -> pipeline (+ data-quality report) -> tests -> backtest (+ SHAP) -> Power BI export -> EDA notebook -> README tables
"""
import argparse
import subprocess
import sys
import time

STEPS = [
    ("download", [sys.executable, "-m", "src.pipeline.download"], "skip_download"),
    ("pipeline + data-quality report", [sys.executable, "-m", "src.pipeline.run"], None),
    ("tests", [sys.executable, "-m", "pytest", "-q"], "skip_tests"),
    ("backtest (baselines, SARIMA, LightGBM, intervals, SHAP)", [sys.executable, "-m", "src.models.backtest"], None),
    ("Power BI export", [sys.executable, "-m", "src.export.powerbi"], None),
    ("EDA notebook", [sys.executable, "scripts/build_eda_notebook.py"], "skip_notebook"),
    ("README results", [sys.executable, "scripts/render_readme.py"], None),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    for f in ("skip_download", "skip_tests", "skip_notebook"):
        ap.add_argument("--" + f.replace("_", "-"), action="store_true")
    args = ap.parse_args()
    for name, cmd, flag in STEPS:
        if flag and getattr(args, flag):
            print(f"== skipping {name}")
            continue
        print(f"\n== {name}", flush=True)
        t = time.time()
        subprocess.run(cmd, check=True)
        print(f"   done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
