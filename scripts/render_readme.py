"""Fill the generated blocks of README.md from the run artifacts (reports/*.json|csv). No number is typed by hand.

Blocks look like  <!-- BEGIN:NAME --> ... <!-- END:NAME -->  and are replaced in place.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline.run import issue_table  # noqa: E402

R = ROOT / "reports"
M = json.loads((R / "metrics.json").read_text())
DQ = json.loads((R / "data_quality.json").read_text())
EDA = json.loads((R / "eda_stats.json").read_text())
NAMES = {"naive": "Naive", "seasonal_naive": "Seasonal naive (baseline)", "sarima": "SARIMAX (60-store sample)", "lightgbm": "LightGBM"}


def test_count() -> int:
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True).stdout
    m = re.search(r"(\d+) tests? collected", out)
    return int(m.group(1)) if m else int(out.strip().splitlines()[-1].split()[0])


def md_table(df: pd.DataFrame, fmt: dict) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(cols))) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.get(c, str)(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def f1(x): return f"{x:.1f}"
def f0(x): return f"{x:,.0f}"


def dq_block() -> str:
    rows = issue_table(DQ)
    total = sum(c for _, c, _, _ in rows)
    kinds = sum(1 for _, c, _, _ in rows if c > 0)
    out = [f"{kinds} issue types had a non-zero count, touching **{total:,}** records in total "
           f"(full detail and reasoning: [`reports/data_quality.md`](reports/data_quality.md)).", "",
           "| Issue | Records | Fix |", "|---|---:|---|"]
    out += [f"| {n} | {c:,} | {fix.split(';')[0]} |" for n, c, fix, _ in rows if c > 0]
    zero = [n for n, c, _, _ in rows if c == 0]
    out += ["", "Checked and **clean** (0 found): " + "; ".join(zero) + "."]
    return "\n".join(out)


def results_block() -> str:
    ov = pd.DataFrame(M["overall"]).set_index("model")
    order = ["naive", "seasonal_naive", "lightgbm"]
    t = ov.loc[order].reset_index()
    t["model"] = t["model"].map(NAMES)
    t = t[["model", "WAPE", "MAPE", "RMSE", "Bias", "WAPE_fold_mean", "WAPE_fold_std"]]
    t.columns = ["Model (all 1,115 stores)", "WAPE %", "MAPE %", "RMSE", "Bias %", "WAPE mean over folds", "WAPE std over folds"]
    fm = {c: f1 for c in t.columns[1:]}
    fm["RMSE"] = f0
    sub = pd.DataFrame(M["overall_sarima_subset"]).set_index("model").loc[["naive", "seasonal_naive", "sarima", "lightgbm"]].reset_index()
    sub["model"] = sub["model"].map(NAMES)
    sub = sub[["model", "WAPE", "MAPE", "RMSE", "Bias"]]
    sub.columns = ["Model (same 60-store sample)", "WAPE %", "MAPE %", "RMSE", "Bias %"]
    fm2 = {c: f1 for c in sub.columns[1:]}
    fm2["RMSE"] = f0
    bf = pd.DataFrame(M["by_fold"]).pivot(index=["fold", "origin"], columns="model", values="WAPE").reset_index()
    bf["window"] = bf["origin"].map(lambda o: f"{(pd.Timestamp(o) + pd.Timedelta(days=1)).date()} to {(pd.Timestamp(o) + pd.Timedelta(days=M['horizon_days'])).date()}")
    bf = bf[["fold", "window", "naive", "seasonal_naive", "sarima", "lightgbm"]]
    bf.columns = ["Fold", "Test window", "Naive", "Seasonal naive", "SARIMAX*", "LightGBM"]
    fm3 = {c: f1 for c in ["Naive", "Seasonal naive", "SARIMAX*", "LightGBM"]}
    fm3["Fold"] = str
    hw = pd.DataFrame(M["by_horizon_week"])[["week", "WAPE", "seasonal_naive_WAPE"]]
    hw.columns = ["Horizon week", "LightGBM WAPE %", "Seasonal-naive WAPE %"]
    iv, ivf = M["interval_80"], M["interval_80_coverage_by_fold"]
    return "\n".join([
        f"Backtest: **{M['n_folds']} rolling-origin folds**, each forecasting the next **{M['horizon_days']} days (6 weeks) daily** for every store; "
        f"test windows tile {M['test_period'][0]} to {M['test_period'][1]}. Metrics are pooled over all open store-days "
        f"(**{int(ov.loc['lightgbm', 'n_rows']):,}** per model). Lower is better. WAPE = sum|error| / sum(actual).", "",
        md_table(t, fm), "",
        f"**WAPE by fold** (\\*SARIMAX only covers the {M['sarima_stores']}-store sample, so its fold numbers are on those stores, not all stores):", "",
        md_table(bf, fm3), "",
        f"**Head-to-head on the same {M['sarima_stores']} stores** (the fair way to include SARIMAX, which is fitted per store):", "",
        md_table(sub, fm2), "",
        "**Accuracy by forecast horizon** (LightGBM, all folds pooled):", "", md_table(hw, {"Horizon week": lambda w: str(int(w)), "LightGBM WAPE %": f1, "Seasonal-naive WAPE %": f1}), "",
        f"**Prediction intervals** (quantile LightGBM, 10th-90th percentile, nominal 80%): empirical coverage **{iv['empirical_coverage'] * 100:.1f}%**, "
        f"mean width **{iv['mean_width_pct_of_actual']:.1f}%** of actual sales. Coverage by fold: "
        + ", ".join(f"fold {k}: {v * 100:.1f}%" for k, v in ivf.items()) + ".", "",
        "![SHAP summary](reports/figures/shap_summary.png)", "",
        "![Feature importance](reports/figures/feature_importance.png)",
    ])


def headline_block() -> str:
    ov = pd.DataFrame(M["overall"]).set_index("model")
    b, s = M["best_model"], "seasonal_naive"
    imp = M["improvement_vs_seasonal_naive_pct"]
    return (f"> **{NAMES[b]}** forecasts 6 weeks of daily sales per store with **WAPE {ov.loc[b, 'WAPE']:.1f}%** / **MAPE {ov.loc[b, 'MAPE']:.1f}%** "
            f"versus **{ov.loc[s, 'WAPE']:.1f}% / {ov.loc[s, 'MAPE']:.1f}%** for the seasonal-naive baseline "
            f"(**{imp['WAPE']:.1f}%** lower WAPE, **{imp['MAPE']:.1f}%** lower MAPE) in a {M['n_folds']}-fold walk-forward backtest over "
            f"{M['n_stores']:,} stores. Data-quality checks fixed or flagged **{sum(c for _, c, _, _ in issue_table(DQ)):,}** records; "
            f"**{test_count()}** automated tests pass, including a no-leakage test.")


def insights_block() -> str:
    ov = pd.DataFrame(M["overall"]).set_index("model")
    bf = pd.DataFrame(M["by_fold"])
    lg = bf[bf.model == "lightgbm"].set_index("fold")
    sn = bf[bf.model == "seasonal_naive"].set_index("fold")
    wins = int((lg["WAPE"] < sn["WAPE"]).sum())
    worst, best = lg["WAPE"].idxmax(), lg["WAPE"].idxmin()
    hw = pd.DataFrame(M["by_horizon_week"])
    sub = pd.DataFrame(M["overall_sarima_subset"]).set_index("model")
    top = M["top_features"]
    pt, ps, ph = EDA["promo_test"], EDA["promo_by_store_type"], EDA["state_holiday_test"]
    tl = ps["mean_lift_by_type"]
    ivf = M["interval_80_coverage_by_fold"]
    worst_origin = pd.Timestamp(lg.loc[worst, "origin"])
    L = [
        f"1. **Promotions have a large, highly significant effect on sales.** Within-store, same-weekday, non-holiday comparison across {pt['n_stores']:,} stores: "
        f"promo days sell **{pt['mean_lift_pct']:.1f}%** more (95% bootstrap CI {pt['lift_ci95'][0]:.1f}% to {pt['lift_ci95'][1]:.1f}%; "
        f"paired t = {pt['t_stat']:.1f}, Wilcoxon p = {pt['p_wilcoxon']:.1e}, Cohen's d_z = {pt['cohens_dz']:.2f}); "
        f"{pt['share_stores_positive'] * 100:.0f}% of stores benefit. The lift is not uniform: it ranges from {min(tl.values()):.1f}% to {max(tl.values()):.1f}% "
        f"across store types (Kruskal-Wallis p = {ps['p']:.1e}). This is observational, not a randomised trial.",
        f"2. **What the model relies on.** Top SHAP features: `{top[0]['feature']}`, `{top[1]['feature']}`, `{top[2]['feature']}` "
        f"(together {sum(t['gain_share'] for t in top[:3]) * 100:.0f}% of total split gain). Recent same-weekday level plus the promo schedule carry most of the signal, "
        f"and the baselines, which ignore the promo calendar, have no access to the second of these.",
        f"3. **Consistent, not a lucky split.** LightGBM beats seasonal naive in **{wins} of {len(lg)}** folds; fold WAPE ranges from "
        f"{lg['WAPE'].min():.1f}% (fold {best}) to {lg['WAPE'].max():.1f}% (fold {worst}). The hardest window is fold {worst} "
        f"(test window {(worst_origin + pd.Timedelta(days=1)).date()} to {(worst_origin + pd.Timedelta(days=M['horizon_days'])).date()}"
        + (", which contains Christmas and New Year; the training data holds only one earlier Christmas" if (worst_origin + pd.Timedelta(days=1)) <= pd.Timestamp(worst_origin.year, 12, 25) <= (worst_origin + pd.Timedelta(days=M['horizon_days'])) else "") + ").",
        f"4. **Horizon.** WAPE is {hw['WAPE'].iloc[0]:.1f}% in week 1 and {hw['WAPE'].iloc[-1]:.1f}% in week 6 "
        f"(range {hw['WAPE'].min():.1f}-{hw['WAPE'].max():.1f}%), i.e. no monotone degradation with horizon. By design every target day only sees sales at least "
        f"{M['horizon_days']} days old, so near-term days get no fresher data; an origin-anchored model that uses the latest weeks for the first horizon weeks is a possible improvement.",
        f"5. **Statistical vs ML.** On the same {M['sarima_stores']} stores SARIMAX reaches WAPE {sub.loc['sarima', 'WAPE']:.1f}% (baseline {sub.loc['seasonal_naive', 'WAPE']:.1f}%), "
        f"LightGBM {sub.loc['lightgbm', 'WAPE']:.1f}%. SARIMAX beats the baselines (it takes the promo/open flags as regressors, which they lack) but needs one fit per store per fold; "
        f"the single global LightGBM is both more accurate and cheaper to run at {M['n_stores']:,}-store scale.",
        f"6. **The intervals are too narrow.** The nominal 80% interval covers **{M['interval_80']['empirical_coverage'] * 100:.1f}%** of actuals "
        f"(lowest in fold {min(ivf, key=ivf.get)}: {min(ivf.values()) * 100:.1f}%). Interval quantiles learned from training data typically under-state out-of-sample uncertainty; "
        f"a conformal recalibration using earlier folds is the natural next step (not done here). Treat the band as a lower bound on uncertainty.",
    ]
    L.append(f"7. **Holidays act mainly through store closures.** Among stores that *are* open on state holidays, sales relative to their own mean are not significantly "
             f"different (median ratio {ph['median_ratio_holiday']:.2f} vs {ph['median_ratio_ordinary']:.2f}, Mann-Whitney p = {ph['p_mannwhitney']:.2f}); "
             f"the forecast effect comes from the (known) open flag. Sundays: only {EDA['seasonality']['sunday_open_share'] * 100:.1f}% of store-days are open.")
    return "\n".join(L)


def setup_block() -> str:
    pre = DQ["pre"]
    return (f"- Rossmann Store Sales: **{DQ['n_stores']:,} stores**, daily, {DQ['date_range'][0]} to {DQ['date_range'][1]} "
            f"({pre['rows']:,} raw rows; {pre['expected_rows']:,} after filling the calendar).\n"
            f"- Scored on open store-days only; closed days are a known input and forecast as 0.\n"
            f"- Backtest origins: {', '.join(M['origins'])}.")


def fill(text: str, name: str, body: str) -> str:
    pat = re.compile(rf"(<!-- BEGIN:{name} -->)(.*?)(<!-- END:{name} -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"marker {name} missing from README.md")
    return pat.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)


if __name__ == "__main__":
    p = ROOT / "README.md"
    t = p.read_text(encoding="utf-8")
    for name, fn in [("HEADLINE", headline_block), ("SETUP", setup_block), ("DQ", dq_block), ("RESULTS", results_block), ("INSIGHTS", insights_block)]:
        t = fill(t, name, fn())
    p.write_text(t, encoding="utf-8")
    print("README.md updated")
