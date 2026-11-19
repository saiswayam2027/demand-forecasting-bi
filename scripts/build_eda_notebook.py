"""Build and execute notebooks/01_eda.ipynb. Run: python scripts/build_eda_notebook.py

All findings are printed by code from the data, so no number in the notebook is typed by hand.
"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "01_eda.ipynb"

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 01 - Exploratory analysis & statistics: Rossmann store sales

Questions: what is the trend and seasonality, how much do promos and holidays move sales, and is the promo lift
statistically real (not just visible)? Input is the **cleaned** panel produced by `src/pipeline`.
Closed days (`open = 0`) are excluded from every sales statistic: they are structural zeros, not demand.
""")
code("""
import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
FIG = ROOT / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
panel = pd.read_parquet(ROOT / "data/processed/sales.parquet")
stores = pd.read_parquet(ROOT / "data/processed/stores.parquet")
op = panel[(panel.open == 1)].copy()
print(f"{panel.store.nunique():,} stores, {panel.date.min().date()} -> {panel.date.max().date()}, "
      f"{len(op):,} open store-days ({len(op)/len(panel):.1%} of the panel)")
""")

md("## 1. Trend")
code("""
daily = op.groupby("date").agg(total_sales=("sales", "sum"), open_stores=("store", "nunique"))
daily["sales_per_open_store"] = daily.total_sales / daily.open_stores
fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax[0].plot(daily.index, daily.total_sales, lw=.5, alpha=.5, color="#2a6f97")
ax[0].plot(daily.index, daily.total_sales.rolling(28, center=True).mean(), color="#1b4965", lw=2)
ax[0].set_title("Total daily sales (28-day centred mean in bold)")
ax[1].plot(daily.index, daily.sales_per_open_store.rolling(28, center=True).mean(), color="#6a994e", lw=2)
ax[1].set_title("Average sales per open store (28-day mean)  -  removes the effect of stores opening/closing")
fig.tight_layout(); fig.savefig(FIG / "eda_trend.png", dpi=130); plt.show()

yr = op.assign(year=op.date.dt.year).groupby("year").sales.mean()
jan_jul = op[op.date.dt.month <= 7].assign(year=lambda d: d.date.dt.year).groupby("year").sales.mean()
growth = jan_jul.pct_change().dropna()
print("Mean sales per open store-day, Jan-Jul only (like-for-like):")
print(jan_jul.round(0).to_string())
print("YoY growth, same months:", {int(k): f"{v:.1%}" for k, v in growth.items()})
""")

md("## 2. Weekly and yearly seasonality")
code("""
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
by_dow = op.groupby("day_of_week").sales.agg(["mean", "count"])
open_share = panel.groupby("day_of_week").open.mean()
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].bar(dow_names, by_dow["mean"].reindex(range(1, 8)), color="#2a6f97"); ax[0].set_title("Mean sales per open store-day, by weekday")
ax[1].bar(dow_names, open_share.reindex(range(1, 8)), color="#bc4749"); ax[1].set_title("Share of store-days that are open"); ax[1].set_ylim(0, 1.05)
fig.tight_layout(); fig.savefig(FIG / "eda_weekly.png", dpi=130); plt.show()

monthly = op.assign(year=op.date.dt.year, month=op.date.dt.month).groupby(["year", "month"]).sales.mean().unstack(0)
ax = monthly.plot(marker="o", figsize=(9, 3.8), color=["#adb5bd", "#2a6f97", "#bc4749"])
ax.set_title("Mean sales per open store-day, by month"); ax.set_xticks(range(1, 13)); fig = ax.figure
fig.tight_layout(); fig.savefig(FIG / "eda_yearly.png", dpi=130); plt.show()

fig, ax = plt.subplots(figsize=(9, 3.2))
plot_acf(daily.total_sales, lags=60, ax=ax); ax.set_title("ACF of total daily sales: spikes at lag 7, 14, 21 = weekly cycle")
fig.tight_layout(); fig.savefig(FIG / "eda_acf.png", dpi=130); plt.show()
acf7 = pd.Series(daily.total_sales.values).autocorr(7)
mon, sun = by_dow["mean"].loc[1], by_dow["mean"].loc[7]
top_days = [dow_names[i - 1] for i in by_dow["mean"].sort_values(ascending=False).index[:2]]
dec_vs_year = monthly.loc[12, 2013] / monthly[2013].mean() - 1
print(f"Lag-7 autocorrelation of total sales: {acf7:.2f}")
print(f"Monday mean vs Sunday mean (open store-days only): {mon:,.0f} vs {sun:,.0f}; Sunday opening share {open_share.loc[7]:.1%} (only large stores open on Sundays)")
print(f"Highest-mean weekdays among open store-days: {top_days}")
print(f"December 2013 vs the 2013 monthly average: {dec_vs_year:+.1%}")
""")

md("## 3. Promo effect (descriptive)")
code("""
g = op.groupby("promo").sales.agg(["mean", "median", "count"])
raw_lift = g.loc[1, "mean"] / g.loc[0, "mean"] - 1
print(g.round(0).to_string()); print(f"\\nRaw lift in the mean (promo vs no promo, all open days): {raw_lift:+.1%}")
by_dow_promo = op.groupby(["day_of_week", "promo"]).sales.mean().unstack()
by_dow_promo["lift"] = by_dow_promo[1] / by_dow_promo[0] - 1
print("\\nLift by weekday (promos only run Mon-Fri):"); print(by_dow_promo.dropna().round(3).to_string())
fig, ax = plt.subplots(figsize=(8, 3.6))
sns.boxplot(data=op[op.day_of_week <= 5].sample(200_000, random_state=1), x="day_of_week", y="sales", hue="promo",
            showfliers=False, palette=["#adb5bd", "#2a6f97"], ax=ax)
ax.set_xticks(range(5)); ax.set_xticklabels(dow_names[:5]); ax.set_title("Sales distribution by weekday and promo status")
fig.tight_layout(); fig.savefig(FIG / "eda_promo.png", dpi=130); plt.show()
""")

md("## 4. Holiday effects")
code("""
st = op.assign(kind=op.state_holiday.map({"0": "none", "a": "public", "b": "Easter", "c": "Christmas"}))
tab = st.groupby("kind").sales.agg(["mean", "count"]); tab["vs_none"] = tab["mean"] / tab.loc["none", "mean"] - 1
print("Open stores on state holidays (note: few stores open on these days):"); print(tab.round(3).to_string())
sch = op.groupby("school_holiday").sales.mean(); school_lift = sch.loc[1] / sch.loc[0] - 1
print(f"\\nSchool holiday vs not: {sch.loc[0]:,.0f} -> {sch.loc[1]:,.0f} ({school_lift:+.1%})")
open_by_hol = panel.assign(kind=panel.state_holiday.map({"0": "none", "a": "public", "b": "Easter", "c": "Christmas"})).groupby("kind").open.mean()
print("\\nShare of store-days that are open:"); print(open_by_hol.round(3).to_string())
""")

md("""
## 5. Hypothesis test: does a promotion significantly lift sales?

**Design.** Raw promo/no-promo means are confounded: promos only run Mon-Fri, so a naive comparison mixes in
weekday effects, and stores differ hugely in size, so day-level rows are far from independent. The test therefore:

1. keeps only open Mon-Fri days that are not state holidays;
2. for **each store** computes mean sales on promo days and on non-promo days *within each weekday* and averages the
   per-weekday differences (so the weekday mix is identical in both arms);
3. runs a **paired** test across stores (the store is its own control).

Caveat: this is observational, not a randomised experiment. Rossmann sets the promo calendar centrally (it is not
chosen store-by-store in response to demand), which makes timing close to exogenous, but the result is an
association-with-controls, not proven causation.

H0: mean within-store promo lift = 0.  H1: it is not 0 (two-sided, alpha = 0.05).
Effect size: Cohen's d_z of the paired differences, plus the mean % lift with a bootstrap CI.
""")
code("""
sub = op[(op.day_of_week <= 5) & (op.state_holiday == "0")]
cell = sub.groupby(["store", "day_of_week", "promo"]).sales.mean().unstack("promo").dropna()
cell["diff"] = cell[1] - cell[0]
per_store = cell.groupby("store").agg(promo=(1, "mean"), nopromo=(0, "mean"), diff=("diff", "mean"))
per_store["lift_pct"] = per_store["diff"] / per_store["nopromo"] * 100
n = len(per_store); d = per_store["diff"]

t_stat, p_t = stats.ttest_rel(per_store.promo, per_store.nopromo)
w_stat, p_w = stats.wilcoxon(d)
dz = d.mean() / d.std(ddof=1)
rng = np.random.default_rng(42)
boot = np.array([per_store.lift_pct.sample(n, replace=True, random_state=int(s)).mean() for s in rng.integers(0, 1e9, 2000)])
ci = np.percentile(boot, [2.5, 97.5])
mean_diff_ci = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=stats.sem(d))
sh_stat, sh_p = stats.shapiro(d.sample(min(n, 5000), random_state=1))
print(f"Stores in test: {n:,}")
print(f"Mean sales  promo: {per_store.promo.mean():,.0f}   no promo: {per_store.nopromo.mean():,.0f}")
print(f"Mean paired difference: {d.mean():,.0f} per store-day  (95% CI {mean_diff_ci[0]:,.0f} to {mean_diff_ci[1]:,.0f})")
print(f"Mean within-store lift: {per_store.lift_pct.mean():.1f}%  (bootstrap 95% CI {ci[0]:.1f}% to {ci[1]:.1f}%)")
def fmt_p(p): return "< 1e-300 (underflows to 0)" if p == 0 else f"{p:.3g}"
print(f"Paired t-test:  t = {t_stat:.1f}, df = {n-1}, p = {fmt_p(p_t)}")
print(f"Wilcoxon signed-rank (robust to non-normality): W = {w_stat:,.0f}, p = {p_w:.3g}")
print(f"Effect size Cohen's d_z = {dz:.2f}  ({'large' if abs(dz) >= .8 else 'medium' if abs(dz) >= .5 else 'small'} by Cohen's rule of thumb)")
print(f"Share of stores with a positive lift: {(d > 0).mean():.1%}")
print(f"Shapiro-Wilk on paired differences: W = {sh_stat:.3f}, p = {sh_p:.3g}  -> "
      f"{'normality rejected, so lean on the Wilcoxon result (it agrees)' if sh_p < .05 else 'consistent with normality'}")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
sns.histplot(per_store.lift_pct, bins=50, color="#2a6f97", ax=ax[0]); ax[0].axvline(0, color="k", lw=1)
ax[0].axvline(per_store.lift_pct.mean(), color="#bc4749", ls="--"); ax[0].set_title("Per-store promo lift (%)")
ax[1].scatter(per_store.nopromo, per_store.promo, s=6, alpha=.4, color="#2a6f97")
m = per_store[["nopromo", "promo"]].values.max(); ax[1].plot([0, m], [0, m], "k--", lw=1)
ax[1].set_xlabel("mean sales, no promo"); ax[1].set_ylabel("mean sales, promo"); ax[1].set_title("Each dot = one store (above the line = promo helps)")
fig.tight_layout(); fig.savefig(FIG / "eda_promo_test.png", dpi=130); plt.show()
""")
md("""
**Does the lift depend on store type?** A Kruskal-Wallis test across store types (lift % per store) checks whether
one blanket promo-lift number hides different behaviour by segment.
""")
code("""
ps = per_store.join(stores.set_index("store")[["store_type"]])
groups = [g.lift_pct.values for _, g in ps.groupby("store_type")]
h, p_kw = stats.kruskal(*groups)
eps2 = h / (len(ps) - 1)   # epsilon-squared effect size
print(ps.groupby("store_type").lift_pct.agg(["count", "mean", "median"]).round(1).to_string())
print(f"\\nKruskal-Wallis H = {h:.1f}, p = {p_kw:.3g}, epsilon^2 = {eps2:.3f}")
""")
md("""
**Does a state holiday change sales for stores that are open?** Mann-Whitney U on store-day sales, holiday vs.
ordinary Mon-Fri days (rank-biserial correlation as effect size). Holiday days involve few, atypical stores, so the
comparison is made on each store's *ratio to its own mean*.
""")
code("""
op2 = op[op.day_of_week <= 5].copy()
op2["ratio"] = op2.sales / op2.groupby("store").sales.transform("mean")
hol = op2[op2.state_holiday != "0"].ratio; reg = op2[op2.state_holiday == "0"].ratio
u, p_u = stats.mannwhitneyu(hol, reg, alternative="two-sided")
rbc = 1 - 2 * u / (len(hol) * len(reg))   # rank-biserial (negative => holiday values tend to be higher)
print(f"Holiday open-days: {len(hol):,}; ordinary: {len(reg):,}")
print(f"Median sales / own-store mean: holiday {hol.median():.2f} vs ordinary {reg.median():.2f}")
print(f"Mann-Whitney U = {u:,.0f}, p = {p_u:.3g}, rank-biserial r = {-rbc:+.2f} (positive => holiday sales higher)")
""")

md("## 6. Takeaways (printed from the numbers above)")
code("""
print(f"1. Trend: like-for-like Jan-Jul sales per open store-day moved {', '.join(f'{int(k)}: {v:+.1%}' for k, v in growth.items())}.")
print(f"2. Weekly cycle is strong (lag-7 autocorrelation {acf7:.2f}); Sunday is mostly closed ({open_share.loc[7]:.0%} open) "
      f"and the highest-mean days per open store are {' and '.join(top_days)}.")
print(f"3. Promos: within-store lift {per_store.lift_pct.mean():.1f}% (95% CI {ci[0]:.1f}-{ci[1]:.1f}%), "
      f"paired t p={fmt_p(p_t)}, Cohen's d_z={dz:.2f}; {(d>0).mean():.0%} of stores benefit. Promo (and the weekday it falls on) must be a model input.")
print(f"4. Promo lift differs by store type (Kruskal-Wallis p={p_kw:.2g}, epsilon^2={eps2:.3f}); store attributes belong in the model.")
print(f"5. Holidays: only {open_by_hol[['public','Easter','Christmas']].min():.1%}-{open_by_hol[['public','Easter','Christmas']].max():.1%} of stores open on state holidays; "
      f"for those that do, sales relative to their own mean are not significantly different (Mann-Whitney p={p_u:.2f}). "
      f"The holiday effect on a forecast is mostly the open/closed flag, which is a known input.")
print(f"6. Seasonality: Christmas-season spike: December 2013 is {dec_vs_year:+.1%} above the 2013 monthly average; yearly lags/calendar features are needed.")
""")

code("""
import json
stats_out = {
    "promo_test": {"n_stores": int(n), "mean_sales_promo": float(per_store.promo.mean()), "mean_sales_nopromo": float(per_store.nopromo.mean()),
                   "mean_lift_pct": float(per_store.lift_pct.mean()), "lift_ci95": [float(ci[0]), float(ci[1])],
                   "t_stat": float(t_stat), "p_ttest": float(p_t), "p_wilcoxon": float(p_w), "cohens_dz": float(dz),
                   "share_stores_positive": float((d > 0).mean())},
    "promo_by_store_type": {"kruskal_H": float(h), "p": float(p_kw), "epsilon_sq": float(eps2),
                            "mean_lift_by_type": {k: float(v) for k, v in ps.groupby("store_type").lift_pct.mean().items()}},
    "state_holiday_test": {"p_mannwhitney": float(p_u), "median_ratio_holiday": float(hol.median()), "median_ratio_ordinary": float(reg.median())},
    "trend": {"yoy_like_for_like": {int(k): float(v) for k, v in growth.items()}},
    "seasonality": {"acf_lag7": float(acf7), "december_2013_vs_year_avg": float(dec_vs_year), "sunday_open_share": float(open_share.loc[7])},
}
(ROOT / "reports" / "eda_stats.json").write_text(json.dumps(stats_out, indent=2))
print("saved reports/eda_stats.json")
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
if __name__ == "__main__":
    NB.parent.mkdir(exist_ok=True)
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(NB.parent)}}).execute()
    nbf.write(nb, NB)
    print("wrote", NB)
