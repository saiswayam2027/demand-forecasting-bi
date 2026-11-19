# Power BI dashboard guide

Build the `.pbix` in **Power BI Desktop** (Windows) from the CSVs in this folder. About 60-90 minutes.
Save the finished file as `powerbi/Retail_Demand_Forecasting.pbix` and export screenshots to `docs/screenshots/`
(file names are listed at the end).

The CSVs are produced by `python -m src.export.powerbi` (also part of `python run_all.py`).

| File | Grain | Role |
|------|-------|------|
| `dim_date.csv` | 1 row per calendar day | Date dimension |
| `dim_store.csv` | 1 row per store | Store dimension |
| `dim_model.csv` | 1 row per model | Model dimension (slicer + best-model flag) |
| `fact_sales_actuals.csv` | 1 row per **open** store-day | Actual sales |
| `fact_forecasts.csv` | 1 row per model x fold x open store-day | Forecasts, 80% interval (LightGBM only), actuals, abs error |
| `model_metrics.csv` | 1 row per model x fold x scope (`fold = 0` is all folds pooled) | Pre-computed accuracy from the Python backtest |
| `feature_importance.csv` | 1 row per feature | LightGBM mean abs SHAP + gain share |

Things worth knowing before you start:

* Closed days are not in `fact_sales_actuals`; a closed store-day is simply a missing row. Averages are therefore
  "per **open** store-day".
* Forecasts exist only for the backtest windows (six consecutive 42-day windows ending on the last date in the data).
  Forecast visuals are blank before the first window. That is expected.
* The six folds do not overlap, so each (model, store, date) appears **once**. Summing `forecast` across *models* would
  double-count, so every forecast measure below is guarded to a single model per visual row.
* `sarima` covers only a 60-store sample (`dim_store[in_sarima_sample] = 1`). Compare it with other models on that
  sample only (`model_metrics[scope] = "sarima_sample"`).
* `lower_80` / `upper_80` are only filled for `lightgbm`.

---

## 1. Load the data

1. **Home > Get data > Text/CSV**, pick `dim_date.csv`, click **Transform Data** (not Load). Repeat for the other six
   files; they all land in Power Query.
2. In Power Query, set column types (the **Detect data type** guess is usually right; fix the ones below).

| Table | Column(s) | Type |
|-------|-----------|------|
| dim_date | `date` | Date |
| dim_date | `date_key`, `year`, `quarter`, `month`, `week_of_year`, `day_of_week`, `day_of_month`, `is_weekend`, `state_holiday_any_store` | Whole number |
| dim_store | `store_key`, `promo2`, `promo2_since_week`, `promo2_since_year`, `in_sarima_sample`, `competition_distance_missing`, `competition_open_since_missing` | Whole number |
| dim_store | `competition_distance` | Decimal number |
| dim_store | `competition_open_since` | Date |
| fact_sales_actuals | `store_key`, `date_key`, `sales`, `customers`, `promo`, `school_holiday`, `is_outlier` | Whole number |
| fact_sales_actuals | `state_holiday` | Text (`0`, `a`, `b`, `c`) |
| fact_forecasts | `fold`, `origin_date_key`, `store_key`, `date_key`, `horizon_day` | Whole number |
| fact_forecasts | `actual_sales`, `forecast`, `lower_80`, `upper_80`, `abs_error` | Decimal number |
| model_metrics | `fold`, `n_rows` | Whole number |
| model_metrics | `wape_pct`, `mape_pct`, `rmse`, `bias_pct`, `coverage_80`, `interval_width_pct_of_actual` | Decimal number |
| model_metrics | `origin_date` | Date |
| feature_importance | `mean_abs_shap`, `gain`, `gain_share` | Decimal number |

3. **Close & Apply**.
4. **File > Options and settings > Options > Current File > Data Load > uncheck "Auto date/time"** (you have your own
   date table).

## 2. Data model

Open **Model view** and create these relationships (drag column to column). All are **one-to-many**, cross-filter
direction **Single** (dimension filters fact), active.

| From (one side) | To (many side) |
|-----------------|----------------|
| `dim_date[date_key]` | `fact_sales_actuals[date_key]` |
| `dim_date[date_key]` | `fact_forecasts[date_key]` |
| `dim_store[store_key]` | `fact_sales_actuals[store_key]` |
| `dim_store[store_key]` | `fact_forecasts[store_key]` |
| `dim_model[model]` | `fact_forecasts[model]` |
| `dim_model[model]` | `model_metrics[model]` |

`feature_importance` is a stand-alone table (no relationships).

Then:

1. Select `dim_date` > **Table tools > Mark as date table** > date column `date`.
2. Sort columns: `dim_date[month_name]` > **Sort by column** `month`; `dim_date[day_name]` > sort by `day_of_week`.
3. Hide from Report view (right-click > Hide) every `*_key` column in the fact tables, plus `dim_date[date_key]`,
   `dim_store[store_key]`.
4. Create a **Horizon Week** calculated column on `fact_forecasts` (Table tools > New column):

```DAX
Horizon Week = ROUNDUP ( fact_forecasts[horizon_day] / 7, 0 )
```

5. Create a measure table: **Home > Enter data**, leave it empty, name it `_Measures`, Load. Put all measures below in
   it (right-click the table > New measure). Set formats as noted.

## 3. DAX measures

### 3.1 Actuals and year-over-year (format: whole number / %)

```DAX
Actual Sales = SUM ( fact_sales_actuals[sales] )

Open Store Days = COUNTROWS ( fact_sales_actuals )

Avg Sales per Open Store-Day = DIVIDE ( [Actual Sales], [Open Store Days] )

Customers = SUM ( fact_sales_actuals[customers] )

Sales PY =
CALCULATE ( [Actual Sales], SAMEPERIODLASTYEAR ( dim_date[date] ) )

Sales YoY % =
DIVIDE ( [Actual Sales] - [Sales PY], [Sales PY] )          -- format as %, 1 decimal

Sales YoY Abs = [Actual Sales] - [Sales PY]

Promo Lift % =
VAR PromoAvg   = CALCULATE ( [Avg Sales per Open Store-Day], fact_sales_actuals[promo] = 1 )
VAR NoPromoAvg = CALCULATE ( [Avg Sales per Open Store-Day], fact_sales_actuals[promo] = 0 )
RETURN DIVIDE ( PromoAvg, NoPromoAvg ) - 1
```

The data starts on 2013-01-01, so `Sales PY` is blank for 2013 (expected). Promo Lift here is the *raw* lift; the
notebook's per-store paired estimate controls for weekday and store size, so the two numbers differ.

### 3.2 Forecast vs actual (format: whole number / %)

```DAX
Selected Model =
-- the slicer choice, or the best model if nothing / several are selected
IF (
    HASONEVALUE ( dim_model[model] ),
    VALUES ( dim_model[model] ),
    LOOKUPVALUE ( dim_model[model], dim_model[is_best_model], 1 )
)

Forecast =
IF (
    HASONEVALUE ( dim_model[model] ),
    SUM ( fact_forecasts[forecast] )
)

Actual (Forecast Window) = SUM ( fact_forecasts[actual_sales] )
-- same rows as [Forecast], so the comparison is like-for-like

Forecast Error = [Forecast] - [Actual (Forecast Window)]

Forecast Bias % = DIVIDE ( [Forecast Error], [Actual (Forecast Window)] )

Forecast vs Actual % = DIVIDE ( [Forecast], [Actual (Forecast Window)] ) - 1

Forecast YoY % =
-- forecast compared with actual sales in the same period last year
DIVIDE ( [Forecast] - [Sales PY], [Sales PY] )

Lower 80 = IF ( HASONEVALUE ( dim_model[model] ), SUM ( fact_forecasts[lower_80] ) )
Upper 80 = IF ( HASONEVALUE ( dim_model[model] ), SUM ( fact_forecasts[upper_80] ) )
```

> The `Selected Model` measure is only needed for a card/title; add a card showing it so people know which model
> they are looking at. If you want charts to default to the best model, set the `dim_model` slicer's default selection
> by selecting `is_best_model = 1` once and saving the file.

### 3.3 Accuracy (format: %, 1 decimal unless noted)

```DAX
WAPE =
IF (
    HASONEVALUE ( dim_model[model] ),
    DIVIDE ( SUM ( fact_forecasts[abs_error] ), SUM ( fact_forecasts[actual_sales] ) )
)

MAPE =
IF (
    HASONEVALUE ( dim_model[model] ),
    AVERAGEX (
        fact_forecasts,
        DIVIDE ( fact_forecasts[abs_error], fact_forecasts[actual_sales] )
    )
)

RMSE =
IF (
    HASONEVALUE ( dim_model[model] ),
    SQRT (
        AVERAGEX (
            fact_forecasts,
            ( fact_forecasts[forecast] - fact_forecasts[actual_sales] ) ^ 2
        )
    )
)                                                   -- format: whole number

Interval Coverage 80 =
VAR Rows = FILTER ( fact_forecasts, NOT ISBLANK ( fact_forecasts[lower_80] ) )
RETURN
    DIVIDE (
        COUNTROWS ( FILTER ( Rows, fact_forecasts[actual_sales] >= fact_forecasts[lower_80]
                              && fact_forecasts[actual_sales] <= fact_forecasts[upper_80] ) ),
        COUNTROWS ( Rows )
    )

-- Pre-computed Python numbers (from model_metrics; use "fold = 0" for all folds pooled)
Model WAPE (Python) =
CALCULATE ( MAX ( model_metrics[wape_pct] ) / 100,
            model_metrics[fold] = 0, model_metrics[scope] = "all_stores" )

Baseline WAPE (Seasonal Naive) =
CALCULATE ( MAX ( model_metrics[wape_pct] ) / 100,
            REMOVEFILTERS ( dim_model ),
            model_metrics[model] = "seasonal_naive",
            model_metrics[fold] = 0, model_metrics[scope] = "all_stores" )

WAPE Improvement vs Seasonal Naive =
DIVIDE ( [Baseline WAPE (Seasonal Naive)] - [Model WAPE (Python)], [Baseline WAPE (Seasonal Naive)] )

WAPE Status Color =
SWITCH ( TRUE (), [WAPE] <= 0.10, "#2A9D8F", [WAPE] <= 0.15, "#E9C46A", "#E76F51" )
-- thresholds are examples; set them to what the business would accept
```

**Reconciliation check (do this before building visuals):** put `dim_model[model]` and `[WAPE]` in a table with the
`fold` slicer cleared. The values must match `Model WAPE (Python)` / `reports/metrics_overall.csv` for `naive`,
`seasonal_naive` and `lightgbm` (differences < 0.01 percentage points = rounding). If they do not, a filter is leaking
into the visual.

## 4. Report pages

Use **View > Page view > Fit to page**, canvas 16:9 (1280 x 720). Theme: **View > Themes** and pick a clean light theme
(or import your own JSON). Use one accent colour for Actual (dark blue), one for Forecast (orange), grey for baselines.
Rename the three pages exactly as below. Add a text box title on each page.

### Page 1 - Executive Overview

Purpose: "How are sales trending and how good is the forecast?" in 10 seconds.

Layout (x, y, width, height in px; use **Format > Position** to set exactly):

| # | Visual | Position | Fields / settings |
|---|--------|----------|-------------------|
| 1 | Slicer (Between) | 20, 60, 300, 50 | `dim_date[date]` |
| 2 | Slicer (Dropdown) | 330, 60, 200, 50 | `dim_model[model]` |
| 3 | Slicer (Dropdown) | 540, 60, 200, 50 | `dim_store[store_type]` |
| 4 | Slicer (Dropdown) | 750, 60, 200, 50 | `fact_forecasts[fold]` |
| 5 | Card | 20, 120, 300, 90 | `[Actual Sales]` (label "Total sales") |
| 6 | Card | 330, 120, 300, 90 | `[Sales YoY %]` |
| 7 | Card | 640, 120, 300, 90 | `[WAPE]` + conditional font colour: Format > Callout value > fx > Field value > `[WAPE Status Color]` |
| 8 | Card | 950, 120, 310, 90 | `[WAPE Improvement vs Seasonal Naive]` (label "WAPE improvement vs baseline") |
| 9 | **Line chart** | 20, 220, 820, 280 | X: `dim_date[date]` (continuous). Y: `[Actual Sales]`, `[Forecast]`. See "Interval band" below |
| 10 | Clustered column | 850, 220, 410, 280 | X: `dim_store[store_type]`, Y: `[Avg Sales per Open Store-Day]` |
| 11 | Clustered column | 20, 510, 620, 200 | X: `dim_date[year_month]`, Y: `[Actual Sales]`, line: `[Sales YoY %]` (use a *Line and clustered column chart*) |
| 12 | Bar chart | 650, 510, 610, 200 | Y: `dim_store[store_label]`, X: `[Actual Sales]`, **Filters on this visual > Top N = 10 by `[Actual Sales]`** |

**Interval band (visual 9).** Add `[Lower 80]` and `[Upper 80]` to the Y axis too; style them as thin dashed grey lines
(or: *Analytics pane > Error bars > Enable* and set *Upper bound* = `[Upper 80]`, *Lower bound* = `[Lower 80]` for the
Forecast series, with a semi-transparent fill). The band is only available while a single model with intervals
(`lightgbm`) is selected, which is the default. For other models the band is blank by design.

Interactions: select visual 1 > **Format > Edit interactions** and make sure the slicers filter every visual. Turn
interactions **off** from visual 12 to visual 9 if you want the line to stay at total level.

### Page 2 - Store Drilldown

Purpose: "Why is store X off, and how reliable is its forecast?"

| # | Visual | Position | Fields / settings |
|---|--------|----------|-------------------|
| 1 | Slicer (Dropdown, single select, search on) | 20, 60, 250, 50 | `dim_store[store_label]`. Format > Selection controls > **Single select = On** |
| 2 | Slicer (Dropdown) | 280, 60, 200, 50 | `dim_model[model]` |
| 3 | Slicer (Between) | 490, 60, 300, 50 | `dim_date[date]` |
| 4 | Multi-row card | 800, 55, 460, 90 | `dim_store[store_type]`, `[assortment]`, `[competition_distance]`, `[promo2]` |
| 5 | Card x4 | 20, 120, 4 x 300 wide, 90 | `[Actual Sales]`, `[Avg Sales per Open Store-Day]`, `[WAPE]`, `[Forecast Bias %]` |
| 6 | Line chart | 20, 220, 820, 260 | X: `dim_date[date]`, Y: `[Actual Sales]`, `[Forecast]`, `[Lower 80]`, `[Upper 80]` |
| 7 | Column chart | 850, 220, 410, 260 | X: `dim_date[day_name]`, Y: `[Avg Sales per Open Store-Day]`; legend/small multiples optional: `fact_sales_actuals[promo]` |
| 8 | Matrix (heat-map) | 20, 490, 620, 220 | Rows: `dim_date[year]`, `dim_date[month_name]` (expand). Columns: `dim_date[day_name]`. Values: `[Avg Sales per Open Store-Day]`. **Format > Cell elements > Background colour > Gradient** |
| 9 | Table | 650, 490, 610, 220 | `dim_date[year_month]`, `[Actual (Forecast Window)]`, `[Forecast]`, `[Forecast Error]`, `[WAPE]`. Conditional formatting on `[WAPE]` using `[WAPE Status Color]` |

**Drill-through:** create this page's drill-through from Page 1. On Page 2, drag `dim_store[store_label]` into
**Drill through > Add drill-through fields here**, keep "Keep all filters" on. Then right-click a bar in Page 1 visual
12 > *Drill through > Store Drilldown*.

### Page 3 - Model Performance

Purpose: "Which model is best, and can I trust its intervals?"

| # | Visual | Position | Fields / settings |
|---|--------|----------|-------------------|
| 1 | Table | 20, 60, 620, 170 | From `model_metrics`: `model`, `wape_pct`, `mape_pct`, `rmse`, `bias_pct`. **Page/visual filter:** `fold = 0`, `scope = "all_stores"`. Conditional formatting: data bars on `wape_pct`; sort ascending |
| 2 | Table | 650, 60, 610, 170 | Same fields. Filter `fold = 0`, `scope = "sarima_sample"` (fair comparison including SARIMA on its 60 stores). Title: "Same 60 stores, all models" |
| 3 | Clustered column | 20, 240, 620, 230 | X: `fact_forecasts[fold]`, Legend: `dim_model[model]`, Y: `[WAPE]`. Clear the model slicer so all models show. Title: "WAPE by backtest fold" |
| 4 | Line chart | 650, 240, 610, 230 | X: `fact_forecasts[Horizon Week]`, Legend: `dim_model[model]`, Y: `[WAPE]`. Title: "WAPE by forecast horizon week" (check whether error degrades with horizon; do not assume it) |
| 5 | Card (gauge optional) | 20, 480, 200, 110 | `[Interval Coverage 80]`. Gauge: min 0, target 0.8, max 1. Filter `dim_model[model] = lightgbm` |
| 6 | Card | 20, 600, 200, 110 | `[WAPE Improvement vs Seasonal Naive]` |
| 7 | Bar chart | 230, 480, 560, 230 | `feature_importance`: Y = `feature`, X = `mean_abs_shap`; **Top N = 15**; sort descending. Title: "What drives the LightGBM forecast (mean abs SHAP)" |
| 8 | Table | 800, 480, 460, 230 | `model_metrics` where `model = lightgbm`, `scope = all_stores`: `fold`, `origin_date`, `wape_pct`, `coverage_80`, `interval_width_pct_of_actual` |

Note on visual 5: an 80% interval should contain about 80% of actuals. The Python backtest reports the real coverage per
fold; if your gauge is far below 0.8 that is a genuine finding (intervals too narrow), not a bug. See the README.

## 5. Finishing touches

* **View > Sync slicers**: sync the date and model slicers across all three pages.
* **Bookmarks / navigation:** Insert > Buttons > Navigator > Page navigator.
* **Tooltips:** add `[Forecast Bias %]` and `[Open Store Days]` to the tooltip well of the line charts.
* Alt text on every visual (Format > General > Alt text) for accessibility.
* Number formats: sales `#,0`; percentages `0.0%`; RMSE `#,0`.

## 6. QA checklist

1. `[Actual Sales]` over the full range equals `SUM(sales)` of `data/processed/sales.parquet` (open days) - the Python
   check is `pd.read_parquet(...).query("open==1").sales.sum()`.
2. `[WAPE]` for `lightgbm`, `seasonal_naive`, `naive` (no fold filter) matches `reports/metrics_overall.csv`.
3. `[Interval Coverage 80]` for `lightgbm` matches `coverage_80` in `model_metrics` for `fold = 0` (note `coverage_80` and the `*_pct` columns are stored in percent, the DAX measures are fractions).
4. No visual shows a total for `[Forecast]` while more than one model is in filter context (it must be blank).
5. Page 2 with the store slicer cleared equals Page 1.

## 7. Screenshots to export (for the README)

File > Export > or use the Windows snipping tool at 100% zoom. Save as PNG:

* `docs/screenshots/01_executive_overview.png`
* `docs/screenshots/02_store_drilldown.png`
* `docs/screenshots/03_model_performance.png`
