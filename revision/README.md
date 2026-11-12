# Reproducibility package — InCIT 2026, Paper 185

This folder reproduces all tables (I–V), the robustness checks and Fig. 1–6 of the paper

> **Green Signals in E-Commerce: Explainable Modeling of Price Elasticity and Social Proof**
> (InCIT 2026, Paper 185)

All numbers in the paper come from the two scripts in this folder. The Streamlit dashboard in the
repository root (`app.py`) is a separate exploratory tool, and its settings differ from the paper
(see the root [README](../README.md#paper-reproducibility-incit-2026)).

## Data

- Source: Kaggle, *Amazon Products Sales Dataset 2025*,
  <https://www.kaggle.com/datasets/ikramshah512/amazon-products-sales-dataset-42k-items-2025>
- File: [`data/amazon_products_sales_data_cleaned.csv`](../data/amazon_products_sales_data_cleaned.csv) (included in this repository)

## Key design decision: product-level analysis

The 30,304 listing rows that have both sales and price are **repeated scrapes of 7,291 unique
products**. The analysis therefore de-duplicates to **one row per product (latest scrape)**. All
main results (Tables I–IV, PSM, SHAP) are at the product level.

Listing-level estimates with product-clustered standard errors are kept only as a robustness
check (Model C). The listing-level data are also used to show that a random K-fold split leaks
information, because the same product appears in both training and test folds.

## How to run

Use Python 3.11. Run from inside `revision/`:

```bash
pip install -r requirements-paper.txt
python revision_analysis.py ../data/amazon_products_sales_data_cleaned.csv > results.txt
python make_figures.py ../data/amazon_products_sales_data_cleaned.csv
```

`revision_analysis.py` writes `revision_results.json`, and `make_figures.py` reads it, so run the
scripts in this order. The nested cross-validation (section 7) takes several minutes.

The random seed is **42** everywhere (CV splits, randomized hyper-parameter search, Random Forest,
XGBoost).

## Outputs

| File | Content |
|---|---|
| `results.txt` | Full text log of every table and test (sections 1–8 below) |
| `revision_results.json` | Machine-readable results; input to `make_figures.py` |
| `fig1.png` – `fig6.png` | Figures at 300 dpi, sized to the IEEE column |
| `shap_seed_sensitivity.py` → `shap_seed_sensitivity.txt` | Seed-averaged SHAP values reported in the paper: refits the fixed-parameter XGBoost model of `make_figures.py` with seeds 0–19. Run with `python shap_seed_sensitivity.py ../data/amazon_products_sales_data_cleaned.csv > shap_seed_sensitivity.txt` |

`make_figures.py` writes `fig1.png`–`fig6.png` to the current directory. The copies used in the
paper are committed in [`figures/`](figures/).

## Mapping from `results.txt` to the paper

| Section in `results.txt` | Paper |
|---|---|
| 1. Sample flow and duplication | Sample flow (listings → unique products) |
| 2. Descriptives | Table I |
| 3. PSM | Propensity-score matching + Table II (balance) |
| 4. Moderated log-log regression (Models A–C) | Table III |
| 5. Additional controls (Models D–H) | Table V (robustness) |
| 6. Ordered logit on sales buckets | Outcome-measurement robustness check |
| 7. ML validation (nested CV) | Table IV + leakage check (random KFold vs GroupKFold) |
| 8. TreeSHAP on tuned XGBoost | SHAP analysis (Fig. 5–6) |

## Reference values

A clean run with `requirements-paper.txt` should give (see `results.txt`):

- 7,291 unique products, 362 with an eco-label badge; 360 matched pairs; mean |SMD| after matching 0.0198
- Model A (product level): green × price 0.1426 (p = 0.0295), green × reviews 0.1531, R² 0.4640, badge effect at medians +39.3%
- Model C (listing level): green × price 0.4473
- Nested CV R²: OLS 0.4603, Random Forest 0.4805, XGBoost 0.4794
- Leakage check (listing level, Random Forest): random KFold R² 0.865 vs GroupKFold R² 0.202
- mean |SHAP| of the badge: 0.0261 for the single seed-42 draw in Fig. 5 (Linux), 0.024 seed-averaged (see Platform note)

### Platform note

- **Sections 1–6** (sample flow, Table I, PSM, all regressions, ordered logit) reproduce exactly on any platform.
- **Sections 7–8** (nested CV, leakage check, SHAP) depend on the platform. XGBoost's random stream for row/column subsampling (`subsample`, `colsample_bytree`) differs by OS, so the same seed gives a slightly different model. Thread count was tested and ruled out on Linux: `n_jobs` = 1, 2 and 8 give identical results.
- **Original environment:** Linux x86_64, Python 3.11.15, OpenBLAS, 2 CPU threads, with the versions in `requirements-paper.txt`.
- **Example (Windows 11, same pinned versions):** XGBoost nested-CV R² 0.4789 (vs 0.4794), GroupKFold Random Forest R² 0.1998 (vs 0.2023). The badge mean |SHAP| was 0.0239 (vs 0.0261), both in section 8 and in `make_figures.py`.
- **Fig. 5–6** show one draw (seed 42, Linux: badge mean |SHAP| 0.0261).
- **Seed-averaged values in the paper:** the paper reports seed-averaged SHAP values from `shap_seed_sensitivity.py` (20 seeds, Linux).
  - Badge mean |SHAP|: 0.024 (range 0.022–0.026).
  - Mean badge attribution among badged products: +0.236 log points ≈ +27% (range +24% to +30%).
  - On Windows the same script gives means of 0.024 and +0.235 (≈ +26.5%), with a range of 0.022–0.026 for the badge mean |SHAP| (min 0.0215).

