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
- mean |SHAP| of the badge: 0.026

The ML and SHAP values can differ in the last decimal across platforms and BLAS builds.
