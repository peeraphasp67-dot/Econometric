"""SHAP seed sensitivity for InCIT 2026 Paper 185 (Section IV-C).
Refits the fixed-parameter XGBoost model of make_figures.py with 20 random seeds and reports the
spread of the badge SHAP statistics. XGBoost's row/column subsampling draws differ across
platforms, so a single seed is one draw from this distribution.
Run from inside revision/:  python shap_seed_sensitivity.py ../data/amazon_products_sales_data_cleaned.csv"""
import sys
import numpy as np, pandas as pd, shap
from xgboost import XGBRegressor

PATH = sys.argv[1] if len(sys.argv) > 1 else "../data/amazon_products_sales_data_cleaned.csv"
G = ['Carbon impact', 'Energy efficiency', 'Manufacturing practices', 'Forestry practices', 'Recycled materials']
df = pd.read_csv(PATH).dropna(subset=['purchased_last_month', 'discounted_price'])
df['is_green'] = df.sustainability_tags.apply(lambda x: int(any(g in str(x) for g in G)))
df['total_reviews'] = df.total_reviews.fillna(0)
df['rating'] = df.product_rating.fillna(df.product_rating.mean())
df['coupon'] = (df.has_coupon != 'No Coupon').astype(int)
df['sponsored'] = (df.is_sponsored == 'Sponsored').astype(int)
df['log_sales'] = np.log1p(df.purchased_last_month)
df['log_price'] = np.log1p(df.discounted_price)
df['log_reviews'] = np.log1p(df.total_reviews)
p = df.sort_values('data_collected_at').drop_duplicates('product_title', keep='last').reset_index(drop=True)
FE = ['is_green', 'log_price', 'log_reviews', 'rating', 'coupon', 'sponsored']
g = p.is_green.values == 1
rows = []
for s in range(20):
    m = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.02, subsample=0.7, colsample_bytree=0.7,
                     min_child_weight=5, random_state=s, n_jobs=2, verbosity=0).fit(p[FE], p.log_sales)
    sv = shap.TreeExplainer(m).shap_values(p[FE])
    d = pd.DataFrame({'price': p.discounted_price.values[g], 'shap': sv[g, 0]})
    b = d.groupby(pd.qcut(d.price, 6), observed=True).shap.mean()
    rows.append({'seed': s, 'badge_mean_abs_shap': np.abs(sv[:, 0]).mean(),
                 'sponsored_mean_abs_shap': np.abs(sv[:, 5]).mean(),
                 'badged_mean_attribution': sv[g, 0].mean(),
                 'sextile_min': b.min(), 'sextile_max': b.max()})
r = pd.DataFrame(rows)
print(r.round(4).to_string(index=False))
print('\nSummary over 20 seeds (mean [min, max]):')
for c in r.columns[1:]:
    print(f'  {c:28s} {r[c].mean():.4f} [{r[c].min():.4f}, {r[c].max():.4f}]')
m = r.badged_mean_attribution
print(f'  badged attribution in %      {100*(np.exp(m.mean())-1):.1f}% [{100*(np.exp(m.min())-1):.1f}%, {100*(np.exp(m.max())-1):.1f}%]')
