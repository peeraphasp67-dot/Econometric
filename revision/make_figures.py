"""Regenerate Fig. 1-6 for Paper 185 at the exact size/aspect of the figures in the manuscript.
Run after revision_analysis.py (reads revision_results.json).  python make_figures.py <csv>"""
import sys, json, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import statsmodels.api as sm
from xgboost import XGBRegressor
import shap

PATH = sys.argv[1] if len(sys.argv) > 1 else "data/amazon_products_sales_data_cleaned.csv"
RES = json.load(open('revision_results.json'))
BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1f1f1f', '#6b6b6b', '#e3e3e3'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
                     'font.size': 8, 'axes.edgecolor': MUTED, 'axes.linewidth': 0.6, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK, 'ytick.labelcolor': INK,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300})
SIZES = {1: (2.90, 3.61), 2: (3.31, 2.11), 3: (3.35, 2.24), 4: (3.38, 2.11), 5: (3.24, 2.26), 6: (3.22, 2.14)}
def fig(n): return plt.figure(figsize=SIZES[n])
def save(f, n): f.savefig(f'fig{n}.png', dpi=300); plt.close(f); print('saved fig%d.png' % n)

# ---- data (same rules as revision_analysis.py)
GREEN = ['Carbon impact', 'Energy efficiency', 'Manufacturing practices', 'Forestry practices', 'Recycled materials']
raw = pd.read_csv(PATH)
df = raw.dropna(subset=['purchased_last_month', 'discounted_price']).copy()
df['is_green'] = df.sustainability_tags.apply(lambda x: int(any(g in str(x) for g in GREEN)))
df['total_reviews'] = df.total_reviews.fillna(0)
df['rating'] = df.product_rating.fillna(df.product_rating.mean())
df['coupon'] = (df.has_coupon != 'No Coupon').astype(int)
df['sponsored'] = (df.is_sponsored == 'Sponsored').astype(int)
df['log_sales'] = np.log1p(df.purchased_last_month)
df['log_price'] = np.log1p(df.discounted_price)
df['log_reviews'] = np.log1p(df.total_reviews)
prod = df.sort_values('data_collected_at').drop_duplicates('product_title', keep='last').reset_index(drop=True)
prod['tag1'] = prod.sustainability_tags.astype(str).str.replace(r'\s*\+\d+ more', '', regex=True)

# ---- Fig. 1 pipeline
f = fig(1); ax = f.add_axes([0, 0, 1, 1]); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
steps = [("Raw listings", "N = 42,675 rows"),
         ("Listings with sales and price", "N = 30,304 rows"),
         ("De-duplication to unique products", "N = 7,291 (362 badged, 4.97%)"),
         ("PSM: exact Sponsored × Coupon strata,\n1:1 NN, caliper 0.02", "N = 720 (360 pairs)"),
         ("Moderated log–log OLS (full & PSM)\n+ nested-CV RF / XGBoost + TreeSHAP", "HC3 / pair-clustered SEs"),
         ("Hypothesis tests, effect sizes,\nrobustness (Tables III–V)", "")]
h, gap = 0.125, 0.038; top = 0.975
for i, (a, b) in enumerate(steps):
    y = top - i * (h + gap) - h
    ax.add_patch(FancyBboxPatch((0.06, y), 0.88, h, boxstyle="round,pad=0.004,rounding_size=0.02",
                                fc='#eef4fc' if i != 2 else '#fdebe3', ec=BLUE if i != 2 else ORANGE, lw=0.8))
    ax.text(0.5, y + h * (0.62 if b else 0.5), a, ha='center', va='center', fontsize=7.2, color=INK, linespacing=1.1)
    if b: ax.text(0.5, y + h * 0.2, b, ha='center', va='center', fontsize=6.8, color=MUTED, style='italic')
    if i < len(steps) - 1:
        ax.annotate('', xy=(0.5, y - gap + 0.004), xytext=(0.5, y - 0.004),
                    arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=0.8, mutation_scale=7))
save(f, 1)

# ---- Fig. 2 balance
COV = ['log_price', 'log_reviews', 'rating', 'sponsored', 'coupon']
ps = sm.Logit(prod.is_green, sm.add_constant(prod[COV])).fit(disp=0)
prod['ps'] = ps.predict(sm.add_constant(prod[COV]))
pairs = []
for _, g in prod.groupby(['sponsored', 'coupon']):
    t = g[g.is_green == 1].sort_values('ps', ascending=False); c = g[g.is_green == 0]
    cps, cidx, av = c.ps.values, c.index.values, np.ones(len(c), bool)
    for i, p in zip(t.index, t.ps.values):
        d = np.abs(cps - p); d[~av] = np.inf
        if len(d) and d.min() <= 0.02: j = d.argmin(); av[j] = False; pairs.append((i, cidx[j]))
psm = pd.concat([prod.loc[[a for a, b in pairs]], prod.loc[[b for a, b in pairs]]])
def smd(d, c):
    a, b = d[d.is_green == 1][c], d[d.is_green == 0][c]; s = np.sqrt((a.var() + b.var()) / 2)
    return 0 if s == 0 else abs(a.mean() - b.mean()) / s
lab = ['ln(Price)', 'ln(Reviews)', 'Rating', 'Sponsored', 'Coupon']
r = [smd(prod, c) for c in COV]; m = [smd(psm, c) for c in COV]
f = fig(2); ax = f.add_axes([0.13, 0.2, 0.84, 0.72]); x = np.arange(5); w = 0.36
ax.bar(x - w / 2 - 0.01, r, w, color=MUTED, label='Before matching', zorder=2)
ax.bar(x + w / 2 + 0.01, m, w, color=BLUE, label='After matching', zorder=2)
ax.axhline(0.10, color=ORANGE, lw=1, ls='--', zorder=3); ax.text(4.55, 0.108, 'Threshold 0.10', ha='right', fontsize=6.5, color=INK)
for i, v in enumerate(m): ax.text(i + w / 2 + 0.01, v + 0.01, f'{v:.3f}', ha='center', fontsize=5.8, color=INK)
ax.set_xticks(x, lab); ax.set_ylabel('Abs. standardized difference'); ax.set_ylim(0, 0.56)
ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0); ax.legend(frameon=False, fontsize=6.5, loc='upper right')
save(f, 2)

# ---- Fig. 3 median sales by tag (product level)
gp = prod[prod.is_green == 1].groupby('tag1').purchased_last_month.agg(['median', 'size']).sort_values('median')
f = fig(3); ax = f.add_axes([0.40, 0.18, 0.55, 0.74]); y = np.arange(len(gp))
ax.barh(y, gp['median'], color=BLUE, height=0.6, zorder=2)
base = prod[prod.is_green == 0].purchased_last_month.median()
ax.axvline(base, color=MUTED, lw=0.9, ls='--', zorder=3)
ax.text(base, len(gp) - 0.35, f' Unbadged median = {base:.0f}', fontsize=6.3, color=INK, va='bottom')
for i, (v, n) in enumerate(zip(gp['median'], gp['size'])): ax.text(v + 8, i, f'{v:.0f}', va='center', fontsize=6.3, color=INK)
ax.set_yticks(y, [f'{t} (n={n})' for t, n in zip(gp.index, gp['size'])], fontsize=6.5)
ax.set_xlabel('Median monthly units sold (bucketed)'); ax.set_ylim(-0.6, len(gp) - 0.1)
ax.set_xlim(0, max(gp['median'].max(), base) * 1.18); ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
save(f, 3)

# ---- Fig. 4 nested-CV performance (3 small multiples)
ml = RES['ml_product']; models = ['OLS', 'Ridge', 'XGBoost', 'RandomForest']; names = ['OLS', 'Ridge', 'XGBoost', 'RF']
f = fig(4); cols = [MUTED, MUTED, BLUE, BLUE]
for k, (met, lo, hi) in enumerate([('R2', 0.40, 0.52), ('RMSE', 0.90, 1.02), ('MAE', 0.70, 0.82)]):
    ax = f.add_axes([0.1 + k * 0.33, 0.22, 0.215, 0.66])
    v = [ml[met][mm] for mm in models]; s = [ml[met + '_sd'][mm] for mm in models]
    for i in range(4):
        ax.errorbar(i, v[i], yerr=s[i], fmt='o', ms=4, color=cols[i], ecolor=cols[i], elinewidth=0.9, capsize=2, zorder=3)
    ax.set_ylim(lo, hi); ax.set_xlim(-0.5, 3.5); ax.set_xticks(range(4), names, rotation=45, fontsize=6.3)
    ax.set_title({'R2': 'R² (higher is better)', 'RMSE': 'RMSE (lower is better)', 'MAE': 'MAE (lower is better)'}[met], fontsize=6.8, color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0); ax.tick_params(axis='y', labelsize=6)
save(f, 4)

# ---- SHAP (same model as revision_analysis.py: modal tuned params)
FE = ['is_green', 'log_price', 'log_reviews', 'rating', 'coupon', 'sponsored']
xgb = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.02, subsample=0.7, colsample_bytree=0.7,
                   min_child_weight=5, random_state=42, n_jobs=-1, verbosity=0).fit(prod[FE], prod.log_sales)
sv = shap.TreeExplainer(xgb).shap_values(prod[FE])
imp = pd.Series(np.abs(sv).mean(0), index=['Badge', 'ln(Price)', 'ln(Reviews)', 'Rating', 'Coupon', 'Sponsored']).sort_values()
print(imp.round(4).to_dict())
f = fig(5); ax = f.add_axes([0.2, 0.18, 0.72, 0.78]); y = np.arange(len(imp))
ax.barh(y, imp.values, color=[ORANGE if n == 'Badge' else BLUE for n in imp.index], height=0.6, zorder=2)
for i, v in enumerate(imp.values): ax.text(v + 0.006, i, f'{v:.3f}', va='center', fontsize=6.3, color=INK)
ax.set_yticks(y, imp.index); ax.set_xlabel('Mean |SHAP value| (log units sold)'); ax.set_xlim(0, imp.max() * 1.18)
ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
save(f, 5)

# ---- Fig. 6 badge SHAP vs price, badged products
g = prod.is_green.values == 1
d = pd.DataFrame({'p': prod.discounted_price.values[g], 's': sv[g, 0]})
d['bin'] = pd.qcut(d.p, 6); b = d.groupby('bin', observed=True).agg(p=('p', 'median'), s=('s', 'mean'))
f = fig(6); ax = f.add_axes([0.14, 0.2, 0.82, 0.74])
ax.scatter(d.p, d.s, s=6, color=BLUE, alpha=0.45, lw=0, zorder=2, label='Badged product')
ax.plot(b.p, b.s, color=ORANGE, lw=1.4, marker='o', ms=3.5, zorder=3, label='Price-sextile mean')
ax.axhline(0, color=MUTED, lw=0.7); ax.set_xscale('log')
ax.set_xlabel('Discounted price (USD, log scale)'); ax.set_ylabel('SHAP value of Badge')
ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0); ax.legend(frameon=False, fontsize=6.3, loc='lower right')
save(f, 6)
