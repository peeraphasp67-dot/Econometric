"""
Revision analysis for InCIT 2026 Paper 185
"Green Signals in E-Commerce: Explainable Modeling of Price Elasticity and Social Proof"

Reproduces every number used in the revised manuscript from the public Kaggle file
(amazon_products_sales_data_cleaned.csv). Run:  python revision_analysis.py <path_to_csv>

Primary analytic unit = unique product (one row per product title, latest scrape),
because the listing-level file repeats the same product across scrape dates.
Listing-level estimates with product-clustered SEs are reported for comparability.
"""
import sys, json, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.model_selection import KFold, GroupKFold, RandomizedSearchCV, cross_validate
from sklearn.linear_model import LinearRegression, RidgeCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
warnings.filterwarnings("ignore")
SEED = 42
PATH = sys.argv[1] if len(sys.argv) > 1 else "data/amazon_products_sales_data_cleaned.csv"
OUT = {}

GREEN = ['Carbon impact', 'Energy efficiency', 'Manufacturing practices',
         'Forestry practices', 'Recycled materials']
X_VARS = ['is_green', 'log_price', 'g_x_price', 'log_reviews', 'g_x_reviews',
          'coupon', 'g_x_coupon', 'rating', 'sponsored']
ML_FEATS = ['is_green', 'log_price', 'log_reviews', 'rating', 'coupon', 'sponsored']

def section(t): print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)

# ---------------------------------------------------------------- 1. data
raw = pd.read_csv(PATH)
df = raw.dropna(subset=['purchased_last_month', 'discounted_price']).copy()
df['is_green'] = df.sustainability_tags.apply(lambda x: int(any(g in str(x) for g in GREEN)))
df['multi_tag'] = df.sustainability_tags.astype(str).str.contains(r'\+\d+ more').astype(int)
df['total_reviews'] = df.total_reviews.fillna(0)
df['rating'] = df.product_rating.fillna(df.product_rating.mean())
df['coupon'] = (df.has_coupon != 'No Coupon').astype(int)
df['sponsored'] = (df.is_sponsored == 'Sponsored').astype(int)
df['best_seller'] = (df.is_best_seller == 'Best Seller').astype(int)
df['log_sales'] = np.log1p(df.purchased_last_month)
df['log_price'] = np.log1p(df.discounted_price)
df['log_reviews'] = np.log1p(df.total_reviews)
df['brand'] = df.product_title.str.split().str[0].str.upper().str.strip(',.-:')
df['pid'] = df.product_title.factorize()[0]
def add_inter(d):
    d = d.copy()
    d['g_x_price'] = d.is_green * d.log_price
    d['g_x_reviews'] = d.is_green * d.log_reviews
    d['g_x_coupon'] = d.is_green * d.coupon
    return d
df = add_inter(df)
prod = (df.sort_values('data_collected_at').drop_duplicates('product_title', keep='last')
          .reset_index(drop=True))
bc = prod.brand.value_counts()
prod['brand_grp'] = np.where(prod.brand.map(bc) >= 20, prod.brand, 'OTHER')

section("1. SAMPLE FLOW AND DUPLICATION")
flow = {
    'raw_rows': len(raw),
    'rows_with_sales_and_price': len(df),
    'unique_products': int(prod.shape[0]),
    'exact_duplicate_rows(title,price,reviews,rating)': int(df.duplicated(
        ['product_title', 'discounted_price', 'total_reviews', 'product_rating']).sum()),
    'green_rows': int(df.is_green.sum()),
    'green_unique_products': int(prod.is_green.sum()),
    'green_rows_from_top4_products': int(df[df.is_green == 1].product_title.value_counts().head(4).sum()),
    'raw_share_any_tag_%': round(raw.sustainability_tags.notna().mean() * 100, 2),
    'raw_share_green_%': round(raw.sustainability_tags.apply(
        lambda x: any(g in str(x) for g in GREEN)).mean() * 100, 2),
    'listing_share_green_%': round(df.is_green.mean() * 100, 2),
    'product_share_green_%': round(prod.is_green.mean() * 100, 2),
    'green_multi_tag_rows': int(df[df.is_green == 1].multi_tag.sum()),
    'green_multi_tag_products': int(prod[prod.is_green == 1].multi_tag.sum()),
}
for k, v in flow.items(): print(f"  {k:55s} {v}")
print("  Top green products by row count:")
print(df[df.is_green == 1].groupby('product_title').agg(
    rows=('pid', 'size'), price=('discounted_price', 'median')).sort_values('rows', ascending=False)
    .head(6).rename(index=lambda s: s[:60]).to_string())
OUT['flow'] = flow
print("\n  Outcome support (purchased_last_month):",
      sorted(df.purchased_last_month.unique().astype(int).tolist())[:12], "...")

# ---------------------------------------------------------------- 2. Table I
section("2. TABLE I - DESCRIPTIVES (product level, N=%d)" % len(prod))
def desc(d):
    rows = []
    for name, col in [('Sales volume (units/month, bucketed)', 'purchased_last_month'),
                      ('Discounted price (USD)', 'discounted_price'),
                      ('Total reviews', 'total_reviews'), ('Rating (1-5)', 'rating')]:
        s = d[col]; rows.append([name, s.mean(), s.std(), s.min(), s.median(), s.max()])
    for name, col in [('Eco-label badge', 'is_green'), ('Sponsored', 'sponsored'),
                      ('Coupon', 'coupon'), ('Best Seller', 'best_seller')]:
        rows.append([name, d[col].mean() * 100, np.nan, 0, np.nan, 1])
    return pd.DataFrame(rows, columns=['Variable', 'Mean/Share%', 'SD', 'Min', 'Median', 'Max']).round(2)
t1p = desc(prod); t1l = desc(df)
print(t1p.to_string(index=False))
print("\n  (listing level, N=%d, for reference)" % len(df)); print(t1l.to_string(index=False))
print("\n  Pre-matching means by badge (product level):")
print(prod.groupby('is_green')[['discounted_price', 'total_reviews', 'rating']].mean().round(2).to_string())
print("  Product categories:", prod.product_category.nunique(), "| brands (>=20 products):",
      (bc >= 20).sum(), "| brands total:", len(bc))

# ---------------------------------------------------------------- 3. PSM
section("3. PSM (product level): logit PS; exact strata Sponsored x Coupon; 1:1 NN, no replacement, caliper 0.02")
COV = ['log_price', 'log_reviews', 'rating', 'sponsored', 'coupon']
ps_fit = sm.Logit(prod.is_green, sm.add_constant(prod[COV])).fit(disp=0)
prod['ps'] = ps_fit.predict(sm.add_constant(prod[COV]))
print(pd.DataFrame({'coef': ps_fit.params, 'p': ps_fit.pvalues}).round(4).to_string())
def match(d, caliper=0.02):
    pairs = []
    for _, g in d.groupby(['sponsored', 'coupon']):
        t = g[g.is_green == 1].sort_values('ps', ascending=False)
        c = g[g.is_green == 0]
        cps, cidx, avail = c.ps.values, c.index.values, np.ones(len(c), bool)
        for i, p in zip(t.index, t.ps.values):
            dist = np.abs(cps - p); dist[~avail] = np.inf
            if len(dist) == 0: continue
            j = dist.argmin()
            if dist[j] <= caliper:
                avail[j] = False; pairs.append((i, cidx[j]))
    ti, ci = zip(*pairs)
    return pd.concat([d.loc[list(ti)].assign(pair=range(len(ti))),
                      d.loc[list(ci)].assign(pair=range(len(ci)))])
psm = match(prod)
n_t = int(prod.is_green.sum()); n_m = int(psm.is_green.sum())
print(f"\n  treated={n_t}, matched pairs={n_m}, dropped treated={n_t - n_m}, N matched={len(psm)}")
def smd(a, b):
    s = np.sqrt((a.var() + b.var()) / 2); return 0 if s == 0 else abs(a.mean() - b.mean()) / s
bal = []
for c in COV:
    r = smd(prod[prod.is_green == 1][c], prod[prod.is_green == 0][c])
    m = smd(psm[psm.is_green == 1][c], psm[psm.is_green == 0][c])
    vt, vc = psm[psm.is_green == 1][c].var(), psm[psm.is_green == 0][c].var()
    bal.append([c, r, m, vt / vc if vc > 0 else 1.0])
bal = pd.DataFrame(bal, columns=['Covariate', 'SMD_raw', 'SMD_matched', 'VarRatio_matched']).round(3)
print(bal.to_string(index=False)); print("  mean SMD raw=%.4f matched=%.4f" % (bal.SMD_raw.mean(), bal.SMD_matched.mean()))
OUT['psm'] = {'treated': n_t, 'pairs': n_m, 'mean_smd_raw': bal.SMD_raw.mean(), 'mean_smd_matched': bal.SMD_matched.mean()}

# ---------------------------------------------------------------- 4. regressions
section("4. MODERATED LOG-LOG REGRESSION")
def fit(d, xvars, cov='HC3', groups=None, fe=None):
    X = d[xvars].astype(float)
    if fe:
        X = pd.concat([X, pd.get_dummies(d[fe], drop_first=True, prefix=fe, dtype=float)], axis=1)
    X = sm.add_constant(X)
    if groups is not None:
        return sm.OLS(d.log_sales, X).fit(cov_type='cluster', cov_kwds={'groups': groups})
    return sm.OLS(d.log_sales, X).fit(cov_type=cov)

def lincom(m, w):
    names = list(w); c = np.zeros(len(m.params)); idx = list(m.params.index)
    for k, v in w.items(): c[idx.index(k)] = v
    est = float(c @ m.params.values); se = float(np.sqrt(c @ m.cov_params().values @ c))
    from scipy import stats
    return est, se, 2 * (1 - stats.norm.cdf(abs(est / se)))

def report(m, label, ref):
    k = ['const'] + X_VARS
    tab = pd.DataFrame({'b': m.params.reindex(k), 'se': m.bse.reindex(k), 'p': m.pvalues.reindex(k)})
    print(f"\n--- {label}: N={int(m.nobs)}, R2={m.rsquared:.4f}, adjR2={m.rsquared_adj:.4f}")
    print(tab.round(4).to_string())
    e, s, p = lincom(m, {'log_price': 1, 'g_x_price': 1})
    print(f"  Badged price gradient b2+b4 = {e:.4f} (SE {s:.4f}, p={p:.4f})")
    lp, lr = np.log1p(ref.discounted_price.median()), np.log1p(ref.total_reviews.median())
    e2, s2, p2 = lincom(m, {'is_green': 1, 'g_x_price': lp, 'g_x_reviews': lr})
    print(f"  Badge effect at medians (P=${ref.discounted_price.median():.2f}, R={ref.total_reviews.median():.0f}): "
          f"{e2:.4f} log pts (SE {s2:.4f}, p={p2:.4f}) -> {100*(np.exp(e2)-1):+.1f}% "
          f"[95% CI {100*(np.exp(e2-1.96*s2)-1):+.1f}%, {100*(np.exp(e2+1.96*s2)-1):+.1f}%]")
    b = m.params
    be = (-(b.is_green + b.g_x_reviews * lr)) / b.g_x_price
    print(f"  Break-even price (badge effect=0 at median reviews): ${np.expm1(be):.2f}")
    # effect at price quartiles
    for q in [0.25, 0.5, 0.75]:
        lpq = np.log1p(ref.discounted_price.quantile(q))
        eq, sq, pq = lincom(m, {'is_green': 1, 'g_x_price': lpq, 'g_x_reviews': lr})
        print(f"    badge effect at price Q{int(q*100)} (${np.expm1(lpq):.2f}): {100*(np.exp(eq)-1):+.1f}% (p={pq:.3f})")
    return {'N': int(m.nobs), 'R2': m.rsquared, 'params': m.params.reindex(k).to_dict(),
            'p': m.pvalues.reindex(k).to_dict(), 'se': m.bse.reindex(k).to_dict(),
            'badged_gradient': [e, s, p], 'badge_at_median': [e2, s2, p2], 'breakeven_price': float(np.expm1(be))}

R = {}
R['product_full'] = report(fit(prod, X_VARS), "A. Product level, full (HC3)", prod)
R['product_psm'] = report(fit(psm, X_VARS, groups=psm.pair), "B. Product level, PSM-matched (SE clustered by pair)", psm)
R['listing_cluster'] = report(fit(df, X_VARS, groups=df.pid), "C. Listing level as in submitted paper (SE clustered by product)", df)

section("5. ADDITIONAL CONTROLS (Reviewer 2) - product level")
xc = X_VARS + ['best_seller']
m_cat = fit(prod, xc, fe='product_category'); R['ctrl_category'] = report(m_cat, "D. + Best Seller + category FE", prod)
p2 = prod.copy(); p2['cat_brand'] = p2.product_category  # placeholder
Xb = pd.concat([prod[xc].astype(float),
                pd.get_dummies(prod.product_category, drop_first=True, prefix='cat', dtype=float),
                pd.get_dummies(prod.brand_grp, drop_first=True, prefix='br', dtype=float)], axis=1)
m_cb = sm.OLS(prod.log_sales, sm.add_constant(Xb)).fit(cov_type='HC3')
R['ctrl_cat_brand'] = report(m_cb, "E. + Best Seller + category FE + brand FE (brands>=20 products)", prod)
psm_c = psm.copy()
Xp = pd.concat([psm_c[xc].astype(float), pd.get_dummies(psm_c.product_category, drop_first=True, prefix='cat', dtype=float)], axis=1)
R['psm_ctrl_category'] = report(sm.OLS(psm_c.log_sales, sm.add_constant(Xp)).fit(cov_type='cluster', cov_kwds={'groups': psm_c.pair}),
                                "F. PSM + Best Seller + category FE (clustered by pair)", psm)
nos = [v for v in X_VARS if v != 'sponsored']
R['no_sponsored'] = report(fit(prod, nos), "G. Product level without Sponsored", prod)
# alt dedup: product means across scrapes
agg = df.groupby('pid').agg({**{c: 'mean' for c in ['log_sales', 'log_price', 'log_reviews', 'rating']},
                             **{c: 'max' for c in ['is_green', 'coupon', 'sponsored']},
                             'discounted_price': 'median', 'total_reviews': 'median'}).reset_index()
agg = add_inter(agg)
R['product_mean_agg'] = report(fit(agg, X_VARS), "H. Product-mean aggregation (alt. dedup rule)", agg)

section("6. OUTCOME MEASUREMENT: ordered logit on sales buckets (product level)")
lev = sorted(prod.purchased_last_month.unique())
yb = pd.Categorical(prod.purchased_last_month, categories=lev, ordered=True)
# collapse very sparse top buckets
yc = prod.purchased_last_month.clip(upper=10000)
yb = pd.Categorical(yc, categories=sorted(yc.unique()), ordered=True)
om = OrderedModel(yb, prod[X_VARS].astype(float), distr='logit').fit(method='bfgs', disp=0, maxiter=2000)
print(pd.DataFrame({'b': om.params[X_VARS], 'se': om.bse[X_VARS], 'p': om.pvalues[X_VARS]}).round(4).to_string())
R['ordered_logit'] = {'params': om.params[X_VARS].to_dict(), 'p': om.pvalues[X_VARS].to_dict()}
print("  left-truncation: min observed bucket =", int(prod.purchased_last_month.min()),
      "; share at 50 =", round((prod.purchased_last_month == 50).mean() * 100, 1), "%")

# ---------------------------------------------------------------- 7. ML validation
section("7. ML VALIDATION: nested CV (outer 5-fold, inner 3-fold randomized search)")
def metrics(y, p): return r2_score(y, p), np.sqrt(mean_squared_error(y, p)), mean_absolute_error(y, p)
rf_grid = {'n_estimators': [200, 400], 'max_depth': [None, 8, 12, 16], 'min_samples_leaf': [1, 3, 5, 10],
           'max_features': [0.5, 0.8, 1.0]}
xgb_grid = {'n_estimators': [200, 400, 800], 'max_depth': [3, 4, 6], 'learning_rate': [0.02, 0.05, 0.1],
            'subsample': [0.7, 0.9, 1.0], 'colsample_bytree': [0.7, 1.0], 'min_child_weight': [1, 5, 10]}
def nested(d, feats_lin, feats_ml, groups=None, n_iter=12):
    y = d.log_sales.values
    outer = GroupKFold(5) if groups is not None else KFold(5, shuffle=True, random_state=SEED)
    split = outer.split(d, y, groups) if groups is not None else outer.split(d)
    res = {k: [] for k in ['OLS', 'Ridge', 'RandomForest', 'XGBoost']}; best = {'RandomForest': [], 'XGBoost': []}
    for tr, te in split:
        dtr, dte = d.iloc[tr], d.iloc[te]
        for name, mdl, fs in [('OLS', LinearRegression(), feats_lin),
                              ('Ridge', RidgeCV(alphas=np.logspace(-3, 3, 13)), feats_lin)]:
            mdl.fit(dtr[fs], dtr.log_sales); res[name].append(metrics(dte.log_sales, mdl.predict(dte[fs])))
        inner = KFold(3, shuffle=True, random_state=SEED)
        for name, base, grid in [('RandomForest', RandomForestRegressor(random_state=SEED, n_jobs=-1), rf_grid),
                                 ('XGBoost', XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0), xgb_grid)]:
            s = RandomizedSearchCV(base, grid, n_iter=n_iter, cv=inner, scoring='neg_mean_squared_error',
                                   random_state=SEED, n_jobs=1).fit(dtr[feats_ml], dtr.log_sales)
            res[name].append(metrics(dte.log_sales, s.predict(dte[feats_ml]))); best[name].append(s.best_params_)
    tab = pd.DataFrame({k: np.mean(v, 0) for k, v in res.items()}, index=['R2', 'RMSE', 'MAE']).T
    sd = pd.DataFrame({k: np.std(v, 0) for k, v in res.items()}, index=['R2_sd', 'RMSE_sd', 'MAE_sd']).T
    return pd.concat([tab, sd], axis=1).round(4), best
t4, best = nested(prod, X_VARS, ML_FEATS)
print("\n  Product level (primary), out-of-sample mean over 5 outer folds:"); print(t4.to_string())
print("  Selected hyper-parameters per fold:")
for k, v in best.items():
    for i, b in enumerate(v): print(f"    {k} fold{i+1}: {b}")
OUT['ml_product'] = t4.to_dict()

print("\n  Leakage demonstration at listing level (default hyper-parameters, 5-fold):")
rows = []
for label, cv, g in [('random KFold (as submitted)', KFold(5, shuffle=True, random_state=SEED), None),
                     ('GroupKFold by product', GroupKFold(5), df.pid.values)]:
    for name, mdl, fs in [('OLS', LinearRegression(), X_VARS),
                          ('RandomForest', RandomForestRegressor(n_estimators=150, random_state=SEED, n_jobs=-1), ML_FEATS),
                          ('XGBoost', XGBRegressor(n_estimators=150, learning_rate=0.1, random_state=SEED, n_jobs=-1, verbosity=0), ML_FEATS)]:
        sc = cross_validate(mdl, df[fs], df.log_sales, cv=cv, groups=g, scoring=('r2',))
        rows.append([label, name, sc['test_r2'].mean()])
lk = pd.DataFrame(rows, columns=['CV scheme', 'Model', 'R2']).round(4); print(lk.to_string(index=False))
OUT['leakage'] = lk.to_dict('records')

# ---------------------------------------------------------------- 8. SHAP
section("8. TreeSHAP on tuned XGBoost (product level, all products)")
import shap
from collections import Counter
bp = Counter(tuple(sorted(b.items())) for b in best['XGBoost']).most_common(1)[0][0]
xgb = XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0, **dict(bp)).fit(prod[ML_FEATS], prod.log_sales)
expl = shap.TreeExplainer(xgb)
sv = expl.shap_values(prod[ML_FEATS])
imp = pd.Series(np.abs(sv).mean(0), index=ML_FEATS).sort_values(ascending=False)
print("  mean|SHAP| importance:"); print(imp.round(4).to_string())
gain = pd.Series(xgb.get_booster().get_score(importance_type='gain')); print("  gain share:"); print((gain / gain.sum()).round(3).sort_values(ascending=False).to_string())
g = prod.is_green.values == 1
bi = ML_FEATS.index('is_green')
dep = pd.DataFrame({'price': prod.discounted_price.values[g], 'shap_badge': sv[g, bi]})
dep['bin'] = pd.qcut(dep.price, 6)
print("\n  Badge SHAP among badged products by price sextile:")
print(dep.groupby('bin', observed=True).agg(n=('price', 'size'), mean_price=('price', 'mean'),
      mean_shap=('shap_badge', 'mean')).round(4).to_string())
print(f"  Mean badge SHAP: badged={sv[g, bi].mean():.4f}, non-badged={sv[~g, bi].mean():.4f}, all={sv[:, bi].mean():.4f}")
# counterfactual toggle
X1, X0 = prod[ML_FEATS].copy(), prod[ML_FEATS].copy(); X1['is_green'] = 1; X0['is_green'] = 0
cf = xgb.predict(X1) - xgb.predict(X0)
cfd = pd.DataFrame({'price': prod.discounted_price, 'cf': cf})
cfd['bin'] = pd.qcut(cfd.price, 8)
print("\n  XGB counterfactual badge effect (pred(badge=1)-pred(badge=0)), all products, by price octile:")
print(cfd.groupby('bin', observed=True).cf.agg(['mean', 'size']).round(4).to_string())
# how the SHAP interaction captures price moderation
siv = expl.shap_interaction_values(prod[ML_FEATS].iloc[np.where(g)[0]])
pi = ML_FEATS.index('log_price')
print(f"  Mean |SHAP interaction| badge x price among badged: {np.abs(siv[:, bi, pi]).mean():.4f}")
OUT['shap_importance'] = imp.to_dict()

json.dump({'flow': OUT['flow'], 'psm': OUT['psm'], 'regressions': R, 'ml_product': OUT['ml_product'],
           'leakage': OUT['leakage'], 'shap_importance': OUT['shap_importance']},
          open('revision_results.json', 'w'), indent=2, default=float)
print("\nSaved revision_results.json")
