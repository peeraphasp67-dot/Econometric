import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import i18n

# -----------------------------------------------------------------------------
# 0. LANGUAGE STATE — forced default, before any widget (incl. set_page_config)
# -----------------------------------------------------------------------------
# Every fresh session MUST start in English. setdefault() only writes when the
# key is absent, so an explicit user selection later in the same session is
# never overwritten by this line on subsequent reruns.
st.session_state.setdefault("current_lang", i18n.DEFAULT_LANG)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & ENTERPRISE DESIGN CSS
# -----------------------------------------------------------------------------
# Widget keys below are read from session_state here (before the widgets that
# own them are instantiated further down) so the browser tab title can reflect
# the active dataset on the very first rerun after a file is uploaded.
FILE_UPLOAD_KEY = "pricelens_uploaded_file"
USE_DEMO_KEY = "pricelens_use_demo"

_prev_upload = st.session_state.get(FILE_UPLOAD_KEY)
_prev_use_demo = st.session_state.get(USE_DEMO_KEY, True)
if _prev_upload is not None and not _prev_use_demo:
    _dynamic_page_title = f"PriceLens — {_prev_upload.name}"
else:
    _dynamic_page_title = "PriceLens — Demo Amazon Dataset"

st.set_page_config(
    page_title=_dynamic_page_title,
    layout="wide"
)

# Custom CSS — "High-End SaaS Dashboard" design system (Electric Blue accent)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* 1. Global Typography */
    body, p, label, input, select, textarea, .stMarkdown {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        line-height: 1.7 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        line-height: 1.5 !important;
        padding-top: 4px !important;
        padding-bottom: 4px !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        overflow: visible !important;
        color: #0F172A !important;
    }

    .stMarkdown p, .stMarkdown span {
        line-height: 1.75 !important;
    }

    /* 2. Main Content Surface — light slate background for card contrast */
    [data-testid="stAppViewContainer"] {
        background-color: #F1F5F9 !important;
    }
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    .block-container {
        padding-top: 2.5rem !important;
        max-width: 1280px;
    }

    /* 3. Hide Stray Icon Text Glitches at Sidebar Top */
    [data-testid="stSidebarHeader"] button span {
        display: none !important;
    }
    [data-testid="stSidebar"] button[title*="collapse"] span {
        visibility: hidden !important;
    }

    /* 4. Dark Navy Sidebar (SaaS nav-rail) with Electric Blue accents */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #1E293B !important;
    }

    /* 5. Force Readable White Text for Radio Buttons & Sidebar Labels */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stRadioButton"] label p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }

    /* 6. Selectbox (Dropdown) on Sidebar */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] * {
        color: #0F172A !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }
    [data-testid="stSidebar"] [data-testid="stCheckbox"] label p {
        color: #E2E8F0 !important;
    }

    /* Dropdown Popover Menu */
    div[data-baseweb="popover"] div[role="listbox"],
    div[data-baseweb="popover"] ul {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px rgba(15,23,42,.10) !important;
    }
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] [role="option"],
    div[data-baseweb="popover"] [role="option"] * {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        font-size: 0.95rem !important;
    }
    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="popover"] [role="option"]:hover,
    div[data-baseweb="popover"] [role="option"]:hover * {
        background-color: #EFF6FF !important;
        color: #2563EB !important;
        font-weight: 600 !important;
    }

    /* 7. Dashed Drop-Zone File Uploader (sidebar) — selector repeated to win
       specificity battles against Streamlit's own built-in dropzone border */
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"][data-testid="stFileUploaderDropzone"] {
        background-color: #1E293B !important;
        border-width: 2px !important;
        border-style: dashed !important;
        border-color: #3B82F6 !important;
        border-radius: 16px !important;
        padding: 22px 16px !important;
        transition: all 0.2s ease-in-out !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"][data-testid="stFileUploaderDropzone"]:hover {
        background-color: #24324a !important;
        border-color: #60A5FA !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] svg {
        fill: #60A5FA !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] div {
        color: #E2E8F0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        color: #E2E8F0 !important;
    }

    /* Schema teaching card (empty state) */
    .schema-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 14px 16px;
        margin-top: 12px;
        margin-bottom: 4px;
    }
    .schema-card .schema-title {
        color: #60A5FA;
        font-weight: 700;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .schema-card ul {
        margin: 0;
        padding-left: 18px;
    }
    .schema-card li {
        color: #CBD5E1;
        font-size: 0.82rem;
        line-height: 1.6;
    }
    .schema-card li b {
        color: #F8FAFC;
    }

    /* 8. Flat KPI Metric Cards + Top Accent Bar (no shadow) */
    [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-top: 4px solid #2563EB !important;
        border-radius: 16px !important;
        padding: 20px 22px !important;
        box-shadow: none !important;
        transition: border-color 0.2s ease-in-out !important;
    }
    [data-testid="stColumn"]:nth-of-type(2) [data-testid="stMetric"] {
        border-top-color: #7C3AED !important;
    }
    [data-testid="stColumn"]:nth-of-type(3) [data-testid="stMetric"] {
        border-top-color: #D97706 !important;
    }
    [data-testid="stColumn"]:nth-of-type(4) [data-testid="stMetric"] {
        border-top-color: #059669 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        color: #64748B !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        letter-spacing: -0.02em !important;
    }

    /* Blue for Delta / Subtext (was green) */
    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] *,
    [data-testid="stMetricDelta"] div,
    [data-testid="stMetricDelta"] span {
        font-weight: 700 !important;
        color: #2563EB !important;
    }
    [data-testid="stMetricDelta"] svg {
        fill: #2563EB !important;
        color: #2563EB !important;
    }

    @media (prefers-color-scheme: dark) {
        [data-testid="stMetric"] {
            background-color: #1E293B !important;
            border-left-color: #334155 !important;
            border-right-color: #334155 !important;
            border-bottom-color: #334155 !important;
        }
        [data-testid="stMetricValue"] {
            color: #F8FAFC !important;
        }
        [data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
        }
        [data-testid="stMetricDelta"],
        [data-testid="stMetricDelta"] *,
        [data-testid="stMetricDelta"] div,
        [data-testid="stMetricDelta"] span {
            color: #60A5FA !important;
        }
        [data-testid="stMetricDelta"] svg {
            fill: #60A5FA !important;
            color: #60A5FA !important;
        }
    }

    .stCaption,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p,
    .stCaption p {
        color: #475569 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    /* 9. Segmented Control Tab Navigation (ARIA-role selectors — stable across
       Streamlit/BaseWeb versions, unlike data-baseweb attributes) */
    .stTabs [role="tablist"] {
        gap: 4px;
        background-color: #E2E8F0;
        border-bottom: none !important;
        border-radius: 14px;
        padding: 5px;
        width: fit-content;
    }
    .stTabs [role="tab"] {
        padding: 10px 20px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #475569 !important;
        border-radius: 10px !important;
        background-color: transparent;
        transition: all 0.15s ease-in-out;
    }
    .stTabs [role="tab"] p {
        color: inherit !important;
        font-weight: 600 !important;
    }
    .stTabs [role="tab"]:hover {
        color: #2563EB !important;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #2563EB !important;
        box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 4px 10px rgba(15,23,42,.08) !important;
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Callout Card Boxes */
    .hypothesis-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        padding: 16px 20px;
        border-radius: 12px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(15,23,42,.03);
    }
    .elm-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px rgba(15,23,42,.05);
    }
    .engine-badge {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        padding: 7px 16px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 14px;
    }
    .data-source-badge {
        display: inline-block;
        background-color: #F5F3FF;
        color: #6D28D9;
        border: 1px solid #DDD6FE;
        padding: 7px 16px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


GREEN_TAGS_DEFAULT = [
    'Carbon impact',
    'Energy efficiency',
    'Manufacturing practices',
    'Forestry practices',
    'Recycled materials'
]

DEMO_MAPPING = {
    "target": "purchased_last_month",
    "treatment": "sustainability_tags",
    "price": "discounted_price",
    "reviews": "total_reviews",
    "rating": "product_rating",
    "coupon": "has_coupon",
    "sponsored": "is_sponsored",
    "category": "product_category",
}

MODEL_OPTIONS_ORDER = ["ols", "ridge", "rf", "xgboost"]
MODEL_CLEAN_NAMES = {
    "ols": "OLS Regression",
    "ridge": "Ridge Regression",
    "rf": "Random Forest Regressor",
    "xgboost": "XGBoost Regressor"
}

X_VARS = [
    'is_green_binary',
    'log_price',
    'interaction_price_green',
    'log_reviews',
    'interaction_reviews_green',
    'has_coupon_binary',
    'interaction_coupon_green',
    'product_rating',
    'is_sponsored_binary'
]

# Modern SaaS Interactive chart theme — Electric Blue colorway on light, card-like tooltips
PLOTLY_FONT = dict(family="'Inter', sans-serif", size=12, color="#334155")
PLOTLY_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'scrollZoom': True
}
CHART_COLORWAY = ["#2563EB", "#7C3AED", "#D97706", "#059669", "#DB2777", "#0891B2"]
CHART_GRIDCOLOR = 'rgba(100,116,139,0.14)'
CHART_HOVERLABEL = dict(bgcolor='#FFFFFF', bordercolor='#E2E8F0', font=dict(family="'Inter', sans-serif", size=12, color='#0F172A'))


LANG_KEY = "current_lang"


def t(key):
    """Retrieve a UI string in the active language, falling back to English."""
    return i18n.t(key, st.session_state.get(LANG_KEY, i18n.DEFAULT_LANG))


def get_sig_stars(p_val):
    """Return significance stars based on p-value threshold."""
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    elif p_val < 0.1:
        return "."
    else:
        return "ns"


def fmt_p(p_val):
    if p_val is None or not np.isfinite(p_val):
        return "p = N/A"
    return "p < 0.001" if p_val < 0.001 else f"p = {p_val:.4f}"


# -----------------------------------------------------------------------------
# 3B. LANGUAGE SELECTOR (top of sidebar, before everything else)
# -----------------------------------------------------------------------------
# current_lang was already forced to i18n.DEFAULT_LANG ("en") via setdefault()
# at the very top of the script if this is a fresh session, so the index
# computed here always lands on "🇺🇸 English" the first time the app loads.
_current_lang_value = st.session_state.get(LANG_KEY, i18n.DEFAULT_LANG)
if _current_lang_value not in i18n.LANGUAGE_CODES:
    _current_lang_value = i18n.DEFAULT_LANG

st.sidebar.selectbox(
    i18n.t("lang_selector_label", _current_lang_value),
    options=i18n.LANGUAGE_CODES,
    format_func=i18n.language_label,
    index=i18n.LANGUAGE_CODES.index(_current_lang_value),
    key=LANG_KEY,
)
active_lang = st.session_state.get(LANG_KEY, i18n.DEFAULT_LANG)

# RTL is applied STRICTLY and ONLY when Arabic is the explicitly active
# language; every other language (including on first load) gets an explicit
# LTR reset so no RTL styling can ever leak from a prior state/rerun.
if active_lang == "ar" and i18n.is_rtl(active_lang):
    st.markdown(i18n.RTL_CSS, unsafe_allow_html=True)
else:
    st.markdown(i18n.LTR_CSS, unsafe_allow_html=True)

st.sidebar.markdown("---")

# -----------------------------------------------------------------------------
# 4. DATA SOURCE: DEMO DATASET OR USER UPLOAD
# -----------------------------------------------------------------------------
st.sidebar.title(t('data_source_title'))

uploaded_file = st.sidebar.file_uploader(
    t('upload_label'),
    type=["csv", "xlsx"],
    help=t('upload_help'),
    key=FILE_UPLOAD_KEY
)
use_demo = st.sidebar.checkbox(t('use_demo_toggle'), value=(uploaded_file is None), key=USE_DEMO_KEY)

# Empty-state schema teaching card — shown while the demo dataset is active,
# so a new user knows exactly which column roles their own file should have.
if use_demo or uploaded_file is None:
    _schema_rows = [
        (t('schema_target'), t('required_tag')),
        (t('schema_treatment'), t('required_tag')),
        (t('schema_price'), t('required_tag')),
        (t('schema_reviews'), t('optional_tag')),
        (t('schema_rating'), t('optional_tag')),
        (t('schema_coupon'), t('optional_tag')),
        (t('schema_sponsored'), t('optional_tag')),
        (t('schema_category'), t('optional_tag')),
    ]
    _schema_items = "".join(f"<li>{label} <i>({tag})</i></li>" for label, tag in _schema_rows)
    st.sidebar.markdown(
        f"""
        <div class="schema-card">
            <div class="schema-title">{t('schema_card_title')}</div>
            <ul>{_schema_items}</ul>
        </div>
        """,
        unsafe_allow_html=True
    )


@st.cache_data
def load_demo_raw():
    return pd.read_csv('data/amazon_products_sales_data_cleaned.csv')


@st.cache_data
def load_uploaded_raw(file_bytes, file_name):
    import io
    buf = io.BytesIO(file_bytes)
    if file_name.lower().endswith(".xlsx"):
        return pd.read_excel(buf)
    return pd.read_csv(buf)


is_demo_active = use_demo or uploaded_file is None
raw_df = None
data_source_error = None

if is_demo_active:
    try:
        raw_df = load_demo_raw()
    except Exception as e:
        data_source_error = f"Unable to read demo data file: {e}"
else:
    try:
        raw_df = load_uploaded_raw(uploaded_file.getvalue(), uploaded_file.name)
    except Exception as e:
        data_source_error = f"Unable to read uploaded file: {e}"

if data_source_error:
    st.error(data_source_error)
    st.stop()
if raw_df is None or len(raw_df) == 0:
    st.error("The active dataset is empty.")
    st.stop()

# Dynamic branding: reflect the active dataset in the page title and header
active_dataset_label = t('demo_dataset_label') if is_demo_active else uploaded_file.name

st.title(f"PriceLens — {active_dataset_label}")
st.markdown(f'<p style="color: #333333; font-size: 1.05rem; font-weight: 500; margin-top: -10px; margin-bottom: 20px;">{t("header_subtitle")}</p>', unsafe_allow_html=True)
st.markdown(
    f'<div class="data-source-badge">{t("active_dataset_badge").format(label=active_dataset_label, n=f"{len(raw_df):,}")}</div>',
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 5. COLUMN MAPPING & VALIDATION
# -----------------------------------------------------------------------------
mapping = {}
optional_missing_roles = []

if is_demo_active:
    mapping = dict(DEMO_MAPPING)
    treated_values = list(GREEN_TAGS_DEFAULT)
    coupon_positive_values = None   # numeric/text auto rule handled below (demo uses legacy rule)
    sponsored_positive_values = None
else:
    st.sidebar.markdown("---")
    st.sidebar.subheader(t('mapping_title'))
    st.sidebar.caption(t('mapping_help'))

    cols = list(raw_df.columns)
    none_opt = t('map_none')

    def col_picker(label, key, optional=False):
        options = ([none_opt] if optional else []) + cols
        return st.sidebar.selectbox(label, options, key=f"map_{key}")

    mapping["target"] = col_picker(t('map_target'), "target")
    mapping["treatment"] = col_picker(t('map_treatment'), "treatment")
    mapping["price"] = col_picker(t('map_price'), "price")
    mapping["reviews"] = col_picker(t('map_reviews'), "reviews", optional=True)
    mapping["rating"] = col_picker(t('map_rating'), "rating", optional=True)
    mapping["coupon"] = col_picker(t('map_coupon'), "coupon", optional=True)
    mapping["sponsored"] = col_picker(t('map_sponsored'), "sponsored", optional=True)
    mapping["category"] = col_picker(t('map_category'), "category", optional=True)

    for role in ["reviews", "rating", "coupon", "sponsored", "category"]:
        if mapping.get(role) == none_opt:
            mapping[role] = None
            optional_missing_roles.append(role)

    treated_values = []
    if mapping["treatment"]:
        treat_series = raw_df[mapping["treatment"]]
        if pd.api.types.is_bool_dtype(treat_series):
            treated_values = [True]
        elif pd.api.types.is_numeric_dtype(treat_series):
            uniq_vals = sorted(treat_series.dropna().unique().tolist())
            treated_values = st.sidebar.multiselect(t('map_treated_values'), uniq_vals, default=uniq_vals[-1:] if uniq_vals else [])
        else:
            uniq_vals = sorted(treat_series.dropna().astype(str).unique().tolist())[:200]
            treated_values = st.sidebar.multiselect(t('map_treated_values'), uniq_vals)

    coupon_positive_values = None
    if mapping.get("coupon"):
        s = raw_df[mapping["coupon"]]
        if not (pd.api.types.is_bool_dtype(s) or (pd.api.types.is_numeric_dtype(s) and set(s.dropna().unique()).issubset({0, 1}))):
            uniq_vals = sorted(s.dropna().astype(str).unique().tolist())[:200]
            mode_val = s.astype(str).mode().iloc[0] if len(s.dropna()) else None
            default_vals = [v for v in uniq_vals if v != mode_val]
            coupon_positive_values = st.sidebar.multiselect(
                f"{t('map_positive_values')} — {mapping['coupon']}", uniq_vals, default=default_vals, key="coupon_pos"
            )

    sponsored_positive_values = None
    if mapping.get("sponsored"):
        s = raw_df[mapping["sponsored"]]
        if not (pd.api.types.is_bool_dtype(s) or (pd.api.types.is_numeric_dtype(s) and set(s.dropna().unique()).issubset({0, 1}))):
            uniq_vals = sorted(s.dropna().astype(str).unique().tolist())[:200]
            mode_val = s.astype(str).mode().iloc[0] if len(s.dropna()) else None
            default_vals = [v for v in uniq_vals if v != mode_val]
            sponsored_positive_values = st.sidebar.multiselect(
                f"{t('map_positive_values')} — {mapping['sponsored']}", uniq_vals, default=default_vals, key="sponsored_pos"
            )

# Required-field validation
required_ok = bool(mapping.get("target")) and bool(mapping.get("treatment")) and bool(mapping.get("price"))
if not is_demo_active and not required_ok:
    st.error(t('err_missing_required'))
    st.stop()
if not is_demo_active and len(treated_values) == 0:
    st.error(t('err_no_treated_values'))
    st.stop()

for role in optional_missing_roles:
    st.sidebar.warning(t('warn_optional_missing').format(role=role))

# -----------------------------------------------------------------------------
# 6. BUILD CANONICAL DATAFRAME (preprocessing, log-transforms, interactions)
# -----------------------------------------------------------------------------


def _binarize_generic(series, positive_values):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0) != 0).astype(int)
    if positive_values:
        return series.astype(str).apply(lambda x: 1 if x in positive_values else 0)
    return pd.Series(0, index=series.index)


@st.cache_data(show_spinner=False)
def build_canonical_df(raw, mapping_dict, treated_vals, coupon_pos, sponsored_pos, is_demo):
    warnings = []
    df_clean = raw.copy()

    # ---- Target ----
    target_series = pd.to_numeric(df_clean[mapping_dict["target"]], errors="coerce")
    n_bad_target = target_series.isna().sum() - df_clean[mapping_dict["target"]].isna().sum()

    # ---- Price ----
    price_series = pd.to_numeric(df_clean[mapping_dict["price"]], errors="coerce")
    n_bad_price = price_series.isna().sum() - df_clean[mapping_dict["price"]].isna().sum()

    dtype_errors = []
    if n_bad_target > 0:
        dtype_errors.append(("target", mapping_dict["target"], int(n_bad_target)))
    if n_bad_price > 0:
        dtype_errors.append(("price", mapping_dict["price"], int(n_bad_price)))

    df_clean["purchased_last_month"] = target_series
    df_clean["discounted_price"] = price_series

    # ---- Treatment ----
    treat_raw_col = df_clean[mapping_dict["treatment"]]
    if is_demo:
        df_clean["sustainability_tags"] = treat_raw_col
        df_clean["is_green_binary"] = treat_raw_col.apply(
            lambda x: 1 if any(tg in str(x) for tg in treated_vals) else 0
        )
    else:
        df_clean["sustainability_tags"] = treat_raw_col.astype(str)
        if pd.api.types.is_bool_dtype(treat_raw_col) or pd.api.types.is_numeric_dtype(treat_raw_col):
            df_clean["is_green_binary"] = treat_raw_col.apply(lambda x: 1 if x in treated_vals else 0)
        else:
            df_clean["is_green_binary"] = treat_raw_col.astype(str).apply(
                lambda x: 1 if any(str(tv) in x for tv in treated_vals) else 0
            )

    # ---- Optional: reviews ----
    if mapping_dict.get("reviews"):
        reviews_series = pd.to_numeric(df_clean[mapping_dict["reviews"]], errors="coerce")
        n_bad = reviews_series.isna().sum() - df_clean[mapping_dict["reviews"]].isna().sum()
        if n_bad > 0:
            dtype_errors.append(("reviews", mapping_dict["reviews"], int(n_bad)))
        df_clean["total_reviews"] = reviews_series.fillna(0)
    else:
        df_clean["total_reviews"] = 0.0
        warnings.append(("optional_missing", "Social Proof / Reviews"))

    # ---- Optional: rating ----
    if mapping_dict.get("rating"):
        rating_series = pd.to_numeric(df_clean[mapping_dict["rating"]], errors="coerce")
        n_bad = rating_series.isna().sum() - df_clean[mapping_dict["rating"]].isna().sum()
        if n_bad > 0:
            dtype_errors.append(("rating", mapping_dict["rating"], int(n_bad)))
        fallback_rating = rating_series.mean() if rating_series.notna().any() else 4.0
        df_clean["product_rating"] = rating_series.fillna(fallback_rating)
    else:
        df_clean["product_rating"] = 4.0
        warnings.append(("optional_missing", "Rating"))

    # ---- Optional: coupon ----
    if mapping_dict.get("coupon"):
        df_clean["has_coupon_binary"] = _binarize_generic(df_clean[mapping_dict["coupon"]], coupon_pos)
    else:
        df_clean["has_coupon_binary"] = 0
        warnings.append(("optional_missing", "Promo / Coupon"))

    # ---- Optional: sponsored ----
    if mapping_dict.get("sponsored"):
        df_clean["is_sponsored_binary"] = _binarize_generic(df_clean[mapping_dict["sponsored"]], sponsored_pos)
    else:
        df_clean["is_sponsored_binary"] = 0
        warnings.append(("optional_missing", "Sponsored / Ad"))

    # ---- Optional: category ----
    if mapping_dict.get("category"):
        df_clean["product_category"] = df_clean[mapping_dict["category"]].astype(str)
    else:
        df_clean["product_category"] = "All"

    # ---- Drop rows with missing core numeric fields ----
    df_clean = df_clean.dropna(subset=["purchased_last_month", "discounted_price"]).copy()

    if len(df_clean) == 0:
        return None, dtype_errors, warnings, 0, 0

    # ---- Handle negative / zero values before log transform ----
    neg_price = int((df_clean["discounted_price"] < 0).sum())
    neg_target = int((df_clean["purchased_last_month"] < 0).sum())
    df_clean["discounted_price"] = df_clean["discounted_price"].clip(lower=0)
    df_clean["purchased_last_month"] = df_clean["purchased_last_month"].clip(lower=0)
    df_clean["total_reviews"] = df_clean["total_reviews"].clip(lower=0)

    # ---- Log transforms (log1p handles zeros/skewness) ----
    df_clean["log_purchased"] = np.log1p(df_clean["purchased_last_month"])
    df_clean["log_price"] = np.log1p(df_clean["discounted_price"])
    df_clean["log_reviews"] = np.log1p(df_clean["total_reviews"])

    # ---- Interaction terms (H1, H2, H3) ----
    df_clean["interaction_price_green"] = df_clean["log_price"] * df_clean["is_green_binary"]
    df_clean["interaction_reviews_green"] = df_clean["log_reviews"] * df_clean["is_green_binary"]
    df_clean["interaction_coupon_green"] = df_clean["has_coupon_binary"] * df_clean["is_green_binary"]

    return df_clean, dtype_errors, warnings, neg_price, neg_target


canonical_result = build_canonical_df(
    raw_df, mapping, tuple(treated_values),
    tuple(coupon_positive_values) if coupon_positive_values else None,
    tuple(sponsored_positive_values) if sponsored_positive_values else None,
    is_demo_active
)
df_clean, dtype_errors, build_warnings, neg_price_n, neg_target_n = canonical_result

if dtype_errors:
    for role, col, n_bad in dtype_errors:
        st.error(t('err_dtype').format(col=col, role=role, n_bad=n_bad))
    st.stop()

if df_clean is None or len(df_clean) == 0:
    st.error(t('err_empty_after_clean'))
    st.stop()

if neg_price_n > 0 and not is_demo_active:
    st.sidebar.warning(t('warn_negative_values').format(col=mapping.get("price")))
if neg_target_n > 0 and not is_demo_active:
    st.sidebar.warning(t('warn_negative_values').format(col=mapping.get("target")))

if len(df_clean) < 30:
    st.warning(t('warn_small_sample_generic').format(n=len(df_clean)))

df = df_clean

# -----------------------------------------------------------------------------
# 7. PROPENSITY SCORE MATCHING (with fallback for small samples)
# -----------------------------------------------------------------------------
PSM_COVARIATES = ['log_price', 'log_reviews', 'product_rating', 'is_sponsored_binary', 'has_coupon_binary']


def standardized_diff(a, b):
    """Mean Absolute Standardized Difference (MASD) between two covariate samples."""
    a = a.dropna()
    b = b.dropna()
    if len(a) == 0 or len(b) == 0:
        return 0.0
    pooled_std = np.sqrt((a.var() + b.var()) / 2)
    if pooled_std == 0 or not np.isfinite(pooled_std):
        return 0.0
    return abs((a.mean() - b.mean()) / pooled_std)


@st.cache_data(show_spinner=False)
def run_psm(df_in, covariates, data_key):
    treated = df_in[df_in['is_green_binary'] == 1].copy()
    control = df_in[df_in['is_green_binary'] == 0].copy()

    min_group = min(len(treated), len(control))
    if min_group < 5:
        return df_in.copy(), False

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import NearestNeighbors

        X_cov = df_in[covariates].fillna(0)
        ps_model = LogisticRegression(random_state=42, max_iter=1000).fit(X_cov, df_in['is_green_binary'])
        df_in = df_in.copy()
        df_in['pscore'] = ps_model.predict_proba(X_cov)[:, 1]

        treated = df_in[df_in['is_green_binary'] == 1].copy()
        control = df_in[df_in['is_green_binary'] == 0].copy()

        cap = min(len(treated), 5000)
        treated_sample = treated.sample(cap, random_state=42) if len(treated) > cap else treated

        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        nn.fit(control[['pscore']])
        distances, indices = nn.kneighbors(treated_sample[['pscore']])

        matched_control = control.iloc[indices.flatten()].copy()
        psm_df = pd.concat([treated_sample, matched_control]).reset_index(drop=True)
        return psm_df, True
    except Exception:
        return df_in.copy(), False


with st.spinner(t('spinner_psm')):
    _data_key = f"{len(df)}_{int(df['is_green_binary'].sum())}_{hash(tuple(df.columns))}"
    psm_df, psm_matched = run_psm(df, PSM_COVARIATES, _data_key)

if not psm_matched:
    st.sidebar.warning(t('warn_psm_fallback'))

# -----------------------------------------------------------------------------
# 7B. DATASET HEALTH & SCARCITY DIAGNOSTICS (shown for user-uploaded datasets)
# -----------------------------------------------------------------------------
if not is_demo_active:
    _raw_treated_full = df[df['is_green_binary'] == 1]
    _raw_control_full = df[df['is_green_binary'] == 0]
    _matched_treated_full = psm_df[psm_df['is_green_binary'] == 1]
    _matched_control_full = psm_df[psm_df['is_green_binary'] == 0]

    _raw_masd_vals = [standardized_diff(_raw_treated_full[c], _raw_control_full[c]) for c in PSM_COVARIATES]
    _mean_masd_raw = float(np.mean(_raw_masd_vals)) if _raw_masd_vals else 0.0
    if psm_matched:
        _matched_masd_vals = [standardized_diff(_matched_treated_full[c], _matched_control_full[c]) for c in PSM_COVARIATES]
        _mean_masd_matched = float(np.mean(_matched_masd_vals)) if _matched_masd_vals else _mean_masd_raw
    else:
        _mean_masd_matched = _mean_masd_raw

    _scarcity_pct = (df['is_green_binary'].mean() * 100) if len(df) > 0 else 0.0
    _masd_ok = _mean_masd_matched < 0.10

    st.markdown(f"##### 🩺 {t('dataset_health_title')}")
    st.markdown(f'<p style="color: #4A5568; font-size: 13px; margin-top: -6px; margin-bottom: 10px;">{t("dataset_health_desc")}</p>', unsafe_allow_html=True)

    _health_col1, _health_col2 = st.columns(2)
    _health_col1.metric(t('dataset_health_scarcity_label'), f"{_scarcity_pct:.1f}%")
    _health_col2.metric(
        t('dataset_health_masd_label'),
        t('dataset_health_masd_pass') if _masd_ok else t('dataset_health_masd_fail'),
        t('dataset_health_masd_detail').format(before=_mean_masd_raw, after=_mean_masd_matched)
    )

    if not psm_matched:
        st.warning(t('warn_psm_fallback'))

    st.markdown("---")

# -----------------------------------------------------------------------------
# 8. SIDEBAR FILTERS & DYNAMIC MODEL SELECTION
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.title(t('sidebar_title'))

if mapping.get("category") or is_demo_active:
    categories = [t('all_cats')] + sorted(df['product_category'].dropna().unique().tolist())
    selected_category = st.sidebar.selectbox(t('cat_filter'), categories)
else:
    selected_category = t('all_cats')

min_p = float(df['discounted_price'].min())
max_p = float(df['discounted_price'].quantile(0.98)) if df['discounted_price'].nunique() > 1 else float(df['discounted_price'].max()) + 1.0
if max_p <= min_p:
    max_p = min_p + 1.0
price_range = st.sidebar.slider(t('price_filter'), min_p, max_p, (min_p, max_p))

filtered_df = df[(df['discounted_price'] >= price_range[0]) & (df['discounted_price'] <= price_range[1])].copy()
filtered_psm_df = psm_df[(psm_df['discounted_price'] >= price_range[0]) & (psm_df['discounted_price'] <= price_range[1])].copy()

if selected_category != t('all_cats'):
    filtered_df = filtered_df[filtered_df['product_category'] == selected_category]
    filtered_psm_df = filtered_psm_df[filtered_psm_df['product_category'] == selected_category]

st.sidebar.markdown("---")
st.sidebar.subheader(t('predictive_model_header'))

model_options = {
    "ols": t('model_ols'),
    "ridge": t('model_ridge'),
    "rf": t('model_rf'),
    "xgboost": t('model_xgboost')
}

selected_model_key = st.sidebar.selectbox(
    t('select_model_engine'),
    options=MODEL_OPTIONS_ORDER,
    format_func=lambda x: model_options[x],
    key="active_model_engine"
)

active_model_display_name = MODEL_CLEAN_NAMES[selected_model_key]

# -----------------------------------------------------------------------------
# 9. DYNAMIC VARIABLE LABELS
# -----------------------------------------------------------------------------
if is_demo_active:
    VAR_LABELS = {
        'const': t('var_const'),
        'is_green_binary': t('var_is_green_binary'),
        'log_price': t('var_log_price'),
        'log_reviews': t('var_log_reviews'),
        'product_rating': t('var_product_rating'),
        'is_sponsored_binary': t('var_is_sponsored_binary'),
        'has_coupon_binary': t('var_has_coupon_binary'),
        'interaction_price_green': t('var_interaction_price_green'),
        'interaction_reviews_green': t('var_interaction_reviews_green'),
        'interaction_coupon_green': t('var_interaction_coupon_green'),
    }
    TARGET_LABEL = t('demo_target_label')
    TREAT_LABEL = t('var_is_green_binary')
    PRICE_LABEL = t('demo_price_label')
    REVIEWS_LABEL = t('demo_reviews_label')
    COUPON_LABEL = t('demo_coupon_label')
else:
    TARGET_LABEL = mapping["target"]
    TREAT_LABEL = mapping["treatment"]
    PRICE_LABEL = mapping["price"]
    REVIEWS_LABEL = mapping.get("reviews") or "Reviews"
    COUPON_LABEL = mapping.get("coupon") or "Coupon"
    VAR_LABELS = {
        'const': t('var_const'),
        'is_green_binary': f"{TREAT_LABEL} (Treated)",
        'log_price': f"Log({PRICE_LABEL} + 1)",
        'log_reviews': f"Log({REVIEWS_LABEL} + 1)",
        'product_rating': mapping.get("rating") or "Rating",
        'is_sponsored_binary': mapping.get("sponsored") or "Sponsored",
        'has_coupon_binary': COUPON_LABEL,
        'interaction_price_green': f"{TREAT_LABEL} × Log({PRICE_LABEL}) [H1]",
        'interaction_reviews_green': f"{TREAT_LABEL} × Log({REVIEWS_LABEL}) [H2]",
        'interaction_coupon_green': f"{TREAT_LABEL} × {COUPON_LABEL} [H3]",
    }


def translate_var_name(var):
    return VAR_LABELS.get(var, var)


def get_var_group(var_name):
    if var_name == 'const':
        return '—'
    if var_name in ['is_green_binary', 'interaction_price_green', 'interaction_reviews_green', 'interaction_coupon_green']:
        return t('group_green_badges')
    return t('group_model_controls')

# -----------------------------------------------------------------------------
# 10. MODEL TRAINING (live, cached — replaces static pretrained artifacts)
# -----------------------------------------------------------------------------


def _df_hash(frame, cols):
    try:
        return int(pd.util.hash_pandas_object(frame[cols]).sum())
    except Exception:
        return len(frame)


@st.cache_resource(show_spinner=False)
def fit_ols(_X, _y, data_hash):
    # has_constant='add' is required (not the 'skip' default): datasets with
    # several unmapped optional columns can produce multiple constant feature
    # columns (e.g. a fixed rating default), which would otherwise make
    # statsmodels silently skip adding the intercept — desyncing the fitted
    # parameter count from the exog shape used later at prediction time.
    X_const = sm.add_constant(_X, has_constant='add')
    return sm.OLS(_y, X_const).fit()


@st.cache_resource(show_spinner=False)
def fit_ridge(_X, _y, data_hash):
    from sklearn.linear_model import Ridge as SKRidge
    model = SKRidge(alpha=1.0)
    model.fit(_X, _y)
    return model


@st.cache_resource(show_spinner=False)
def fit_tree_model(_X, _y, model_key, data_hash):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(_X, _y, test_size=0.20, random_state=42)
    if model_key == "rf":
        from sklearn.ensemble import RandomForestRegressor as SKRF
        model = SKRF(n_estimators=150, random_state=42, n_jobs=-1)
    else:
        from xgboost import XGBRegressor as SKXGB
        model = SKXGB(n_estimators=150, learning_rate=0.1, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


@st.cache_resource(show_spinner=False)
def compute_shap_values(_model, _X_sample, model_key, data_hash, sample_size):
    import shap
    explainer = shap.TreeExplainer(_model)
    return explainer.shap_values(_X_sample)


def estimate_shap_price_boundary(df_in, model_key="xgboost"):
    """Dynamically estimate the non-linear price inflection boundary for the
    treatment effect, straight from a real TreeSHAP dependence curve fitted on
    the active dataset (demo or user-uploaded) — never hardcoded.

    Returns (boundary_lo, boundary_hi, low_val, high_val, binned_df, r2) or
    (None, None, None, None, None, None) if there isn't enough data/variation.
    """
    if df_in is None or len(df_in) <= len(X_VARS) + 2:
        return None, None, None, None, None, None

    X_full = df_in[X_VARS]
    y_full = df_in['log_purchased']
    data_hash = _df_hash(df_in, X_VARS + ['log_purchased'])

    try:
        tree_model, X_train, X_test, y_train, y_test = fit_tree_model(X_full, y_full, model_key, data_hash)
        from sklearn.metrics import r2_score
        test_r2 = float(r2_score(y_test, tree_model.predict(X_test)))

        sample_size = min(500, len(X_full))
        X_shap_sample = X_full.sample(sample_size, random_state=42) if len(X_full) > sample_size else X_full
        shap_values = compute_shap_values(tree_model, X_shap_sample, model_key, data_hash, sample_size)

        treat_idx = X_VARS.index('is_green_binary')
        treat_shap = shap_values[:, treat_idx]
        price_vals = X_shap_sample['discounted_price'] if 'discounted_price' in X_shap_sample.columns else df_in.loc[X_shap_sample.index, 'discounted_price']

        dep_df = pd.DataFrame({'Price': price_vals.values, 'SHAP': treat_shap}).sort_values('Price')
        if dep_df['Price'].nunique() <= 5:
            return None, None, None, None, None, test_r2

        n_bins = min(20, dep_df['Price'].nunique())
        dep_df['bin'] = pd.qcut(dep_df['Price'], q=n_bins, duplicates='drop')
        binned = dep_df.groupby('bin', observed=True).agg(mean_price=('Price', 'mean'), mean_shap=('SHAP', 'mean')).reset_index()
        if len(binned) <= 2:
            return None, None, None, None, None, test_r2

        binned['shap_delta'] = binned['mean_shap'].diff().abs()
        steepest_idx = binned['shap_delta'].idxmax()
        if steepest_idx is None or steepest_idx <= 0:
            return None, None, None, None, None, test_r2

        boundary_lo = float(binned.loc[steepest_idx - 1, 'mean_price'])
        boundary_hi = float(binned.loc[steepest_idx, 'mean_price'])
        low_val = float(binned['mean_shap'].iloc[0])
        high_val = float(binned['mean_shap'].iloc[-1])
        return boundary_lo, boundary_hi, low_val, high_val, binned, test_r2
    except Exception:
        return None, None, None, None, None, None


@st.cache_resource(show_spinner=False)
def get_headline_ols(_X, _y, data_hash):
    """A stable OLS fit on the active sample used to power dynamic hypothesis text,
    regardless of which engine (OLS/Ridge/RF/XGBoost) is selected for display."""
    # has_constant='add' is required (not the 'skip' default): datasets with
    # several unmapped optional columns can produce multiple constant feature
    # columns (e.g. a fixed rating default), which would otherwise make
    # statsmodels silently skip adding the intercept — desyncing the fitted
    # parameter count from the exog shape used later at prediction time.
    X_const = sm.add_constant(_X, has_constant='add')
    return sm.OLS(_y, X_const).fit()


def predict_scenario(model_key, active_data, price, is_green, rating, reviews_count, coupon=0, sponsored=0):
    """Predict the target using the model trained on the current active sample."""
    X = active_data[X_VARS]
    y = active_data['log_purchased']
    data_hash = _df_hash(active_data, X_VARS + ['log_purchased'])

    log_p = np.log1p(price)
    log_rev = np.log1p(reviews_count)
    row = pd.DataFrame([{
        'is_green_binary': float(is_green),
        'log_price': log_p,
        'interaction_price_green': log_p * is_green,
        'log_reviews': log_rev,
        'interaction_reviews_green': log_rev * is_green,
        'has_coupon_binary': float(coupon),
        'interaction_coupon_green': float(coupon) * is_green,
        'product_rating': float(rating),
        'is_sponsored_binary': float(sponsored),
    }])[X_VARS]

    if model_key == "ols":
        model_fit = fit_ols(X, y, data_hash)
        pred_log = model_fit.predict(sm.add_constant(row, has_constant='add'))[0]
        rmse = np.sqrt(model_fit.mse_resid)
        r2 = model_fit.rsquared
    elif model_key == "ridge":
        model_fit = fit_ridge(X, y, data_hash)
        pred_log = model_fit.predict(row)[0]
        y_pred_all = model_fit.predict(X)
        rmse = float(np.sqrt(np.mean((y - y_pred_all) ** 2)))
        from sklearn.metrics import r2_score
        r2 = float(r2_score(y, y_pred_all))
    else:
        model_fit, X_train, X_test, y_train, y_test = fit_tree_model(X, y, model_key, data_hash)
        pred_log = model_fit.predict(row)[0]
        from sklearn.metrics import r2_score, mean_squared_error
        y_test_pred = model_fit.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
        r2 = float(r2_score(y_test, y_test_pred))

    predicted = float(max(0, np.expm1(pred_log)))
    return predicted, r2, rmse


def get_compare_predictions(active_data, price, is_green, rating, reviews_count, coupon=0, sponsored=0):
    results = []
    for k in MODEL_OPTIONS_ORDER:
        pred, r2, rmse = predict_scenario(k, active_data, price, is_green, rating, reviews_count, coupon, sponsored)
        results.append({
            "model_name": k,
            "display_name": MODEL_CLEAN_NAMES[k],
            "predicted_sales": pred,
            "metrics": {"R2_Score": r2, "RMSE": rmse}
        })
    return results


# ---- Headline OLS stats (used for dynamic hypothesis text & Tab1 crossover metric) ----
_headline_X = filtered_df[X_VARS]
_headline_y = filtered_df['log_purchased']
_headline_hash = _df_hash(filtered_df, X_VARS + ['log_purchased'])
headline_stats = None
if len(filtered_df) > len(X_VARS) + 2:
    try:
        headline_stats = get_headline_ols(_headline_X, _headline_y, _headline_hash)
    except Exception:
        headline_stats = None


def compute_crossover_price(stats):
    """Price at which the treatment's net log-sales contribution flips from negative to positive."""
    if stats is None:
        return None
    try:
        beta_g = stats.params.get('is_green_binary')
        beta_pg = stats.params.get('interaction_price_green')
        if beta_g is None or beta_pg is None or beta_pg == 0:
            return None
        crossover = np.expm1(-beta_g / beta_pg)
        if crossover <= 0 or not np.isfinite(crossover):
            return None
        return float(crossover)
    except Exception:
        return None


crossover_price = compute_crossover_price(headline_stats)

# ---- Dynamic non-linear price boundary, estimated from a real TreeSHAP
# dependence curve on the active dataset (never hardcoded). Computed once
# here with a fixed canonical engine (XGBoost) so the Tab 1 KPI stays stable
# regardless of which model the user later picks in the sidebar; Tab 3 reuses
# this exact function (and the cache) when XGBoost is the active selection.
with st.spinner(t('spinner_shap')):
    shap_boundary_lo, shap_boundary_hi, _shap_low_val, _shap_high_val, _shap_binned, _shap_boundary_r2 = estimate_shap_price_boundary(filtered_df, model_key="xgboost")

# -----------------------------------------------------------------------------
# 11. DASHBOARD TABS
# -----------------------------------------------------------------------------
tabs = st.tabs([
    t('tab_overview'),
    t('tab_model'),
    t('tab_shap'),
    t('tab_simulator')
])

# =============================================================================
# --- TAB 1: OVERVIEW & GREEN SCARCITY ---
# =============================================================================
with tabs[0]:
    st.subheader(t('exec_summary'))

    col1, col2, col3, col4 = st.columns(4)

    tot_n = len(filtered_df)
    green_count = int(filtered_df['is_green_binary'].sum())
    green_pct = (green_count / tot_n * 100) if tot_n > 0 else 0.0

    col1.metric(t('metric_total_products'), f"{tot_n:,}")
    col2.metric(t('metric_green_share'), f"{green_count:,} ({green_pct:.1f}%)")
    n_pairs = len(filtered_psm_df) // 2 if psm_matched else 0
    col3.metric(t('metric_psm_sample'), f"{len(filtered_psm_df):,} ({n_pairs:,} Pairs)" if psm_matched else f"{len(filtered_psm_df):,} (Unmatched)")
    if shap_boundary_lo is not None:
        col4.metric(t('metric_critical_boundary'), f"\\${shap_boundary_lo:,.0f} – \\${shap_boundary_hi:,.0f}")
    else:
        col4.metric(t('metric_critical_boundary'), t('critical_boundary_na'))

    st.markdown("---")
    st.subheader(t('overview_chart_title'))
    st.markdown(f'<p style="color: #4A5568; font-size: 14px; font-weight: 500; margin-top: -6px; margin-bottom: 14px;">{t("overview_chart_subtitle")}</p>', unsafe_allow_html=True)

    ctrl_col1, ctrl_col2 = st.columns([1, 1])
    with ctrl_col1:
        exclude_outliers = st.checkbox(t('outlier_filter_toggle'), value=True)
    with ctrl_col2:
        use_log_scale = st.checkbox(t('scale_log'), value=False)

    multi_group_mode = len(treated_values) > 1
    if multi_group_mode:
        tag_data = filtered_df[filtered_df['sustainability_tags'].apply(lambda x: any(str(tv) in str(x) for tv in treated_values))].copy()
        group_col = 'sustainability_tags'
    else:
        tag_data = filtered_df.copy()
        tag_data['group_label'] = tag_data['is_green_binary'].map({1: t('sim_green_product'), 0: t('sim_standard_product')})
        group_col = 'group_label'

    if len(tag_data) > 0:
        tag_sales = tag_data.groupby(group_col).agg(
            median_sales=('purchased_last_month', 'median'),
            product_count=('purchased_last_month', 'count')
        ).reset_index().rename(columns={group_col: 'sustainability_tags'})

        if exclude_outliers:
            tag_sales_filtered = tag_sales[tag_sales['product_count'] >= 10]
            if len(tag_sales_filtered) > 0:
                tag_sales = tag_sales_filtered

        tag_sales = tag_sales.sort_values(by='median_sales', ascending=True).tail(10).copy()

        tag_sales['tag_display'] = tag_sales.apply(
            lambda r: f"⚠️ {r['sustainability_tags']} (N={r['product_count']:,})"
            if r['product_count'] < 30
            else f"{r['sustainability_tags']} (N={r['product_count']:,})",
            axis=1
        )

        if use_log_scale:
            tag_sales['plot_sales'] = np.maximum(1.0, tag_sales['median_sales'])
        else:
            tag_sales['plot_sales'] = tag_sales['median_sales']

        fig_bar = px.bar(
            tag_sales,
            x='plot_sales',
            y='tag_display',
            orientation='h',
            color_discrete_sequence=['#2563EB'],
            log_x=use_log_scale,
            text='median_sales',
            labels={
                'plot_sales': t('x_axis_median_sales'),
                'tag_display': t('y_axis_tag')
            }
        )
        fig_bar.update_traces(
            texttemplate='%{text:,.0f}',
            textposition='auto',
            marker_line_color='#1D4ED8',
            marker_line_width=1,
            hovertemplate='<b>%{y}</b><br>Median: %{text:,.0f}<extra></extra>'
        )
        fig_bar.update_layout(
            height=400,
            font=PLOTLY_FONT,
            hoverlabel=CHART_HOVERLABEL,
            colorway=CHART_COLORWAY,
            showlegend=False,
            margin=dict(l=10, r=40, t=20, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor=CHART_GRIDCOLOR),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_bar, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)

        csv_bytes = tag_sales[['sustainability_tags', 'median_sales', 'product_count']].to_csv(index=False).encode('utf-8')
        st.download_button(t('table_download'), data=csv_bytes, file_name="group_summary.csv", mime="text/csv")
    else:
        st.info(t('no_tag_data'))

    st.markdown("---")
    # PSM Covariate Balance Section
    st.subheader(t('psm_balance_title'))
    st.markdown(f'<p style="color: #4A5568; font-size: 14px; font-weight: 500; margin-top: -6px; margin-bottom: 14px;">{t("psm_balance_subtitle")}</p>', unsafe_allow_html=True)

    if psm_matched:
        st.markdown(
            f'<div class="hypothesis-card">{t("psm_info_box").format(n_matched=len(filtered_psm_df), n_pairs=n_pairs)}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(f'<div class="hypothesis-card">{t("psm_unmatched_info_box")}</div>', unsafe_allow_html=True)

    covariate_labels = {
        'log_price': translate_var_name('log_price'),
        'log_reviews': translate_var_name('log_reviews'),
        'product_rating': translate_var_name('product_rating'),
        'is_sponsored_binary': translate_var_name('is_sponsored_binary'),
        'has_coupon_binary': translate_var_name('has_coupon_binary'),
    }

    raw_treated = filtered_df[filtered_df['is_green_binary'] == 1]
    raw_control = filtered_df[filtered_df['is_green_binary'] == 0]
    matched_treated = filtered_psm_df[filtered_psm_df['is_green_binary'] == 1]
    matched_control = filtered_psm_df[filtered_psm_df['is_green_binary'] == 0]

    masd_rows = []
    for covar, label in covariate_labels.items():
        raw_masd = standardized_diff(raw_treated[covar], raw_control[covar])
        matched_masd = standardized_diff(matched_treated[covar], matched_control[covar]) if psm_matched else raw_masd
        masd_rows.append({'Covariate': label, 'Raw Sample MASD': raw_masd, 'PSM Matched MASD': matched_masd})
    masd_data = pd.DataFrame(masd_rows)

    fig_masd = go.Figure()
    fig_masd.add_trace(go.Bar(
        x=masd_data['Covariate'],
        y=masd_data['Raw Sample MASD'],
        name=t('legend_raw_sample').format(n=f"{len(filtered_df):,}"),
        marker_color='#EF4444'
    ))
    if psm_matched:
        fig_masd.add_trace(go.Bar(
            x=masd_data['Covariate'],
            y=masd_data['PSM Matched MASD'],
            name=t('legend_psm_matched').format(n=f"{len(filtered_psm_df):,}"),
            marker_color='#10B981'
        ))

    fig_masd.add_hline(
        y=0.05,
        line_dash="dash",
        line_color="#F59E0B",
        annotation_text=t('masd_threshold_annotation'),
        annotation_position="top right"
    )

    fig_masd.update_layout(
        height=380,
        font=PLOTLY_FONT,
            hoverlabel=CHART_HOVERLABEL,
            colorway=CHART_COLORWAY,
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(title=t('masd_yaxis_title'), showgrid=True, gridcolor=CHART_GRIDCOLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_masd, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)


# =============================================================================
# --- TAB 2: MODERATED ECONOMETRIC & MODEL SPECIFICATION ---
# =============================================================================
with tabs[1]:
    if selected_model_key in ["ols", "ridge"]:
        badge_html = f'<div class="engine-badge">{t("badge_parametric").format(model_name=active_model_display_name)}</div>'
    else:
        badge_html = f'<div class="engine-badge" style="background-color:#FFFBEB; color:#B45309; border-color:#FDE68A;">{t("badge_nonparametric").format(model_name=active_model_display_name)}</div>'
    st.markdown(badge_html, unsafe_allow_html=True)

    # ---- Methodological Rationale: why two modeling frameworks? ----
    _r2_note = t('methodology_ml_r2_note').format(r2=_shap_boundary_r2) if _shap_boundary_r2 is not None else ""
    st.markdown(
        f"""
        <div class="hypothesis-card">
            <h4 style="color:#1D4ED8; margin-top:0;">{t('methodology_card_title')}</h4>
            <p style="font-size:0.9rem; margin-bottom:6px;"><b>{t('methodology_econometric_label')}</b> — {t('methodology_econometric_desc')}</p>
            <p style="font-size:0.9rem; margin-bottom:0;"><b>{t('methodology_ml_label')}</b> — {t('methodology_ml_desc').format(r2_note=_r2_note)}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    sample_choice = st.radio(
        t('model_sample_selector'),
        options=[t('sample_full').format(n=len(filtered_df)), t('sample_psm').format(n=len(filtered_psm_df))],
        horizontal=True
    )

    active_df = filtered_psm_df if "PSM" in sample_choice else filtered_df

    if len(active_df) > len(X_VARS) + 2:
        st.subheader(t('model_title').format(model_name=active_model_display_name))

        X = active_df[X_VARS]
        y = active_df['log_purchased']
        data_hash = _df_hash(active_df, X_VARS + ['log_purchased'])

        if len(active_df) < 30:
            st.warning(t('small_sample_warning').format(n=len(active_df)))

        display_df = None
        results_df = None

        if selected_model_key in ["ols", "ridge"]:
            st.markdown(t('dep_var_label').format(target_label=TARGET_LABEL))

            if selected_model_key == "ols":
                with st.spinner(t('spinner_ols')):
                    model_fit = fit_ols(X, y, data_hash)

                var_display_list = [translate_var_name(v) for v in model_fit.params.index]
                groups_list = [get_var_group(v) for v in model_fit.params.index]

                results_df = pd.DataFrame({
                    '_orig_var': list(model_fit.params.index),
                    '_p_val_raw': model_fit.pvalues.values,
                    t('col_group'): groups_list,
                    t('col_variable'): var_display_list,
                    t('col_coef'): model_fit.params.values,
                    t('col_std_err'): model_fit.bse.values,
                    t('col_t_val'): model_fit.tvalues.values,
                    t('col_p_val'): model_fit.pvalues.values,
                    t('col_sig'): [get_sig_stars(p) for p in model_fit.pvalues.values]
                })

                display_df = results_df[[t('col_group'), t('col_variable'), t('col_coef'),
                                        t('col_std_err'), t('col_t_val'), t('col_p_val'), t('col_sig')]].copy()

                display_df[t('col_coef')] = results_df.apply(
                    lambda r: f"{r[t('col_coef')]:.4f} {r[t('col_sig')]}", axis=1
                )
                display_df[t('col_std_err')] = results_df[t('col_std_err')].apply(lambda x: f"{x:.4f}")
                display_df[t('col_t_val')] = results_df[t('col_t_val')].apply(lambda x: f"{x:.3f}")
                display_df[t('col_p_val')] = results_df.apply(
                    lambda r: "< 0.001" if r['_p_val_raw'] < 0.001 else f"{r['_p_val_raw']:.4f}",
                    axis=1
                )

                def highlight_sig_rows(row):
                    idx = row.name
                    p_val = results_df.loc[idx, '_p_val_raw']
                    if p_val < 0.05:
                        return ['background-color: rgba(37, 99, 235, 0.08); font-weight: 500;'] * len(row)
                    return [''] * len(row)

                st.dataframe(
                    display_df.style.apply(highlight_sig_rows, axis=1),
                    use_container_width=True,
                    hide_index=True
                )
                st.markdown(t('sig_legend'))

                m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
                m_col1.metric(t('metric_r2'), f"{model_fit.rsquared:.4f}")
                m_col2.metric(t('metric_adj_r2'), f"{model_fit.rsquared_adj:.4f}")
                m_col3.metric(t('metric_f_stat'), f"{model_fit.fvalue:.2f}")
                m_col4.metric(t('metric_n_obs'), f"{int(model_fit.nobs):,}")
                m_col5.metric(t('metric_residual_var'), f"{model_fit.mse_resid:.4f}")

            else:  # Ridge Regression
                from sklearn.metrics import r2_score as sk_r2

                with st.spinner(t('spinner_ridge')):
                    ridge_sk = fit_ridge(X, y, data_hash)

                var_names = ['const'] + X_VARS
                coefs = [ridge_sk.intercept_] + list(ridge_sk.coef_)

                results_df = pd.DataFrame({
                    '_orig_var': var_names,
                    t('col_group'): [get_var_group(v) for v in var_names],
                    t('col_variable'): [translate_var_name(v) for v in var_names],
                    t('col_coef'): coefs,
                    t('col_std_err'): "N/A (Regularized)",
                    t('col_t_val'): "N/A",
                    t('col_p_val'): "—",
                    t('col_sig'): "ns"
                })

                display_df = results_df[[t('col_group'), t('col_variable'), t('col_coef'),
                                        t('col_std_err'), t('col_t_val'), t('col_p_val'), t('col_sig')]].copy()
                display_df[t('col_coef')] = results_df[t('col_coef')].apply(lambda x: f"{x:.4f}")

                st.dataframe(display_df, use_container_width=True, hide_index=True)
                st.info(t('ridge_info_box'))

                y_pred = ridge_sk.predict(X)
                r2_val = sk_r2(y, y_pred)
                n_obs = len(y)
                p_vars = X.shape[1]
                adj_r2_val = 1 - (1 - r2_val) * (n_obs - 1) / (n_obs - p_vars - 1) if n_obs - p_vars - 1 > 0 else float('nan')
                mse_resid = np.mean((y - y_pred) ** 2)

                m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
                m_col1.metric(t('metric_r2'), f"{r2_val:.4f}")
                m_col2.metric(t('metric_adj_r2'), f"{adj_r2_val:.4f}")
                m_col3.metric(t('metric_f_stat'), "N/A (L2 Penalty)")
                m_col4.metric(t('metric_n_obs'), f"{n_obs:,}")
                m_col5.metric(t('metric_residual_var'), f"{mse_resid:.4f}")

            if display_df is not None:
                csv_bytes = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(t('download_regression_csv'), data=csv_bytes, file_name=f"regression_results_{selected_model_key}.csv", mime="text/csv")

            # ---- Dynamic Hypotheses Callout Blocks (driven by headline OLS stats) ----
            st.markdown("---")
            h_col1, h_col2, h_col3 = st.columns(3)

            if headline_stats is not None:
                params = headline_stats.params
                pvalues = headline_stats.pvalues

                beta_h1 = params.get('interaction_price_green', 0.0)
                p_h1 = pvalues.get('interaction_price_green', 1.0)
                beta_h2 = params.get('interaction_reviews_green', 0.0)
                p_h2 = pvalues.get('interaction_reviews_green', 1.0)
                beta_h3 = params.get('interaction_coupon_green', 0.0)
                p_h3 = pvalues.get('interaction_coupon_green', 1.0)

                wb = i18n.get_word_bank(active_lang)

                sign_word = wb["pos"] if beta_h1 >= 0 else wb["neg"]
                verb = wb["confirm_yes"] if p_h1 < 0.05 else wb["confirm_no"]
                effect_word = wb["dampens"] if beta_h1 >= 0 else wb["amplifies"]
                h1_text = t('h1_desc').format(
                    sign_word=sign_word, treat_label=TREAT_LABEL, price_label=PRICE_LABEL,
                    beta=beta_h1, p_str=fmt_p(p_h1), verb=verb, effect_word=effect_word
                )

                sign_word2 = wb["pos"] if beta_h2 >= 0 else wb["neg"]
                verb2 = wb["indicate_yes"] if p_h2 < 0.05 else wb["indicate_no"]
                effect_word2 = wb["may"] if p_h2 >= 0.05 else wb["does"]
                h2_text = t('h2_desc').format(
                    sign_word=sign_word2, treat_label=TREAT_LABEL, reviews_label=REVIEWS_LABEL,
                    beta=beta_h2, p_str=fmt_p(p_h2), verb=verb2, effect_word2=effect_word2
                )

                h3_text = t('h3_desc').format(
                    treat_label=TREAT_LABEL, coupon_label=COUPON_LABEL, beta=beta_h3, p_str=fmt_p(p_h3)
                )
            else:
                h1_text = h2_text = h3_text = "N/A — insufficient data to estimate this interaction."

            with h_col1:
                st.markdown(f"""
                <div class="hypothesis-card">
                    <h4 style="color:#1D4ED8; margin-top:0;">{t('h1_title')}</h4>
                    <p style="font-size:0.9rem;">{h1_text}</p>
                </div>
                """, unsafe_allow_html=True)
            with h_col2:
                st.markdown(f"""
                <div class="hypothesis-card">
                    <h4 style="color:#1D4ED8; margin-top:0;">{t('h2_title')}</h4>
                    <p style="font-size:0.9rem;">{h2_text}</p>
                </div>
                """, unsafe_allow_html=True)
            with h_col3:
                st.markdown(f"""
                <div class="hypothesis-card">
                    <h4 style="color:#1D4ED8; margin-top:0;">{t('h3_title')}</h4>
                    <p style="font-size:0.9rem;">{h3_text}</p>
                </div>
                """, unsafe_allow_html=True)

        else:
            # ----------------------------------------------------
            # BRANCH B: Non-Parametric Tree Models (RF / XGBoost)
            # ----------------------------------------------------
            st.markdown(t('nonparam_active_note').format(model_name=active_model_display_name))

            with st.spinner(t('spinner_train_model').format(model_name=active_model_display_name)):
                tree_model, X_train, X_test, y_train, y_test = fit_tree_model(X, y, selected_model_key, data_hash)

            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            y_pred = tree_model.predict(X_test)

            r2_val = r2_score(y_test, y_pred)
            rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
            mae_val = mean_absolute_error(y_test, y_pred)

            met_col1, met_col2, met_col3, met_col4 = st.columns(4)
            met_col1.metric(t('metric_r2_test'), f"{r2_val:.4f}")
            met_col2.metric(t('metric_rmse_test'), f"{rmse_val:.4f}")
            met_col3.metric(t('metric_mae_test'), f"{mae_val:.4f}")
            met_col4.metric(t('metric_samples_traintest'), f"{len(X_train):,} / {len(X_test):,}")

            st.markdown("---")
            st.subheader(t('feature_importance_map_title').format(model_name=active_model_display_name))

            importances = tree_model.feature_importances_
            col_feature, col_importance = t('col_feature'), t('col_importance')
            feature_imp_df = pd.DataFrame({
                col_feature: [translate_var_name(v) for v in X_VARS],
                col_importance: importances
            }).sort_values(col_importance, ascending=True)

            fig_tree_imp = px.bar(
                feature_imp_df,
                x=col_importance,
                y=col_feature,
                orientation='h',
                color=col_importance,
                color_continuous_scale='Blues',
                text=col_importance
            )
            fig_tree_imp.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig_tree_imp.update_layout(
                height=420,
                font=PLOTLY_FONT,
            hoverlabel=CHART_HOVERLABEL,
            colorway=CHART_COLORWAY,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor=CHART_GRIDCOLOR),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_tree_imp, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)

            csv_bytes = feature_imp_df.to_csv(index=False).encode('utf-8')
            st.download_button(t('download_regression_csv'), data=csv_bytes, file_name=f"feature_importance_{selected_model_key}.csv", mime="text/csv")

    else:
        st.warning(t('warn_insufficient_sample'))


# =============================================================================
# --- TAB 3: SHAP & NON-LINEAR BOUNDARY ANALYSIS ---
# =============================================================================
with tabs[2]:
    st.markdown(f'<div class="engine-badge">{t("badge_shap_engine").format(model_name=active_model_display_name)}</div>', unsafe_allow_html=True)

    st.subheader(t('shap_title'))
    st.markdown(f'<p style="color: #4A5568; font-size: 14px; font-weight: 500; margin-top: -6px; margin-bottom: 18px;">{t("shap_subtitle")}</p>', unsafe_allow_html=True)

    shap_sample_df = filtered_df if len(filtered_df) > len(X_VARS) + 2 else None

    if selected_model_key in ["ols", "ridge"]:
        st.info(t('linear_shap_note'))

        st.markdown(f"#### {t('shap_summary_title').format(model_name=active_model_display_name)}")
        st.markdown(f'<p style="color: #4A5568; font-size: 13px; margin-top: -4px;">{t("shap_summary_desc")}</p>', unsafe_allow_html=True)

        if headline_stats is not None:
            beta_features = [v for v in X_VARS]
            col_feature, col_beta = t('col_feature'), t('col_beta_magnitude')
            linear_imp_df = pd.DataFrame({
                col_feature: [translate_var_name(v) for v in beta_features],
                col_beta: [abs(headline_stats.params.get(v, 0.0)) for v in beta_features]
            }).sort_values(col_beta, ascending=True)

            fig_lin_imp = px.bar(
                linear_imp_df,
                x=col_beta,
                y=col_feature,
                orientation='h',
                color=col_beta,
                color_continuous_scale='Blues',
                text=col_beta
            )
            fig_lin_imp.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_lin_imp.update_layout(
                height=380,
                font=PLOTLY_FONT,
            hoverlabel=CHART_HOVERLABEL,
            colorway=CHART_COLORWAY,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor=CHART_GRIDCOLOR),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_lin_imp, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
        else:
            st.info(t('info_not_enough_coef'))

    else:  # Tree Models (Random Forest / XGBoost) — real TreeSHAP
        if shap_sample_df is None:
            st.warning(t('warn_insufficient_shap'))
        else:
            X_full = shap_sample_df[X_VARS]
            y_full = shap_sample_df['log_purchased']
            data_hash = _df_hash(shap_sample_df, X_VARS + ['log_purchased'])

            with st.spinner(t('spinner_train').format(model_name=active_model_display_name)):
                tree_model, X_train, X_test, y_train, y_test = fit_tree_model(X_full, y_full, selected_model_key, data_hash)

            shap_sample_size = min(500, len(X_full))
            X_shap_sample = X_full.sample(shap_sample_size, random_state=42) if len(X_full) > shap_sample_size else X_full

            with st.spinner(t('spinner_shap')):
                shap_values = compute_shap_values(tree_model, X_shap_sample, selected_model_key, data_hash, shap_sample_size)

            mean_abs_shap = np.abs(shap_values).mean(axis=0)

            st.markdown(f"#### {t('shap_summary_title').format(model_name=active_model_display_name)}")
            st.markdown(f'<p style="color: #4A5568; font-size: 13px; margin-top: -4px;">{t("shap_summary_desc")}</p>', unsafe_allow_html=True)

            col_feature, col_shap = t('col_feature'), t('col_mean_shap')
            shap_importance_df = pd.DataFrame({
                col_feature: [translate_var_name(v) for v in X_VARS],
                col_shap: mean_abs_shap
            }).sort_values(col_shap, ascending=True)

            fig_shap_imp = px.bar(
                shap_importance_df,
                x=col_shap,
                y=col_feature,
                orientation='h',
                color=col_shap,
                color_continuous_scale='Blues',
                text=col_shap
            )
            fig_shap_imp.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_shap_imp.update_layout(
                height=380,
                font=PLOTLY_FONT,
            hoverlabel=CHART_HOVERLABEL,
            colorway=CHART_COLORWAY,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor=CHART_GRIDCOLOR),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_shap_imp, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)

            st.markdown("---")

            # ---- Real SHAP Dependence Plot for the treatment feature vs price ----
            # Reuses estimate_shap_price_boundary() — the same function that powers
            # the Tab 1 KPI card — so the boundary is never hardcoded and stays
            # consistent across the dashboard. Cache hits instantly when the
            # currently selected engine matches the canonical one used in Tab 1.
            boundary_lo, boundary_hi, low_val, high_val, binned, _ = estimate_shap_price_boundary(shap_sample_df, model_key=selected_model_key)

            if boundary_lo is not None:
                # Plain $ for Plotly (not markdown-parsed) vs escaped \$ for any
                # st.markdown() call, since Streamlit renders a matched "$...$"
                # pair as LaTeX math and silently drops the literal dollar signs.
                lo_str, hi_str = f"${boundary_lo:,.0f}", f"${boundary_hi:,.0f}"
                lo_str_md, hi_str_md = f"\\${boundary_lo:,.0f}", f"\\${boundary_hi:,.0f}"
                st.markdown(f"#### {t('shap_dep_title').format(lo=lo_str_md, hi=hi_str_md)}")
                st.markdown(f'<p style="color: #4A5568; font-size: 13px; margin-top: -4px;">{t("shap_dep_subtitle").format(price_label=PRICE_LABEL)}</p>', unsafe_allow_html=True)

                fig_dep = go.Figure()
                fig_dep.add_trace(go.Scatter(
                    x=binned['mean_price'], y=binned['mean_shap'],
                    mode='lines+markers',
                    name=f'Treatment SHAP Utility ({active_model_display_name})',
                    line=dict(color='#2563EB', width=3)
                ))
                fig_dep.add_vrect(
                    x0=boundary_lo, x1=boundary_hi,
                    fillcolor="#F59E0B", opacity=0.22,
                    layer="below", line_width=0,
                    annotation_text=t('critical_boundary_annotation').format(lo=lo_str, hi=hi_str),
                    annotation_position="top left",
                    annotation=dict(font=dict(size=12, color="#B45309", family="Inter"))
                )
                fig_dep.update_layout(
                    height=420,
                    font=PLOTLY_FONT,
            hoverlabel=CHART_HOVERLABEL,
            colorway=CHART_COLORWAY,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(title=PRICE_LABEL, showgrid=True, gridcolor=CHART_GRIDCOLOR),
                    yaxis=dict(title=t('shap_dep_utility_yaxis'), showgrid=True, gridcolor=CHART_GRIDCOLOR)
                )
                st.plotly_chart(fig_dep, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)

                st.markdown(f"### 💡 {t('elm_card_title')}")
                elm_col1, elm_col2, elm_col3 = st.columns(3)
                with elm_col1:
                    st.markdown(f"""
                    <div class="elm-card">
                        <h4 style="color:#64748B; margin-top:0;">{t('elm_low_title').format(lo=lo_str_md)}</h4>
                        <p style="font-size:0.88rem; color:#334155;">{t('elm_low_desc').format(low_val=low_val)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with elm_col2:
                    st.markdown(f"""
                    <div class="elm-card" style="border: 2px solid #F59E0B; background-color: #FFFBEB;">
                        <h4 style="color:#B45309; margin-top:0;">{t('elm_mid_title').format(lo=lo_str_md, hi=hi_str_md)}</h4>
                        <p style="font-size:0.88rem; color:#78350F;">{t('elm_mid_desc')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with elm_col3:
                    st.markdown(f"""
                    <div class="elm-card" style="border: 2px solid #2563EB; background-color: #EFF6FF;">
                        <h4 style="color:#1D4ED8; margin-top:0;">{t('elm_high_title').format(hi=hi_str_md)}</h4>
                        <p style="font-size:0.88rem; color:#115E59;">{t('elm_high_desc').format(high_val=high_val)}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(t('shap_boundary_unavailable'))


# =============================================================================
# --- TAB 4: INTERACTIVE SCENARIO & PRICING SIMULATOR ---
# =============================================================================
with tabs[3]:
    st.markdown(f'<div class="engine-badge">{t("badge_sim_engine").format(model_name=active_model_display_name)}</div>', unsafe_allow_html=True)

    st.subheader(t('sim_title'))
    st.write(t('sim_desc'))

    if len(filtered_df) > len(X_VARS) + 2:
        col_a, col_b = st.columns(2)
        default_price = float(np.clip(filtered_df['discounted_price'].median(), min_p, max_p))
        with col_a:
            sim_price = st.number_input(t('sim_price_label'), value=default_price, min_value=0.0, step=5.0)
            sim_rating = st.slider(t('sim_rating_label'), 0.0, 5.0, float(np.clip(filtered_df['product_rating'].mean(), 0, 5)), 0.1)
            sim_reviews = st.number_input(t('sim_reviews_label'), value=int(filtered_df['total_reviews'].median()), min_value=0, step=50)

        with col_b:
            sim_sponsored = st.checkbox(t('sim_sponsored_label'), value=False)
            sim_coupon = st.checkbox(t('sim_coupon_label'), value=False)
            group_choice_options = treated_values if len(treated_values) > 1 else ["Treated"]
            sim_tag = st.selectbox(t('sim_tag_label'), [t('sim_none_tag')] + [str(v) for v in group_choice_options])

        with st.spinner(t('spinner_predict')):
            pred_units_base, r2, rmse = predict_scenario(
                selected_model_key, filtered_df, sim_price, 0, sim_rating, sim_reviews,
                coupon=int(sim_coupon), sponsored=int(sim_sponsored)
            )

        std_err = rmse if rmse is not None else 0.8

        pred_log_base = np.log1p(pred_units_base)
        ci_lower_base = max(0, np.exp(pred_log_base - 1.96 * std_err) - 1)
        ci_upper_base = max(0, np.exp(pred_log_base + 1.96 * std_err) - 1)

        st.markdown("---")

        st.subheader(t('heterogeneity_header'))
        m_metric_1, m_metric_2 = st.columns(2)

        if headline_stats is not None:
            beta_p = headline_stats.params.get('log_price', 0.0)
            beta_pg = headline_stats.params.get('interaction_price_green', 0.0)
            eta_std = beta_p
            eta_green = beta_p + beta_pg
        else:
            eta_std, eta_green = 0.0, 0.0

        with m_metric_1:
            st.markdown(f"""
            <div class="hypothesis-card">
                <h4 style="color:#1D4ED8; margin-top:0;">{t('elasticity_title')}</h4>
                <p style="font-size:0.95rem; font-weight:600; color:#0F172A; margin-bottom:4px;">
                    {t('elasticity_standard').format(eta_std=eta_std)} &nbsp;|&nbsp;
                    <span style="color:#2563EB;">{t('elasticity_green').format(eta_green=eta_green)}</span>
                </p>
                <p style="font-size:0.85rem; color:#475569;">{t('elasticity_desc').format(eta_std=eta_std, eta_green=eta_green)}</p>
            </div>
            """, unsafe_allow_html=True)

        beta_rg = headline_stats.params.get('interaction_reviews_green', 0.0) if headline_stats is not None else 0.0
        eq_reviews = int(max(0, beta_rg) * 800)
        max_reviews = max(1.0, float(filtered_df['total_reviews'].quantile(0.95)))
        pct_lift = max(0.5, min(50.0, abs(beta_rg) * 100 * (1 - min(sim_reviews, max_reviews) / max_reviews)))

        with m_metric_2:
            st.markdown(f"""
            <div class="hypothesis-card">
                <h4 style="color:#1D4ED8; margin-top:0;">{t('substitution_title')}</h4>
                <p style="font-size:0.88rem; color:#334155;">
                    {t('substitution_desc').format(reviews=int(sim_reviews), eq_reviews=eq_reviews, pct_lift=pct_lift)}
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader(t('sim_result_header'))

        pred_units_tagged = None
        if sim_tag != t('sim_none_tag'):
            with st.spinner(t('spinner_predict')):
                pred_units_tagged, _, _ = predict_scenario(
                    selected_model_key, filtered_df, sim_price, 1, sim_rating, sim_reviews,
                    coupon=int(sim_coupon), sponsored=int(sim_sponsored)
                )

            pred_log_tagged = np.log1p(pred_units_tagged)
            ci_lower_tagged = max(0, np.exp(pred_log_tagged - 1.96 * std_err) - 1)
            ci_upper_tagged = max(0, np.exp(pred_log_tagged + 1.96 * std_err) - 1)

            diff_units = pred_units_tagged - pred_units_base
            pct_diff = (diff_units / pred_units_base * 100) if pred_units_base > 0 else 0

            res_col1, res_col2, res_col3 = st.columns(3)

            res_col1.metric(t('sim_standard_product'), f"{int(pred_units_base):,} {t('sim_units_unit')}")
            res_col1.markdown(
                f'<p style="color: #333333; font-size: 14px; font-weight: 500; margin-top: 4px;"><b>{t("ci_label")}:</b> [{int(ci_lower_base):,} – {int(ci_upper_base):,}]</p>',
                unsafe_allow_html=True
            )

            res_col2.metric(
                t('sim_green_product'),
                f"{int(pred_units_tagged):,} {t('sim_units_unit')}",
                f"{diff_units:+,.0f} {t('sim_units_unit')}"
            )
            res_col2.markdown(
                f'<p style="color: #333333; font-size: 14px; font-weight: 500; margin-top: 4px;"><b>{t("ci_label")}:</b> [{int(ci_lower_tagged):,} – {int(ci_upper_tagged):,}]</p>',
                unsafe_allow_html=True
            )

            res_col3.metric(t('sim_conditional_diff'), f"{pct_diff:+.1f}%", f"{diff_units:+,.0f} {t('sim_units_unit')}")

            col_predicted_value = t('col_predicted_value')
            col_product_type = t('col_product_type')
            comp_df = pd.DataFrame({
                col_product_type: [t('sim_standard_label'), f"{t('sim_tagged_label')}: {sim_tag}"],
                col_predicted_value: [pred_units_base, pred_units_tagged]
            })

            fig_comp = px.bar(
                comp_df,
                x=col_product_type,
                y=col_predicted_value,
                color=col_product_type,
                color_discrete_sequence=['#94A3B8', '#2563EB'],
                text=col_predicted_value,
                title=t('sim_comparison_chart_title') + f" ({active_model_display_name})"
            )
            fig_comp.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig_comp.update_layout(
                height=380,
                font=PLOTLY_FONT,
            hoverlabel=CHART_HOVERLABEL,
            colorway=CHART_COLORWAY,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(showgrid=True, gridcolor=CHART_GRIDCOLOR, title=col_predicted_value)
            )
            st.plotly_chart(fig_comp, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
        else:
            st.info(f"**{t('sim_standard_product')}:** **{int(pred_units_base):,} {t('sim_units_unit')}**")
            st.markdown(
                f'<p style="color: #333333; font-size: 14px; font-weight: 500; margin-top: 4px;"><b>{t("ci_label")}:</b> [{int(ci_lower_base):,} – {int(ci_upper_base):,}]</p>',
                unsafe_allow_html=True
            )
            st.info(t('sim_select_tag_prompt'))

        st.markdown("---")
        st.subheader(t('model_compare_header'))

        compare_green_status = 1 if sim_tag != t('sim_none_tag') else 0
        with st.spinner(t('spinner_compare')):
            compare_data = get_compare_predictions(
                filtered_df, sim_price, compare_green_status, sim_rating, sim_reviews,
                coupon=int(sim_coupon), sponsored=int(sim_sponsored)
            )

        selected_pred = pred_units_tagged if pred_units_tagged is not None else pred_units_base

        col_model, col_predicted_value, col_diff = t('col_model'), t('col_predicted_value'), t('col_diff_vs_active')
        compare_rows = []
        for item in compare_data:
            m_pred = item.get("predicted_sales", 0.0)
            pct_diff = ((m_pred - selected_pred) / selected_pred * 100) if selected_pred > 0 else 0.0
            is_active = (item["model_name"] == selected_model_key)
            compare_rows.append({
                col_model: f"⭐ {item['display_name']} (Active)" if is_active else item["display_name"],
                col_predicted_value: round(m_pred, 2),
                col_diff: t('active_engine_label') if is_active else f"{pct_diff:+.1f}%",
                t('col_r2_score'): f"{item['metrics']['R2_Score']:.4f}" if item["metrics"].get("R2_Score") is not None else "N/A",
                t('col_rmse'): f"{item['metrics']['RMSE']:.4f}" if item["metrics"].get("RMSE") is not None else "N/A"
            })

        compare_df = pd.DataFrame(compare_rows)

        fig_all_comp = px.bar(
            compare_df,
            x=col_model,
            y=col_predicted_value,
            color=col_model,
            color_discrete_sequence=['#2563EB' if selected_model_key == k else '#94A3B8' for k in MODEL_OPTIONS_ORDER],
            text=col_predicted_value,
            title=t('model_pred_comparison_title')
        )
        fig_all_comp.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig_all_comp.update_layout(
            height=380,
            font=PLOTLY_FONT,
            hoverlabel=CHART_HOVERLABEL,
            colorway=CHART_COLORWAY,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(showgrid=True, gridcolor=CHART_GRIDCOLOR, title=col_predicted_value)
        )
        st.plotly_chart(fig_all_comp, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)

        st.dataframe(compare_df, use_container_width=True, hide_index=True)

        csv_bytes = compare_df.to_csv(index=False).encode('utf-8')
        st.download_button(t('download_compare_csv'), data=csv_bytes, file_name="model_comparison.csv", mime="text/csv")

        # =====================================================================
        # ---- GREEN CERTIFICATION STRATEGY & ROI SIMULATOR ----
        # Uses the fitted econometric coefficients (H1 price-elasticity shift,
        # H2 social-proof substitution) via the same OLS model that powers the
        # hypothesis cards in Tab 2 — not ad-hoc rules — to give a managerial
        # recommendation on whether green certification is likely to pay off
        # for a candidate product at a given price / review count.
        st.markdown("---")
        st.markdown(f"### 🌱 {t('roi_simulator_title')}")
        st.write(t('roi_simulator_desc'))

        _price_budget = float(filtered_df['discounted_price'].quantile(0.25))
        _price_premium = crossover_price if (crossover_price is not None and min_p <= crossover_price <= max_p) else float(filtered_df['discounted_price'].quantile(0.75))
        _reviews_low = float(filtered_df['total_reviews'].quantile(0.25))
        _reviews_high = float(filtered_df['total_reviews'].quantile(0.75))
        _rating_mean = float(np.clip(filtered_df['product_rating'].mean(), 0, 5))

        roi_col_a, roi_col_b = st.columns(2)
        with roi_col_a:
            roi_price = st.number_input(t('roi_price_label'), value=float(np.clip(_price_premium, min_p, max_p)), min_value=0.0, step=5.0, key="roi_price_input")
        with roi_col_b:
            roi_reviews = st.number_input(t('roi_reviews_label'), value=int(_reviews_low), min_value=0, step=10, key="roi_reviews_input")

        if headline_stats is not None:
            with st.spinner(t('spinner_predict')):
                _roi_pred_without, _, _ = predict_scenario("ols", filtered_df, roi_price, 0, _rating_mean, roi_reviews, coupon=0, sponsored=0)
                _roi_pred_with, _, _ = predict_scenario("ols", filtered_df, roi_price, 1, _rating_mean, roi_reviews, coupon=0, sponsored=0)
            _roi_pct_lift = ((_roi_pred_with - _roi_pred_without) / _roi_pred_without * 100) if _roi_pred_without > 0 else 0.0

            if roi_price < _price_budget or roi_reviews >= _reviews_high:
                _roi_tier = "low"
            elif roi_price >= _price_premium and roi_reviews <= _reviews_low:
                _roi_tier = "high"
            else:
                _roi_tier = "moderate"

            _roi_tier_map = {
                "high": (t('roi_rec_high'), t('roi_rationale_high').format(threshold=_price_premium), "#2563EB"),
                "moderate": (t('roi_rec_moderate'), t('roi_rationale_moderate'), "#D97706"),
                "low": (t('roi_rec_low'), t('roi_rationale_low').format(threshold=_price_budget), "#DC2626"),
            }
            _roi_label, _roi_rationale, _roi_color = _roi_tier_map[_roi_tier]

            roi_result_col1, roi_result_col2 = st.columns([1, 2])
            with roi_result_col1:
                st.metric(t('roi_lift_label'), f"{_roi_pct_lift:+.1f}%")
            with roi_result_col2:
                st.markdown(
                    f"""
                    <div class="hypothesis-card" style="border-left-color:{_roi_color};">
                        <h4 style="color:{_roi_color}; margin-top:0;">{t('roi_recommendation_header')}: {_roi_label}</h4>
                        <p style="font-size:0.9rem;">{_roi_rationale}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info(t('info_not_enough_coef'))

        st.markdown("---")
        st.markdown(
            f'<p style="color: #4A5568; font-style: italic; font-size: 13px; margin-top: 12px;"><b>{t("note_label")}</b> {t("disclaimer_text")}</p>',
            unsafe_allow_html=True
        )
    else:
        st.warning(t('warn_insufficient_sim'))
