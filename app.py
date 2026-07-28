import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import os
import json
import requests
import joblib

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & ENTERPRISE DESIGN CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Amazon Green Electronics Research Dashboard",
    layout="wide"
)

# Custom CSS for Thai Typography, Sidebar Styling, Dropdown & High-Contrast Metrics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&family=Sarabun:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* 1. Global Typography & Thai Font Line-Height Fix */
    body, p, label, input, select, textarea, .stMarkdown {
        font-family: 'Prompt', 'Sarabun', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        line-height: 1.7 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Prompt', 'Sarabun', 'Inter', sans-serif !important;
        line-height: 1.5 !important;
        padding-top: 4px !important;
        padding-bottom: 4px !important;
        font-weight: 600 !important;
        overflow: visible !important;
    }
    
    .stMarkdown p, .stMarkdown span {
        line-height: 1.75 !important;
    }
    
    /* 2. Hide Stray 'keyboard_double' / 'ub' Icon Text Glitch at Sidebar Top */
    [data-testid="stSidebarHeader"] button span {
        display: none !important;
    }
    [data-testid="stSidebar"] button[title*="collapse"] span {
        visibility: hidden !important;
    }
    
    /* 3. Enterprise Dark Slate Sidebar Base Styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }
    
    /* 4. Force Readable White Text for Radio Buttons & Sidebar Labels */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stRadioButton"] label p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }
    
    /* 5. Force Dark Text (#0F172A) on White Background for Selectbox (Dropdown) on Sidebar */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] * {
        color: #0F172A !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
    }
    
    /* Dropdown Popover Menu (Options List) */
    div[data-baseweb="popover"] div[role="listbox"],
    div[data-baseweb="popover"] ul {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
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
        background-color: #F1F5F9 !important;
        color: #0D9488 !important;
        font-weight: 600 !important;
    }
    
    /* 6. Enterprise Styled Metric Cards & High-Contrast Text */
    [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        padding: 16px 20px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.2s ease-in-out !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: #CBD5E1 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
    }
    
    /* High-Contrast Metric Delta Text Fix (Vivid Light Green for Delta / Subtext) */
    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] *,
    [data-testid="stMetricDelta"] div,
    [data-testid="stMetricDelta"] span {
        font-weight: 700 !important;
        color: #16A34A !important; /* Vivid Kelly Green in light mode */
    }
    [data-testid="stMetricDelta"] svg {
        fill: #16A34A !important;
        color: #16A34A !important;
    }

    /* Dark Mode Adaptivity for Metrics & Bright High-Contrast Text */
    @media (prefers-color-scheme: dark) {
        [data-testid="stMetric"] {
            background-color: #1E293B !important;
            border-color: #334155 !important;
        }
        [data-testid="stMetricValue"] {
            color: #F8FAFC !important;
        }
        [data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
        }
        /* Bright High-Contrast Light Emerald Green (#4ADE80) for Dark Backgrounds */
        [data-testid="stMetricDelta"],
        [data-testid="stMetricDelta"] *,
        [data-testid="stMetricDelta"] div,
        [data-testid="stMetricDelta"] span {
            color: #4ADE80 !important;
        }
        [data-testid="stMetricDelta"] svg {
            fill: #4ADE80 !important;
            color: #4ADE80 !important;
        }
    /* 7. High-Contrast Global Caption & Subtext Contrast Fix */
    .stCaption, 
    [data-testid="stCaptionContainer"], 
    [data-testid="stCaptionContainer"] p, 
    .stCaption p {
        color: #334155 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    /* Clean Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 500;
        border-radius: 6px 6px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. BILINGUAL TRANSLATION DICTIONARY
# -----------------------------------------------------------------------------
TRANSLATIONS = {
    "EN": {
        "page_title": "Amazon Green Electronics Research Dashboard",
        "header_title": "Amazon Green E-Commerce Research Dashboard",
        "header_subtitle": "Econometric Analysis of Sustainability Tags and Electronics Sales Performance",
        "sidebar_title": "Control Panel & Filters",
        "lang_selector": "Language / ภาษา",
        "cat_filter": "Product Category",
        "all_cats": "All Categories",
        "price_filter": "Price Range ($)",
        # Tabs
        "tab_overview": "Overview",
        "tab_model": "Econometric Model",
        "tab_breakdown": "Tag Breakdown",
        "tab_simulator": "Association Simulator",
        # Tab 1: Overview
        "exec_summary": "Executive Summary",
        "metric_total_products": "Total Products Analysed",
        "metric_green_share": "Green Products Share",
        "metric_avg_price": "Avg Discounted Price",
        "metric_avg_sales": "Avg Monthly Units Sold",
        "overview_chart_title": "Median Monthly Sales Volume by Sustainability Tag",
        "overview_chart_subtitle": "Comparing median monthly sales across genuine environmental sustainability certifications",
        "scale_toggle": "Axis Scaling Mode",
        "scale_linear": "Linear Scale",
        "scale_log": "Logarithmic Scale (Recommended for Outliers)",
        "outlier_filter_toggle": "Exclude low-sample tags (<10 products)",
        "x_axis_median_sales": "Median Monthly Units Purchased",
        "y_axis_tag": "Sustainability Tag",
        "no_tag_data": "No tag data available for the current filter selection.",
        # Tab 2: Model
        "model_title": "OLS Regression Analysis (Academic Research Standard)",
        "dep_var_label": "**Dependent Variable:** $\\ln(\\text{Purchased Last Month} + 1)$",
        "col_group": "Variable Group",
        "col_variable": "Variable / Predictor",
        "col_coef": "Coefficient (β)",
        "col_std_err": "Std Error",
        "col_t_val": "t-statistic",
        "col_p_val": "P-value",
        "col_sig": "Significance",
        "sig_legend": "**Significance levels:** `***` p < 0.001 | `**` p < 0.01 | `*` p < 0.05 | `.` p < 0.1 | `ns` Not Significant | `⚠️ N<30` Insufficient sample",
        "metric_r2": "R-squared",
        "metric_adj_r2": "Adjusted R-squared",
        "metric_f_stat": "F-statistic",
        "metric_n_obs": "Observations (N)",
        "metric_residual_var": "Residual Variance (σ²)",
        "group_green_badges": "🌿 Green Sustainability Badges",
        "group_seller_controls": "🏷️ Seller/Functional Controls",
        "group_model_controls": "📊 Market & Product Controls",
        "interaction_hypothesis": "**Interaction Term Interpretation:** A significant positive coefficient on `Log(Price) × Green Badge` suggests that sustainability labelling moderates (reduces) consumer price sensitivity — i.e., green-tagged products suffer less sales penalty from higher prices.",
        # Tab 3: Breakdown
        "breakdown_title": "Deep-Dive Tag Performance",
        "select_tag_label": "Select Green Sustainability Tag to Inspect",
        "tag_metrics_summary": "Summary Metrics for Selected Tag",
        "metric_tag_count": "Products Count",
        "metric_tag_avg_price": "Avg Price",
        "metric_tag_avg_sales": "Avg Monthly Sales",
        "metric_tag_avg_rating": "Avg Rating",
        "scatter_title": "Price vs. Monthly Sales for '{tag}' Products",
        "scatter_xlabel": "Price ($)",
        "scatter_ylabel": "Monthly Purchased (Units)",
        "no_tag_products_found": "No products found for the selected tag and filters.",
        # Tab 4: Simulator (Causal-language-free)
        "sim_title": "Model-Based Sales Association Simulator",
        "sim_desc": "Simulate model-predicted monthly sales based on product attributes and compare conditional differences with green sustainability tagging.",
        "sim_price_label": "Target Price ($)",
        "sim_rating_label": "Target Rating (1.0 - 5.0)",
        "sim_reviews_label": "Expected Total Reviews",
        "sim_sponsored_label": "Run Sponsored Ad Campaign?",
        "sim_coupon_label": "Offer Discount Coupon?",
        "sim_tag_label": "Select Green Sustainability Tag",
        "sim_none_tag": "None (Standard Product)",
        "sim_result_header": "Model-Predicted Association Results",
        "sim_standard_product": "Standard Product (No Tag)",
        "sim_green_product": "Green Tagged Product",
        "sim_conditional_diff": "Conditional Difference (%)",
        "sim_units_unit": "units / month",
        "sim_comparison_chart_title": "Model-Predicted Monthly Sales: Standard vs. Tagged",
        "sim_standard_label": "Standard Product",
        "sim_tagged_label": "Tagged Product",
        "sim_select_tag_prompt": "Select a Green Sustainability Tag above to see the model-predicted conditional difference.",
        "ci_label": "95% CI",
        "disclaimer_text": "Note: Results reflect observational associations derived from cross-sectional data, not guaranteed causal impacts.",
        "small_sample_warning": "⚠️ Warning: Sample size is too small (N={n} < 30). Statistical inferences and p-values may be uninterpretable.",
        # Variable name mappings
        "var_const": "Intercept (Const)",
        "var_log_price": "Log(Discounted Price + 1)",
        "var_product_rating": "Product Rating (1-5)",
        "var_log_reviews": "Log(Total Reviews + 1)",
        "var_is_sponsored_binary": "Sponsored Product (Dummy)",
        "var_has_coupon_binary": "Has Discount Coupon (Dummy)",
        "var_interaction_price_green": "Log(Price) × Green Badge (Interaction)",
        "var_tag_prefix": "Tag: "
    },
    "TH": {
        "page_title": "แดชบอร์ดการวิจัยสินค้าอิเล็กทรอนิกส์รักษ์โลก Amazon",
        "header_title": "แดชบอร์ดการวิจัยอีคอมเมิร์ซสีเขียว Amazon",
        "header_subtitle": "การวิเคราะห์เชิงเศรษฐมิติความสัมพันธ์ระหว่างป้ายความยั่งยืนกับยอดขายสินค้ากลุ่มอิเล็กทรอนิกส์",
        "sidebar_title": "แผงควบคุมและตัวกรอง",
        "lang_selector": "เลือกภาษา / Language",
        "cat_filter": "หมวดหมู่สินค้า",
        "all_cats": "หมวดหมู่ทั้งหมด",
        "price_filter": "ช่วงราคา ($)",
        # Tabs
        "tab_overview": "ภาพรวม",
        "tab_model": "โมเดลเศรษฐมิติ",
        "tab_breakdown": "วิเคราะห์รายป้าย",
        "tab_simulator": "เครื่องมือจำลองความสัมพันธ์",
        # Tab 1: Overview
        "exec_summary": "สรุปภาพรวมผู้บริหาร",
        "metric_total_products": "จำนวนสินค้าทั้งหมดที่วิเคราะห์",
        "metric_green_share": "สัดส่วนสินค้ามีป้ายความยั่งยืน",
        "metric_avg_price": "ราคาลดเฉลี่ย",
        "metric_avg_sales": "ยอดขายเฉลี่ยต่อเดือน",
        "overview_chart_title": "มัธยฐานยอดขายต่อเดือนแยกตามป้ายความยั่งยืนทางสิ่งแวดล้อม",
        "overview_chart_subtitle": "เปรียบเทียบมัธยฐานยอดขายรายเดือนเฉพาะป้ายรับรองความยั่งยืนทางสิ่งแวดล้อมที่แท้จริง",
        "scale_toggle": "โหมดสเกลแกนยอดขาย",
        "scale_linear": "สเกลเชิงเส้น (Linear)",
        "scale_log": "สเกลลอการิทึม (Log Scale - แนะนำสำหรับปรับสเกลค่าสุดโต่ง)",
        "outlier_filter_toggle": "กรองป้ายที่มีตัวอย่างน้อยออก (<10 สินค้า)",
        "x_axis_median_sales": "มัธยฐานจำนวนหน่วยที่ขายได้ต่อเดือน",
        "y_axis_tag": "ป้ายกำกับความยั่งยืนทางสิ่งแวดล้อม",
        "no_tag_data": "ไม่มีข้อมูลป้ายความยั่งยืนสำหรับเงื่อนไขตัวกรองนี้",
        # Tab 2: Model
        "model_title": "การวิเคราะห์การถดถอย OLS (มาตรฐานงานวิจัยเชิงวิชาการ)",
        "dep_var_label": "**ตัวแปรตาม (Dependent Variable):** $\\ln(\\text{Purchased Last Month} + 1)$",
        "col_group": "กลุ่มตัวแปร",
        "col_variable": "ตัวแปรพยากรณ์",
        "col_coef": "สัมประสิทธิ์ (β)",
        "col_std_err": "ความคลาดเคลื่อนมาตรฐาน (Std Error)",
        "col_t_val": "ค่า t-statistic",
        "col_p_val": "ค่า P-value",
        "col_sig": "ระดับนัยสำคัญ",
        "sig_legend": "**สัญลักษณ์นัยสำคัญทางสถิติ:** `***` p < 0.001 | `**` p < 0.01 | `*` p < 0.05 | `.` p < 0.1 | `ns` ไม่มีนัยสำคัญ | `⚠️ N<30` ตัวอย่างไม่เพียงพอ",
        "metric_r2": "R-squared",
        "metric_adj_r2": "Adjusted R-squared",
        "metric_f_stat": "F-statistic",
        "metric_n_obs": "จำนวนตัวอย่าง (N)",
        "metric_residual_var": "ความแปรปรวนของส่วนที่เหลือ (σ²)",
        "group_green_badges": "🌿 ป้ายความยั่งยืนด้านสิ่งแวดล้อม",
        "group_seller_controls": "🏷️ ตัวแปรควบคุมด้านผู้ขาย/ฟังก์ชัน",
        "group_model_controls": "📊 ตัวแปรควบคุมด้านตลาดและสินค้า",
        "interaction_hypothesis": "**การตีความ Interaction Term:** สัมประสิทธิ์ที่เป็นบวกและมีนัยสำคัญของ `Log(ราคา) × ป้ายสีเขียว` บ่งชี้ว่าการติดป้ายความยั่งยืนช่วยลดความไวต่อราคา (Price Sensitivity) ของผู้บริโภค กล่าวคือสินค้าที่ติดป้ายสีเขียวได้รับผลกระทบจากราคาที่สูงน้อยกว่า",
        # Tab 3: Breakdown
        "breakdown_title": "วิเคราะห์เจาะลึกประสิทธิภาพป้ายความยั่งยืน",
        "select_tag_label": "เลือกป้ายความยั่งยืนทางสิ่งแวดล้อมที่ต้องการวิเคราะห์",
        "tag_metrics_summary": "สรุปข้อมูลเฉพาะป้ายที่เลือก",
        "metric_tag_count": "จำนวนสินค้าในป้ายนี้",
        "metric_tag_avg_price": "ราคาเฉลี่ย",
        "metric_tag_avg_sales": "ยอดขายเฉลี่ยต่อเดือน",
        "metric_tag_avg_rating": "คะแนนรีวิวเฉลี่ย",
        "scatter_title": "ความสัมพันธ์ระหว่างราคากับยอดขายสำหรับสินค้าป้าย '{tag}'",
        "scatter_xlabel": "ราคา ($)",
        "scatter_ylabel": "ยอดขายต่อเดือน (ชิ้น)",
        "no_tag_products_found": "ไม่พบสินค้าตามป้ายและเงื่อนไขตัวกรองที่เลือก",
        # Tab 4: Simulator (Causal-language-free)
        "sim_title": "เครื่องมือจำลองความสัมพันธ์ยอดขายเชิงแบบจำลอง",
        "sim_desc": "จำลองยอดขายที่พยากรณ์จากแบบจำลอง ML ตามคุณลักษณะสินค้า พร้อมเปรียบเทียบผลต่างแบบมีเงื่อนไขจากการติดป้ายความยั่งยืนทางสิ่งแวดล้อม",
        "sim_price_label": "ราคาเป้าหมาย ($)",
        "sim_rating_label": "คะแนนรีวิวเป้าหมาย (1.0 - 5.0)",
        "sim_reviews_label": "จำนวนรีวิวที่คาดหวัง",
        "sim_sponsored_label": "ลงโฆษณา Sponsored?",
        "sim_coupon_label": "มีคูปองส่วนลด?",
        "sim_tag_label": "เลือกป้ายความยั่งยืนทางสิ่งแวดล้อม",
        "sim_none_tag": "ไม่มี (สินค้าทั่วไป)",
        "sim_result_header": "ผลการพยากรณ์ความสัมพันธ์จากแบบจำลอง",
        "sim_standard_product": "สินค้าทั่วไป (ไม่มีป้าย)",
        "sim_green_product": "สินค้าติดป้ายความยั่งยืน",
        "sim_conditional_diff": "ผลต่างแบบมีเงื่อนไข (%)",
        "sim_units_unit": "ชิ้น / เดือน",
        "sim_comparison_chart_title": "ยอดขายพยากรณ์จากแบบจำลอง: สินค้าทั่วไป VS สินค้าติดป้าย",
        "sim_standard_label": "สินค้าทั่วไป",
        "sim_tagged_label": "สินค้าติดป้าย",
        "sim_select_tag_prompt": "เลือกป้ายความยั่งยืนด้านบนเพื่อดูผลต่างแบบมีเงื่อนไขจากแบบจำลอง",
        "ci_label": "ช่วงความเชื่อมั่น 95%",
        "disclaimer_text": "หมายเหตุ: ผลลัพธ์สะท้อนความสัมพันธ์เชิงสังเกตการณ์จากข้อมูลภาคตัดขวาง มิใช่ผลกระทบเชิงสาเหตุที่การันตีได้",
        "small_sample_warning": "⚠️ คำเตือน: ขนาดตัวอย่างน้อยเกินไป (N={n} < 30) การอนุมานทางสถิติและค่า p-value อาจไม่น่าเชื่อถือ",
        # Variable name mappings
        "var_const": "จุดตัดแกน (Const)",
        "var_log_price": "Log(ราคาลด + 1)",
        "var_product_rating": "คะแนนรีวิวสินค้า (1-5)",
        "var_log_reviews": "Log(จำนวนรีวิว + 1)",
        "var_is_sponsored_binary": "โฆษณา Sponsored (Dummy)",
        "var_has_coupon_binary": "มีคูปองส่วนลด (Dummy)",
        "var_interaction_price_green": "Log(ราคา) × ป้ายสีเขียว (Interaction)",
        "var_tag_prefix": "ป้าย: "
    }
}

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONFIGURATION (LANGUAGE, FILTERS & MULTI-MODEL SELECTION)
# -----------------------------------------------------------------------------
st.sidebar.title("Language / ภาษา")
lang_choice = st.sidebar.radio(
    "Select Language / เลือกภาษา",
    options=["ภาษาไทย", "English"],
    index=0,
    horizontal=True,
    label_visibility="collapsed"
)
lang = "TH" if "ไทย" in lang_choice else "EN"

def t(key):
    """Retrieve translated string for key based on active language."""
    return TRANSLATIONS[lang].get(key, TRANSLATIONS["EN"].get(key, key))

def translate_var_name(var, lang_code):
    """Convert code variable names into academic research standard labels."""
    mapping = {
        'const': t('var_const'),
        'log_price': t('var_log_price'),
        'product_rating': t('var_product_rating'),
        'log_reviews': t('var_log_reviews'),
        'is_sponsored_binary': t('var_is_sponsored_binary'),
        'has_coupon_binary': t('var_has_coupon_binary'),
        'interaction_price_green': t('var_interaction_price_green'),
    }
    if var in mapping:
        return mapping[var]
    if var.startswith('tag_'):
        tag_raw = var.replace('tag_', '')
        return f"{t('var_tag_prefix')}{tag_raw}"
    return var

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

# Clean Header
st.title(t('header_title'))
st.markdown(f'<p style="color: #333333; font-size: 1.05rem; font-weight: 500; margin-top: -10px; margin-bottom: 20px;">{t("header_subtitle")}</p>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. DATA LOADING & PREPROCESSING
# -----------------------------------------------------------------------------
# Strict Academic Tag Categorisation:
# GREEN_TAGS  = genuine environmental sustainability badges (Primary IVs & UI Dropdowns)
# CONTROL_TAGS = seller / functional non-environmental badges (OLS Model Controls ONLY)
GREEN_TAGS = [
    'Carbon impact',
    'Energy efficiency',
    'Manufacturing practices',
    'Forestry practices',
    'Recycled materials'
]
CONTROL_TAGS = ['Works with Alexa', 'Small Business']
ALL_MODEL_TAGS = GREEN_TAGS + CONTROL_TAGS

@st.cache_data
def load_data():
    df = pd.read_csv('data/amazon_products_sales_data_cleaned.csv')
    df_clean = df.dropna(subset=['purchased_last_month', 'discounted_price', 'product_rating', 'total_reviews']).copy()
    
    # Binary indicator for genuine green sustainability tag
    df_clean['has_sustainability_tag'] = df_clean['sustainability_tags'].isin(GREEN_TAGS).astype(int)
    df_clean['is_sponsored_binary'] = (df_clean['is_sponsored'] == 'Sponsored').astype(int)
    df_clean['has_coupon_binary'] = (df_clean['has_coupon'] != 'No Coupon').astype(int)
    
    df_clean['log_purchased'] = np.log1p(df_clean['purchased_last_month'])
    df_clean['log_price'] = np.log1p(df_clean['discounted_price'])
    df_clean['log_reviews'] = np.log1p(df_clean['total_reviews'])
    
    # Create dummy columns for each model tag (both green and control tags)
    for tag in ALL_MODEL_TAGS:
        df_clean[f'tag_{tag}'] = (df_clean['sustainability_tags'] == tag).astype(int)
    
    # Binary indicator: 1 if product has ANY genuine green sustainability tag
    df_clean['is_green_binary'] = df_clean[[f'tag_{tg}' for tg in GREEN_TAGS]].max(axis=1)
    
    # Interaction term: log(price+1) × green badge — tests price sensitivity moderation
    df_clean['interaction_price_green'] = df_clean['log_price'] * df_clean['is_green_binary']
        
    return df_clean

try:
    df = load_data()
except Exception as e:
    st.error(f"Unable to read data file: {e}")
    st.info("Please verify that 'amazon_products_sales_data_cleaned.csv' is placed inside the 'data' folder.")
    st.stop()

# -----------------------------------------------------------------------------
# 5. SIDEBAR FILTERS & DYNAMIC MODEL SELECTION
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.title(t('sidebar_title'))

categories = [t('all_cats')] + list(df['product_category'].dropna().unique())
selected_category = st.sidebar.selectbox(t('cat_filter'), categories)

min_p = float(df['discounted_price'].min())
max_p = float(df['discounted_price'].quantile(0.98))
price_range = st.sidebar.slider(t('price_filter'), min_p, max_p, (min_p, max_p))

filtered_df = df[(df['discounted_price'] >= price_range[0]) & (df['discounted_price'] <= price_range[1])].copy()
if selected_category != t('all_cats'):
    filtered_df = filtered_df[filtered_df['product_category'] == selected_category]

# Dynamic Model Selection Dropdown
st.sidebar.markdown("---")
st.sidebar.subheader("Predictive Model / โมเดลพยากรณ์")

model_options = {
    "ols": "OLS Regression (แนะนำสำหรับการวิเคราะห์เชิงเศรษฐมิติ)" if lang == "TH" else "OLS Regression (Recommended for Econometrics)",
    "ridge": "Ridge Regression",
    "rf": "Random Forest Regressor",
    "xgboost": "XGBoost Regressor (ความแม่นยำสูง)" if lang == "TH" else "XGBoost Regressor (High Accuracy)"
}

selected_model_key = st.sidebar.selectbox(
    "Select Model for Simulator" if lang == "EN" else "เลือกโมเดลพยากรณ์ขาย",
    options=list(model_options.keys()),
    format_func=lambda x: model_options[x]
)

# Shared Plotly Configuration: ENABLE MODEBAR 100% (Zoom, Pan, Reset, Download)
PLOTLY_FONT = dict(family="'Prompt', 'Sarabun', 'Inter', sans-serif", size=12)
PLOTLY_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'scrollZoom': True
}

# -----------------------------------------------------------------------------
# 6. LOCAL BACKEND UTILITIES (FALLBACK FOR API)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_local_model(model_name):
    """Loads a pre-trained ML model locally if the API backend is not reachable."""
    model_filenames = {
        "ols": "ols_model.joblib",
        "ridge": "ridge_model.joblib",
        "rf": "rf_model.joblib",
        "xgboost": "xgb_model.joblib"
    }
    path = f"models/{model_filenames[model_name]}"
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            pass
    return None

@st.cache_data
def get_local_metrics():
    """Loads model performance metrics JSON locally."""
    path = "models/metrics.json"
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def get_model_prediction(model_name, price, is_green, rating, reviews_count):
    """Executes a prediction using the backend API, falling back to local model loading."""
    api_success = False
    predicted_sales = 0.0
    r2 = None
    rmse = None
    
    # Try calling backend API first
    try:
        url = "http://127.0.0.1:8000/api/predict"
        res = requests.post(url, json={
            "model_name": model_name,
            "price": price,
            "is_green": int(is_green),
            "rating": rating,
            "reviews_count": reviews_count
        }, timeout=0.8)
        if res.status_code == 200:
            data = res.json()
            predicted_sales = data["predicted_sales"]
            r2 = data["metrics"]["R2_Score"]
            rmse = data["metrics"]["RMSE"]
            api_success = True
    except Exception:
        pass
    
    # Fallback to local prediction
    if not api_success:
        local_metrics = get_local_metrics()
        model_display_names = {
            "ols": "OLS Regression",
            "ridge": "Ridge Regression",
            "rf": "Random Forest",
            "xgboost": "XGBoost"
        }
        m_info = local_metrics.get(model_display_names[model_name], {})
        r2 = m_info.get("R2_Score")
        rmse = m_info.get("RMSE")
        
        model = load_local_model(model_name)
        if model is not None:
            log_p = np.log1p(price)
            log_p_x_g = log_p * is_green
            df_input = pd.DataFrame([{
                'log_price': log_p,
                'is_green': float(is_green),
                'log_price_x_is_green': log_p_x_g,
                'rating': float(rating),
                'reviews_count': float(reviews_count)
            }])
            pred_log = model.predict(df_input)[0]
            predicted_sales = float(max(0, np.expm1(pred_log)))
            
    return predicted_sales, r2, rmse

def get_compare_predictions(price, is_green, rating, reviews_count):
    """Queries predictions across all models simultaneously via /api/compare or local fallback."""
    try:
        url = "http://127.0.0.1:8000/api/compare"
        res = requests.post(url, json={
            "price": price,
            "is_green": int(is_green),
            "rating": rating,
            "reviews_count": reviews_count
        }, timeout=1.0)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    
    # Local fallback comparison
    keys = ["ols", "ridge", "rf", "xgboost"]
    display_names = {
        "ols": "OLS Regression",
        "ridge": "Ridge Regression",
        "rf": "Random Forest",
        "xgboost": "XGBoost"
    }
    results = []
    for k in keys:
        pred, r2, rmse = get_model_prediction(k, price, is_green, rating, reviews_count)
        results.append({
            "model_name": k,
            "display_name": display_names[k],
            "predicted_sales": pred,
            "metrics": {
                "R2_Score": r2,
                "RMSE": rmse
            }
        })
    return results

# -----------------------------------------------------------------------------
# 7. HELPER: Variable Group Assignment for OLS Table
# -----------------------------------------------------------------------------
def get_var_group(var_name):
    """Assign each OLS variable to a conceptual group for the results table."""
    if var_name == 'const':
        return '—'
    if var_name.startswith('tag_'):
        tag_raw = var_name.replace('tag_', '')
        if tag_raw in GREEN_TAGS:
            return t('group_green_badges')
        if tag_raw in CONTROL_TAGS:
            return t('group_seller_controls')
    if var_name == 'interaction_price_green':
        return t('group_green_badges')
    return t('group_model_controls')

# Pre-compute tag sample sizes in filtered data for N<30 checks
def get_tag_sample_sizes(data):
    """Return dict of tag column name → sample count."""
    sizes = {}
    for tag in ALL_MODEL_TAGS:
        col = f'tag_{tag}'
        sizes[col] = int(data[col].sum())
    return sizes

# -----------------------------------------------------------------------------
# 8. DASHBOARD TABS
# -----------------------------------------------------------------------------
tabs = st.tabs([
    t('tab_overview'),
    t('tab_model'),
    t('tab_breakdown'),
    t('tab_simulator')
])

# --- TAB 1: OVERVIEW ---
with tabs[0]:
    st.subheader(t('exec_summary'))
    col1, col2, col3, col4 = st.columns(4)
    
    green_count = int(filtered_df['is_green_binary'].sum())
    green_pct = (filtered_df['is_green_binary'].mean() * 100) if len(filtered_df) > 0 else 0
    avg_price = filtered_df['discounted_price'].mean() if len(filtered_df) > 0 else 0
    avg_sales = filtered_df['purchased_last_month'].mean() if len(filtered_df) > 0 else 0
    
    col1.metric(t('metric_total_products'), f"{len(filtered_df):,}")
    col2.metric(t('metric_green_share'), f"{green_count:,} ({green_pct:.1f}%)")
    col3.metric(t('metric_avg_price'), f"${avg_price:.2f}")
    col4.metric(t('metric_avg_sales'), f"{avg_sales:,.0f}")
    
    st.markdown("---")
    st.subheader(t('overview_chart_title'))
    st.markdown(f'<p style="color: #4A5568; font-size: 14px; font-weight: 500; margin-top: -6px; margin-bottom: 14px;">{t("overview_chart_subtitle")}</p>', unsafe_allow_html=True)
    
    ctrl_col1, ctrl_col2 = st.columns([1, 1])
    with ctrl_col1:
        exclude_outliers = st.checkbox(t('outlier_filter_toggle'), value=True)
    with ctrl_col2:
        use_log_scale = st.checkbox(t('scale_log'), value=False)
        
    # Strictly filter for genuine green sustainability tags ONLY
    tag_data = filtered_df[filtered_df['sustainability_tags'].isin(GREEN_TAGS)].copy()
    
    if len(tag_data) > 0:
        tag_sales = tag_data.groupby('sustainability_tags').agg(
            median_sales=('purchased_last_month', 'median'),
            product_count=('purchased_last_month', 'count')
        ).reset_index()
        
        if exclude_outliers:
            tag_sales_filtered = tag_sales[tag_sales['product_count'] >= 10]
            if len(tag_sales_filtered) > 0:
                tag_sales = tag_sales_filtered
                
        tag_sales = tag_sales.sort_values(by='median_sales', ascending=True).tail(10).copy()
        
        # Add ⚠️ warning icon for tags with N < 30
        tag_sales['tag_display'] = tag_sales.apply(
            lambda r: f"⚠️ {r['sustainability_tags']} (N={r['product_count']:,})"
            if r['product_count'] < 30
            else f"{r['sustainability_tags']} (N={r['product_count']:,})",
            axis=1
        )
        
        # Ensure log scale safety (avoid log(0))
        if use_log_scale:
            tag_sales['plot_sales'] = np.maximum(1.0, tag_sales['median_sales'])
        else:
            tag_sales['plot_sales'] = tag_sales['median_sales']
            
        fig_bar = px.bar(
            tag_sales,
            x='plot_sales',
            y='tag_display',
            orientation='h',
            color_discrete_sequence=['#0D9488'], # Vivid solid Teal bar color
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
            marker_line_color='#0F766E',
            marker_line_width=1,
            hovertemplate='<b>%{y}</b><br>Median Monthly Sales: %{text:,.0f} units<extra></extra>'
        )
        fig_bar.update_layout(
            height=450,
            font=PLOTLY_FONT,
            showlegend=False,
            margin=dict(l=10, r=40, t=20, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(148,163,184,0.2)'),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_bar, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
    else:
        st.info(t('no_tag_data'))

# --- TAB 2: ECONOMETRIC MODEL & FEATURE IMPORTANCE ---
with tabs[1]:
    if len(filtered_df) > 10:
        # Sample size safeguard for overall model
        if len(filtered_df) < 30:
            st.warning(t('small_sample_warning').format(n=len(filtered_df)))
        
        # Build regressor matrix with conceptual grouping:
        # (1) Green Sustainability Badges + Interaction
        # (2) Seller/Functional Controls
        # (3) Market & Product Controls
        green_tag_cols = [f'tag_{tag}' for tag in GREEN_TAGS]
        control_tag_cols = [f'tag_{tag}' for tag in CONTROL_TAGS]
        
        X_vars = (green_tag_cols
                  + ['interaction_price_green']
                  + control_tag_cols
                  + ['log_price', 'product_rating', 'log_reviews',
                     'is_sponsored_binary', 'has_coupon_binary'])
        
        # Define variable display order for grouped table
        var_order = (['const']
                     + green_tag_cols
                     + ['interaction_price_green']
                     + control_tag_cols
                     + ['log_price', 'product_rating', 'log_reviews',
                        'is_sponsored_binary', 'has_coupon_binary'])
        var_order_map = {v: i for i, v in enumerate(var_order)}
        
        # ----------------------------------------------------
        # CONDITION A: Parametric Regression Models (OLS / Ridge)
        # ----------------------------------------------------
        if selected_model_key in ["ols", "ridge"]:
            st.subheader(t('model_title') if selected_model_key == "ols" else "Ridge Regression Analysis / การถดถอยแบบ Ridge")
            st.markdown(t('dep_var_label'))
            
            X = filtered_df[X_vars]
            X_const = sm.add_constant(X)
            y = filtered_df['log_purchased']
            
            tag_n_sizes = get_tag_sample_sizes(filtered_df)
            small_sample_vars = {col for col, n in tag_n_sizes.items() if n < 30}
            
            # Warn if any tag has N<30
            small_tags_list = [col.replace('tag_', '') for col in small_sample_vars]
            if small_tags_list:
                tags_str = ', '.join(small_tags_list)
                st.info(f"ℹ️ Tags with N < 30 in current filter: **{tags_str}**. "
                        f"Coefficients for these tags are annotated with ⚠️ and significance is suppressed.")
            
            # Build variable display names with N annotation for tags
            def get_var_display(var):
                base = translate_var_name(var, lang)
                if var.startswith('tag_'):
                    n = tag_n_sizes.get(var, 0)
                    if var in small_sample_vars:
                        return f"{base} ⚠️ (N={n})"
                    return f"{base} (N={n})"
                return base

            if selected_model_key == "ols":
                model_fit = sm.OLS(y, X_const).fit()
                
                results_df = pd.DataFrame({
                    '_orig_var': list(model_fit.params.index),
                    '_p_val_raw': model_fit.pvalues.values,
                    '_sort': [var_order_map.get(v, 999) for v in model_fit.params.index],
                    t('col_group'): [get_var_group(v) for v in model_fit.params.index],
                    t('col_variable'): [get_var_display(v) for v in model_fit.params.index],
                    t('col_coef'): model_fit.params.values,
                    t('col_std_err'): model_fit.bse.values,
                    t('col_t_val'): model_fit.tvalues.values,
                    t('col_p_val'): model_fit.pvalues.values,
                    t('col_sig'): [
                        '⚠️ N<30' if v in small_sample_vars else get_sig_stars(p)
                        for v, p in zip(model_fit.params.index, model_fit.pvalues.values)
                    ]
                })
                
                results_df = results_df.sort_values('_sort').reset_index(drop=True)
                
                display_cols = [t('col_group'), t('col_variable'), t('col_coef'),
                                t('col_std_err'), t('col_t_val'), t('col_p_val'), t('col_sig')]
                display_df = results_df[display_cols].copy()
                
                display_df[t('col_coef')] = results_df.apply(
                    lambda r: f"{r[t('col_coef')]:.4f} {r[t('col_sig')]}", axis=1
                )
                display_df[t('col_std_err')] = results_df[t('col_std_err')].apply(lambda x: f"{x:.4f}")
                display_df[t('col_t_val')] = results_df[t('col_t_val')].apply(lambda x: f"{x:.3f}")
                display_df[t('col_p_val')] = results_df.apply(
                    lambda r: "—" if r['_orig_var'] in small_sample_vars
                    else ("< 0.001" if r['_p_val_raw'] < 0.001 else f"{r['_p_val_raw']:.4f}"),
                    axis=1
                )
                
                def highlight_significant(row):
                    idx = row.name
                    orig_var = results_df.loc[idx, '_orig_var']
                    p_val = results_df.loc[idx, '_p_val_raw']
                    n_cols = len(row)
                    if orig_var in small_sample_vars:
                        return ['color: #94A3B8; font-style: italic;'] * n_cols
                    if p_val < 0.05:
                        return ['background-color: rgba(13, 148, 136, 0.12); font-weight: 500;'] * n_cols
                    return [''] * n_cols

                st.dataframe(
                    display_df.style.apply(highlight_significant, axis=1),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown(t('sig_legend'))
                st.markdown(t('interaction_hypothesis'))
                
                m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
                m_col1.metric(t('metric_r2'), f"{model_fit.rsquared:.4f}")
                m_col2.metric(t('metric_adj_r2'), f"{model_fit.rsquared_adj:.4f}")
                m_col3.metric(t('metric_f_stat'), f"{model_fit.fvalue:.2f}")
                m_col4.metric(t('metric_n_obs'), f"{int(model_fit.nobs):,}")
                m_col5.metric(t('metric_residual_var'), f"{model_fit.mse_resid:.4f}")

            else: # ridge
                from sklearn.linear_model import Ridge as SKRidge
                
                # Fit Ridge Regressor
                ridge_sk = SKRidge(alpha=1.0)
                ridge_sk.fit(X, y)
                
                # Coefficients mapping
                coefs = [ridge_sk.intercept_] + list(ridge_sk.coef_)
                var_names = ['const'] + X_vars
                
                results_df = pd.DataFrame({
                    '_orig_var': var_names,
                    '_sort': [var_order_map.get(v, 999) for v in var_names],
                    t('col_group'): [get_var_group(v) for v in var_names],
                    t('col_variable'): [get_var_display(v) for v in var_names],
                    t('col_coef'): coefs,
                    t('col_std_err'): "N/A (Regularized)",
                    t('col_t_val'): "N/A",
                    t('col_p_val'): "—",
                    t('col_sig'): "ns"
                })
                
                results_df = results_df.sort_values('_sort').reset_index(drop=True)
                
                display_cols = [t('col_group'), t('col_variable'), t('col_coef'),
                                t('col_std_err'), t('col_t_val'), t('col_p_val'), t('col_sig')]
                display_df = results_df[display_cols].copy()
                
                display_df[t('col_coef')] = results_df[t('col_coef')].apply(lambda x: f"{x:.4f}")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                st.info("ℹ️ Ridge regression is a regularized estimator. Standard errors and P-values are not analytically defined due to regularization bias.")
                
                # Compute training metrics for Ridge
                y_pred = ridge_sk.predict(X)
                from sklearn.metrics import r2_score as sk_r2
                r2_val = sk_r2(y, y_pred)
                n_obs = len(y)
                p_vars = X.shape[1]
                adj_r2_val = 1 - (1 - r2_val) * (n_obs - 1) / (n_obs - p_vars - 1)
                mse_resid = np.mean((y - y_pred) ** 2)
                
                m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
                m_col1.metric(t('metric_r2'), f"{r2_val:.4f}")
                m_col2.metric(t('metric_adj_r2'), f"{adj_r2_val:.4f}")
                m_col3.metric(t('metric_f_stat'), "N/A")
                m_col4.metric(t('metric_n_obs'), f"{n_obs:,}")
                m_col5.metric(t('metric_residual_var'), f"{mse_resid:.4f}")

        # ----------------------------------------------------
        # CONDITION B: Non-parametric Tree Models (RF / XGB)
        # ----------------------------------------------------
        else:
            st.subheader(f"Feature Importance & Metrics - {model_options[selected_model_key]}")
            st.markdown("Non-parametric model: Coefficient estimates, standard errors, and P-values are not defined.")
            
            X = filtered_df[X_vars]
            y = filtered_df['log_purchased']
            
            # Dynamic Train-Test Split (80/20) for metrics calculation on current filters
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42
            )
            
            # Model execution
            if selected_model_key == "rf":
                from sklearn.ensemble import RandomForestRegressor as SKRF
                model_fit = SKRF(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                from xgboost import XGBRegressor as SKXGB
                model_fit = SKXGB(n_estimators=50, learning_rate=0.1, random_state=42, n_jobs=-1)
                
            model_fit.fit(X_train, y_train)
            y_pred = model_fit.predict(X_test)
            
            # Metrics
            r2_val = r2_score(y_test, y_pred)
            rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
            mae_val = mean_absolute_error(y_test, y_pred)
            
            met_col1, met_col2, met_col3, met_col4 = st.columns(4)
            met_col1.metric("R² Score (Test set)", f"{r2_val:.4f}")
            met_col2.metric("RMSE (Test set)", f"{rmse_val:.4f}")
            met_col3.metric("MAE (Test set)", f"{mae_val:.4f}")
            met_col4.metric("Samples N (Train/Test)", f"{len(X_train):,} / {len(X_test):,}")
            
            # Feature Importance
            importances = model_fit.feature_importances_
            feature_imp_df = pd.DataFrame({
                "Feature / ตัวแปรพยากรณ์": [translate_var_name(v, lang) for v in X_vars],
                "Importance / ระดับความสำคัญ": importances
            }).sort_values("Importance / ระดับความสำคัญ", ascending=True)
            
            # Feature Importance Plot
            fig_importance = px.bar(
                feature_imp_df,
                x='Importance / ระดับความสำคัญ',
                y='Feature / ตัวแปรพยากรณ์',
                orientation='h',
                color='Importance / ระดับความสำคัญ',
                color_continuous_scale='Tealgrn',
                title=f"Feature Importance Map ({model_options[selected_model_key]})"
            )
            fig_importance.update_layout(
                height=450,
                font=PLOTLY_FONT,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='rgba(148,163,184,0.2)'),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_importance, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
            
    else:
        st.warning("Insufficient data under selected filters to train or fit models.")

# --- TAB 3: TAG BREAKDOWN ---
with tabs[2]:
    st.subheader(t('breakdown_title'))
    # Strictly display genuine GREEN_TAGS ONLY in the selectbox
    selected_tag = st.selectbox(t('select_tag_label'), GREEN_TAGS)
    
    sub_df = filtered_df[filtered_df['sustainability_tags'] == selected_tag].copy()
    
    if len(sub_df) > 0:
        # N < 30 safeguard for selected tag
        tag_n = len(sub_df)
        if tag_n < 30:
            st.warning(t('small_sample_warning').format(n=tag_n))
        
        # 1. Batch Prediction on sub_df using selected_model_key
        sub_df_X = pd.DataFrame({
            'log_price': np.log1p(sub_df['discounted_price']),
            'is_green': 1.0,
            'log_price_x_is_green': np.log1p(sub_df['discounted_price']),
            'rating': sub_df['product_rating'].astype(float),
            'reviews_count': sub_df['total_reviews'].astype(float)
        })
        
        model_obj = load_local_model(selected_model_key)
        if model_obj is not None:
            pred_log_arr = model_obj.predict(sub_df_X)
            sub_df['predicted_sales'] = np.maximum(0, np.expm1(pred_log_arr))
        else:
            sub_df['predicted_sales'] = sub_df['purchased_last_month']
            
        tag_pred_avg_sales = sub_df['predicted_sales'].mean()
        
        # 2. Metric cards summary (5 Cards including Predicted Sales KPI)
        st.markdown(f"#### {t('tag_metrics_summary')}: `{selected_tag}`")
        tc1, tc2, tc3, tc4, tc5 = st.columns(5)
        
        tag_prod_count = len(sub_df)
        tag_share = (tag_prod_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        tag_avg_price = sub_df['discounted_price'].mean()
        overall_avg_price = filtered_df['discounted_price'].mean()
        price_diff = tag_avg_price - overall_avg_price
        
        tag_avg_sales = sub_df['purchased_last_month'].mean()
        overall_avg_sales = filtered_df['purchased_last_month'].mean()
        sales_diff = tag_avg_sales - overall_avg_sales
        
        pred_diff = tag_pred_avg_sales - tag_avg_sales
        
        tag_avg_rating = sub_df['product_rating'].mean()
        overall_avg_rating = filtered_df['product_rating'].mean()
        rating_diff = tag_avg_rating - overall_avg_rating
        
        tc1.metric(t('metric_tag_count'), f"{tag_prod_count:,}", f"{tag_share:.1f}% of total")
        tc2.metric(t('metric_tag_avg_price'), f"${tag_avg_price:.2f}", f"${price_diff:+.2f} vs avg")
        tc3.metric(t('metric_tag_avg_sales'), f"{tag_avg_sales:,.0f}", f"{sales_diff:+,.0f} vs avg")
        tc4.metric(
            f"Predicted Sales ({selected_model_key.upper()})" if lang == "EN" else f"ยอดขายพยากรณ์ ({model_options[selected_model_key].split()[0]})",
            f"{tag_pred_avg_sales:,.0f}",
            f"{pred_diff:+,.0f} vs actual"
        )
        tc5.metric(t('metric_tag_avg_rating'), f"{tag_avg_rating:.2f}", f"{rating_diff:+.2f} vs avg")
        
        # 3. Badge Impact Analysis (Coefficient or Feature Importance)
        st.markdown("---")
        st.markdown(f"##### 🏷️ Badge Impact Analysis / ผลกระทบของป้าย `{selected_tag}` ({model_options[selected_model_key]})")
        
        impact_col1, impact_col2 = st.columns(2)
        
        if selected_model_key in ["ols", "ridge"]:
            col_name = f"tag_{selected_tag}"
            beta_val = 0.0
            sig_tag = "ns"
            
            if selected_model_key == "ols":
                green_tag_cols = [f'tag_{tag}' for tag in GREEN_TAGS]
                control_tag_cols = [f'tag_{tag}' for tag in CONTROL_TAGS]
                X_vars_full = (green_tag_cols + ['interaction_price_green'] + control_tag_cols
                               + ['log_price', 'product_rating', 'log_reviews', 'is_sponsored_binary', 'has_coupon_binary'])
                X_full = sm.add_constant(filtered_df[X_vars_full])
                y_full = filtered_df['log_purchased']
                ols_fit = sm.OLS(y_full, X_full).fit()
                if col_name in ols_fit.params:
                    beta_val = ols_fit.params[col_name]
                    p_val_tag = ols_fit.pvalues[col_name]
                    sig_tag = get_sig_stars(p_val_tag)
            else: # ridge
                from sklearn.linear_model import Ridge as SKRidge
                green_tag_cols = [f'tag_{tag}' for tag in GREEN_TAGS]
                control_tag_cols = [f'tag_{tag}' for tag in CONTROL_TAGS]
                X_vars_full = (green_tag_cols + ['interaction_price_green'] + control_tag_cols
                               + ['log_price', 'product_rating', 'log_reviews', 'is_sponsored_binary', 'has_coupon_binary'])
                ridge_sk = SKRidge(alpha=1.0)
                ridge_sk.fit(filtered_df[X_vars_full], filtered_df['log_purchased'])
                if col_name in X_vars_full:
                    beta_val = ridge_sk.coef_[X_vars_full.index(col_name)]
                sig_tag = "ns (Regularized)"
                
            pct_impact = (np.exp(beta_val) - 1) * 100
            
            with impact_col1:
                st.metric("Regression Coefficient (β)", f"{beta_val:.4f}", f"{sig_tag}")
            with impact_col2:
                st.metric("Model-Predicted Conditional Effect", f"{pct_impact:+.1f}%")
                
        else: # rf or xgboost
            X_vars_ml = ['log_price', 'is_green', 'log_price_x_is_green', 'rating', 'reviews_count']
            if hasattr(model_obj, 'feature_importances_'):
                imp = model_obj.feature_importances_
                is_green_imp = imp[X_vars_ml.index('is_green')] if 'is_green' in X_vars_ml else 0.0
                interaction_imp = imp[X_vars_ml.index('log_price_x_is_green')] if 'log_price_x_is_green' in X_vars_ml else 0.0
                total_green_imp = is_green_imp + interaction_imp
            else:
                is_green_imp, interaction_imp, total_green_imp = 0.0, 0.0, 0.0
                
            with impact_col1:
                st.metric("Green Badge Feature Importance", f"{is_green_imp * 100:.2f}%")
            with impact_col2:
                st.metric("Total Green Effect Importance (with Price Interaction)", f"{total_green_imp * 100:.2f}%")

        # 4. Scatter Plot Overlay (Actual vs Predicted Sales)
        st.markdown("---")
        fig_scatter = go.Figure()
        
        # Trace 1: Actual Monthly Sales
        fig_scatter.add_trace(go.Scatter(
            x=sub_df['discounted_price'],
            y=sub_df['purchased_last_month'],
            mode='markers',
            name='Actual Sales (ยอดขายจริง)',
            marker=dict(size=8, color='#0D9488', opacity=0.75),
            text=sub_df['product_title'],
            hovertemplate='<b>%{text}</b><br>Price: $%{x:.2f}<br>Actual Sales: %{y:,.0f} units<extra></extra>'
        ))
        
        # Trace 2: Model-Predicted Monthly Sales
        fig_scatter.add_trace(go.Scatter(
            x=sub_df['discounted_price'],
            y=sub_df['predicted_sales'],
            mode='markers',
            name=f'Predicted Sales ({model_options[selected_model_key].split()[0]})',
            marker=dict(size=8, color='#F59E0B', symbol='diamond', opacity=0.85),
            text=sub_df['product_title'],
            hovertemplate=f'<b>%{{text}}</b><br>Price: $%{{x:.2f}}<br>Predicted Sales ({model_options[selected_model_key].split()[0]}): %{{y:,.0f}} units<extra></extra>'
        ))
        
        fig_scatter.update_layout(
            title=t('scatter_title').format(tag=selected_tag) + f" — Actual vs. Predicted ({model_options[selected_model_key]})",
            height=500,
            font=PLOTLY_FONT,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(148,163,184,0.2)', title=t('scatter_xlabel')),
            yaxis=dict(showgrid=True, gridcolor='rgba(148,163,184,0.2)', title=t('scatter_ylabel'))
        )
        st.plotly_chart(fig_scatter, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
    else:
        st.warning(t('no_tag_products_found'))

# --- TAB 4: ASSOCIATION SIMULATOR (Causal-language-free) ---
with tabs[3]:
    st.subheader(t('sim_title'))
    st.write(t('sim_desc'))
    
    if len(filtered_df) > 10:
        col_a, col_b = st.columns(2)
        with col_a:
            sim_price = st.number_input(t('sim_price_label'), value=49.99, min_value=1.0, step=5.0)
            sim_rating = st.slider(t('sim_rating_label'), 1.0, 5.0, 4.5, 0.1)
            sim_reviews = st.number_input(t('sim_reviews_label'), value=500, min_value=0, step=50)
        
        with col_b:
            sim_sponsored = st.checkbox(t('sim_sponsored_label'), value=True)
            sim_coupon = st.checkbox(t('sim_coupon_label'), value=False)
            # Strictly display genuine GREEN_TAGS ONLY in the simulator selectbox
            sim_tag = st.selectbox(t('sim_tag_label'), [t('sim_none_tag')] + GREEN_TAGS)
            
        # Get selected model prediction for standard product (is_green=0)
        pred_units_base, r2, rmse = get_model_prediction(
            selected_model_key, sim_price, 0, sim_rating, sim_reviews
        )
        
        # Compute standard error bounds (fallback to RMSE from training metrics if available)
        std_err = rmse if rmse is not None else 0.8
        
        # Base Prediction 95% Confidence Interval
        pred_log_base = np.log1p(pred_units_base)
        ci_lower_base = max(0, np.exp(pred_log_base - 1.96 * std_err) - 1)
        ci_upper_base = max(0, np.exp(pred_log_base + 1.96 * std_err) - 1)
        
        st.markdown("---")
        
        # Model Info Card displaying R2 and RMSE of currently selected model
        inf_col1, inf_col2, inf_col3 = st.columns([2, 1, 1])
        with inf_col1:
            st.info(f"**Selected Model / แบบจำลองพยากรณ์:** `{model_options[selected_model_key]}`")
        with inf_col2:
            st.metric("R² Score (Test set)", f"{r2:.4f}" if r2 is not None else "N/A")
        with inf_col3:
            st.metric("RMSE (Test set)", f"{rmse:.4f}" if rmse is not None else "N/A")
            
        st.subheader(t('sim_result_header'))
        
        if sim_tag != t('sim_none_tag'):
            clean_tag_name = sim_tag
            
            # Predict sales for green product (is_green=1)
            pred_units_tagged, _, _ = get_model_prediction(
                selected_model_key, sim_price, 1, sim_rating, sim_reviews
            )
            
            # Tagged Prediction 95% Confidence Interval
            pred_log_tagged = np.log1p(pred_units_tagged)
            ci_lower_tagged = max(0, np.exp(pred_log_tagged - 1.96 * std_err) - 1)
            ci_upper_tagged = max(0, np.exp(pred_log_tagged + 1.96 * std_err) - 1)
            
            diff_units = pred_units_tagged - pred_units_base
            pct_diff = (diff_units / pred_units_base * 100) if pred_units_base > 0 else 0
            
            # N < 30 safeguard for the selected tag
            tag_n = int(filtered_df[f'tag_{clean_tag_name}'].sum())
            if tag_n < 30:
                st.warning(t('small_sample_warning').format(n=tag_n))
            
            res_col1, res_col2, res_col3 = st.columns(3)
            
            res_col1.metric(
                t('sim_standard_product'),
                f"{int(pred_units_base):,} {t('sim_units_unit')}"
            )
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
            
            res_col3.metric(
                t('sim_conditional_diff'),
                f"{pct_diff:+.1f}%",
                f"{diff_units:+,.0f} units"
            )
            
            # Side-by-side bar chart comparison
            comp_df = pd.DataFrame({
                'Product Type': [t('sim_standard_label'), f"{t('sim_tagged_label')}: {clean_tag_name}"],
                'Predicted Monthly Sales': [pred_units_base, pred_units_tagged]
            })
            
            fig_comp = px.bar(
                comp_df,
                x='Product Type',
                y='Predicted Monthly Sales',
                color='Product Type',
                color_discrete_sequence=['#64748B', '#0D9488'],
                text='Predicted Monthly Sales',
                title=t('sim_comparison_chart_title')
            )
            fig_comp.update_traces(
                texttemplate='%{text:,.0f} units',
                textposition='outside'
            )
            fig_comp.update_layout(
                height=380,
                font=PLOTLY_FONT,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(showgrid=True, gridcolor='rgba(148,163,184,0.2)', title='Units Sold / Month')
            )
            st.plotly_chart(fig_comp, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
        else:
            st.info(f"**{t('sim_standard_product')}:** **{int(pred_units_base):,} {t('sim_units_unit')}**")
            st.markdown(
                f'<p style="color: #333333; font-size: 14px; font-weight: 500; margin-top: 4px;"><b>{t("ci_label")}:</b> [{int(ci_lower_base):,} – {int(ci_upper_base):,}]</p>',
                unsafe_allow_html=True
            )
            st.info(t('sim_select_tag_prompt'))
            
        # --- MODEL PREDICTION COMPARISON (BAR CHART & DATA TABLE) ---
        st.markdown("---")
        st.subheader("Model Prediction Comparison" if lang == "EN" else "เปรียบเทียบผลพยากรณ์ระหว่างโมเดล")
        
        # Query /api/compare or fallback
        compare_green_status = 1 if sim_tag != t('sim_none_tag') else 0
        compare_data = get_compare_predictions(
            sim_price, compare_green_status, sim_rating, sim_reviews
        )
        
        compare_rows = []
        selected_pred = pred_units_tagged if sim_tag != t('sim_none_tag') else pred_units_base
        
        for item in compare_data:
            m_pred = item.get("predicted_sales", 0.0)
            pct_diff = ((m_pred - selected_pred) / selected_pred * 100) if selected_pred > 0 else 0.0
            compare_rows.append({
                "Model / แบบจำลอง": item["display_name"],
                "Predicted Sales / พยากรณ์ยอดขาย (ชิ้น)": int(round(m_pred)),
                "Difference vs Selected / ต่างจากโมเดลปัจจุบัน": f"{pct_diff:+.1f}%" if item["model_name"] != selected_model_key else "Selected (0.0%)",
                "R² Score": f"{item['metrics']['R2_Score']:.4f}" if item["metrics"].get("R2_Score") is not None else "N/A",
                "RMSE": f"{item['metrics']['RMSE']:.4f}" if item["metrics"].get("RMSE") is not None else "N/A"
            })
            
        compare_df = pd.DataFrame(compare_rows)
        
        fig_all_comp = px.bar(
            compare_df,
            x='Model / แบบจำลอง',
            y='Predicted Sales / พยากรณ์ยอดขาย (ชิ้น)',
            color='Model / แบบจำลอง',
            color_discrete_sequence=['#64748B', '#475569', '#0D9488', '#10B981'],
            text='Predicted Sales / พยากรณ์ยอดขาย (ชิ้น)',
            title="Model-Predicted Sales Volume Comparison" if lang == "EN" else "เปรียบเทียบปริมาณยอดขายพยากรณ์ระหว่างแบบจำลอง"
        )
        fig_all_comp.update_traces(
            texttemplate='%{text:,.0f} units',
            textposition='outside'
        )
        fig_all_comp.update_layout(
            height=380,
            font=PLOTLY_FONT,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(showgrid=True, gridcolor='rgba(148,163,184,0.2)', title='Predicted Units / Month')
        )
        st.plotly_chart(fig_all_comp, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
        
        # Display summary comparison table
        st.dataframe(compare_df, use_container_width=True, hide_index=True)
        
        # Academic disclaimer (always shown)
        st.markdown("---")
        st.markdown(
            f'<p style="color: #4A5568; font-style: italic; font-size: 13px; margin-top: 12px;"><b>หมายเหตุ / Note:</b> {t("disclaimer_text")}</p>',
            unsafe_allow_html=True
        )
    else:
        st.warning("Insufficient data under selected filters to run simulator.")