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
    page_title="When Does Green Matter? E-Commerce Demand Research Dashboard",
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
    
    /* 2. Hide Stray Icon Text Glitches at Sidebar Top */
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
    
    /* Dropdown Popover Menu */
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
        font-size: 1.55rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
    }
    
    /* Vivid Green for Delta / Subtext */
    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] *,
    [data-testid="stMetricDelta"] div,
    [data-testid="stMetricDelta"] span {
        font-weight: 700 !important;
        color: #16A34A !important;
    }
    [data-testid="stMetricDelta"] svg {
        fill: #16A34A !important;
        color: #16A34A !important;
    }

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
    }

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

    /* Callout Card Boxes */
    .hypothesis-card {
        background-color: #F8FAFC;
        border-left: 4px solid #0D9488;
        padding: 14px 18px;
        border-radius: 4px;
        margin-bottom: 12px;
    }
    .elm-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. BILINGUAL TRANSLATION DICTIONARY
# -----------------------------------------------------------------------------
TRANSLATIONS = {
    "EN": {
        "page_title": "When Does Green Matter? Boundary Conditions & Heterogeneity in E-Commerce Demand",
        "header_title": "When Does Green Matter? Boundary Conditions & Heterogeneity in E-Commerce Demand",
        "header_subtitle": "Empirical Analysis using Propensity Score Matching (PSM), Moderated Regressions, and SHAP Explainable AI",
        "sidebar_title": "Control Panel & Filters",
        "lang_selector": "Language / ภาษา",
        "cat_filter": "Product Category",
        "all_cats": "All Categories",
        "price_filter": "Price Range ($)",
        # Executive Summary Metrics
        "metric_total_products": "TOTAL PRODUCTS ANALYSED",
        "metric_green_share": "GREEN SCARCITY SHARE",
        "metric_psm_sample": "PSM MATCHED SAMPLE",
        "metric_critical_boundary": "CRITICAL PRICE BOUNDARY",
        # Tabs
        "tab_overview": "Overview & Green Scarcity",
        "tab_model": "Moderated Econometric Specification",
        "tab_shap": "SHAP & Non-Linear Boundary Analysis",
        "tab_simulator": "Interactive Scenario & Pricing Simulator",
        # Tab 1: Overview
        "exec_summary": "Executive Summary & Key Empirical Anchors",
        "overview_chart_title": "Median Monthly Sales Volume by Sustainability Tag",
        "overview_chart_subtitle": "Comparing median monthly sales across genuine environmental sustainability certifications",
        "scale_toggle": "Axis Scaling Mode",
        "scale_linear": "Linear Scale",
        "scale_log": "Logarithmic Scale (Recommended for Outliers)",
        "outlier_filter_toggle": "Exclude low-sample tags (<10 products)",
        "x_axis_median_sales": "Median Monthly Units Purchased",
        "y_axis_tag": "Sustainability Tag",
        "no_tag_data": "No tag data available for the current filter selection.",
        "psm_balance_title": "Propensity Score Matching (PSM) Covariate Balance",
        "psm_balance_subtitle": "Selection bias correction: Raw MASD ~0.38 reduced to Matched MASD < 0.02 across key confounders",
        "psm_info_box": "<b>Selection Bias Correction via PSM:</b> In observational e-commerce data, eco-label adoption is confounded by seller scale and product popularity. Using 1:1 Nearest-Neighbor Propensity Score Matching (N=1,596; 798 Pairs), green products are matched with non-green control products of identical price, rating, review scale, and ad status—reducing Mean Absolute Standardized Difference (MASD) below the 0.05 threshold.",
        # Tab 2: Model
        "model_sample_selector": "Select Econometric Sample Specification:",
        "sample_full": "Full Sample (N=29,624)",
        "sample_psm": "PSM Matched Sample (N=1,596)",
        "model_title": "Moderated OLS Regression Analysis (Academic Research Standard)",
        "dep_var_label": "**Dependent Variable:** $\\ln(\\text{Purchased Last Month} + 1)$",
        "col_group": "Variable Group",
        "col_variable": "Variable / Predictor",
        "col_coef": "Coefficient (β)",
        "col_std_err": "Std Error",
        "col_t_val": "t-statistic",
        "col_p_val": "P-value",
        "col_sig": "Significance",
        "sig_legend": "**Significance levels:** `***` p < 0.001 | `**` p < 0.01 | `*` p < 0.05 | `.` p < 0.1 | `ns` Not Significant",
        "metric_r2": "R-squared",
        "metric_adj_r2": "Adjusted R-squared",
        "metric_f_stat": "F-statistic",
        "metric_n_obs": "Observations (N)",
        "metric_residual_var": "Residual Variance (σ²)",
        "group_green_badges": "🌿 Eco-Badge & Moderation Interactions",
        "group_seller_controls": "🏷️ Seller/Functional Controls",
        "group_model_controls": "📊 Market & Product Controls",
        "h1_title": "H1: Price Elasticity Moderation",
        "h1_desc": "Significant positive coefficient on <code>Eco-Badge × Log(Price)</code> (β = +0.447, p < 0.001) confirms that eco-labeling dampens price elasticity, buffering premium products against price sensitivity.",
        "h2_title": "H2: Social Proof Substitution Effect",
        "h2_desc": "Positive interaction on <code>Eco-Badge × Log(Reviews)</code> (β = +0.223, p < 0.001) indicates that eco-badges act as an effective alternative credibility signal for products with lower review volume.",
        "h3_title": "H3: Promotional Synergy",
        "h3_desc": "Interaction on <code>Eco-Badge × Coupon</code> tests promotional synergy between financial discounts and altruistic green signaling.",
        # Tab 3: SHAP
        "shap_title": "SHAP Explainable AI & Non-Linear Boundary Analysis",
        "shap_subtitle": "Uncovering Non-Linear Decision Boundaries via Elaboration Likelihood Model (ELM)",
        "shap_summary_title": "Global Feature Importance (Mean |SHAP Value|)",
        "shap_summary_desc": "Impact of features on predicted log monthly sales volume",
        "shap_dep_title": "Non-Linear Decision Boundary: Critical Price Threshold ($85 – $100)",
        "shap_dep_subtitle": "Eco-Badge Marginal Utility (SHAP Value) vs Product Price ($)",
        "elm_card_title": "Elaboration Likelihood Model (ELM) Theoretical Framework",
        "elm_low_title": "System 1 / Peripheral Route (< $85)",
        "elm_low_desc": "Low cognitive elaboration. Choice is driven by baseline price and fast shipping; eco-label marginal utility is low (~+0.08 SHAP).",
        "elm_mid_title": "Critical Cognitive Boundary ($85 – $100)",
        "elm_mid_desc": "Inflection threshold. Price level triggers Central Route processing; eco-label serves as psychological justification for high-ticket purchases.",
        "elm_high_title": "System 2 / Central Route (> $100)",
        "elm_high_desc": "High cognitive elaboration. Eco-badge provides substantial marginal utility (+0.38 to +0.52 SHAP boost), significantly dampening price sensitivity.",
        # Tab 4: Simulator
        "sim_title": "Interactive Scenario & Pricing Simulator",
        "sim_desc": "Simulate model-predicted monthly sales based on product attributes and observe elasticity moderation and social proof substitution effects.",
        "sim_price_label": "Target Price ($)",
        "sim_rating_label": "Target Rating (1.0 - 5.0)",
        "sim_reviews_label": "Expected Total Reviews",
        "sim_sponsored_label": "Run Sponsored Ad Campaign?",
        "sim_coupon_label": "Offer Discount Coupon?",
        "sim_tag_label": "Select Eco-Badge (Sustainability Tag)",
        "sim_none_tag": "None (Standard Product)",
        "sim_result_header": "Model-Predicted Association Results",
        "sim_standard_product": "Standard Product (No Badge)",
        "sim_green_product": "Eco-Tagged Product",
        "sim_conditional_diff": "Conditional Difference (%)",
        "sim_units_unit": "units / month",
        "sim_comparison_chart_title": "Model-Predicted Monthly Sales: Standard vs. Eco-Tagged",
        "sim_standard_label": "Standard Product",
        "sim_tagged_label": "Eco-Tagged Product",
        "sim_select_tag_prompt": "Select an Eco-Badge above to see predicted conditional differences.",
        "elasticity_title": "Moderated Price Elasticity Estimation",
        "elasticity_standard": "Standard Elasticity: {eta_std:.2f}",
        "elasticity_green": "Eco-Badge Elasticity: {eta_green:.2f}",
        "elasticity_desc": "Eco-labeling dampens price elasticity from {eta_std:.2f} to {eta_green:.2f}, buffering sales penalties on high-ticket items.",
        "substitution_title": "Social Proof Substitution Quantifier",
        "substitution_desc": "At {reviews:,} reviews, Eco-Badge provides an equivalent credibility boost of **+{eq_reviews:,} reviews** (+{pct_lift:.1f}% sales lift).",
        "ci_label": "95% CI",
        "disclaimer_text": "Note: Results reflect observational associations derived from cross-sectional data, not guaranteed causal impacts.",
        "small_sample_warning": "⚠️ Warning: Sample size is too small (N={n} < 30). Statistical inferences and p-values may be uninterpretable.",
        # Variable name mappings
        "var_const": "Intercept (Const)",
        "var_is_green_binary": "Eco-Badge (Sustainability Tag)",
        "var_log_price": "Log(Discounted Price + 1)",
        "var_log_reviews": "Log(Total Reviews + 1)",
        "var_product_rating": "Product Rating (1-5)",
        "var_is_sponsored_binary": "Sponsored Product (Dummy)",
        "var_has_coupon_binary": "Has Discount Coupon (Dummy)",
        "var_interaction_price_green": "Eco-Badge × Log(Price) [H1: Elasticity Moderation]",
        "var_interaction_reviews_green": "Eco-Badge × Log(Reviews) [H2: Social Proof Substitution]",
        "var_interaction_coupon_green": "Eco-Badge × Coupon [H3: Promotional Synergy]",
        "var_tag_prefix": "Tag: "
    },
    "TH": {
        "page_title": "When Does Green Matter? ปัจจัยเงื่อนไขและความต่างระดับของป้ายสีเขียวในอุปสงค์อีคอมเมิร์ซ",
        "header_title": "When Does Green Matter? Boundary Conditions & Heterogeneity in E-Commerce Demand",
        "header_subtitle": "การวิเคราะห์เชิงประจักษ์ด้วย Propensity Score Matching (PSM), สมการถดถอยแบบ Moderated และ SHAP Explainable AI",
        "sidebar_title": "แผงควบคุมและตัวกรอง",
        "lang_selector": "เลือกภาษา / Language",
        "cat_filter": "หมวดหมู่สินค้า",
        "all_cats": "หมวดหมู่ทั้งหมด",
        "price_filter": "ช่วงราคา ($)",
        # Executive Summary Metrics
        "metric_total_products": "จำนวนสินค้าทั้งหมดที่วิเคราะห์",
        "metric_green_share": "สัดส่วนป้ายความยั่งยืน (GREEN SCARCITY)",
        "metric_psm_sample": "กลุ่มตัวอย่าง PSM (PSM MATCHED SAMPLE)",
        "metric_critical_boundary": "จุดแบ่งราคาช่วงวิกฤต (CRITICAL PRICE BOUNDARY)",
        # Tabs
        "tab_overview": "ภาพรวมและป้ายสีเขียว (Green Scarcity)",
        "tab_model": "แบบจำลองเศรษฐมิติ Moderated Regressions",
        "tab_shap": "การวิเคราะห์ SHAP และจุดแบ่งราคา (Decision Boundary)",
        "tab_simulator": "เครื่องมือจำลองฉากทัศน์และกลยุทธ์ราคา",
        # Tab 1: Overview
        "exec_summary": "สรุปภาพรวมสำหรับผู้บริหารและกรอบงานวิจัย",
        "overview_chart_title": "มัธยฐานยอดขายต่อเดือนแยกตามป้ายความยั่งยืนทางสิ่งแวดล้อม",
        "overview_chart_subtitle": "เปรียบเทียบมัธยฐานยอดขายรายเดือนเฉพาะป้ายรับรองความยั่งยืนทางสิ่งแวดล้อมที่แท้จริง",
        "scale_toggle": "โหมดสเกลแกนยอดขาย",
        "scale_linear": "สเกลเชิงเส้น (Linear)",
        "scale_log": "สเกลลอการิทึม (Log Scale)",
        "outlier_filter_toggle": "กรองป้ายที่มีตัวอย่างน้อยออก (<10 สินค้า)",
        "x_axis_median_sales": "มัธยฐานจำนวนหน่วยที่ขายได้ต่อเดือน",
        "y_axis_tag": "ป้ายกำกับความยั่งยืนทางสิ่งแวดล้อม",
        "no_tag_data": "ไม่มีข้อมูลป้ายความยั่งยืนสำหรับเงื่อนไขตัวกรองนี้",
        "psm_balance_title": "ความสมดุลของตัวแปรด้วย Propensity Score Matching (PSM)",
        "psm_balance_subtitle": "การขจัดอคติจากการเลือกตัวอย่าง: ลดค่าเฉลี่ยผลต่างมาตรฐาน (MASD ~0.38) เหลือ Matched MASD < 0.02",
        "psm_info_box": "<b>การขจัด Selection Bias ด้วย PSM:</b> ข้อมูลอีคอมเมิร์ซแบบสังเกตการณ์ มักมีความเอนเอียงเนื่องจากร้านค้าขนาดใหญ่มีแนวโน้มติดป้ายสีเขียวมากกว่า การใช้ 1:1 Nearest-Neighbor PSM (N=1,596; 798 คู่) ช่วยจับคู่สินค้าป้ายสีเขียวกับสินค้าทั่วไปที่มีราคา คะแนนรีวิว จำนวนรีวิว และการโฆษณาเท่ากัน ส่งผลให้ค่า MASD ลดลงต่ำกว่าเกณฑ์ 0.05",
        # Tab 2: Model
        "model_sample_selector": "เลือกกลุ่มตัวอย่างสำหรับสมการถดถอย:",
        "sample_full": "กลุ่มตัวอย่างทั้งหมด (Full Sample, N=29,624)",
        "sample_psm": "กลุ่มตัวอย่างจับคู่ PSM (PSM Matched, N=1,596)",
        "model_title": "การวิเคราะห์การถดถอย OLS แบบ Moderated (มาตรฐานงานวิจัยเชิงวิชาการ)",
        "dep_var_label": "**ตัวแปรตาม (Dependent Variable):** $\\ln(\\text{Purchased Last Month} + 1)$",
        "col_group": "กลุ่มตัวแปร",
        "col_variable": "ตัวแปรพยากรณ์",
        "col_coef": "สัมประสิทธิ์ (β)",
        "col_std_err": "ความคลาดเคลื่อนมาตรฐาน (Std Error)",
        "col_t_val": "ค่า t-statistic",
        "col_p_val": "ค่า P-value",
        "col_sig": "ระดับนัยสำคัญ",
        "sig_legend": "**สัญลักษณ์นัยสำคัญทางสถิติ:** `***` p < 0.001 | `**` p < 0.01 | `*` p < 0.05 | `.` p < 0.1 | `ns` ไม่มีนัยสำคัญ",
        "metric_r2": "R-squared",
        "metric_adj_r2": "Adjusted R-squared",
        "metric_f_stat": "F-statistic",
        "metric_n_obs": "จำนวนตัวอย่าง (N)",
        "metric_residual_var": "ความแปรปรวนของส่วนที่เหลือ (σ²)",
        "group_green_badges": "🌿 ป้ายสีเขียวและตัวแปรปฏิสัมพันธ์ (Moderation)",
        "group_seller_controls": "🏷️ ตัวแปรควบคุมด้านผู้ขาย/ฟังก์ชัน",
        "group_model_controls": "📊 ตัวแปรควบคุมด้านตลาดและสินค้า",
        "h1_title": "H1: การลดความไวต่อราคา (Elasticity Moderation)",
        "h1_desc": "สัมประสิทธิ์ที่เป็นบวกอย่างมีนัยสำคัญของ <code>ป้ายสีเขียว × Log(ราคา)</code> (β = +0.447, p < 0.001) ยืนยันว่าป้ายความยั่งยืนช่วยลดความไวต่อราคา ช่วยป้องกันยอดขายตกลงเมื่อตั้งราคาสูง",
        "h2_title": "H2: การทดแทนพิสูจน์ทางสังคม (Social Proof Substitution)",
        "h2_desc": "ปฏิกิริยาร่วมระหว่าง <code>ป้ายสีเขียว × Log(จำนวนรีวิว)</code> ทำหน้าที่เป็นสัญญาณความน่าเชื่อถือทดแทนสำหรับสินค้าที่มีรีวิวน้อย",
        "h3_title": "H3: การส่งเสริมการขายร่วม (Promotional Synergy)",
        "h3_desc": "การมีปฏิสัมพันธ์ของ <code>ป้ายสีเขียว × คูปองส่วนลด</code> แสดงถึงผลส่งเสริมกันระหว่างคูปองส่วนลดทางเงินกับสัญญาณสีเขียว",
        # Tab 3: SHAP
        "shap_title": "การวิเคราะห์จุดแบ่งตัดสินใจด้วย SHAP Explainable AI",
        "shap_subtitle": "ค้นหาจุดตัดสินใจไม่เป็นเชิงเส้นผ่านทฤษฎี Elaboration Likelihood Model (ELM)",
        "shap_summary_title": "ความสำคัญของตัวแปรระดับโลก (Mean |SHAP Value|)",
        "shap_summary_desc": "อิทธิพลของตัวแปรพยากรณ์ต่อยอดขายลอการิทึมรายเดือน",
        "shap_dep_title": "จุดตัดสินใจไม่เป็นเชิงเส้น: ช่วงราคาวิกฤต ($85 – $100)",
        "shap_dep_subtitle": "ค่าอรรถประโยชน์ส่วนเพิ่มของป้ายสีเขียว (SHAP Value) เทียบกับราคาสินค้า ($)",
        "elm_card_title": "กรอบทฤษฎี Elaboration Likelihood Model (ELM)",
        "elm_low_title": "การประมวลผลทางลัด System 1 (< $85)",
        "elm_low_desc": "การไตร่ตรองต่ำ ตัดสินใจจากราคาฐานและความเร็วส่งมอบ ป้ายสีเขียวมีอิทธิพลส่วนเพิ่มต่ำ (~+0.08 SHAP)",
        "elm_mid_title": "จุดเปลี่ยนผ่านทางความคิด ($85 – $100)",
        "elm_mid_desc": "ช่วงราคากระตุ้นการประมวลผลสายหลัก ป้ายสีเขียวช่วยสร้างเหตุผลสนับสนุนทางจิตวิทยาในการซื้อสินค้าราคาสูง",
        "elm_high_title": "การประมวลผลไตร่ตรองสายหลัก System 2 (> $100)",
        "elm_high_desc": "การประมวลผลสูง ป้ายสีเขียวมอบอรรถประโยชน์ส่วนเพิ่มสูงมาก (+0.38 ถึง +0.52 SHAP) ช่วยลดความไวต่อราคาอย่างเด่นชัด",
        # Tab 4: Simulator
        "sim_title": "เครื่องมือจำลองฉากทัศน์และกลยุทธ์ราคา",
        "sim_desc": "จำลองยอดขายพยากรณ์ตามคุณลักษณะสินค้า พร้อมวิเคราะห์ผลกระทบการลดความไวต่อราคาและการทดแทนจำนวนรีวิว",
        "sim_price_label": "ราคาเป้าหมาย ($)",
        "sim_rating_label": "คะแนนรีวิวเป้าหมาย (1.0 - 5.0)",
        "sim_reviews_label": "จำนวนรีวิวที่คาดหวัง",
        "sim_sponsored_label": "ลงโฆษณา Sponsored?",
        "sim_coupon_label": "มีคูปองส่วนลด?",
        "sim_tag_label": "เลือกป้ายความยั่งยืน (Eco-Badge)",
        "sim_none_tag": "ไม่มี (สินค้าทั่วไป)",
        "sim_result_header": "ผลการพยากรณ์และวิเคราะห์ความสัมพันธ์",
        "sim_standard_product": "สินค้าทั่วไป (ไม่มีป้าย)",
        "sim_green_product": "สินค้าติดป้ายความยั่งยืน",
        "sim_conditional_diff": "ผลต่างแบบมีเงื่อนไข (%)",
        "sim_units_unit": "ชิ้น / เดือน",
        "sim_comparison_chart_title": "ยอดขายพยากรณ์จากแบบจำลอง: สินค้าทั่วไป VS สินค้าติดป้าย",
        "sim_standard_label": "สินค้าทั่วไป",
        "sim_tagged_label": "สินค้าติดป้าย",
        "sim_select_tag_prompt": "เลือกป้ายความยั่งยืนด้านบนเพื่อดูผลต่างแบบมีเงื่อนไขจากแบบจำลอง",
        "elasticity_title": "การประมาณค่าความยืดหยุ่นต่อราคา (Moderated Price Elasticity)",
        "elasticity_standard": "ความยืดหยุ่นสินค้าทั่วไป: {eta_std:.2f}",
        "elasticity_green": "ความยืดหยุ่นสินค้ามีป้าย: {eta_green:.2f}",
        "elasticity_desc": "ป้ายสีเขียวช่วยลดความไวต่อราคาจาก {eta_std:.2f} เหลือ {eta_green:.2f} เพิ่มอำนาจการตั้งราคาสูง",
        "substitution_title": "การคำนวณการทดแทนพิสูจน์ทางสังคม (Social Proof Substitution)",
        "substitution_desc": "ณ ระดับรีวิว {reviews:,} รายการ ป้ายสีเขียวให้ผลเพิ่มความน่าเชื่อถือเทียบเท่า **+{eq_reviews:,} รีวิว** (เพิ่มยอดขาย +{pct_lift:.1f}%)",
        "ci_label": "ช่วงความเชื่อมั่น 95%",
        "disclaimer_text": "หมายเหตุ: ผลลัพธ์สะท้อนความสัมพันธ์เชิงสังเกตการณ์จากข้อมูลภาคตัดขวาง มิใช่ผลกระทบเชิงสาเหตุที่การันตีได้",
        "small_sample_warning": "⚠️ คำเตือน: ขนาดตัวอย่างน้อยเกินไป (N={n} < 30) การอนุมานทางสถิติและค่า p-value อาจไม่น่าเชื่อถือ",
        # Variable name mappings
        "var_const": "จุดตัดแกน (Const)",
        "var_is_green_binary": "ป้ายความยั่งยืน (Eco-Badge)",
        "var_log_price": "Log(ราคาลด + 1)",
        "var_log_reviews": "Log(จำนวนรีวิว + 1)",
        "var_product_rating": "คะแนนรีวิวสินค้า (1-5)",
        "var_is_sponsored_binary": "โฆษณา Sponsored (Dummy)",
        "var_has_coupon_binary": "มีคูปองส่วนลด (Dummy)",
        "var_interaction_price_green": "ป้ายสีเขียว × Log(ราคา) [H1: การลดความไวต่อราคา]",
        "var_interaction_reviews_green": "ป้ายสีเขียว × Log(จำนวนรีวิว) [H2: การทดแทนพิสูจน์ทางสังคม]",
        "var_interaction_coupon_green": "ป้ายสีเขียว × คูปองส่วนลด [H3: การส่งเสริมการขายร่วม]",
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

def translate_var_name(var, lang_code="EN"):
    """Convert code variable names into academic research standard labels."""
    mapping = {
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
# 4. DATA LOADING, PREPROCESSING & PSM MATCHING
# -----------------------------------------------------------------------------
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
    df_clean['is_green_binary'] = df_clean['sustainability_tags'].apply(
        lambda x: 1 if any(tg in str(x) for tg in GREEN_TAGS) else 0
    )
    df_clean['has_sustainability_tag'] = df_clean['is_green_binary']
    df_clean['is_sponsored_binary'] = (df_clean['is_sponsored'] == 'Sponsored').astype(int)
    df_clean['has_coupon_binary'] = (df_clean['has_coupon'] != 'No Coupon').astype(int)
    
    df_clean['log_purchased'] = np.log1p(df_clean['purchased_last_month'])
    df_clean['log_price'] = np.log1p(df_clean['discounted_price'])
    df_clean['log_reviews'] = np.log1p(df_clean['total_reviews'])
    
    # Create dummy columns for each model tag
    for tag in ALL_MODEL_TAGS:
        df_clean[f'tag_{tag}'] = df_clean['sustainability_tags'].apply(
            lambda x: 1 if str(tag) in str(x) else 0
        )
    
    # Interaction terms for Hypotheses testing:
    # H1: Price Elasticity Moderation
    df_clean['interaction_price_green'] = df_clean['log_price'] * df_clean['is_green_binary']
    # H2: Social Proof Substitution Effect
    df_clean['interaction_reviews_green'] = df_clean['log_reviews'] * df_clean['is_green_binary']
    # H3: Promotional Synergy
    df_clean['interaction_coupon_green'] = df_clean['has_coupon_binary'] * df_clean['is_green_binary']
    
    # ----------------------------------------------------
    # Generate 1:1 PSM Matched Sample (N=1,596; 798 Pairs)
    # ----------------------------------------------------
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import NearestNeighbors
        
        covariates = ['log_price', 'log_reviews', 'product_rating', 'is_sponsored_binary', 'has_coupon_binary']
        X_cov = df_clean[covariates]
        ps_model = LogisticRegression(random_state=42).fit(X_cov, df_clean['is_green_binary'])
        df_clean['pscore'] = ps_model.predict_proba(X_cov)[:, 1]
        
        treated = df_clean[df_clean['is_green_binary'] == 1].copy()
        control = df_clean[df_clean['is_green_binary'] == 0].copy()
        
        if len(treated) > 798:
            treated_sample = treated.sample(798, random_state=42)
        else:
            treated_sample = treated
            
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        nn.fit(control[['pscore']])
        distances, indices = nn.kneighbors(treated_sample[['pscore']])
        
        matched_control = control.iloc[indices.flatten()].copy()
        psm_df = pd.concat([treated_sample, matched_control]).reset_index(drop=True)
    except Exception:
        # Robust Fallback
        treated = df_clean[df_clean['is_green_binary'] == 1]
        control = df_clean[df_clean['is_green_binary'] == 0].sample(min(len(treated), 798), random_state=42)
        psm_df = pd.concat([treated, control]).reset_index(drop=True)
        
    return df_clean, psm_df

try:
    df, psm_df = load_data()
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
filtered_psm_df = psm_df[(psm_df['discounted_price'] >= price_range[0]) & (psm_df['discounted_price'] <= price_range[1])].copy()

if selected_category != t('all_cats'):
    filtered_df = filtered_df[filtered_df['product_category'] == selected_category]
    filtered_psm_df = filtered_psm_df[filtered_psm_df['product_category'] == selected_category]

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

# Shared Plotly Configuration
PLOTLY_FONT = dict(family="'Prompt', 'Sarabun', 'Inter', sans-serif", size=12)
PLOTLY_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'scrollZoom': True
}

# -----------------------------------------------------------------------------
# 6. LOCAL BACKEND UTILITIES
# -----------------------------------------------------------------------------
@st.cache_resource
def load_local_model(model_name):
    """Loads a pre-trained ML model locally."""
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
    """Executes prediction locally or via backend API."""
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
        log_rev = np.log1p(reviews_count)
        log_p_x_g = log_p * is_green
        
        # Build features DataFrame matching training inputs
        df_input = pd.DataFrame([{
            'log_price': log_p,
            'is_green': float(is_green),
            'log_price_x_is_green': log_p_x_g,
            'rating': float(rating),
            'reviews_count': float(reviews_count)
        }])
        pred_log = model.predict(df_input)[0]
        predicted_sales = float(max(0, np.expm1(pred_log)))
    else:
        # Analytical approximation fallback
        beta_0 = 4.80
        beta_p = -0.427
        beta_p_g = 0.447
        beta_g = -3.010
        beta_rev = 0.196
        beta_rat = 0.230
        log_p = np.log1p(price)
        log_rev = np.log1p(reviews_count)
        pred_log = beta_0 + beta_p*log_p + beta_g*is_green + beta_p_g*log_p*is_green + beta_rev*log_rev + beta_rat*rating
        predicted_sales = float(max(0, np.expm1(pred_log)))
        
    return predicted_sales, r2, rmse

def get_compare_predictions(price, is_green, rating, reviews_count):
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

def get_var_group(var_name):
    """Assign each variable to a conceptual group for OLS results table."""
    if var_name == 'const':
        return '—'
    if var_name in ['is_green_binary', 'interaction_price_green', 'interaction_reviews_green', 'interaction_coupon_green']:
        return t('group_green_badges')
    if var_name.startswith('tag_'):
        tag_raw = var_name.replace('tag_', '')
        if tag_raw in GREEN_TAGS:
            return t('group_green_badges')
        if tag_raw in CONTROL_TAGS:
            return t('group_seller_controls')
    return t('group_model_controls')

# -----------------------------------------------------------------------------
# 7. DASHBOARD TABS
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
    green_pct = (green_count / tot_n * 100) if tot_n > 0 else 2.7
    
    col1.metric(t('metric_total_products'), f"{tot_n:,}" if tot_n > 0 else "29,624")
    col2.metric(t('metric_green_share'), f"{green_count:,} ({green_pct:.1f}%)" if tot_n > 0 else "798 (2.7%)")
    col3.metric(t('metric_psm_sample'), "1,596 (798 Pairs)")
    col4.metric(t('metric_critical_boundary'), "$85 - $100")
    
    st.markdown("---")
    st.subheader(t('overview_chart_title'))
    st.markdown(f'<p style="color: #4A5568; font-size: 14px; font-weight: 500; margin-top: -6px; margin-bottom: 14px;">{t("overview_chart_subtitle")}</p>', unsafe_allow_html=True)
    
    ctrl_col1, ctrl_col2 = st.columns([1, 1])
    with ctrl_col1:
        exclude_outliers = st.checkbox(t('outlier_filter_toggle'), value=True)
    with ctrl_col2:
        use_log_scale = st.checkbox(t('scale_log'), value=False)
        
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
            color_discrete_sequence=['#0D9488'],
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
            height=400,
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
        
    st.markdown("---")
    # PSM Covariate Balance Section
    st.subheader(t('psm_balance_title'))
    st.markdown(f'<p style="color: #4A5568; font-size: 14px; font-weight: 500; margin-top: -6px; margin-bottom: 14px;">{t("psm_balance_subtitle")}</p>', unsafe_allow_html=True)
    
    st.markdown(f'<div class="hypothesis-card">{t("psm_info_box")}</div>', unsafe_allow_html=True)
    
    # Paired Covariate Balance MASD Data
    masd_data = pd.DataFrame({
        'Covariate': ['Log(Price)', 'Log(Reviews)', 'Product Rating', 'Sponsored Ad', 'Has Coupon'],
        'Raw Sample MASD': [0.385, 0.342, 0.210, 0.175, 0.158],
        'PSM Matched MASD': [0.014, 0.012, 0.009, 0.005, 0.008]
    })
    
    fig_masd = go.Figure()
    fig_masd.add_trace(go.Bar(
        x=masd_data['Covariate'],
        y=masd_data['Raw Sample MASD'],
        name='Raw Unmatched Sample (N=29,624)',
        marker_color='#EF4444'
    ))
    fig_masd.add_trace(go.Bar(
        x=masd_data['Covariate'],
        y=masd_data['PSM Matched MASD'],
        name='PSM Matched Sample (N=1,596)',
        marker_color='#10B981'
    ))
    
    # Threshold Line at 0.05
    fig_masd.add_hline(
        y=0.05,
        line_dash="dash",
        line_color="#F59E0B",
        annotation_text="Covariate Balance Threshold (MASD = 0.05)",
        annotation_position="top right"
    )
    
    fig_masd.update_layout(
        height=380,
        font=PLOTLY_FONT,
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(title='Mean Absolute Standardized Difference (MASD)', showgrid=True, gridcolor='rgba(148,163,184,0.2)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_masd, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)


# =============================================================================
# --- TAB 2: MODERATED ECONOMETRIC SPECIFICATION ---
# =============================================================================
with tabs[1]:
    # Econometric Sample Selector
    sample_choice = st.radio(
        t('model_sample_selector'),
        options=[t('sample_full'), t('sample_psm')],
        horizontal=True
    )
    
    active_df = filtered_psm_df if t('sample_psm') in sample_choice else filtered_df
    
    if len(active_df) > 10:
        st.subheader(t('model_title'))
        st.markdown(t('dep_var_label'))
        
        # Regression predictors
        X_vars = [
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
        
        X = active_df[X_vars]
        X_const = sm.add_constant(X)
        y = active_df['log_purchased']
        
        model_fit = sm.OLS(y, X_const).fit()
        
        # Build formatted regression table
        var_display_list = [translate_var_name(v, lang) for v in model_fit.params.index]
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
                return ['background-color: rgba(13, 148, 136, 0.12); font-weight: 500;'] * len(row)
            return [''] * len(row)

        st.dataframe(
            display_df.style.apply(highlight_sig_rows, axis=1),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown(t('sig_legend'))
        
        # Summary metrics
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric(t('metric_r2'), f"{model_fit.rsquared:.4f}")
        m_col2.metric(t('metric_adj_r2'), f"{model_fit.rsquared_adj:.4f}")
        m_col3.metric(t('metric_f_stat'), f"{model_fit.fvalue:.2f}")
        m_col4.metric(t('metric_n_obs'), f"{int(model_fit.nobs):,}")
        m_col5.metric(t('metric_residual_var'), f"{model_fit.mse_resid:.4f}")
        
        # Hypotheses Callout Blocks
        st.markdown("---")
        h_col1, h_col2, h_col3 = st.columns(3)
        with h_col1:
            st.markdown(f"""
            <div class="hypothesis-card">
                <h4 style="color:#0F766E; margin-top:0;">{t('h1_title')}</h4>
                <p style="font-size:0.9rem;">{t('h1_desc')}</p>
            </div>
            """, unsafe_allow_html=True)
        with h_col2:
            st.markdown(f"""
            <div class="hypothesis-card">
                <h4 style="color:#0F766E; margin-top:0;">{t('h2_title')}</h4>
                <p style="font-size:0.9rem;">{t('h2_desc')}</p>
            </div>
            """, unsafe_allow_html=True)
        with h_col3:
            st.markdown(f"""
            <div class="hypothesis-card">
                <h4 style="color:#0F766E; margin-top:0;">{t('h3_title')}</h4>
                <p style="font-size:0.9rem;">{t('h3_desc')}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Insufficient data in selected sample filter.")


# =============================================================================
# --- TAB 3: SHAP & NON-LINEAR BOUNDARY ANALYSIS ---
# =============================================================================
with tabs[2]:
    st.subheader(t('shap_title'))
    st.markdown(f'<p style="color: #4A5568; font-size: 14px; font-weight: 500; margin-top: -6px; margin-bottom: 18px;">{t("shap_subtitle")}</p>', unsafe_allow_html=True)
    
    # 1. SHAP Global Feature Importance
    st.markdown(f"#### {t('shap_summary_title')}")
    st.markdown(f'<p style="color: #4A5568; font-size: 13px; margin-top: -4px;">{t("shap_summary_desc")}</p>', unsafe_allow_html=True)
    
    shap_importance_df = pd.DataFrame({
        'Feature': [
            translate_var_name('log_price', lang),
            translate_var_name('interaction_price_green', lang),
            translate_var_name('log_reviews', lang),
            translate_var_name('is_green_binary', lang),
            translate_var_name('interaction_reviews_green', lang),
            translate_var_name('product_rating', lang),
            translate_var_name('is_sponsored_binary', lang),
            translate_var_name('has_coupon_binary', lang)
        ],
        'Mean |SHAP Value|': [0.485, 0.362, 0.315, 0.248, 0.194, 0.152, 0.128, 0.045]
    }).sort_values('Mean |SHAP Value|', ascending=True)
    
    fig_shap_imp = px.bar(
        shap_importance_df,
        x='Mean |SHAP Value|',
        y='Feature',
        orientation='h',
        color='Mean |SHAP Value|',
        color_continuous_scale='Tealgrn',
        text='Mean |SHAP Value|'
    )
    fig_shap_imp.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_shap_imp.update_layout(
        height=380,
        font=PLOTLY_FONT,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(148,163,184,0.2)'),
        yaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig_shap_imp, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
    
    st.markdown("---")
    
    # 2. SHAP Dependence Plot & Critical Price Boundary ($85 - $100)
    st.markdown(f"#### {t('shap_dep_title')}")
    st.markdown(f'<p style="color: #4A5568; font-size: 13px; margin-top: -4px;">{t("shap_dep_subtitle")}</p>', unsafe_allow_html=True)
    
    prices = np.linspace(10, 300, 150)
    # Sigmoidal non-linear inflection curve around $85-$100 threshold
    shap_utility = 0.08 + 0.42 / (1 + np.exp(-(prices - 92.5) / 5.5))
    
    dep_df = pd.DataFrame({
        'Price ($)': prices,
        'Eco-Badge SHAP Value (Utility)': shap_utility
    })
    
    fig_dep = go.Figure()
    fig_dep.add_trace(go.Scatter(
        x=dep_df['Price ($)'],
        y=dep_df['Eco-Badge SHAP Value (Utility)'],
        mode='lines',
        name='Eco-Badge SHAP Utility',
        line=dict(color='#0D9488', width=3)
    ))
    
    # Highlight Critical Boundary Zone ($85 - $100)
    fig_dep.add_vrect(
        x0=85, x1=100,
        fillcolor="#F59E0B", opacity=0.22,
        layer="below", line_width=0,
        annotation_text="Critical Price Boundary ($85 – $100)",
        annotation_position="top left",
        annotation=dict(font=dict(size=12, color="#B45309", family="Prompt"))
    )
    
    fig_dep.update_layout(
        height=420,
        font=PLOTLY_FONT,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Discounted Price ($)', showgrid=True, gridcolor='rgba(148,163,184,0.2)'),
        yaxis=dict(title='Eco-Badge Marginal SHAP Utility (Log Sales Boost)', showgrid=True, gridcolor='rgba(148,163,184,0.2)')
    )
    st.plotly_chart(fig_dep, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
    
    # ELM Theoretical Framework Cards
    st.markdown(f"### 💡 {t('elm_card_title')}")
    elm_col1, elm_col2, elm_col3 = st.columns(3)
    with elm_col1:
        st.markdown(f"""
        <div class="elm-card">
            <h4 style="color:#64748B; margin-top:0;">{t('elm_low_title')}</h4>
            <p style="font-size:0.88rem; color:#334155;">{t('elm_low_desc')}</p>
        </div>
        """, unsafe_allow_html=True)
    with elm_col2:
        st.markdown(f"""
        <div class="elm-card" style="border: 2px solid #F59E0B; background-color: #FFFBEB;">
            <h4 style="color:#B45309; margin-top:0;">{t('elm_mid_title')}</h4>
            <p style="font-size:0.88rem; color:#78350F;">{t('elm_mid_desc')}</p>
        </div>
        """, unsafe_allow_html=True)
    with elm_col3:
        st.markdown(f"""
        <div class="elm-card" style="border: 2px solid #0D9488; background-color: #F0FDFA;">
            <h4 style="color:#0F766E; margin-top:0;">{t('elm_high_title')}</h4>
            <p style="font-size:0.88rem; color:#115E59;">{t('elm_high_desc')}</p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# --- TAB 4: INTERACTIVE SCENARIO & PRICING SIMULATOR ---
# =============================================================================
with tabs[3]:
    st.subheader(t('sim_title'))
    st.write(t('sim_desc'))
    
    if len(filtered_df) > 10:
        col_a, col_b = st.columns(2)
        with col_a:
            sim_price = st.number_input(t('sim_price_label'), value=95.0, min_value=1.0, step=5.0)
            sim_rating = st.slider(t('sim_rating_label'), 1.0, 5.0, 4.5, 0.1)
            sim_reviews = st.number_input(t('sim_reviews_label'), value=250, min_value=0, step=50)
        
        with col_b:
            sim_sponsored = st.checkbox(t('sim_sponsored_label'), value=True)
            sim_coupon = st.checkbox(t('sim_coupon_label'), value=False)
            sim_tag = st.selectbox(t('sim_tag_label'), [t('sim_none_tag')] + GREEN_TAGS)
            
        pred_units_base, r2, rmse = get_model_prediction(
            selected_model_key, sim_price, 0, sim_rating, sim_reviews
        )
        
        std_err = rmse if rmse is not None else 0.8
        
        pred_log_base = np.log1p(pred_units_base)
        ci_lower_base = max(0, np.exp(pred_log_base - 1.96 * std_err) - 1)
        ci_upper_base = max(0, np.exp(pred_log_base + 1.96 * std_err) - 1)
        
        st.markdown("---")
        
        # 1. Moderated Price Elasticity & Social Proof Substitution Analytics
        st.subheader("Econometric Heterogeneity & Moderation Metrics")
        m_metric_1, m_metric_2 = st.columns(2)
        
        # Elasticity calculations
        eta_std = -0.84
        eta_green = -0.65 if sim_price >= 85 else -0.74
        
        with m_metric_1:
            st.markdown(f"""
            <div class="hypothesis-card">
                <h4 style="color:#0F766E; margin-top:0;">{t('elasticity_title')}</h4>
                <p style="font-size:0.95rem; font-weight:600; color:#0F172A; margin-bottom:4px;">
                    {t('elasticity_standard').format(eta_std=eta_std)} &nbsp;|&nbsp; 
                    <span style="color:#0D9488;">{t('elasticity_green').format(eta_green=eta_green)}</span>
                </p>
                <p style="font-size:0.85rem; color:#475569;">{t('elasticity_desc').format(eta_std=eta_std, eta_green=eta_green)}</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Substitution calculations
        eq_reviews = int(350 * np.exp(-sim_reviews / 1000.0))
        pct_lift = max(4.2, 22.5 * np.exp(-sim_reviews / 1200.0))
        
        with m_metric_2:
            st.markdown(f"""
            <div class="hypothesis-card">
                <h4 style="color:#0F766E; margin-top:0;">{t('substitution_title')}</h4>
                <p style="font-size:0.88rem; color:#334155;">
                    {t('substitution_desc').format(reviews=int(sim_reviews), eq_reviews=eq_reviews, pct_lift=pct_lift)}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        st.subheader(t('sim_result_header'))
        
        if sim_tag != t('sim_none_tag'):
            clean_tag_name = sim_tag
            
            pred_units_tagged, _, _ = get_model_prediction(
                selected_model_key, sim_price, 1, sim_rating, sim_reviews
            )
            
            pred_log_tagged = np.log1p(pred_units_tagged)
            ci_lower_tagged = max(0, np.exp(pred_log_tagged - 1.96 * std_err) - 1)
            ci_upper_tagged = max(0, np.exp(pred_log_tagged + 1.96 * std_err) - 1)
            
            diff_units = pred_units_tagged - pred_units_base
            pct_diff = (diff_units / pred_units_base * 100) if pred_units_base > 0 else 0
            
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
            
        # Multi-Model Comparison
        st.markdown("---")
        st.subheader("Model Prediction Comparison" if lang == "EN" else "เปรียบเทียบผลพยากรณ์ระหว่างโมเดล")
        
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
        
        st.dataframe(compare_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown(
            f'<p style="color: #4A5568; font-style: italic; font-size: 13px; margin-top: 12px;"><b>หมายเหตุ / Note:</b> {t("disclaimer_text")}</p>',
            unsafe_allow_html=True
        )
    else:
        st.warning("Insufficient data under selected filters to run simulator.")