# 🌿 PriceLens — Amazon Green E-Commerce Research Dashboard

งานวิจัยและระบบวิเคราะห์อิทธิพลของ **ป้ายกำกับความยั่งยืน (Sustainability Tags)** ต่อประสิทธิภาพการขายและราคาสินค้าอิเล็กทรอนิกส์บน Amazon โดยใช้กรอบงานวิจัย Q1 (Green Scarcity, Price Sensitivity Meter, Moderated Regressions, SHAP Boundary Analysis)

## 📄 Paper reproducibility (InCIT 2026)

The results in the paper *"Green Signals in E-Commerce: Explainable Modeling of Price Elasticity and Social Proof"* (InCIT 2026, Paper 185) are produced **only** by the scripts in [`revision/`](revision/). See [`revision/README.md`](revision/README.md) for data, environment and run instructions.

The Streamlit dashboard (`app.py`) is an exploratory tool. Its PSM and model settings differ from the paper:

| | Dashboard (`app.py`) | Paper (`revision/`) |
|---|---|---|
| Unit of analysis | Listing-level rows | Product-level (one row per product, latest scrape) |
| PSM | Nearest-neighbour matching with replacement, no caliper | Exact Sponsored × Coupon strata, 1:1 without replacement, caliper 0.02 |
| ML validation | Dashboard model settings | Nested cross-validation |

## 🚀 วิธีการติดตั้งและรัน Web App

1. **ติดตั้ง Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **รันแอป:**
   ```bash
   streamlit run app.py
   ```

3. เปิดเบราว์เซอร์ไปที่ URL ที่ Streamlit แสดง (ปกติคือ `http://localhost:8501`)

## ✨ ฟีเจอร์หลัก

- **ชุดข้อมูลแบบไดนามิก** — ใช้ demo dataset ที่มีให้ หรืออัปโหลดไฟล์ของตัวเอง (.csv / .xlsx) พร้อมระบบ column mapping ให้จับคู่ฟิลด์เอง
- **4 แท็บวิเคราะห์หลัก:**
  1. **PSM & Descriptive** — Price Sensitivity Meter และสถิติเชิงพรรณนาของชุดข้อมูล
  2. **Moderated Regression** — ทดสอบสมมติฐาน (Green Scarcity, Hypotheses H1–H3) ด้วยแบบจำลอง Econometric
  3. **SHAP Boundary Analysis** — วิเคราะห์ feature importance และจุดตัดสินใจของโมเดล tree-based (Random Forest / XGBoost)
  4. **ROI Simulator** — จำลองผลกระทบของป้ายกำกับความยั่งยืนต่อยอดขายที่คาดการณ์ พร้อมเปรียบเทียบระหว่างโมเดล (OLS, Ridge, Random Forest, XGBoost)
- **โมเดลพยากรณ์แบบเลือกได้** — สลับเครื่องมือพยากรณ์ระหว่าง OLS Regression, Ridge Regression, Random Forest และ XGBoost ได้จากแถบด้านข้าง
- **รองรับ 10 ภาษา** — English, 中文, हिन्दी, Español, Français, العربية, বাংলা, Português, Русский, Bahasa Indonesia พร้อมระบบ RTL (Right-to-Left) แยกเฉพาะสำหรับภาษาอาหรับ

## 📁 โครงสร้างไฟล์หลัก

- `app.py` — โค้ดหลักของ Streamlit dashboard (ทุกแท็บและ logic การวิเคราะห์)
- `i18n.py` — โมดูลแปลภาษาและการจัดการ RTL/LTR
- `data/` — ชุดข้อมูล demo (Amazon products sales data)
- `requirements.txt` — รายการ Python dependencies
