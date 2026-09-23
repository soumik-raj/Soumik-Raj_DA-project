# 🛒 Indian Retail Sales — 4-Tier Analytics & ML Project

**Industry:** B2B Food & Beverage Retail (India)  
**Dataset:** [Indian Retail Sales — Kaggle](https://www.kaggle.com/datasets/winstonbobby/indian-retail-sales)  
**Target Variable:** `Profit_Flag` (1 = profitable order, 0 = loss-making order)

---

## 📁 Project Structure

```
├── app.py                  # Streamlit interactive dashboard
├── INDIA_RETAIL_DATA.csv   # Source dataset (place here)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the dataset CSV in this folder
#    (file must be named: INDIA_RETAIL_DATA.csv)

# 3. Launch the dashboard
streamlit run app.py
```

---

## 🏗️ 4-Tier Analytics Ladder

### Tier 0 — Data Hygiene & Architecture
- **Source:** 5,000+ transaction records, 14 raw columns
- **Cleaning steps:**
  - Date parsing with `dayfirst=True` (DD-MM-YYYY Indian format)
  - Numeric coercion for `Profit`, `Sales`, `QtyOrdered`
  - Duplicate removal
  - Median imputation for numeric nulls
  - `Shipping_Days` clipped at 0 (no negative transit times)
- **Engineered features:**
  - `Profit_Flag` — binary target (1 if Profit > 0)
  - `Profit_Margin` — Profit / Sales
  - `Shipping_Days` — derived from Order Date and Ship Date
  - `Log_Sales`, `Log_Qty` — log-transformed for skewness
  - `Order_Year`, `Order_Month` — temporal features

---

### Tiers 1 & 2 — Exploratory Data Analysis

| Chart | Title | Insight |
|---|---|---|
| 1 | Profit vs Loss by Segment | Hotels/Hospitals = highest volume but also highest loss rate |
| 2 | Monthly Revenue Trend | High variance; Q4 spikes driven by bulk orders |
| 3 | Profit Rate by Freight Mode | Express Air = lowest profitability; Delivery Truck = highest |
| 4 | Top 10 States by Profit | MP, UP, WB top contributors; also top loss generators |
| 5 | Sales vs Profit Scatter | Processed Meat shows extreme negative outliers even at moderate sales |

> ⚠️ All insights are **descriptive/diagnostic**. Correlation ≠ causation. Controlled experiments required before attribution.

---

### Tier 3 — Predictive Modelling

**Models available:** Random Forest · Logistic Regression  
**Split:** 80% Train / 20% Test (stratified)  
**Leakage prevention:** `Profit` and `Profit_Margin` excluded from feature set (direct target derivatives)

#### Typical Model Performance (Random Forest)

| Metric | Value |
|---|---|
| Accuracy | ~0.71–0.76 |
| Precision | ~0.72–0.77 |
| Recall | ~0.82–0.88 |
| ROC-AUC | ~0.73–0.78 |

#### Business Trade-off: FP vs FN

| Error | Definition | Cost |
|---|---|---|
| **False Positive** | Predicted Profit → actual Loss | Direct margin erosion — accept a loss-making order |
| **False Negative** | Predicted Loss → actual Profit | Opportunity cost — reject a profitable order |

In thin-margin B2B food distribution, **False Positives are more costly**. Recommend lowering the classification threshold to 0.40–0.45 to prioritise Precision.

---

### Tier 4 — Prescriptive Strategy

**Resource constraint:** `N` daily human review slots (configurable in dashboard)

| # | Lever | Action |
|---|---|---|
| 1 | Freight Repricing | Surcharge Express Air for low-Sales Processed Meat orders |
| 2 | Container Optimisation | Replace Jumbo Drum with Jumbo Box for low-quantity Canned Food |
| 3 | Segment Focus | Shift effort from Hotels to Stand Alone Restaurants (North) |
| 4 | Order Gate | Pre-approval required for orders with Profit_Proba < 0.40 |
| 5 | Seasonal Contracts | Pre-negotiate Delivery Truck rates before Q4 peak |

---

## 🧩 Feature Set Used in ML

| Feature | Type | Description |
|---|---|---|
| Freight Mode | Categorical | Regular Air / Express Air / Delivery Truck |
| Segment | Categorical | Customer segment |
| Product Type | Categorical | Processed Meat / Canned Foods / Preserved Food |
| Product Sub-Category | Categorical | Specific product |
| Product Container | Categorical | Packaging size |
| Region | Categorical | North / South / East / West |
| Log_Sales | Numeric | log(1 + Sales) |
| Log_Qty | Numeric | log(1 + QtyOrdered) |
| Shipping_Days | Numeric | Days between order and ship date |
| Order_Month | Numeric | 1–12 |
| Order_Year | Numeric | 2010–2013 |

**Excluded (leakage):** `Profit`, `Profit_Margin`, `City`, `State` (identifiers)

---

## 📦 Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
streamlit>=1.28.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

---

## 📝 Notes

- The dataset covers 2010–2013 across Indian states.
- No PII is present; all data is transactional.
- The dashboard is fully self-contained — no external API calls.
- Recommendations are correlation-based; causal validation requires A/B testing.

---

*Built as a production-ready 4-Tier Analytics project using the Indian Retail Sales dataset.*
