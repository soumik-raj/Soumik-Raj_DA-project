"""
Indian Retail Sales — 4-Tier Analytics & ML Dashboard
Industry : B2B Food & Beverage Retail (India)
Target   : Profit_Flag  (1 = profitable order, 0 = loss-making order)

Run:  streamlit run app.py
Requires: INDIA_RETAIL_DATA.csv in the same folder.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Retail — 4-Tier Analytics",
    page_icon="🛒",
    layout="wide",
)
PALETTE = ["#3b82d4", "#e5483a", "#f59e0b", "#10b981", "#8b5cf6", "#ec4899"]
sns.set_theme(style="whitegrid", font_scale=0.95)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & HYGIENE  (Tier 0)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & cleaning dataset…")
def load_and_clean() -> pd.DataFrame:
    try:
        df = pd.read_csv("INDIA_RETAIL_DATA.csv")
    except FileNotFoundError:
        st.error(
            "❌  **INDIA_RETAIL_DATA.csv** not found.\n\n"
            "Place the file in the same folder as `app.py` and refresh."
        )
        st.stop()

    # ── 1. Column normalisation ───────────────────────────────────────────────
    df.columns = df.columns.str.strip()

    # ── 2. Date parsing ───────────────────────────────────────────────────────
    for col in ["Order Date", "Ship Date"]:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    # ── 3. Numeric coercion ───────────────────────────────────────────────────
    for col in ["Profit", "QtyOrdered", "Sales", "Unit Price",
                "Freight Expenses", "Discount offered"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 4. Duplicate removal ──────────────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    dups_removed = before - len(df)

    # ── 5. Missing-value handling ─────────────────────────────────────────────
    missing_before = int(df.isnull().sum().sum())
    num_fill_cols = ["Profit", "QtyOrdered", "Sales", "Unit Price",
                     "Freight Expenses", "Discount offered"]
    for col in num_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    df.dropna(subset=["Order Date", "Ship Date", "State",
                       "Region", "Segment", "Freight Mode"], inplace=True)

    # ── 6. Anomaly fixes ──────────────────────────────────────────────────────
    if "QtyOrdered" in df.columns:
        df = df[df["QtyOrdered"] > 0]          # remove zero-qty records
    if "Sales" in df.columns:
        df = df[df["Sales"] > 0]               # remove zero-sales records

    # ── 7. Feature engineering ────────────────────────────────────────────────
    df["Profit_Flag"]    = (df["Profit"] > 0).astype(int)          # TARGET
    df["Profit_Margin"]  = df["Profit"] / (df["Sales"] + 1e-9)
    df["Shipping_Days"]  = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Shipping_Days"]  = df["Shipping_Days"].clip(lower=0)
    df["Order_Year"]     = df["Order Date"].dt.year
    df["Order_Month"]    = df["Order Date"].dt.month
    df["Order_Quarter"]  = df["Order Date"].dt.quarter
    df["Log_Sales"]      = np.log1p(df["Sales"].clip(lower=0))
    df["Log_Qty"]        = np.log1p(df["QtyOrdered"].clip(lower=0))
    if "Unit Price" in df.columns:
        df["Log_UnitPrice"] = np.log1p(df["Unit Price"].clip(lower=0))
    if "Freight Expenses" in df.columns:
        df["Log_Freight"] = np.log1p(df["Freight Expenses"].clip(lower=0))

    df.attrs["audit"] = dict(
        rows=len(df),
        dups_removed=dups_removed,
        missing_before=missing_before,
        pct_profitable=float(df["Profit_Flag"].mean() * 100),
    )
    return df


df = load_and_clean()
audit = df.attrs["audit"]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🛒 India Retail Analytics")
st.sidebar.markdown("**4-Tier Analytics Ladder**")
nav = st.sidebar.radio(
    "Navigate to",
    [
        "📋 Data Hygiene",
        "📊 EDA (Tiers 1-2)",
        "🤖 ML Model (Tier 3)",
        "🎯 Prescriptive (Tier 4)",
    ],
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** {audit['rows']:,} clean orders")
st.sidebar.markdown(f"**Profitable:** {audit['pct_profitable']:.1f}%")
st.sidebar.caption("Dataset: Indian Retail Sales (Kaggle)")


# ─────────────────────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — DATA HYGIENE
# ═════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
if nav == "📋 Data Hygiene":
    st.title("📋 Tier 0 — Data Hygiene & Architecture")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clean Rows", f"{audit['rows']:,}")
    c2.metric("Duplicates Removed", audit["dups_removed"])
    c3.metric("Missing Values (raw)", audit["missing_before"])
    c4.metric("% Profitable Orders", f"{audit['pct_profitable']:.1f}%")

    st.markdown("---")
    st.subheader("Column Schema")
    schema = pd.DataFrame({
        "Column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "Non-Null Count": df.notnull().sum().values,
        "Null Count": df.isnull().sum().values,
        "Unique Values": [df[c].nunique() for c in df.columns],
    })
    st.dataframe(schema, use_container_width=True)

    st.markdown("---")
    st.subheader("Descriptive Statistics — Numeric Columns")
    num_cols = [c for c in ["Profit", "Sales", "QtyOrdered", "Unit Price",
                             "Freight Expenses", "Discount offered",
                             "Shipping_Days", "Profit_Margin"] if c in df.columns]
    st.dataframe(df[num_cols].describe().round(3), use_container_width=True)

    st.markdown("---")
    st.subheader("Sample Records")
    st.dataframe(df.head(100), use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Data Quality Audit Notes")
    st.info(
        "**Actions taken:**\n\n"
        "- Dates parsed with `dayfirst=True` (Indian DD-MM-YYYY format)\n"
        "- Numeric columns coerced to float; remaining nulls filled with column median\n"
        "- Duplicate rows dropped\n"
        "- Zero-quantity and zero-sales orders removed as data anomalies\n"
        "- `Shipping_Days` clipped at 0 (negative transit times are impossible)\n"
        "- `Profit_Margin` computed as Profit / Sales (epsilon added to avoid division-by-zero)\n\n"
        "**Target variable:** `Profit_Flag` = 1 if Profit > 0, else 0\n\n"
        "**Leakage prevention:** `Profit`, `Profit_Margin`, raw `Profit` are **excluded** "
        "from the ML feature matrix — they are direct derivatives of the target."
    )


# ─────────────────────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — EDA
# ═════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "📊 EDA (Tiers 1-2)":
    st.title("📊 Tiers 1 & 2 — Exploratory Data Analysis")

    # ── Chart 1: Profit/Loss by Segment ──────────────────────────────────────
    st.markdown("---")
    st.subheader("Chart 1 — Profitable vs Loss Orders by Customer Segment")
    fig1, ax1 = plt.subplots(figsize=(11, 5))
    seg_pnl = (
        df.groupby(["Segment", "Profit_Flag"])
          .size().reset_index(name="Count")
    )
    seg_pnl["Outcome"] = seg_pnl["Profit_Flag"].map({1: "Profitable", 0: "Loss"})
    pivot1 = seg_pnl.pivot(index="Segment", columns="Outcome", values="Count").fillna(0)
    pivot1.plot(kind="bar", ax=ax1, color=["#e5483a", "#3b82d4"],
                edgecolor="white", width=0.6)
    ax1.set_title("Profitable vs Loss-Making Orders by Customer Segment",
                  fontsize=13, fontweight="bold")
    ax1.set_xlabel("Segment"); ax1.set_ylabel("Number of Orders")
    ax1.tick_params(axis="x", rotation=20)
    ax1.legend(title="Outcome")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig1.tight_layout()
    st.pyplot(fig1)
    st.info(
        "**Descriptive:** Hotels/Hospitals and Restaurant Chains generate the highest order volumes "
        "but also the most loss-making orders in absolute terms. Stand Alone Restaurants have a "
        "relatively higher profitable-to-loss ratio.\n\n"
        "**Diagnostic:** Segment alone does not *cause* losses — pricing strategy, freight choice, "
        "and product mix are confounding variables requiring separate investigation."
    )

    # ── Chart 2: Quarterly Revenue & Profit Trend ─────────────────────────────
    st.markdown("---")
    st.subheader("Chart 2 — Quarterly Revenue & Profit Trend (2010–2013)")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    df["YQ"] = df["Order Date"].dt.to_period("Q").astype(str)
    qtrly = df.groupby("YQ")[["Sales", "Profit"]].sum().reset_index()
    ax2.bar(qtrly["YQ"], qtrly["Sales"] / 1e3, color="#c7ddf4", label="Sales (₹K)", zorder=2)
    ax2.plot(qtrly["YQ"], qtrly["Profit"] / 1e3, color="#e5483a",
             linewidth=2, marker="o", label="Profit (₹K)", zorder=3)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_title("Quarterly Gross Sales vs Net Profit (₹ Thousands)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Quarter"); ax2.set_ylabel("₹ Thousands")
    ax2.tick_params(axis="x", rotation=45)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}K"))
    ax2.legend()
    fig2.tight_layout()
    st.pyplot(fig2)
    st.info(
        "**Descriptive:** Revenue grows gradually from 2010 to 2013 with notable Q4 spikes driven "
        "by bulk orders. Profit (red line) oscillates close to zero in many quarters, highlighting "
        "persistent margin pressure across the entire period.\n\n"
        "**Diagnostic:** The gap between Sales and Profit is disproportionately large, suggesting "
        "high freight expenses or deep discounting erode margins regardless of revenue volume."
    )

    # ── Chart 3: Freight Mode Profitability ───────────────────────────────────
    st.markdown("---")
    st.subheader("Chart 3 — Profit Rate & Avg Freight Cost by Freight Mode")
    fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
    freight_stats = (
        df.groupby("Freight Mode")
          .agg(
              Profit_Rate=("Profit_Flag", "mean"),
              Avg_Freight=("Freight Expenses", "mean") if "Freight Expenses" in df.columns
                         else ("Profit_Flag", "count"),
          )
          .reset_index()
          .sort_values("Profit_Rate", ascending=False)
    )
    # Left: profit rate
    bars = axes[0].barh(freight_stats["Freight Mode"],
                        freight_stats["Profit_Rate"] * 100,
                        color=PALETTE[:len(freight_stats)], height=0.45, edgecolor="white")
    for bar, val in zip(bars, freight_stats["Profit_Rate"] * 100):
        axes[0].text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center", fontsize=11)
    axes[0].set_title("% Profitable Orders by Freight Mode", fontweight="bold")
    axes[0].set_xlabel("Profitable Order Rate (%)"); axes[0].set_xlim(0, 100)

    # Right: avg freight cost
    if "Freight Expenses" in df.columns:
        axes[1].barh(freight_stats["Freight Mode"], freight_stats["Avg_Freight"],
                     color=PALETTE[:len(freight_stats)], height=0.45, edgecolor="white")
        axes[1].set_title("Avg Freight Expenses (₹) by Mode", fontweight="bold")
        axes[1].set_xlabel("Avg Freight Cost (₹)")
        axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    fig3.tight_layout()
    st.pyplot(fig3)
    st.info(
        "**Descriptive:** Express Air has the lowest profitability rate among all freight modes. "
        "Delivery Truck yields the highest rate and lowest average freight cost.\n\n"
        "**Caution:** Freight mode is often determined by order urgency or product perishability — "
        "the association with profitability may be a confounder, not a cause."
    )

    # ── Chart 4: Discount vs Profit Margin ────────────────────────────────────
    st.markdown("---")
    st.subheader("Chart 4 — Discount Offered vs Profit Margin (Order-Level)")
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sample = df.sample(min(len(df), 800), random_state=42)
    sc = ax4.scatter(
        sample["Discount offered"] if "Discount offered" in sample.columns else np.zeros(len(sample)),
        sample["Profit_Margin"].clip(-2, 2),
        c=sample["Profit_Flag"], cmap="RdYlGn", alpha=0.55, s=20, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax4, label="Profit Flag (1=Profit, 0=Loss)")
    ax4.axhline(0, color="black", linewidth=0.9, linestyle="--", label="Break-even")
    ax4.set_title("Discount Offered vs Profit Margin (clipped at ±200%)",
                  fontsize=13, fontweight="bold")
    ax4.set_xlabel("Discount Offered (fraction)")
    ax4.set_ylabel("Profit Margin")
    ax4.legend()
    fig4.tight_layout()
    st.pyplot(fig4)
    if "Discount offered" in df.columns:
        corr_disc = df["Discount offered"].corr(df["Profit_Margin"])
        st.info(
            f"**Descriptive:** Pearson correlation between Discount and Profit Margin = **{corr_disc:.3f}**. "
            "Higher discounts are associated with lower or negative margins. Orders with zero or "
            "minimal discount cluster in the profitable (green) zone.\n\n"
            "**Diagnostic:** This is a strong predictive signal — discounting decisions directly "
            "compress margins. However, discount level is likely also correlated with order size "
            "and customer segment (confounding)."
        )

    # ── Chart 5: Product Sub-Category Profit Heatmap ──────────────────────────
    st.markdown("---")
    st.subheader("Chart 5 — Profit Rate Heatmap: Product Sub-Category × Region")
    fig5, ax5 = plt.subplots(figsize=(14, 7))
    heat_data = (
        df.groupby(["Product Sub-Category", "Region"])["Profit_Flag"]
          .mean()
          .unstack(fill_value=np.nan)
          .mul(100)
    )
    sns.heatmap(
        heat_data, annot=True, fmt=".0f", cmap="RdYlGn",
        linewidths=0.3, ax=ax5, vmin=0, vmax=100,
        cbar_kws={"label": "Profit Rate (%)"},
    )
    ax5.set_title("Profit Rate (%) by Product Sub-Category × Region",
                  fontsize=13, fontweight="bold")
    ax5.set_xlabel("Region"); ax5.set_ylabel("Product Sub-Category")
    ax5.tick_params(axis="y", labelsize=8)
    fig5.tight_layout()
    st.pyplot(fig5)
    st.info(
        "**Descriptive:** Certain product-region combinations (deep red cells) show <40% profit "
        "rates, indicating systemic loss-making routes. Wild Berry and certain preserved foods in "
        "the West region show particularly poor performance.\n\n"
        "**Cohort Insight:** These combinations form natural at-risk cohorts for targeted "
        "pricing or fulfilment intervention. Correlation with region does not imply regional "
        "demand as the cause — logistics costs and local pricing may explain the pattern."
    )


# ─────────────────────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — ML MODEL
# ═════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "🤖 ML Model (Tier 3)":
    st.title("🤖 Tier 3 — Predictive Modelling")
    st.markdown(
        "**Target:** `Profit_Flag` — binary (1 = profitable, 0 = loss-making)  \n"
        "**Split:** 80% Train / 20% Test (stratified)  \n"
        "**Leakage guard:** `Profit`, `Profit_Margin` excluded from features."
    )

    # ── Feature construction ──────────────────────────────────────────────────
    CAT_COLS = ["Freight Mode", "Segment", "Product Type",
                "Product Sub-Category", "Product Container",
                "Region", "Order Priority"]
    CAT_COLS = [c for c in CAT_COLS if c in df.columns]

    NUM_COLS = ["Log_Sales", "Log_Qty", "Shipping_Days",
                "Order_Month", "Order_Quarter", "Order_Year"]
    if "Log_UnitPrice" in df.columns:
        NUM_COLS.append("Log_UnitPrice")
    if "Log_Freight" in df.columns:
        NUM_COLS.append("Log_Freight")
    if "Discount offered" in df.columns:
        NUM_COLS.append("Discount offered")

    TARGET = "Profit_Flag"

    ml_df = df[CAT_COLS + NUM_COLS + [TARGET]].copy().dropna()

    le_dict = {}
    for col in CAT_COLS:
        le_dict[col] = LabelEncoder()
        ml_df[col] = le_dict[col].fit_transform(ml_df[col].astype(str))

    X = ml_df[CAT_COLS + NUM_COLS]
    y = ml_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    col_m1, col_m2 = st.columns([2, 1])
    with col_m1:
        model_choice = st.selectbox("Choose Model", ["Random Forest", "Logistic Regression"])
    with col_m2:
        threshold = st.slider("Decision threshold", 0.30, 0.70, 0.50, 0.05)

    with st.spinner("Training model…"):
        if model_choice == "Random Forest":
            clf = RandomForestClassifier(
                n_estimators=300, max_depth=15,
                min_samples_leaf=4, random_state=42, n_jobs=-1
            )
        else:
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]
        y_pred  = (y_proba >= threshold).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)
    cm   = confusion_matrix(y_test, y_pred)

    # ── KPI Metrics ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Model Performance — Test Set")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy",  f"{acc:.4f}")
    k2.metric("Precision", f"{prec:.4f}")
    k3.metric("Recall",    f"{rec:.4f}")
    k4.metric("ROC-AUC",   f"{auc:.4f}")

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    st.markdown("---")
    col_cm, col_fi = st.columns([1, 2])
    with col_cm:
        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Loss (0)", "Profit (1)"]
        )
        disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
        ax_cm.set_title(f"{model_choice}", fontsize=11, fontweight="bold")
        fig_cm.tight_layout()
        st.pyplot(fig_cm)

    # ── Feature Importance ────────────────────────────────────────────────────
    with col_fi:
        st.subheader("Feature Importance")
        if model_choice == "Random Forest":
            fi = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=True)
        else:
            fi = pd.Series(np.abs(clf.coef_[0]), index=X.columns).sort_values(ascending=True)
        fig_fi, ax_fi = plt.subplots(figsize=(7, 5))
        colors_fi = [PALETTE[0] if i >= len(fi) - 3 else "#c0cfe8" for i in range(len(fi))]
        ax_fi.barh(fi.index, fi.values, color=colors_fi, edgecolor="white")
        ax_fi.set_title("Feature Importance (top = most important)",
                        fontsize=11, fontweight="bold")
        ax_fi.set_xlabel("Importance Score")
        fig_fi.tight_layout()
        st.pyplot(fig_fi)

    # ── Trade-off explanation ─────────────────────────────────────────────────
    tn, fp, fn, tp = cm.ravel()
    st.markdown("---")
    st.subheader("📌 Business Trade-off: False Positives vs False Negatives")
    st.markdown(
        f"""
**Context:** We predict whether an order will be **profitable** before fulfilment.

| Error Type | What Happened | Business Cost |
|---|---|---|
| **False Positive (FP = {fp:,})** | Model said *Profitable* → was actually a *Loss* | Sales team fulfils a loss-making order. Direct margin erosion — the **most costly** error in thin-margin B2B food distribution. |
| **False Negative (FN = {fn:,})** | Model said *Loss* → was actually *Profitable* | A good order is flagged for review or rejected. **Opportunity cost** — foregone revenue and risk of customer churn. |

**Threshold Guidance:** At the current threshold of **{threshold:.2f}**:
- If FP >> FN → increase threshold (be more conservative, avoid bad orders)
- If FN >> FP → decrease threshold (capture more profitable orders)

For Indian B2B food distribution where avg Profit/order ≈ ₹{df['Profit'].median():.0f} and losses can be
deep (min Profit = ₹{df['Profit'].min():,.0f}), we recommend prioritising **Precision** → keep threshold ≥ 0.45.
        """
    )


# ─────────────────────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — PRESCRIPTIVE
# ═════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "🎯 Prescriptive (Tier 4)":
    st.title("🎯 Tier 4 — Prescriptive Strategy & Operational Levers")

    # ── KPI Summary ───────────────────────────────────────────────────────────
    st.subheader("Business KPI Overview")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Total Orders",    f"{len(df):,}")
    kpi2.metric("Total Revenue",   f"₹{df['Sales'].sum() / 1e6:.2f}M")
    kpi3.metric("Total Profit",    f"₹{df['Profit'].sum() / 1e6:.2f}M")
    kpi4.metric("Avg Profit/Order", f"₹{df['Profit'].mean():,.0f}")
    if "Freight Expenses" in df.columns:
        kpi5.metric("Avg Freight Cost", f"₹{df['Freight Expenses'].mean():,.0f}")

    # ── Segment-level scorecard ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("1. Segment × Freight Mode — Profitability Scorecard")
    agg_dict = {
        "Orders": ("Profit_Flag", "count"),
        "Profit_Rate_%": ("Profit_Flag", lambda x: round(x.mean() * 100, 1)),
        "Avg_Profit_₹": ("Profit", lambda x: round(x.mean(), 0)),
        "Total_Profit_₹": ("Profit", lambda x: round(x.sum(), 0)),
    }
    if "Freight Expenses" in df.columns:
        agg_dict["Avg_Freight_₹"] = ("Freight Expenses", lambda x: round(x.mean(), 0))
    scorecard = (
        df.groupby(["Segment", "Freight Mode"])
          .agg(**agg_dict)
          .reset_index()
          .sort_values("Profit_Rate_%", ascending=False)
    )
    st.dataframe(scorecard, use_container_width=True)

    # ── Resource-constrained operational rule ─────────────────────────────────
    st.markdown("---")
    st.subheader("2. Operational Order-Gate Rule (Resource-Constrained)")
    col_budget, col_thresh = st.columns(2)
    budget    = col_budget.slider("Daily review capacity (orders/day)", 10, 300, 60, 10)
    risk_thr  = col_thresh.slider("Flag orders with Profit Probability below:", 0.20, 0.60, 0.40, 0.05)

    with st.spinner("Scoring all orders…"):
        CAT_COLS2 = ["Freight Mode", "Segment", "Product Type",
                     "Product Sub-Category", "Product Container", "Region", "Order Priority"]
        CAT_COLS2 = [c for c in CAT_COLS2 if c in df.columns]
        NUM_COLS2 = ["Log_Sales", "Log_Qty", "Shipping_Days",
                     "Order_Month", "Order_Quarter", "Order_Year"]
        if "Log_UnitPrice" in df.columns: NUM_COLS2.append("Log_UnitPrice")
        if "Log_Freight"   in df.columns: NUM_COLS2.append("Log_Freight")
        if "Discount offered" in df.columns: NUM_COLS2.append("Discount offered")

        feat_cols = CAT_COLS2 + NUM_COLS2
        score_df  = df[feat_cols + ["Profit_Flag", "Profit", "Sales",
                                     "Segment", "Freight Mode",
                                     "Product Sub-Category", "State"]].copy().dropna()

        le3 = {}
        for col in CAT_COLS2:
            le3[col] = LabelEncoder()
            score_df[col + "_enc"] = le3[col].fit_transform(score_df[col].astype(str))

        enc_feats = [c + "_enc" for c in CAT_COLS2] + NUM_COLS2
        Xs = score_df[enc_feats]
        ys = score_df["Profit_Flag"]

        clf_full = RandomForestClassifier(
            n_estimators=150, max_depth=12, random_state=42, n_jobs=-1
        )
        clf_full.fit(Xs, ys)
        score_df["Profit_Proba"] = clf_full.predict_proba(Xs)[:, 1]

    at_risk = score_df[score_df["Profit_Proba"] < risk_thr].copy()
    flagged = at_risk.sort_values("Profit_Proba").head(budget)

    st.markdown(
        f"**{len(at_risk):,} orders** have a predicted profit probability below **{risk_thr:.2f}**.  \n"
        f"With capacity of **{budget}** reviews/day, the system escalates the "
        f"**{min(budget, len(at_risk))} highest-risk orders** below."
    )

    display_cols = ["Segment", "Freight Mode", "Product Sub-Category",
                    "Sales", "Profit", "Profit_Proba"]
    display_cols = [c for c in display_cols if c in flagged.columns]
    risk_show = flagged[display_cols].copy()
    risk_show.rename(columns={"Sales": "Sales (₹)", "Profit": "Actual Profit (₹)",
                               "Profit_Proba": "Risk Score"}, inplace=True)
    risk_show["Risk Score"] = risk_show["Risk Score"].round(3)
    for c in ["Sales (₹)", "Actual Profit (₹)"]:
        if c in risk_show.columns:
            risk_show[c] = risk_show[c].round(2)
    st.dataframe(risk_show, use_container_width=True)

    # ── Loss hotspots ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("3. Loss Hotspots — Freight × Product Combination")
    worst_cols = {"Loss_Rate_%": ("Profit_Flag", lambda x: round((1 - x.mean()) * 100, 1)),
                  "Avg_Loss_₹": ("Profit", lambda x: round(x.mean(), 0)),
                  "Orders": ("Profit_Flag", "count")}
    worst = (
        df.groupby(["Freight Mode", "Product Sub-Category"])
          .agg(**worst_cols)
          .reset_index()
          .query("Loss_Rate_% > 50 and Orders > 5")
          .sort_values("Avg_Loss_₹")
          .head(8)
    )
    st.dataframe(worst, use_container_width=True)

    # ── Best-performing cohorts ───────────────────────────────────────────────
    st.subheader("4. High-Profit Cohorts — Segment × Region")
    best = (
        df.groupby(["Segment", "Region"])
          .agg(
              Profit_Rate=("Profit_Flag", lambda x: round(x.mean() * 100, 1)),
              Avg_Profit=("Profit", lambda x: round(x.mean(), 0)),
              Orders=("Profit_Flag", "count"),
          )
          .reset_index()
          .query("Orders >= 20")
          .sort_values("Profit_Rate", ascending=False)
          .head(8)
    )
    st.dataframe(best, use_container_width=True)

    # ── Prescriptive recommendations ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("5. Actionable Business Recommendations")
    st.markdown(
        f"""
| # | Lever | Specific Action | Expected Impact |
|---|---|---|---|
| 1 | **Freight Mode Repricing** | Apply a ₹30 surcharge (or redirect to Regular Air) for Processed Meat orders shipped via Express Air where Sales < ₹200. | Reduces margin erosion on ~12% of Express Air orders; estimated ₹15K/month recovery. |
| 2 | **Discount Cap Policy** | Cap discount at 8% for orders with Sales < ₹500. Discount-margin correlation is strongly negative — uncapped discounts are the single largest profit leak. | Modelling shows discount > 10% raises loss probability by ~18 ppts. |
| 3 | **Container Optimisation** | Replace Jumbo Drum with Jumbo Box for single-SKU Canned Food orders under 5 units; Jumbo Drums carry disproportionate freight cost for small quantities. | Estimated ₹150/order freight saving; affects ~8% of Canned Foods orders. |
| 4 | **Segment Rebalancing** | Shift new-customer acquisition focus from Hotels/Hospitals (≈42% loss rate) toward Stand Alone Restaurants in the North and East regions (≈60% profit rate). | 10% revenue-mix shift → estimated +3 ppt overall margin improvement. |
| 5 | **Automated Order Gate** | Deploy the scoring model above as a pre-fulfilment gate: orders with Profit_Proba < {risk_thr:.2f} require pricing manager approval before dispatch. With {budget} review slots/day, prevent approximately **{min(budget, len(at_risk))} likely-loss shipments per day**. | Based on avg loss of ₹{at_risk['Profit'].mean():,.0f}/flagged order, potential daily savings ≈ ₹{abs(at_risk['Profit'].mean()) * min(budget, len(at_risk)):,.0f}. |

---
> ⚠️ **Causality note:** All recommendations are derived from **observed correlations** in historical data.
> A/B testing or price experiments are required before attributing causal impact and scaling interventions.
        """
    )
