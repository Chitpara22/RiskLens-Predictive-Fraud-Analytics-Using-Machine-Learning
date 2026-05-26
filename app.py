"""
app.py  –  RiskLens Fraud Detection Dashboard
Run:  streamlit run app.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

# Resolve all paths relative to this file's directory, regardless of CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)   # ensure relative imports inside joblib / pandas also work

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.metrics import confusion_matrix

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RiskLens – Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: #e94560; font-size: 2.4rem; margin: 0; }
    .main-header p  { color: #a8b2d8; font-size: 1rem; margin: 0.3rem 0 0; }
    .kpi-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460; border-radius: 10px;
        padding: 1.2rem; text-align: center;
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #e94560; }
    .kpi-label { font-size: 0.85rem; color: #a8b2d8; margin-top: 0.2rem; }
    .fraud-badge {
        background: #e94560; color: white;
        padding: 4px 12px; border-radius: 20px; font-weight: 600;
    }
    .safe-badge {
        background: #00b09b; color: white;
        padding: 4px 12px; border-radius: 20px; font-weight: 600;
    }
    .section-title { color: #e94560; font-weight: 700; margin-bottom: 0.5rem; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
    .prediction-box {
        border-radius: 12px; padding: 1.5rem; margin-top: 1rem; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load data & models
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "RiskLens_Dataset.csv"))
    df["Is_fraud"] = df["Is_fraud"].fillna(0).astype(int)
    return df

@st.cache_resource
def load_models():
    if not os.path.exists(os.path.join(BASE_DIR, "models", "best_model.pkl")):
        return None, None, None, None, None
    model    = joblib.load(os.path.join(BASE_DIR, "models", "best_model.pkl"))
    scaler   = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
    encoders = joblib.load(os.path.join(BASE_DIR, "models", "encoders.pkl"))
    meta     = joblib.load(os.path.join(BASE_DIR, "models", "meta.pkl"))
    metrics  = joblib.load(os.path.join(BASE_DIR, "models", "all_metrics.pkl"))
    return model, scaler, encoders, meta, metrics

df = load_data()
model, scaler, encoders, meta, all_metrics = load_models()
MODELS_READY = model is not None

CAT_COLS = ["Gender", "Region", "Acquisition_Channel", "Lead_Source",
            "Device_Type", "Product_Plan", "Subscription_Type",
            "Payment_Mode", "Is_Renewed", "Is_Churned"]
NUM_COLS = ["Age", "Monthly_Revenue", "Discount_Amount", "Net_Revenue",
            "Marketing_Cost", "Sales_Cost", "Support_Cost",
            "Customer_Lifetime_Months", "Customer_Satisfaction_Score",
            "Employees_Count", "Total_Cost", "Profit",
            "Profit_Margin_%", "Effective_Revenue",
            "year", "month", "quarter"]

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🛡️ RiskLens – Fraud Detection System</h1>
  <p>ML-powered transaction fraud analysis & real-time prediction dashboard</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar filters
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/100/shield.png", width=80)
    st.title("🔧 Filters")

    regions = ["All"] + sorted(df["Region"].dropna().unique().tolist())
    sel_region = st.selectbox("Region", regions)

    plans = ["All"] + sorted(df["Product_Plan"].dropna().unique().tolist())
    sel_plan = st.selectbox("Product Plan", plans)

    payment_modes = ["All"] + sorted(df["Payment_Mode"].dropna().unique().tolist())
    sel_payment = st.selectbox("Payment Mode", payment_modes)

    age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
    age_range = st.slider("Age Range", age_min, age_max, (age_min, age_max))

    st.markdown("---")
    st.info("📌 Train the model first using `python train_model.py`")

# Apply filters
dff = df.copy()
if sel_region  != "All": dff = dff[dff["Region"]       == sel_region]
if sel_plan    != "All": dff = dff[dff["Product_Plan"]  == sel_plan]
if sel_payment != "All": dff = dff[dff["Payment_Mode"]  == sel_payment]
dff = dff[(dff["Age"] >= age_range[0]) & (dff["Age"] <= age_range[1])]

fraud_df  = dff[dff["Is_fraud"] == 1]
normal_df = dff[dff["Is_fraud"] == 0]

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🔍 EDA & Insights",
    "🤖 Model Performance",
    "🎯 Predict Transaction",
    "📋 Data Explorer",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    total    = len(dff)
    fraud_n  = int(dff["Is_fraud"].sum())
    normal_n = total - fraud_n
    fraud_pct = fraud_n / total * 100 if total else 0
    avg_rev  = dff["Net_Revenue"].mean()
    fraud_rev = fraud_df["Net_Revenue"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, total,         "Total Transactions"),
        (c2, fraud_n,       "Fraud Cases"),
        (c3, normal_n,      "Legitimate Cases"),
        (c4, f"{fraud_pct:.1f}%", "Fraud Rate"),
        (c5, f"₹{avg_rev:,.0f}", "Avg Net Revenue"),
    ]
    for col, val, label in cards:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-value">{val}</div>
              <div class="kpi-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### 🍩 Fraud vs Legitimate")
        fig = go.Figure(go.Pie(
            labels=["Legitimate", "Fraud"],
            values=[normal_n, fraud_n],
            hole=0.55,
            marker_colors=["#00b09b", "#e94560"],
            textinfo="label+percent",
        ))
        fig.update_layout(showlegend=False, height=320,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### 📅 Monthly Fraud Trend")
        trend = dff.groupby(["year","month"])["Is_fraud"].agg(["sum","count"]).reset_index()
        trend["period"] = pd.to_datetime(trend.assign(day=1)[["year","month","day"]]).dt.strftime("%b %Y")
        trend["rate"] = trend["sum"] / trend["count"] * 100
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Bar(x=trend["period"], y=trend["sum"],
                              name="Fraud Count", marker_color="#e94560", opacity=0.75))
        fig2.add_trace(go.Scatter(x=trend["period"], y=trend["rate"],
                                  name="Fraud Rate %", line=dict(color="#f5a623", width=2)),
                       secondary_y=True)
        fig2.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", font_color="white", showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🌍 Fraud by Region")
        reg = dff.groupby("Region")["Is_fraud"].agg(["sum","count"]).reset_index()
        reg["rate"] = reg["sum"] / reg["count"] * 100
        fig3 = px.bar(reg.sort_values("rate", ascending=True),
                      x="rate", y="Region", orientation="h",
                      color="rate", color_continuous_scale="RdYlGn_r",
                      labels={"rate": "Fraud Rate %"}, height=300)
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.markdown("#### 💳 Fraud by Payment Mode")
        pm = dff.groupby("Payment_Mode")["Is_fraud"].agg(["sum","count"]).reset_index()
        pm["rate"] = pm["sum"] / pm["count"] * 100
        fig4 = px.bar(pm, x="Payment_Mode", y="rate",
                      color="rate", color_continuous_scale="RdYlGn_r",
                      labels={"rate": "Fraud Rate %"}, height=300)
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – EDA
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🔍 Exploratory Data Analysis")

    row1_l, row1_r = st.columns(2)
    with row1_l:
        st.markdown("#### Age Distribution by Fraud")
        fig = px.histogram(dff, x="Age", color=dff["Is_fraud"].map({0:"Legitimate",1:"Fraud"}),
                           barmode="overlay", nbins=30, opacity=0.7,
                           color_discrete_map={"Legitimate":"#00b09b","Fraud":"#e94560"},
                           labels={"color":"Type"}, height=300)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with row1_r:
        st.markdown("#### Net Revenue by Fraud Status")
        fig = px.box(dff, x=dff["Is_fraud"].map({0:"Legitimate",1:"Fraud"}),
                     y="Net_Revenue",
                     color=dff["Is_fraud"].map({0:"Legitimate",1:"Fraud"}),
                     color_discrete_map={"Legitimate":"#00b09b","Fraud":"#e94560"},
                     labels={"x":"","Net_Revenue":"Net Revenue"}, height=300)
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    row2_l, row2_r = st.columns(2)
    with row2_l:
        st.markdown("#### Customer Satisfaction vs Fraud")
        fig = px.violin(dff, x=dff["Is_fraud"].map({0:"Legitimate",1:"Fraud"}),
                        y="Customer_Satisfaction_Score", box=True,
                        color=dff["Is_fraud"].map({0:"Legitimate",1:"Fraud"}),
                        color_discrete_map={"Legitimate":"#00b09b","Fraud":"#e94560"},
                        height=300)
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with row2_r:
        st.markdown("#### Profit Margin by Fraud Status")
        fig = px.box(dff, x=dff["Is_fraud"].map({0:"Legitimate",1:"Fraud"}),
                     y="Profit_Margin_%",
                     color=dff["Is_fraud"].map({0:"Legitimate",1:"Fraud"}),
                     color_discrete_map={"Legitimate":"#00b09b","Fraud":"#e94560"},
                     height=300)
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📈 Correlation Heatmap (Numeric Features)")
    num_df = dff[NUM_COLS + ["Is_fraud"]].corr()
    fig_h = px.imshow(num_df, color_continuous_scale="RdBu_r", aspect="auto",
                      zmin=-1, zmax=1, height=500)
    fig_h.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("#### 🏷️ Fraud Rate by Product Plan & Device Type")
    pivot = dff.pivot_table(values="Is_fraud", index="Product_Plan",
                            columns="Device_Type", aggfunc="mean") * 100
    fig_p = px.imshow(pivot, color_continuous_scale="RdYlGn_r",
                      labels={"color":"Fraud Rate %"}, height=350)
    fig_p.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig_p, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🤖 Model Performance & Evaluation")
    if not MODELS_READY:
        st.warning("⚠️ Models not found. Run `python train_model.py` first.")
    else:
        best_name = meta["best_model_name"]
        st.success(f"✅ Best Model: **{best_name}**")

        # Summary table
        rows = []
        for name, m in all_metrics.items():
            rpt = m["report"]
            rows.append({
                "Model": name,
                "ROC-AUC":      round(m["auc"], 4),
                "Avg Precision":round(m["ap"],  4),
                "Accuracy":     round(rpt["accuracy"], 4),
                "Precision (Fraud)": round(rpt.get("1",rpt.get("1.0",{})).get("precision",0),4),
                "Recall (Fraud)":    round(rpt.get("1",rpt.get("1.0",{})).get("recall",0),4),
                "F1 (Fraud)":        round(rpt.get("1",rpt.get("1.0",{})).get("f1-score",0),4),
            })
        summ_df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
        st.dataframe(summ_df.style.highlight_max(axis=0, color="#1f4e3d"), use_container_width=True)

        col_sel = st.selectbox("Select model to inspect:", list(all_metrics.keys()))
        m = all_metrics[col_sel]

        c1, c2 = st.columns(2)
        with c1:
            # ROC Curve
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=m["fpr"], y=m["tpr"],
                                         name=f"AUC={m['auc']:.4f}",
                                         fill="tozeroy", fillcolor="rgba(233,69,96,0.15)",
                                         line=dict(color="#e94560", width=2)))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1],
                                         line=dict(dash="dash", color="#888")))
            fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR",
                                  height=350, paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig_roc, use_container_width=True)

        with c2:
            # Precision-Recall Curve
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=m["recall"], y=m["precision"],
                                        name=f"AP={m['ap']:.4f}",
                                        fill="tozeroy", fillcolor="rgba(0,176,155,0.15)",
                                        line=dict(color="#00b09b", width=2)))
            fig_pr.update_layout(title="Precision-Recall Curve",
                                  xaxis_title="Recall", yaxis_title="Precision",
                                  height=350, paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig_pr, use_container_width=True)

        # Confusion Matrix
        st.markdown("#### Confusion Matrix")
        cm = np.array(m["cm"])
        labels = ["Non-Fraud", "Fraud"]
        fig_cm = px.imshow(cm, text_auto=True, x=labels, y=labels,
                           color_continuous_scale="Blues",
                           labels=dict(x="Predicted", y="Actual"),
                           height=350)
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_cm, use_container_width=True)

        # Feature importances (Random Forest / XGBoost / GBM)
        if hasattr(model, "feature_importances_"):
            st.markdown("#### 🌟 Feature Importances (Best Model)")
            fi = pd.DataFrame({
                "Feature":    meta["feature_names"],
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=True).tail(20)
            fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                            color="Importance", color_continuous_scale="RdYlGn",
                            height=500)
            fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig_fi, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – PREDICT TRANSACTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🎯 Real-Time Fraud Prediction")
    if not MODELS_READY:
        st.warning("⚠️ Models not found. Run `python train_model.py` first.")
    else:
        st.markdown("Fill in the transaction details below:")

        with st.form("predict_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                gender        = st.selectbox("Gender",        df["Gender"].unique())
                region        = st.selectbox("Region",        df["Region"].unique())
                acquisition   = st.selectbox("Acquisition Channel", df["Acquisition_Channel"].unique())
                lead_source   = st.selectbox("Lead Source",   df["Lead_Source"].unique())
                device        = st.selectbox("Device Type",   df["Device_Type"].unique())

            with col2:
                product_plan  = st.selectbox("Product Plan",   df["Product_Plan"].unique())
                sub_type      = st.selectbox("Subscription",   df["Subscription_Type"].unique())
                payment_mode  = st.selectbox("Payment Mode",   df["Payment_Mode"].unique())
                is_renewed    = st.selectbox("Is Renewed",     df["Is_Renewed"].unique())
                is_churned    = st.selectbox("Is Churned",     df["Is_Churned"].unique())

            with col3:
                age           = st.number_input("Age",           10,  100, 35)
                monthly_rev   = st.number_input("Monthly Revenue",0.0, 1e6, 5000.0, step=100.0)
                discount      = st.number_input("Discount Amount",0.0, 1e5, 200.0, step=10.0)
                marketing_c   = st.number_input("Marketing Cost", 0.0, 1e5, 500.0, step=50.0)
                sales_c       = st.number_input("Sales Cost",     0.0, 1e5, 300.0, step=50.0)
                support_c     = st.number_input("Support Cost",   0.0, 1e5, 100.0, step=10.0)
                cust_life     = st.number_input("Customer Lifetime (months)", 1, 120, 12)
                csat          = st.slider("Customer Satisfaction Score", 0.0, 10.0, 7.0, 0.1)
                employees     = st.number_input("Employees Count", 1, 10000, 50)
                year_val      = st.number_input("Year",  2020, 2030, 2023)
                month_val     = st.number_input("Month", 1,    12,   6)
                quarter_val   = st.number_input("Quarter",1,   4,    2)

            submitted = st.form_submit_button("🔍 Predict Fraud Risk", use_container_width=True)

        if submitted:
            net_rev   = monthly_rev - discount
            total_c   = marketing_c + sales_c + support_c
            profit    = net_rev - total_c
            profit_m  = (profit / net_rev * 100) if net_rev else 0
            eff_rev   = net_rev * 1.05   # simplified effective revenue

            row_cat = {
                "Gender": gender, "Region": region, "Acquisition_Channel": acquisition,
                "Lead_Source": lead_source, "Device_Type": device,
                "Product_Plan": product_plan, "Subscription_Type": sub_type,
                "Payment_Mode": payment_mode, "Is_Renewed": is_renewed, "Is_Churned": is_churned,
            }
            row_num = {
                "Age": age, "Monthly_Revenue": monthly_rev, "Discount_Amount": discount,
                "Net_Revenue": net_rev, "Marketing_Cost": marketing_c, "Sales_Cost": sales_c,
                "Support_Cost": support_c, "Customer_Lifetime_Months": cust_life,
                "Customer_Satisfaction_Score": csat, "Employees_Count": employees,
                "Total_Cost": total_c, "Profit": profit, "Profit_Margin_%": profit_m,
                "Effective_Revenue": eff_rev, "year": year_val,
                "month": month_val, "quarter": quarter_val,
            }

            enc_row = {}
            for col, val in row_cat.items():
                le = encoders.get(col)
                try:
                    enc_row[col] = le.transform([val])[0]
                except Exception:
                    enc_row[col] = 0
            enc_row.update(row_num)

            feature_order = meta["feature_names"]
            X_input = pd.DataFrame([enc_row])[feature_order].fillna(0)
            X_scaled_input = scaler.transform(X_input)

            pred      = model.predict(X_scaled_input)[0]
            prob      = model.predict_proba(X_scaled_input)[0]
            fraud_prob = prob[1] * 100
            safe_prob  = prob[0] * 100

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=fraud_prob,
                number={"suffix": "%", "font": {"size": 36, "color": "#e94560"}},
                delta={"reference": 16.1, "increasing": {"color": "#e94560"},
                       "decreasing": {"color": "#00b09b"}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#e94560" if pred == 1 else "#00b09b"},
                    "steps": [
                        {"range": [0,  30], "color": "#1f4e3d"},
                        {"range": [30, 60], "color": "#4e4e1f"},
                        {"range": [60,100], "color": "#4e1f1f"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 3},
                                  "thickness": 0.8, "value": 50},
                },
                title={"text": "Fraud Probability", "font": {"size": 18, "color": "white"}},
            ))
            fig_gauge.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig_gauge, use_container_width=True)

            if pred == 1:
                st.markdown(f"""
                <div class="prediction-box" style="background:rgba(233,69,96,0.15); border:2px solid #e94560;">
                  <h2 style="color:#e94560">🚨 FRAUD DETECTED</h2>
                  <p style="color:white; font-size:1.1rem;">
                    This transaction has a <strong style="color:#e94560">{fraud_prob:.1f}%</strong>
                    probability of being fraudulent.<br>
                    Recommended action: <strong>Block & Investigate</strong>
                  </p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box" style="background:rgba(0,176,155,0.15); border:2px solid #00b09b;">
                  <h2 style="color:#00b09b">✅ LEGITIMATE TRANSACTION</h2>
                  <p style="color:white; font-size:1.1rem;">
                    This transaction has a <strong style="color:#00b09b">{safe_prob:.1f}%</strong>
                    probability of being legitimate.<br>
                    Recommended action: <strong>Approve</strong>
                  </p>
                </div>""", unsafe_allow_html=True)

            with st.expander("📊 Detailed Probability Breakdown"):
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.metric("Fraud Probability",      f"{fraud_prob:.2f}%")
                with col_p2:
                    st.metric("Legitimate Probability", f"{safe_prob:.2f}%")
                st.markdown("**Derived Features:**")
                st.json({
                    "Net Revenue":    round(net_rev,  2),
                    "Total Cost":     round(total_c,  2),
                    "Profit":         round(profit,   2),
                    "Profit Margin":  f"{profit_m:.2f}%",
                    "Effective Revenue": round(eff_rev, 2),
                })

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 – DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 📋 Data Explorer")

    c1, c2, c3 = st.columns(3)
    with c1: filter_fraud = st.selectbox("Fraud Status", ["All", "Fraud", "Legitimate"])
    with c2: max_rows = st.slider("Max rows to display", 10, 500, 50)
    with c3:
        search_col = st.selectbox("Search by column", ["None"] + CAT_COLS)

    view_df = dff.copy()
    if filter_fraud == "Fraud":       view_df = view_df[view_df["Is_fraud"] == 1]
    elif filter_fraud == "Legitimate":view_df = view_df[view_df["Is_fraud"] == 0]

    if search_col != "None":
        vals   = ["All"] + sorted(view_df[search_col].dropna().unique().tolist())
        chosen = st.selectbox(f"Filter by {search_col}", vals)
        if chosen != "All": view_df = view_df[view_df[search_col] == chosen]

    st.dataframe(
        view_df.head(max_rows).style.apply(
            lambda x: ["background-color: rgba(233,69,96,0.2)" if v == 1
                       else "background-color: rgba(0,176,155,0.1)"
                       for v in x], subset=["Is_fraud"]),
        use_container_width=True
    )

    st.markdown(f"Showing **{min(max_rows, len(view_df))}** of **{len(view_df)}** rows")

    st.download_button(
        "⬇️ Download filtered data as CSV",
        data=view_df.to_csv(index=False),
        file_name="filtered_transactions.csv",
        mime="text/csv",
    )

    st.markdown("#### 📊 Quick Statistics")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**Numeric Summary**")
        st.dataframe(view_df[NUM_COLS[:8]].describe().T.round(2), use_container_width=True)
    with col_s2:
        st.markdown("**Categorical Breakdown**")
        cat_col_pick = st.selectbox("Pick a column", CAT_COLS)
        freq = view_df[cat_col_pick].value_counts().reset_index()
        freq.columns = [cat_col_pick, "Count"]
        freq["Fraud Rate %"] = (
            view_df.groupby(cat_col_pick)["Is_fraud"].mean().reindex(freq[cat_col_pick]).values * 100
        ).round(2)
        st.dataframe(freq, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#a8b2d8; font-size:0.85rem;'>"
    "🛡️ RiskLens Fraud Detection System · Built with Streamlit & Scikit-Learn"
    "</p>", unsafe_allow_html=True
)
