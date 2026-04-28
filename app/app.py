import streamlit as st
import numpy as np
import joblib
import json
import os

# -------------------- LOAD MODELS --------------------
clf = joblib.load("../models/classification.pkl")
reg = joblib.load("../models/regression.pkl")
kmeans = joblib.load("../models/clustering.pkl")
scaler = joblib.load("../models/scaler.pkl")

# -------------------- LOAD METRICS --------------------
metrics_path = "../models/metrics.json"

if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        metrics = json.load(f)
else:
    metrics = {
        "classification": {"accuracy": 0},
        "regression": {"r2": 0, "rmse": 0},
        "clustering": {"silhouette": 0}
    }

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Smart Retail AI", layout="wide")

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.main {
    background-color: #0f172a;
}
h1 {
    color: #38bdf8;
    text-align: center;
}
.card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.5);
    text-align: center;
}
.metric {
    font-size: 28px;
    font-weight: bold;
    color: #facc15;
}
.label {
    font-size: 16px;
    color: #94a3b8;
}
.section {
    padding: 15px;
    border-radius: 10px;
    background: #1e293b;
}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.markdown("<h1>🧠 Smart Retail Intelligence System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#94a3b8;'>AI Powered Customer Insights Dashboard</p>", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.header("📊 Customer Input")

age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.slider("Annual Income", 10000, 100000, 40000)
spending_score = st.sidebar.slider("Spending Score", 1, 100, 50)
purchases = st.sidebar.slider("Purchases", 1, 50, 10)
visits = st.sidebar.slider("Website Visits", 1, 100, 20)
cart_abandon = st.sidebar.slider("Cart Abandon Rate", 0.0, 1.0, 0.3)

predict_btn = st.sidebar.button("🚀 Predict")

# -------------------- PREDICTION --------------------
if predict_btn:

    # Feature engineering
    total_spend = purchases * 500
    engagement_score = (visits * 0.7) + ((1 - cart_abandon) * 0.3)

    # Predictions
    cluster = kmeans.predict(
        scaler.transform([[age, income, spending_score, total_spend, engagement_score]])
    )[0]

    churn = clf.predict(np.array([[
        age, income, spending_score,
        purchases, visits, cart_abandon,
        total_spend, engagement_score
    ]]))[0]

    spend = reg.predict(np.array([[income, purchases, engagement_score]]))[0]

    # Mapping
    segment_map = {
        0: "💎 Premium",
        1: "🛍️ Regular",
        2: "📉 Low Value"
    }

    churn_map = {
        0: "🟢 Low Risk",
        1: "🔴 High Risk"
    }

    # -------------------- RESULT CARDS --------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="label">Customer Segment</div>
            <div class="metric">{segment_map.get(cluster)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="label">Churn Risk</div>
            <div class="metric">{churn_map.get(churn)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
            <div class="label">Predicted Spend</div>
            <div class="metric">₹ {int(spend)}</div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------- RECOMMENDATIONS --------------------
    st.markdown("---")
    st.subheader("📢 AI Recommendations")

    if churn == 1:
        st.error("⚠️ High churn risk! Offer discounts or personalized campaigns.")
    else:
        st.success("✅ Customer likely to stay. Maintain engagement.")

    if cluster == 0:
        st.info("💎 Premium: Provide loyalty rewards & VIP offers.")
    elif cluster == 1:
        st.info("🛍️ Regular: Promote personalized deals.")
    else:
        st.info("📉 Low-value: Increase engagement with offers.")

    # -------------------- MODEL PERFORMANCE --------------------
    st.markdown("---")
    st.subheader("📊 Model Performance")

    m1, m2, m3 = st.columns(3)

    # Classification
    with m1:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### 🟢 Classification")
        st.metric("Accuracy", f"{metrics['classification']['accuracy']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Regression
    with m2:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### 🟣 Regression")
        st.metric("R² Score", f"{metrics['regression']['r2']:.2f}")
        st.metric("RMSE", f"{metrics['regression']['rmse']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Clustering
    with m3:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### 🔵 Clustering")
        st.metric("Silhouette Score", f"{metrics['clustering']['silhouette']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("<p style='text-align:center;color:#64748b;'>🚀 Built with Machine Learning + Streamlit</p>", unsafe_allow_html=True)