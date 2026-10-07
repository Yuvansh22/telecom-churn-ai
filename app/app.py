"""Streamlit dashboard.  Run:  streamlit run app/app.py
Requires: python src/generate_data.py, python src/train_models.py, python src/predict.py"""
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RISK_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
RISK_COLOR = {"LOW": "#2ca02c", "MEDIUM": "#f2c230", "HIGH": "#ff7f0e", "CRITICAL": "#d62728"}

st.set_page_config(page_title="Telecom Churn Intelligence", layout="wide")


@st.cache_data
def load():
    data = pd.read_csv(ROOT / "data" / "telecom_churn.csv")
    pred = pd.read_csv(ROOT / "outputs" / "predictions" / "churn_predictions.csv")
    imp = pd.read_csv(ROOT / "outputs" / "metrics" / "feature_importance.csv", index_col=0)
    comp = pd.read_csv(ROOT / "outputs" / "metrics" / "model_comparison.csv", index_col=0)
    return data, pred, imp, comp


try:
    data, pred, imp, comp = load()
except FileNotFoundError:
    st.error("Run `python src/generate_data.py`, `python src/train_models.py` and `python src/predict.py` first.")
    st.stop()

df = data.merge(pred.drop(columns=[c for c in ["actual_churn"] if c in pred.columns]), on="customer_id")
st.title("Telecom Churn Prediction & Retention Intelligence")
st.caption("Churn probability comes from the ML model. Health score and retention actions are rule-based business analytics.")

scope = st.sidebar.radio("Customers included", ["Held-out test set (honest predictions)", "All customers (train rows are in-sample)"])
if scope.startswith("Held-out") and "split" in df.columns:
    df = df[df["split"] == "test"]
st.sidebar.write(f"{len(df):,} customers shown")

tab1, tab2, tab3 = st.tabs(["Overview", "Charts", "Customer lookup"])

with tab1:
    c = st.columns(4)
    c[0].metric("Total customers", f"{len(df):,}")
    c[1].metric("Actual churn rate", f"{df['churn'].mean():.1%}")
    c[2].metric("Avg churn probability", f"{df['churn_probability'].mean():.1%}")
    c[3].metric("High-risk (HIGH + CRITICAL)", f"{df['risk_level'].isin(['HIGH', 'CRITICAL']).sum():,}")
    left, right = st.columns(2)
    rc = df["risk_level"].value_counts().reindex(RISK_ORDER).fillna(0).reset_index()
    rc.columns = ["risk_level", "customers"]
    left.plotly_chart(px.bar(rc, x="risk_level", y="customers", color="risk_level", color_discrete_map=RISK_COLOR,
                             title="Customers by risk level"), width="stretch")
    right.subheader("Model comparison (test set, threshold 0.5)")
    right.dataframe(comp.round(3))
    st.subheader("Top 20 highest-risk customers")
    st.dataframe(df.sort_values("churn_probability", ascending=False)[
        ["customer_id", "churn_probability", "risk_level", "customer_health_score", "top_risk_factors", "recommended_action"]].head(20),
        width="stretch")

with tab2:
    a, b = st.columns(2)
    d = df.assign(churn_label=df["churn"].map({0: "Stayed", 1: "Churned"}))
    a.plotly_chart(px.pie(d, names="churn_label", title="Churn distribution", hole=0.4), width="stretch")
    for col, title, holder in [("contract_type", "Churn rate by contract type", b), ("plan_type", "Churn rate by plan type", a)]:
        g = df.groupby(col)["churn"].mean().mul(100).reset_index()
        holder.plotly_chart(px.bar(g, x=col, y="churn", title=title, labels={"churn": "Churn rate (%)"}), width="stretch")
    sat = df.dropna(subset=["satisfaction_score"]).groupby("satisfaction_score")["churn"].mean().mul(100).reset_index()
    b.plotly_chart(px.bar(sat, x="satisfaction_score", y="churn", title="Satisfaction vs churn rate",
                          labels={"churn": "Churn rate (%)", "satisfaction_score": "Satisfaction score"}), width="stretch")
    a.plotly_chart(px.box(d, x="churn_label", y="monthly_charges", title="Monthly charges vs churn"), width="stretch")
    nd = df.assign(drops=df["network_drops"].clip(upper=15)).groupby("drops")["churn"].mean().mul(100).reset_index()
    b.plotly_chart(px.line(nd, x="drops", y="churn", markers=True, title="Network drops vs churn rate",
                           labels={"churn": "Churn rate (%)", "drops": "Network drops (15 = 15+)"}), width="stretch")
    cm = df.assign(cmp=df["complaints"].clip(upper=6)).groupby("cmp")["churn"].mean().mul(100).reset_index()
    a.plotly_chart(px.bar(cm, x="cmp", y="churn", title="Complaints vs churn rate",
                          labels={"churn": "Churn rate (%)", "cmp": "Complaints (6 = 6+)"}), width="stretch")
    top = imp.head(15).iloc[::-1].reset_index()
    top.columns = ["feature", "importance"]
    b.plotly_chart(px.bar(top, x="importance", y="feature", orientation="h", title="Feature importance (mean |SHAP|)"),
                   width="stretch")

with tab3:
    cid = st.text_input("Enter customer ID", value=df.sort_values("churn_probability", ascending=False).iloc[0]["customer_id"])
    row = df[df["customer_id"] == cid.strip().upper()]
    if row.empty:
        st.warning("Customer not found in the selected set (the sidebar scope may be hiding it).")
    else:
        r = row.iloc[0]
        st.subheader(f"Customer {r['customer_id']}")
        c = st.columns(4)
        c[0].metric("Churn probability", f"{r['churn_probability']:.0%}")
        c[1].metric("Risk level", r["risk_level"])
        c[2].metric("Health score (0-100)", f"{r['customer_health_score']:.0f}")
        c[3].metric("Actual outcome", "Churned" if r["churn"] == 1 else "Stayed")
        c = st.columns(5)
        c[0].metric("Satisfaction", "n/a" if pd.isna(r["satisfaction_score"]) else f"{r['satisfaction_score']:.0f}/10")
        c[1].metric("Complaints", int(r["complaints"]))
        c[2].metric("Network drops", int(r["network_drops"]))
        c[3].metric("Monthly bill", f"{r['monthly_charges']:,.0f}")
        c[4].metric("Bill change (3m)", f"{r['bill_change_3m']:+.1f}%")
        st.markdown("**Top risk factors (SHAP, from the ML model)**")
        for f in str(r["top_risk_factors"]).split("; "):
            st.write("- " + f)
        st.markdown("**Recommended retention action (rule-based business layer)**")
        for x in str(r["recommended_action"]).split(" | "):
            st.write("- " + x)
