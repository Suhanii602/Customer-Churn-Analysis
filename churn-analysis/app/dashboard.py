import streamlit as st
import pandas as pd

st.set_page_config(page_title="Churn Risk Dashboard", layout="wide", page_icon="◆")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
.stApp { background-color: #0B0E14; }
#MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; }

.dash-title { font-size: 1.5rem; font-weight: 700; color: #F2F4F8; margin-bottom: 2px; }
.dash-sub { color: #7A8296; font-size: 0.85rem; margin-bottom: 20px; }
.model-pill { background: #151922; border: 1px solid #232838; border-radius: 8px;
    padding: 6px 14px; color: #C4C9D4; font-size: 0.8rem; font-weight: 500; }
.model-pill span { color: #545C6E; font-size: 0.72rem; display: block; }

div[role="radiogroup"] { gap: 6px; }
div[role="radiogroup"] label { background: #151922; border: 1px solid #232838;
    border-radius: 20px; padding: 5px 16px !important; margin-right: 4px; }

.kpi-row { display: flex; gap: 14px; margin: 18px 0 26px 0; }
.kpi-card { background: #151922; border: 1px solid #232838; border-radius: 10px; padding: 16px 20px; flex: 1; }
.kpi-label { font-size: 0.75rem; color: #7A8296; margin-bottom: 6px; }
.kpi-value { font-size: 1.5rem; font-weight: 700; color: #F2F4F8; }

.section-header { font-size: 0.95rem; font-weight: 600; color: #E2E5EC; margin: 6px 0 12px 0; }
.section-note { color: #7A8296; font-size: 0.78rem; margin: -8px 0 14px 0; }

.card-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; margin-bottom: 30px; }
.risk-card { background: #151922; border: 1px solid #232838; border-radius: 14px; padding: 18px; }
.card-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; }
.card-id { color: #F2F4F8; font-weight: 600; font-size: 0.95rem; }
.card-sub { color: #7A8296; font-size: 0.75rem; }
.pill { padding: 3px 12px; border-radius: 20px; font-size: 0.7rem; font-weight: 700; }
.pill-High { background: #F87171; color: #1a0d0d; }
.pill-Medium { background: #FBBF24; color: #1a1608; }
.pill-Low { background: #34D399; color: #05140f; }
.prob-big { font-size: 1.7rem; font-weight: 700; color: #F2F4F8; }
.prob-label { color: #7A8296; font-size: 0.72rem; margin-bottom: 10px; }
.gauge-bg { background: #1C2130; border-radius: 4px; height: 7px; width: 100%; overflow: hidden; margin-bottom: 12px; }
.gauge-fill { height: 100%; border-radius: 4px; }
.tag { display: inline-block; background: #1C2130; color: #A8B0C2; border-radius: 5px;
       padding: 3px 8px; font-size: 0.7rem; margin: 0 4px 4px 0; }

.churn-table { width: 100%; border-collapse: collapse; }
.churn-table th { text-align: left; font-size: 0.72rem; color: #7A8296; font-weight: 500;
    padding: 8px 14px; border-bottom: 1px solid #232838; }
.churn-table td { padding: 9px 14px; font-size: 0.83rem; color: #D6DAE3; border-bottom: 1px solid #171B26; }
.churn-table tr:hover td { background: #131722; }
.cust-id-cell { color: #F2F4F8; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

RISK_COLOR = {"High": "#F87171", "Medium": "#FBBF24", "Low": "#34D399"}

df = pd.read_csv("outputs/predictions.csv")

# header
h1, h2 = st.columns([4, 1.4])
with h1:
    st.markdown('<div class="dash-title">Customer Churn Risk</div>', unsafe_allow_html=True)
    st.markdown('<div class="dash-sub">Prioritization tool for retention outreach — not a guarantee of churn. High-risk cases require human review before contact.</div>', unsafe_allow_html=True)
with h2:
    st.markdown('<div class="model-pill"><span>Active model</span>Logistic Regression</div>', unsafe_allow_html=True)

# filters
f1, f2, f3 = st.columns([2, 2.4, 2])
with f1:
    risk_tab = st.radio("Risk band", ["All", "High", "Medium", "Low"], horizontal=True, label_visibility="collapsed")
with f2:
    min_prob = st.slider("Minimum probability", 0.0, 1.0, 0.0, 0.05)
with f3:
    search_id = st.text_input("Search customer ID", placeholder="e.g. CUST-1042")

filtered = df[df["churn_probability"] >= min_prob]
if risk_tab != "All":
    filtered = filtered[filtered["risk_band"] == risk_tab]
if search_id:
    filtered = filtered[filtered["customer_id"].str.contains(search_id, case=False)]
filtered = filtered.sort_values("churn_probability", ascending=False)

# KPI
band_counts = df["risk_band"].value_counts()
st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-card"><div class="kpi-label">Matching filter</div><div class="kpi-value">{len(filtered):,}</div></div>
    <div class="kpi-card"><div class="kpi-label">High risk (total book)</div><div class="kpi-value" style="color:#F87171">{band_counts.get('High', 0):,}</div></div>
    <div class="kpi-card"><div class="kpi-label">Medium risk (total book)</div><div class="kpi-value" style="color:#FBBF24">{band_counts.get('Medium', 0):,}</div></div>
    <div class="kpi-card"><div class="kpi-label">Low risk (total book)</div><div class="kpi-value" style="color:#34D399">{band_counts.get('Low', 0):,}</div></div>
</div>
""", unsafe_allow_html=True)

# card grid
st.markdown('<div class="section-header">Accounts matching current filter</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-note">Showing top {min(6, len(filtered))} of {len(filtered):,} — sorted by probability within the "{risk_tab}" filter. Switch the tab above to see Medium/Low examples.</div>', unsafe_allow_html=True)

card_html = '<div class="card-grid">'
for _, r in filtered.head(6).iterrows():
    reasons = "".join(f'<span class="tag">{t.strip()}</span>' for t in str(r["top_reason_indicators"]).split(";")[:3])
    color = RISK_COLOR.get(r["risk_band"], "#8A93A6")
    card_html += f"""
    <div class="risk-card">
        <div class="card-top">
            <div><div class="card-id">{r['customer_id']}</div><div class="card-sub">{r['recommended_queue']}</div></div>
            <span class="pill pill-{r['risk_band']}">{r['risk_band'].upper()}</span>
        </div>
        <div class="prob-big">{r['churn_probability']:.1%}</div>
        <div class="prob-label">Predicted churn probability</div>
        <div class="gauge-bg"><div class="gauge-fill" style="width:{r['churn_probability']*100:.0f}%; background:{color};"></div></div>
        <div>{reasons}</div>
    </div>"""
card_html += "</div>"
st.markdown(card_html, unsafe_allow_html=True)

# table
st.markdown('<div class="section-header">Full list</div>', unsafe_allow_html=True)
rows_html = ""
for _, r in filtered.head(100).iterrows():
    reasons = "".join(f'<span class="tag">{t.strip()}</span>' for t in str(r["top_reason_indicators"]).split(";"))
    rows_html += f"""<tr>
        <td class="cust-id-cell">{r['customer_id']}</td>
        <td>{r['churn_probability']:.1%}</td>
        <td><span class="pill pill-{r['risk_band']}">{r['risk_band'].upper()}</span></td>
        <td>{reasons}</td>
        <td>{r['recommended_queue']}</td>
    </tr>"""
st.markdown(f"""<table class="churn-table">
    <thead><tr><th>Customer</th><th>Probability</th><th>Risk</th><th>Reason indicators</th><th>Queue</th></tr></thead>
    <tbody>{rows_html}</tbody></table>""", unsafe_allow_html=True)

if len(filtered) > 100:
    st.caption(f"Showing 100 of {len(filtered):,} matching customers.")