import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="FraudShield | Fraud & Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: linear-gradient(135deg, #0a0a1a, #0d1b2a, #0a0a1a); }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #0a1628 100%);
        border-right: 1px solid rgba(0,212,255,0.15);
    }

    .hero-banner {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a3a5c 50%, #0d2137 100%);
        border: 1px solid rgba(0,212,255,0.2);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 0 40px rgba(0,212,255,0.1);
    }
    .hero-banner h1 { color: #00d4ff; font-size: 2.4rem; font-weight: 700; margin: 0; }
    .hero-banner p  { color: rgba(255,255,255,0.7); font-size: 1rem; margin: 0.5rem 0 0; }

    .stat-card {
        background: rgba(0,212,255,0.05);
        border: 1px solid rgba(0,212,255,0.15);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .stat-label { color: rgba(255,255,255,0.5); font-size: 0.75rem; text-transform: uppercase;
                  letter-spacing: 0.1em; margin-bottom: 0.3rem; }
    .stat-value { color: #00d4ff; font-size: 1.8rem; font-weight: 700; }
    .stat-sub   { color: rgba(255,255,255,0.4); font-size: 0.7rem; margin-top: 0.2rem; }

    .result-fraud {
        background: linear-gradient(135deg, rgba(231,76,60,0.2), rgba(192,57,43,0.1));
        border: 2px solid rgba(231,76,60,0.5);
        border-radius: 14px; padding: 1.8rem; text-align: center;
        animation: glow-red 2s ease-in-out infinite;
    }
    .result-safe {
        background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(0,150,200,0.05));
        border: 2px solid rgba(0,212,255,0.4);
        border-radius: 14px; padding: 1.8rem; text-align: center;
    }
    .result-title { font-size: 1.8rem; font-weight: 700; color: #fff; margin-bottom: 0.4rem; }
    .result-sub   { color: rgba(255,255,255,0.7); font-size: 0.9rem; }

    @keyframes glow-red {
        0%, 100% { box-shadow: 0 0 10px rgba(231,76,60,0.3); }
        50%       { box-shadow: 0 0 25px rgba(231,76,60,0.6); }
    }

    .section-title {
        color: #00d4ff; font-size: 0.9rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.1em;
        margin: 1.2rem 0 0.6rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(0,212,255,0.2);
    }

    .risk-badge-high { background: rgba(231,76,60,0.2); border: 1px solid #e74c3c;
                       color: #e74c3c; padding: 0.2rem 0.8rem; border-radius: 20px;
                       font-size: 0.75rem; font-weight: 600; }
    .risk-badge-low  { background: rgba(0,212,255,0.1); border: 1px solid #00d4ff;
                       color: #00d4ff; padding: 0.2rem 0.8rem; border-radius: 20px;
                       font-size: 0.75rem; font-weight: 600; }

    div[data-testid="stMetric"] {
        background: rgba(0,212,255,0.05);
        border: 1px solid rgba(0,212,255,0.1);
        border-radius: 10px; padding: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Model Loading ─────────────────────────────────────────────────────────────
APP_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(APP_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

@st.cache_resource(show_spinner=False)
def load_models():
    cc_model   = joblib.load(os.path.join(MODEL_DIR, 'cc_fraud_model.pkl'))
    cc_scaler  = joblib.load(os.path.join(MODEL_DIR, 'cc_scaler.pkl'))
    cc_feats   = joblib.load(os.path.join(MODEL_DIR, 'cc_features.pkl'))
    upi_model  = joblib.load(os.path.join(MODEL_DIR, 'upi_model.pkl'))
    upi_scaler = joblib.load(os.path.join(MODEL_DIR, 'upi_scaler.pkl'))
    upi_feats  = joblib.load(os.path.join(MODEL_DIR, 'upi_features.pkl'))
    return cc_model, cc_scaler, cc_feats, upi_model, upi_scaler, upi_feats

@st.cache_resource(show_spinner=False)
def load_data():
    data_dir = os.path.join(ROOT_DIR, 'data')
    cc  = pd.read_csv(os.path.join(data_dir, 'creditcard.csv'))
    upi = pd.read_csv(os.path.join(data_dir, 'upi_transactions.csv'))
    return cc, upi

# ── Gauge Chart ───────────────────────────────────────────────────────────────
def fraud_gauge(prob, title="Fraud Risk Score"):
    pct = prob * 100
    color = "#e74c3c" if pct > 60 else "#f39c12" if pct > 30 else "#00d4ff"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'color': 'white', 'size': 15}},
        number={'suffix': '%', 'font': {'color': 'white', 'size': 34}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': "rgba(255,255,255,0.03)",
            'borderwidth': 0,
            'steps': [
                {'range': [0,  30], 'color': 'rgba(0,212,255,0.1)'},
                {'range': [30, 60], 'color': 'rgba(243,156,18,0.1)'},
                {'range': [60,100], 'color': 'rgba(231,76,60,0.1)'},
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}, height=260,
        margin=dict(l=20, r=20, t=40, b=10)
    )
    return fig

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="hero-banner">
        <h1>🛡️ FraudShield AI</h1>
        <p>Real-time fraud & anomaly detection for Credit Card and UPI transactions</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading models..."):
        try:
            cc_model, cc_scaler, cc_feats, upi_model, upi_scaler, upi_feats = load_models()
            cc_df, upi_df = load_data()
        except FileNotFoundError:
            st.error("Models not found. Please run `notebooks/fraud_analysis.py` first.")
            return

    tab1, tab2, tab3 = st.tabs(["💳 Credit Card Fraud", "📱 UPI Anomaly", "📊 Dataset Insights"])

    # ── TAB 1: Credit Card ────────────────────────────────────────────────────
    with tab1:
        col_in, col_out = st.columns([1, 1.3])

        with col_in:
            st.markdown('<p class="section-title">Transaction Details</p>', unsafe_allow_html=True)
            amount  = st.number_input("Transaction Amount (₹)", min_value=1.0, max_value=100000.0, value=2500.0, step=100.0)
            hour    = st.slider("Hour of Day", 0, 23, 14)
            
            st.markdown('<p class="section-title">Behavioural Features</p>', unsafe_allow_html=True)
            merchant_cats = ['Food & Dining', 'Shopping', 'Electronics', 'Travel', 'Fuel', 'Healthcare']
            cc_cat_val = merchant_cats.index(st.selectbox("Merchant Category", merchant_cats))
            
            txn_types = ['Online', 'POS (In-Store)', 'ATM Withdrawal', 'Contactless']
            cc_txn_val = txn_types.index(st.selectbox("Transaction Type", txn_types))
            
            card_types = ['Debit', 'Credit', 'Prepaid']
            cc_card_val = card_types.index(st.selectbox("Card Type", card_types))
            
            cc_intl_val = 1 if st.selectbox("International Transaction?", ["No", "Yes"]) == "Yes" else 0
            
            cc_velocity = st.slider("Transactions in last 24h (Velocity)", 0, 20, 2)
            
            cc_new_merch = 1 if st.selectbox("New Merchant?", ["No (Used before)", "Yes (First time)"]) == "Yes (First time)" else 0
            
            cc_mins_since = st.slider("Minutes since last transaction", 1, 1440, 120)
            
            check_btn = st.button("🔍 Check for Fraud", use_container_width=True, type="primary")

        with col_out:
            if check_btn:
                # amount, hour, merchant_cat, txn_type, card_type, is_international, velocity, is_new_merchant, mins_since_last
                inp = np.array([[amount, hour, cc_cat_val, cc_txn_val, cc_card_val, cc_intl_val, cc_velocity, cc_new_merch, cc_mins_since]])
                inp_sc = cc_scaler.transform(inp)
                prob = cc_model.predict_proba(inp_sc)[0][1]
                pred = cc_model.predict(inp_sc)[0]

                st.plotly_chart(fraud_gauge(prob, "Fraud Probability"), use_container_width=True)

                if pred == 1:
                    st.markdown(f"""
                    <div class="result-fraud">
                        <div class="result-title">🚨 FRAUD DETECTED</div>
                        <div class="result-sub">
                            Risk Score: <strong>{prob*100:.1f}%</strong><br><br>
                            This transaction shows patterns consistent with fraudulent activity.<br>
                            Recommend: <strong>Block & verify with cardholder</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-safe">
                        <div class="result-title">✅ TRANSACTION SAFE</div>
                        <div class="result-sub">
                            Risk Score: <strong>{prob*100:.1f}%</strong><br><br>
                            Transaction patterns appear normal.<br>
                            Recommend: <strong>Approve</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('<p class="section-title">Risk Breakdown</p>', unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Amount Risk", "HIGH" if amount > 3000 else "LOW",
                          delta="↑" if amount > 3000 else "↓", delta_color="inverse")
                c2.metric("Time Risk", "HIGH" if hour in range(0, 5) else "LOW",
                          delta="↑" if hour in range(0, 5) else "↓", delta_color="inverse")
                c3.metric("Velocity Risk", "HIGH" if cc_velocity > 6 else "LOW",
                          delta="↑" if cc_velocity > 6 else "↓", delta_color="inverse")
            else:
                st.info("👈 Enter transaction details and click **Check for Fraud**")
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown("""<div class="stat-card">
                        <div class="stat-label">Training Records</div>
                        <div class="stat-value">10K</div>
                        <div class="stat-sub">transactions</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown("""<div class="stat-card">
                        <div class="stat-label">Fraud Rate</div>
                        <div class="stat-value">1.5%</div>
                        <div class="stat-sub">realistic imbalance</div>
                    </div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown("""<div class="stat-card">
                        <div class="stat-label">Model</div>
                        <div class="stat-value">RF + SMOTE</div>
                        <div class="stat-sub">Random Forest</div>
                    </div>""", unsafe_allow_html=True)

    # ── TAB 2: UPI ────────────────────────────────────────────────────────────
    with tab2:
        col_in2, col_out2 = st.columns([1, 1.3])

        with col_in2:
            st.markdown('<p class="section-title">UPI Transaction Details</p>', unsafe_allow_html=True)
            upi_amount  = st.number_input("Transaction Amount (₹)", min_value=1.0, max_value=200000.0, value=500.0, step=100.0, key="upi_amt")
            upi_hour    = st.slider("Hour of Day", 0, 23, 12, key="upi_hr")
            upi_dow     = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            upi_dow_val = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(upi_dow)

            upi_cat     = st.selectbox("Merchant Category", ["Food", "Shopping", "Recharge", "Utilities", "Transfer", "Entertainment"], key="u_cat")
            upi_cat_val = ["Food", "Shopping", "Recharge", "Utilities", "Transfer", "Entertainment"].index(upi_cat)

            upi_device  = st.selectbox("Device Type", ["Android", "iOS", "Web"])
            upi_dev_val = ["Android", "iOS", "Web"].index(upi_device)
            
            loc_match   = st.selectbox("Location Match", ["Yes (same city)", "No (different city)"])
            loc_val     = 1 if "Yes" in loc_match else 0

            txn_ph      = st.slider("Transactions in Last Hour", 0, 20, 1, key="txn_ph")
            new_benef   = st.selectbox("New Beneficiary?", ["No", "Yes"])
            new_val     = 1 if new_benef == "Yes" else 0

            upi_btn     = st.button("🔍 Detect Anomaly", use_container_width=True, type="primary")

        with col_out2:
            if upi_btn:
                # amount, hour, day_of_week, merchant_cat, device_type, location_match, txn_per_hour, new_beneficiary
                upi_inp    = np.array([[upi_amount, upi_hour, upi_dow_val, upi_cat_val,
                                        upi_dev_val, loc_val, txn_ph, new_val]])
                upi_inp_sc = upi_scaler.transform(upi_inp)
                upi_prob   = upi_model.predict_proba(upi_inp_sc)[0][1]
                upi_pred   = upi_model.predict(upi_inp_sc)[0]

                st.plotly_chart(fraud_gauge(upi_prob, "Anomaly Probability"), use_container_width=True)

                if upi_pred == 1:
                    st.markdown(f"""
                    <div class="result-fraud">
                        <div class="result-title">⚠️ ANOMALY DETECTED</div>
                        <div class="result-sub">
                            Risk Score: <strong>{upi_prob*100:.1f}%</strong><br><br>
                            This UPI transaction shows unusual patterns.<br>
                            Recommend: <strong>Hold & send OTP confirmation</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-safe">
                        <div class="result-title">✅ TRANSACTION NORMAL</div>
                        <div class="result-sub">
                            Risk Score: <strong>{upi_prob*100:.1f}%</strong><br><br>
                            No anomalous patterns detected in this UPI transaction.<br>
                            Recommend: <strong>Approve</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('<p class="section-title">Risk Signals</p>', unsafe_allow_html=True)
                signals = {
                    "High Amount": upi_amount > 10000,
                    "Late Night":  upi_hour in range(0, 5),
                    "No Location Match": loc_val == 0,
                    "High Velocity":     txn_ph > 5,
                    "New Beneficiary":   new_val == 1,
                }
                c1, c2 = st.columns(2)
                for i, (signal, triggered) in enumerate(signals.items()):
                    badge = f'<span class="risk-badge-high">🔴 {signal}</span>' if triggered else \
                            f'<span class="risk-badge-low">🟢 {signal}: OK</span>'
                    (c1 if i % 2 == 0 else c2).markdown(badge, unsafe_allow_html=True)
                    (c1 if i % 2 == 0 else c2).markdown("")
            else:
                st.info("👈 Enter UPI transaction details and click **Detect Anomaly**")
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown("""<div class="stat-card">
                        <div class="stat-label">UPI Records</div>
                        <div class="stat-value">8K</div>
                        <div class="stat-sub">transactions</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown("""<div class="stat-card">
                        <div class="stat-label">Anomaly Rate</div>
                        <div class="stat-value">2.5%</div>
                        <div class="stat-sub">detected cases</div>
                    </div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown("""<div class="stat-card">
                        <div class="stat-label">Model</div>
                        <div class="stat-value">RF + SMOTE</div>
                        <div class="stat-sub">Random Forest</div>
                    </div>""", unsafe_allow_html=True)

    # ── TAB 3: Dataset Insights ───────────────────────────────────────────────
    with tab3:
        st.markdown('<p class="section-title">Credit Card — Fraud by Hour</p>', unsafe_allow_html=True)
        fraud_hr  = cc_df[cc_df['class']==1].groupby('hour').size().reset_index(name='count')
        normal_hr = cc_df[cc_df['class']==0].groupby('hour').size().reset_index(name='count')

        fig_hr = go.Figure()
        fig_hr.add_trace(go.Bar(x=normal_hr['hour'], y=normal_hr['count'] / 60,
                                name='Normal (scaled)', marker_color='rgba(0,212,255,0.6)'))
        fig_hr.add_trace(go.Bar(x=fraud_hr['hour'],  y=fraud_hr['count'],
                                name='Fraud', marker_color='rgba(231,76,60,0.9)'))
        fig_hr.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                             font={'color': 'white'}, barmode='overlay',
                             xaxis_title='Hour of Day', yaxis_title='Count',
                             legend=dict(bgcolor='rgba(0,0,0,0)'), height=320,
                             margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_hr, use_container_width=True)

        st.markdown('<p class="section-title">UPI — Anomaly Signals Distribution</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            loc_data = upi_df.groupby(['location_match', 'is_anomaly']).size().reset_index(name='count')
            fig_loc = px.bar(loc_data, x='location_match', y='count', color='is_anomaly',
                             color_discrete_map={0: 'rgba(0,212,255,0.7)', 1: 'rgba(231,76,60,0.9)'},
                             labels={'location_match': 'Location Match (1=Yes)', 'is_anomaly': 'Type'},
                             title='Anomaly by Location Match')
            fig_loc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  font={'color': 'white'}, showlegend=True, height=280,
                                  margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_loc, use_container_width=True)

        with col2:
            nb_data = upi_df.groupby(['new_beneficiary', 'is_anomaly']).size().reset_index(name='count')
            fig_nb = px.bar(nb_data, x='new_beneficiary', y='count', color='is_anomaly',
                            color_discrete_map={0: 'rgba(0,212,255,0.7)', 1: 'rgba(231,76,60,0.9)'},
                            labels={'new_beneficiary': 'New Beneficiary (1=Yes)', 'is_anomaly': 'Type'},
                            title='Anomaly by New Beneficiary')
            fig_nb.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                 font={'color': 'white'}, showlegend=True, height=280,
                                 margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_nb, use_container_width=True)

if __name__ == "__main__":
    main()
