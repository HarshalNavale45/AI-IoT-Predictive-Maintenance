import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="AI IoT Predictive Maintenance Hub", layout="wide")

# Custom UI Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 10px; }
    .recommendation-box { background-color: #1c2128; border-left: 5px solid #238636; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .alert-box { background-color: #1c2128; border-left: 5px solid #da3633; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .fleet-card { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# 1. SESSION STATE INITIALIZATION (The Simulation Brain)
# ------------------------------------------------------------------
if 'temp' not in st.session_state: st.session_state.temp = 65.0
if 'vibration' not in st.session_state: st.session_state.vibration = 0.05
if 'pressure' not in st.session_state: st.session_state.pressure = 100.0

def reset_simulation():
    st.session_state.temp = 65.0
    st.session_state.vibration = 0.05
    st.session_state.pressure = 100.0

# ------------------------------------------------------------------
# 2. LOAD ASSETS
# ------------------------------------------------------------------
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('models/model.pkl')
        features = joblib.load('models/features.pkl')
        return model, features
    except:
        return None, None

model, feature_list = load_assets()

# ------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# ------------------------------------------------------------------
st.sidebar.title("🛡️ Maintenance AI")
st.sidebar.markdown("### Simulation Engine")
st.sidebar.markdown("Control the live feed for the selected asset.")

temp = st.sidebar.slider("Temperature (°C)", 40.0, 120.0, key="temp")
vibration = st.sidebar.slider("Vibration (mm/s)", 0.0, 1.0, step=0.01, key="vibration")
pressure = st.sidebar.slider("Pressure (PSI)", 50, 150, key="pressure")

if st.sidebar.button("🛠️ Perform Emergency Repair", on_click=reset_simulation):
    st.sidebar.success("Asset repaired! Sensors normalized.")

st.sidebar.markdown("---")
machine_id = st.sidebar.selectbox("Active Asset ID", ["M_001", "M_002", "M_003"])

# ------------------------------------------------------------------
# 4. MAIN DASHBOARD TABS
# ------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Live Telemetry", "🌐 Fleet Operations"])

# --- TAB 1: LIVE TELEMETRY ---
with tab1:
    st.title("🛡️ AI IoT Predictive Maintenance Hub")
    if model and feature_list:
        # Prediction Logic
        input_data = pd.DataFrame([[temp, vibration, pressure, temp, 0.5, vibration, 0.01, pressure, 2.0]], columns=feature_list)
        prob = model.predict_proba(input_data)[0][1]
        health_score = 100 - (prob * 100)
        is_fail = prob > 0.5

        # Gauges Row
        g1, g2, g3 = st.columns(3)
        def draw_gauge(val, name, range_min, range_max, yel, red):
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=val, title={'text': name},
                gauge={'axis':{'range':[range_min, range_max]}, 
                       'bar':{'color': "#238636" if val<yel else ("#d29922" if val<red else "#da3633")},
                       'steps':[{'range':[range_min, red], 'color':"rgba(35, 134, 54, 0.1)"}, {'range':[red, range_max], 'color':"rgba(218, 51, 51, 0.1)"}]}
            ))
            fig.update_layout(height=230, margin=dict(l=20,r=20,t=40,b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color':'white'})
            return fig

        g1.plotly_chart(draw_gauge(temp, "Temperature", 40, 120, 80, 95), use_container_width=True)
        g2.plotly_chart(draw_gauge(vibration, "Vibration", 0, 1, 0.3, 0.5), use_container_width=True)
        g3.plotly_chart(draw_gauge(pressure, "Pressure", 50, 150, 100, 130), use_container_width=True)

        # Health & Trend Row
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("🕸️ Component Health Profile")
            cats = ['Thermal', 'Mechanical', 'Pressure', 'Software', 'Risk']
            r_vals = [min(100, (temp-40)/80*100), min(100, vibration*100), min(100, abs(pressure-100)*2), 10, prob*100]
            fig_rad = go.Figure(go.Scatterpolar(r=r_vals, theta=cats, fill='toself', marker=dict(color='#58a6ff')))
            fig_rad.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=350, paper_bgcolor='rgba(0,0,0,0)', font={'color':'white'})
            st.plotly_chart(fig_rad, use_container_width=True)
        
        with c2:
            st.subheader("💡 Maintenance Intel")
            if is_fail:
                st.markdown(f'<div class="alert-box"><h4>⚠️ CRITICAL RISK DETECTED</h4>Probability: {prob*100:.1f}%<br>Action: Immediate inspection required.</div>', unsafe_allow_html=True)
                st.warning(f"Likely Cause: {'High Temp' if temp > 90 else 'Excessive Vibration'}")
            else:
                st.markdown('<div class="recommendation-box"><h4>✅ SYSTEM HEALTHY</h4>Status: All sensors optimal.<br>Next Check: 14 Days.</div>', unsafe_allow_html=True)
            
            # Download Feature
            report_df = pd.DataFrame([[datetime.now(), machine_id, temp, vibration, pressure, f"{health_score:.1f}%"]], 
                                     columns=["Timestamp", "Asset ID", "Temp", "Vib", "Press", "Health"])
            csv = report_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Generate Maintenance Report", csv, f"report_{machine_id}.csv", "text/csv")

# --- TAB 2: FLEET OPERATIONS ---
with tab2:
    st.title("🌐 Regional Fleet Overview")
    st.markdown("Global Status monitoring for all active assets.")
    
    # Simulate data for other machines
    fleet_data = pd.DataFrame({
        'Asset': ['M_001', 'M_002', 'M_003', 'M_004'],
        'Health': [health_score if machine_id == 'M_001' else 98, 
                   health_score if machine_id == 'M_002' else 85,
                   health_score if machine_id == 'M_003' else 92, 
                   72],
        'Status': ['Critical' if (machine_id == 'M_001' and is_fail) else 'Healthy', 
                   'Warning' if (machine_id == 'M_002' and is_fail) else 'Warning',
                   'Healthy' if (machine_id == 'M_003' and is_fail) else 'Healthy', 
                   'Maintenance Needed']
    })

    f_col1, f_col2 = st.columns([2, 1])

    with f_col1:
        st.subheader("📊 Fleet Performance Matrix")
        fig_fleet = px.bar(fleet_data, x='Asset', y='Health', color='Status', 
                           color_discrete_map={'Healthy':'#238636', 'Warning':'#d29922', 'Critical':'#da3633', 'Maintenance Needed':'#8b949e'})
        fig_fleet.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_fleet, use_container_width=True)
    
    with f_col2:
        st.subheader("📟 Alerts Center")
        for idx, row in fleet_data.iterrows():
            st.markdown(f'<div class="fleet-card"><strong>{row["Asset"]}</strong>: {row["Status"]} ({row["Health"]:.0f}%)</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🗺️ Asset Geo-Distribution")
    # Mock map coordinates
    map_data = pd.DataFrame({'lat': [19.07, 28.61, 12.97, 22.57], 'lon': [72.87, 77.20, 77.59, 88.36]})
    st.map(map_data)

    st.sidebar.markdown("---")
    st.sidebar.info("Developed by Harshal Navale | AI IoT Hub")
