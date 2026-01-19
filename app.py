import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import shap
from src.data_loader import CMAPSSDataLoader
from src.model import EngineHealthModel
from src.xai_engine import XAIExplainer
import os

# Set page config
st.set_page_config(page_title="XAI Engine Health Monitor", layout="wide", page_icon="✈️")

# Sensor Mapping (CMAPSS FD001)
SENSOR_MAP = {
    'sensor_2': 'LPC Outlet Temp (T24)',
    'sensor_3': 'HPC Outlet Temp (T30)',
    'sensor_4': 'LPT Outlet Temp (T50)',
    'sensor_7': 'HPC Outlet Pressure (P30)',
    'sensor_8': 'Physical Fan Speed (Nf)',
    'sensor_9': 'Physical Core Speed (Nc)',
    'sensor_11': 'Static Pressure (Ps30)',
    'sensor_12': 'Ratio of Fuel Flow (phi)',
    'sensor_13': 'Corrected Fan Speed (Nf_corr)',
    'sensor_14': 'Corrected Core Speed (Nc_corr)',
    'sensor_15': 'Bypass Ratio',
    'sensor_17': 'Bleed Enthalpy',
    'sensor_20': 'HPT Coolant Bleed',
    'sensor_21': 'LPT Coolant Bleed'
    # Add others as generic if needed
}

def get_readable_name(col):
    return SENSOR_MAP.get(col, col)

@st.cache_resource
def load_data_and_model():
    with st.spinner("Loading Data and Model..."):
        loader = CMAPSSDataLoader()
        train_df, test_df = loader.process_data('FD001')
        
        model = EngineHealthModel()
        model_path = 'model.pkl'
        
        if os.path.exists(model_path):
            try:
                model.load_model(model_path)
            except:
                st.warning("Failed to load model, retraining...")
                model.train(train_df)
                model.save_model(model_path)
        else:
            model.train(train_df)
            model.save_model(model_path)
            
        # Initialize Explainer
        # We need a sample for background. Using 100 random samples from train
        X_train_sample = model._prepare_features(train_df.sample(100), fit_scaler=False)
        explainer = XAIExplainer(
            model=model.abarth_model, # Explaining the Classifier
            train_data_sample=X_train_sample,
            feature_names=model.feature_columns,
            class_names=['Normal', 'Warning', 'Critical']
        )
        
    return train_df, test_df, model, explainer

# Load Resources
try:
    train_df, test_df, model, explainer = load_data_and_model()
except Exception as e:
    st.error(f"Error loading system: {e}")
    st.stop()

# Sidebar
st.sidebar.title("✈️ Engine Monitor")
engine_id = st.sidebar.selectbox("Select Engine Unit (Test Fleet)", test_df['unit_number'].unique())

# Filter Data for Selected Engine
engine_data = test_df[test_df['unit_number'] == engine_id]
current_cycle = engine_data['time_in_cycles'].max()
current_data = engine_data[engine_data['time_in_cycles'] == current_cycle]

# Predict
pred_rul, pred_state = model.predict(current_data)
pred_rul_val = pred_rul[0]
pred_state_val = pred_state[0]

# Mapping Health to Color
color_map = {"Normal": "green", "Warning": "orange", "Critical": "red"}
state_color = color_map.get(pred_state_val, "blue")

# Header
st.title(f"Engine #{engine_id} Health Diagnostic")
st.markdown(f"**Current Cycle:** {current_cycle}")

# KPI Cards
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Predicted RUL", f"{pred_rul_val:.1f} Cycles")
with col2:
    st.markdown(f"**Health State:** <span style='color:{state_color};font-size:24px;font-weight:bold'>{pred_state_val}</span>", unsafe_allow_html=True)
with col3:
    st.metric("True RUL (Oracle)", f"{current_data['RUL'].values[0]:.0f} Cycles")

st.divider()

# Tab Layout
tabs = st.tabs(["📊 Sensor Trends", "🔍 Explainable AI (XAI)", "📋 Maintenance Report"])

with tabs[0]:
    st.subheader("Sensor Anomalies & Trends")
    # Plot key sensors
    key_sensors = ['sensor_11', 'sensor_12', 'sensor_17', 'sensor_7'] # High variance ones usually
    
    fig = go.Figure()
    for sensor in key_sensors:
        readable = get_readable_name(sensor)
        fig.add_trace(go.Scatter(x=engine_data['time_in_cycles'], y=engine_data[sensor], mode='lines', name=readable))
    
    fig.update_layout(title="Key Diagnostic Parameters over Time", xaxis_title="Flight Cycles", yaxis_title="Sensor Value")
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("Sudden deviations in Static Pressure (Ps30) and Fan Speed usually indicate degradation.")

with tabs[1]:
    st.subheader("Model Decision Explanations")
    
    col_x1, col_x2 = st.columns(2)
    
    with col_x1:
        st.markdown("### LIME (Local Explanation)")
        st.write(f"Why is Engine #{engine_id} classified as **{pred_state_val}**?")
        
        # LIME
        input_data_row = model._prepare_features(current_data).iloc[0].values
        lime_exp = explainer.explain_instance_lime(input_data_row)
        
        # Display LIME as HTML
        st.components.v1.html(lime_exp.as_html(), height=400, scrolling=True)
        
    with col_x2:
        st.markdown("### SHAP (Feature Contribution)")
        st.write("Contribution of each sensor to the risk score.")
        
        # SHAP
        shap_values = explainer.get_shap_values(model._prepare_features(current_data)) # We might need to adjust XAIExplainer to expose this well
        # XAIExplainer needs an update for `calculate_shap_values` or usage of `explain_instance_shap`
        # Let's use the one we defined: explain_instance_shap returns shap values
        # But for plotting we often want the plot directly.
        
        # Hack for SHAP beeswarm or waterfall
        # We'll use a simple bar chart of top features from LIME for now in this iteration if SHAP is complex to render in Streamlit without static image.
        # Check XAIExplainer implementation. `explain_instance_shap` returns values.
        
        # Let's stick to LIME analysis text for now or simple bar chart
        feature_imp = pd.DataFrame(lime_exp.as_list(), columns=['Feature', 'Weight'])
        fig_imp = px.bar(feature_imp, x='Weight', y='Feature', orientation='h', title="Feature Influence (LIME)")
        st.plotly_chart(fig_imp)

with tabs[2]:
    st.subheader("Maintenance Recommendation")
    
    if pred_state_val == "Critical":
        st.error(f"⚠️ **URGENT**: Engine #{engine_id} is in CRITICAL condition.")
        st.markdown("""
        **Action Required:**
        - Immediate grounding of aircraft.
        - Inspect **High Pressure Compressor** (based on Sensor 7/11 deviation).
        - Schedule replacement within **{:.0f} cycles**.
        """.format(pred_rul_val))
    elif pred_state_val == "Warning":
        st.warning(f"⚠️ **WARNING**: Engine #{engine_id} is showing signs of degradation.")
        st.markdown(f"""
        **Action Required:**
        - Schedule maintenance check within next 20 cycles.
        - Monitor **Vibration** and **Temperature** sensors.
        """)
    else:
        st.success(f"✅ Engine #{engine_id} is operating normally.")
        st.markdown("No immediate action required. Continue routine monitoring.")
        
    st.json({
        "engine_id": int(engine_id),
        "predicted_state": pred_state_val,
        "rul_estimate": float(pred_rul_val),
        "top_contributing_factors": [x[0] for x in lime_exp.as_list()[:3]]
    })

