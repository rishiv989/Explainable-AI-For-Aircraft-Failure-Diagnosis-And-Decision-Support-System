import streamlit as st
import os
import numpy as np
import pandas as pd
import joblib
import shap
import json
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# --- CONFIG & CACHING ---
st.set_page_config(layout="wide", page_title="AI Maintenance Dashboard")

@st.cache_resource
def load_models():
    # Try to load deployment ensemble (CPU-optimized)
    models = []
    try:
        # Check deployment filenames
        for i in range(5):
            fname = f"deploy_xgb_ensemble_{i}.pkl"
            if os.path.exists(fname):
                models.append(joblib.load(fname))
            else:
                raise FileNotFoundError
        return models, "Ensemble (5 Models - CPU)"
    except:
        # Fallback to local full models (GPU or whatever exists)
        try:
            for i in range(5):
                 models.append(joblib.load(f"xgb_ensemble_{i}.pkl"))
            return models, "Ensemble (Local)"
        except:
            # Fallback to single deployment model
            if os.path.exists("deployment_model.pkl"):
                 return [joblib.load("deployment_model.pkl")], "Single Model (Dep)"
            else:
                 return [joblib.load("xgb_rul_model.pkl")], "Single Model (Local)"

@st.cache_data
def load_val_data():
    # Try loading small deployment sample first
    if os.path.exists("X_deploy.npy"):
        X_val = np.load("X_deploy.npy")
        y_val = np.load("y_deploy.npy")
        return X_val, y_val
    else:
        # Load local full dataset
        X_val = np.load("X_val.npy")
        y_val = np.load("y_val.npy")
        return X_val, y_val

@st.cache_resource
def get_explainer(model):
    return shap.TreeExplainer(model)

# --- HELPER FUNCTIONS ---
def predict_with_uncertainty(models, X):
    # X shape: (1, 960)
    preds = [m.predict(X)[0] for m in models]
    mean_rul = np.mean(preds)
    std_rul = np.std(preds)
    return mean_rul, std_rul

def get_risk_color(rul):
    if rul > 100: return "green"
    elif rul > 30: return "orange"
    else: return "red"

def map_sensor_names():
    names = []
    names.extend([f"W{i+1}" for i in range(4)])
    names.extend([f"Xs{i+1}" for i in range(14)])
    names.extend([f"Xv{i+1}" for i in range(14)])
    return names

# --- MAIN DASHBOARD ---
def main():
    st.title("✈️ AI Predictive Maintenance Dashboard")
    st.markdown("### Real-time Engine Health Monitoring & Decision Support")

    # Load resources
    with st.spinner("Loading AI Models..."):
        models, model_type = load_models()
        X_val, y_val = load_val_data()
        flat_size = 960 # 30*32
        
    # Sidebar: Engine Selection
    st.sidebar.header("Engine Selection")
    # Simulate Engine IDs from validation set indices
    engine_idx = st.sidebar.number_input("Select Engine Unit Index (0-20000)", min_value=0, max_value=len(X_val)-1, value=0)
    
    # Get Data for Selected Engine
    current_X = X_val[engine_idx].reshape(1, flat_size)
    current_X_3d = X_val[engine_idx] # (30, 32)
    true_rul = y_val[engine_idx]
    
    # 1. Prediction & Uncertainty
    pred_rul, unc = predict_with_uncertainty(models, current_X)
    risk_color = get_risk_color(pred_rul)
    
    # Cast to float for formatting safety
    pred_val = float(pred_rul)
    unc_val = float(unc)
    true_val = float(true_rul)
    
    # Top Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted RUL", f"{pred_val:.1f} cycles", delta=f"{unc_val:.1f} uncertainty")
    c2.metric("True RUL (Ground Truth)", f"{true_val:.1f} cycles")
    c3.markdown(f"**Risk Level**")
    c3.markdown(f"<h2 style='color: {risk_color}'>{'Low' if pred_val>100 else 'Medium' if pred_val>30 else 'High'}</h2>", unsafe_allow_html=True)
    c4.metric("Model Type", model_type)
    
    st.divider()

    # 2. Main Visualizations
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("Time-Series Sensor View")
        # Plot a few key sensors for the 30-cycle window
        # Sensors: 4-17 (Xs), 18-31 (Xv)
        # Let's plot Xs2, Xs3, Xs4, Xs7 (typical jet engine sensors)
        # Indicies: Xs1 is col 4. So Xs2=5, Xs3=6, Xs4=7.
        sensor_data = pd.DataFrame(current_X_3d, columns=map_sensor_names())
        
        sensors_to_plot = st.multiselect("Select Sensors to Monitor", sensor_data.columns, default=["Xs2", "Xs3", "Xs4", "Xs7"])
        
        fig = px.line(sensor_data, y=sensors_to_plot, title="Sensor Readings over Last 30 Cycles")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Global Health Index")
        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = pred_rul,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "RUL (Cycles)"},
            gauge = {
                'axis': {'range': [None, 130]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 30], 'color': "Salmon"},
                    {'range': [30, 100], 'color': "LightYellow"},
                    {'range': [100, 130], 'color': "LightGreen"}],
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Action Recommendation
        if pred_rul < 30:
            st.error("🚨 ACTION REQUIRED: Immediate Inspection")
        elif pred_rul < 100:
            st.warning("⚠️ PLAN MAINTENANCE: Next Window")
        else:
            st.success("✅ MONITOR: System Healthy")

    st.divider()
    
    # 3. Explainability (XAI) Tab
    st.subheader("🔍 Explainable AI Diagnostics")
    tabs = st.tabs(["Global Feature Importance", "Local Explanation (This Engine)", "Sensitivity Analysis"])
    
    with tabs[0]:
        st.image("global_sensor_importance.png", caption="Global Sensor Importance (Phase 3 Aggregation)")
        st.image("shap_summary_plot.png", caption="Detailed SHAP Summary")

    with tabs[1]:
        st.write("Why did the model predict this RUL?")
        if st.button("Compute SHAP for this Engine"):
            explainer = get_explainer(models[0])
            shap_vals = explainer.shap_values(current_X)
            
            # Waterfall Plot (using matplotlib in streamlit)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            fig_shap, ax = plt.subplots()
            # We need feature names for 960 features... messy.
            # Let's aggregate by sensor for the plot
            shap_3d = shap_vals[0].reshape(30, 32)
            shap_sensor = np.sum(shap_3d, axis=0) # Sum over time? Or max?
            
            # Simple bar chart of top contributors for this instance
            top_idx = np.argsort(np.abs(shap_sensor))[::-1][:10]
            top_names = [map_sensor_names()[i] for i in top_idx]
            top_vals = shap_sensor[top_idx]
            
            fig_bar = px.bar(x=top_vals, y=top_names, orientation='h', title="Top Contributing Sensors (Aggregated over window)")
            st.plotly_chart(fig_bar)

    with tabs[2]:
        st.write("### What-If Analysis")
        st.write("Adjust sensor values to see impact on RUL prediction.")
        
        # Select a sensor to perturb
        target_sensor = st.selectbox("Select Sensor", map_sensor_names())
        delta = st.slider(f"Adjust {target_sensor} (%)", -10, 10, 0)
        
        if delta != 0:
            # Perturb data
            # Find index of sensor
            s_idx = map_sensor_names().index(target_sensor)
            
            # Add delta% to all time steps for this sensor
            X_mod = current_X_3d.copy()
            # X_mod is normalized. 1 unit ~ 1 std dev.
            # Let's say 10% = 0.1 sigma shift? Or 10% of value?
            # Assuming normalized data roughly -3 to 3.
            # Let's apply shift = delta * 0.1 (so 10% -> 1.0 sigma)
            shift = delta * 0.1
            X_mod[:, s_idx] += shift
            
            X_mod_flat = X_mod.reshape(1, flat_size)
            new_pred, _ = predict_with_uncertainty(models, X_mod_flat)
            
            diff = new_pred - pred_rul
            st.metric("New Predicted RUL", f"{new_pred:.1f}", delta=f"{diff:.1f}")
            
            if new_pred < pred_rul:
                st.write(f"📉 Degrading {target_sensor} reduces RUL by {abs(diff):.1f} cycles.")
            else:
                st.write(f"📈 Improving {target_sensor} increases RUL by {abs(diff):.1f} cycles.")

if __name__ == "__main__":
    main()
