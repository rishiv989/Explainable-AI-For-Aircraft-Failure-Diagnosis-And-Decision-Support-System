from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
from src.data_loader import CMAPSSDataLoader
from src.model import EngineHealthModel
from src.xai_engine import XAIExplainer

app = FastAPI(title="XAI Engine Monitor API")

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
model = None
explainer = None
train_df = None
test_df = None

class PredictionRequest(BaseModel):
    engine_id: int

@app.on_event("startup")
def load_assets():
    global model, explainer, train_df, test_df
    print("Loading system assets...")
    
    loader = CMAPSSDataLoader()
    # We load data once
    train_df, test_df = loader.process_data('FD001')
    
    model = EngineHealthModel()
    if os.path.exists('model.pkl'):
        model.load_model('model.pkl')
    else:
        print("Training model...")
        model.train(train_df)
        model.save_model('model.pkl')
        
    # Setup XAI
    X_train_sample = model._prepare_features(train_df.sample(100), fit_scaler=False)
    explainer = XAIExplainer(
        model=model.abarth_model,
        train_data_sample=X_train_sample,
        feature_names=model.feature_columns
    )
    print("System ready.")

@app.get("/")
def health_check():
    return {"status": "active", "model_loaded": model is not None}

@app.get("/engines")
def get_engines():
    # Return list of engine IDs in test set
    if test_df is None:
        raise HTTPException(status_code=503, detail="System initializing")
    ids = test_df['unit_number'].unique().tolist()
    return {"engine_ids": ids}

@app.get("/engine/{engine_id}")
def get_engine_data(engine_id: int):
    if test_df is None:
        raise HTTPException(status_code=503, detail="System initializing")
        
    engine_data = test_df[test_df['unit_number'] == engine_id]
    if engine_data.empty:
        raise HTTPException(status_code=404, detail="Engine not found")
        
    # Get just the last cycle for "current" status
    current_cycle = engine_data['time_in_cycles'].max()
    current_row = engine_data[engine_data['time_in_cycles'] == current_cycle]
    
    # Return full history for plotting
    history = engine_data[['time_in_cycles', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_11', 'sensor_12', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']].to_dict(orient='records')
    
    return {
        "engine_id": engine_id,
        "current_cycle": int(current_cycle),
        "true_rul": float(current_row['RUL'].values[0]),
        "history": history
    }


# Sensor mapping based on NASA CMAPSS documentation
SENSOR_MAP = {
    'sensor_1': 'Fan Inlet Temp (T2)',
    'sensor_2': 'LPC Outlet Temp (T24)',
    'sensor_3': 'HPC Outlet Temp (T30)',
    'sensor_4': 'LPT Outlet Temp (T50)',
    'sensor_5': 'Fan Inlet Pressure (P2)',
    'sensor_6': 'Bypass Duct Press (P21)',
    'sensor_7': 'LPC Outlet Press (P24)',
    'sensor_8': 'HPC Outlet Press (Ps30)',
    'sensor_9': 'HPC Outlet Press (P40)',
    'sensor_10': 'LPT Outlet Press (P50)',
    'sensor_11': 'Fuel/Air Ratio (Phi)',
    'sensor_12': 'Fan Speed (Nf)',
    'sensor_13': 'Core Speed (Nc)',
    'sensor_14': 'Bypass Ratio (BPR)',
    'sensor_15': 'Burner Fuel/Air Ratio',
    'sensor_16': 'Relite Pressure',
    'sensor_17': 'Bleed Enthalpy',
    'sensor_18': 'Demanded Fan Speed',
    'sensor_19': 'Demanded Fan Ratio',
    'sensor_20': 'HPT Coolant Bleed',
    'sensor_21': 'LPT Coolant Bleed'
}

MAINTENANCE_ACTIONS = {
    'sensor_2': 'Inspect Low Pressure Compressor (LPC) stator vanes.',
    'sensor_3': 'Check High Pressure Compressor (HPC) for thermal degradation.',
    'sensor_4': 'Inspect Low Pressure Turbine (LPT) blades for creep or corrosion.',
    'sensor_8': 'Verify Static Pressure ports in HPC discharge.',
    'sensor_11': 'Calibrate Fuel Flow controller and check nozzles.',
    'sensor_14': 'Check Bypass Duct seals for leakage.',
    'sensor_17': 'Inspect Bleed Air valves for stuck-open condition.',
}

class SimulationRequest(BaseModel):
    engine_id: int
    adjustments: dict  # e.g., {"sensor_11": 1.05} (5% increase)

@app.get("/system/schema")
def get_schema():
    return {
        "sensors": SENSOR_MAP,
        "actions": MAINTENANCE_ACTIONS
    }

@app.post("/predict_simulated")
def predict_simulated(req: SimulationRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="System initializing")
        
    engine_data = test_df[test_df['unit_number'] == req.engine_id]
    current_cycle = engine_data['time_in_cycles'].max()
    current_row = engine_data[engine_data['time_in_cycles'] == current_cycle].copy()
    
    # Apply adjustments
    for sensor, factor in req.adjustments.items():
        if sensor in current_row.columns:
            current_row[sensor] = current_row[sensor] * factor
            
    # Predict
    pred_rul, pred_state, pred_conf = model.predict(current_row)
    
    return {
        "original_rul": float(model.predict(engine_data[engine_data['time_in_cycles'] == current_cycle])[0][0]),
        "simulated_rul": float(pred_rul[0]),
        "simulated_state": pred_state[0],
        "delta": float(pred_rul[0]) - float(model.predict(engine_data[engine_data['time_in_cycles'] == current_cycle])[0][0])
    }

@app.post("/predict_explain")
def predict_and_explain(req: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="System initializing")
        
    engine_id = req.engine_id
    engine_data = test_df[test_df['unit_number'] == engine_id]
    current_cycle = engine_data['time_in_cycles'].max()
    current_row = engine_data[engine_data['time_in_cycles'] == current_cycle]
    
    if current_row.empty:
        raise HTTPException(status_code=404, detail="Engine data not found")
        
    # Predict
    pred_rul, pred_state, pred_conf = model.predict(current_row)
    
    # Explain
    features = model._prepare_features(current_row)
    input_row = features.iloc[0].values
    
    # LIME
    lime_exp = explainer.explain_instance_lime(input_row)
    lime_list = lime_exp.as_list() # [('feature < X', weight), ...]
    
    # Generate Maintenance Advice
    top_feature_code = lime_list[0][0].split(' ')[0] # Extract 'sensor_11' from 'sensor_11 <= 0.5'
    advice = MAINTENANCE_ACTIONS.get(top_feature_code, "Perform general system diagnostics.")
    readable_feature = SENSOR_MAP.get(top_feature_code, top_feature_code)

    return {
        "prediction": {
            "rul": float(pred_rul[0]),
            "state": pred_state[0],
            "confidence": float(pred_conf[0])
        },
        "explanation": {
            "lime_features": [{"feature": SENSOR_MAP.get(x[0].split(' ')[0], x[0]), "weight": x[1], "raw_feature": x[0]} for x in lime_list],
            "summary": f"Primary Deviant: {readable_feature}. {advice}",
            "raw_top_feature": top_feature_code
        }
    }
