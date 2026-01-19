from src.data_loader import CMAPSSDataLoader
from src.model import EngineHealthModel
from src.xai_engine import XAIExplainer
import pandas as pd
import numpy as np

# Mimic app.py logic
print("Loading data...")
loader = CMAPSSDataLoader()
train_df, test_df = loader.process_data('FD001')

print("Loading model...")
model = EngineHealthModel()
# Load the actual saved model if possible, or retrain to match state
try:
    model.load_model('model.pkl')
    print("Loaded saved model.")
except:
    print("Retraining model...")
    model.train(train_df)

# Engine 1
engine_id = 1
engine_data = test_df[test_df['unit_number'] == engine_id]
current_cycle = engine_data['time_in_cycles'].max()
current_data = engine_data[engine_data['time_in_cycles'] == current_cycle]

print(f"Engine {engine_id}, Cycle {current_cycle}")
print("Input Data:\n", current_data)

# Prepare features
features = model._prepare_features(current_data)
print("Features:\n", features)

if features.isnull().values.any():
    print("NaNs in Prepared Features!")

# XAI
print("Creating Explainer...")
X_train_sample = model._prepare_features(train_df.sample(100), fit_scaler=False)

# Check scaler stats
print("Scaler feature range:", model.scaler.feature_range)
# print("Scaler data min:", model.scaler.data_min_)
# print("Scaler data max:", model.scaler.data_max_)

explainer = XAIExplainer(
    model=model.abarth_model, 
    train_data_sample=X_train_sample,
    feature_names=model.feature_columns
)

print("Running LIME...")
try:
    input_data_row = features.iloc[0].values
    print("Input row:", input_data_row)
    lime_exp = explainer.explain_instance_lime(input_data_row)
    print("LIME Success")
except Exception as e:
    print(f"LIME Failed: {e}")
    import traceback
    traceback.print_exc()
