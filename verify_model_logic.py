import pandas as pd
import numpy as np
import os
from src.data_loader import CMAPSSDataLoader
from src.model import EngineHealthModel
from src.xai_engine import XAIExplainer

def verify():
    print("=== Starting Model Logic Verification ===")
    
    # 1. Load Data
    print("[1] Loading Data...")
    loader = CMAPSSDataLoader()
    _, test_df = loader.process_data('FD001')
    
    # 2. Load Model
    print("[2] Loading Model...")
    model = EngineHealthModel()
    if not os.path.exists('model.pkl'):
        print("ERROR: model.pkl not found. Please train first.")
        return
    model.load_model('model.pkl')
    
    # 3. Test Predictions on Multiple Engines
    print("[3] Testing Predictions on 10 Random Engines...")
    unit_ids = test_df['unit_number'].unique()
    sample_ids = np.random.choice(unit_ids, 10, replace=False)
    
    results = []
    
    for uid in sample_ids:
        engine_data = test_df[test_df['unit_number'] == uid]
        current_cycle = engine_data['time_in_cycles'].max()
        current_row = engine_data[engine_data['time_in_cycles'] == current_cycle]
        
        pred_rul, pred_state, pred_conf = model.predict(current_row)
        
        results.append({
            'unit': uid,
            'rul': pred_rul[0],
            'state': pred_state[0],
            'conf': pred_conf[0]
        })
        
    df_res = pd.DataFrame(results)
    print("\nPrediction Results Sample:")
    print(df_res)
    
    # 4. Verify RUL Variance
    rul_std = df_res['rul'].std()
    print(f"\nRUL Standard Deviation: {rul_std:.2f}")
    if rul_std < 5:
        print("WARNING: RUL seems static or low variance!")
    else:
        print("SUCCESS: RUL varies significantly.")

    # 5. Verify State logic with new thresholds
    # Normal > 75, Warning 30-75, Critical < 30
    print("\n[5] Verifying State Logic...")
    logic_errors = 0
    for _, row in df_res.iterrows():
        rul = row['rul']
        state = row['state']
        
        expected = "Unknown"
        if rul > 75: expected = "Normal"
        elif rul >= 30: expected = "Warning"
        else: expected = "Critical"
        
        if state != expected:
            print(f"ERROR: Unit {row['unit']} RUL {rul:.1f} got {state}, expected {expected}")
            logic_errors += 1
            
    if logic_errors == 0:
        print("SUCCESS: All states match the logical thresholds.")
    else:
        print(f"FAILURE: {logic_errors} state logic errors found.")

    # 6. Verify Confidence Scores
    print("\n[6] Verifying Confidence Scores...")
    conf_std = df_res['conf'].std()
    print(f"Confidence Standard Deviation: {conf_std:.4f}")
    if conf_std == 0:
        print("WARNING: Confidence score is identical for all units!")
    else:
        print("SUCCESS: Confidence scores are dynamic.")

    # 7. Verify XAI - LIME
    print("\n[7] Verifying XAI (LIME)...")
    # Setup explainer
    X_train_sample = model._prepare_features(test_df.sample(50), fit_scaler=False)
    explainer = XAIExplainer(
        model=model.abarth_model,
        train_data_sample=X_train_sample,
        feature_names=model.feature_columns
    )
    
    # Explain two different units and ensure different features
    u1 = sample_ids[0]
    u2 = sample_ids[1]
    
    row1 = test_df[(test_df['unit_number'] == u1) & (test_df['time_in_cycles'] == test_df[test_df['unit_number'] == u1].time_in_cycles.max())]
    row2 = test_df[(test_df['unit_number'] == u2) & (test_df['time_in_cycles'] == test_df[test_df['unit_number'] == u2].time_in_cycles.max())]
    
    exp1 = explainer.explain_instance_lime(model._prepare_features(row1).iloc[0].values)
    exp2 = explainer.explain_instance_lime(model._prepare_features(row2).iloc[0].values)
    
    top1 = exp1.as_list()[0][0]
    top2 = exp2.as_list()[0][0]
    
    print(f"Unit {u1} Top Feature: {top1}")
    print(f"Unit {u2} Top Feature: {top2}")
    
    if top1 == top2:
        print("NOTE: Top feature is identical. This is possible but check if frequently identical.")
    else:
        print("SUCCESS: XAI explanations vary between units.")

    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    verify()
