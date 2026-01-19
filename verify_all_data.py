from src.data_loader import CMAPSSDataLoader
import pandas as pd

loader = CMAPSSDataLoader()
datasets = ['FD001', 'FD002', 'FD003', 'FD004']

print("Starting Data Loading Verification...")
for ds in datasets:
    try:
        print(f"Testing {ds}...")
        train, test = loader.process_data(ds)
        print(f"  - {ds} Loaded. Train: {train.shape}, Test: {test.shape}")
        
        # Check for NaNs immediately to be sure
        if train.isnull().values.any():
            print(f"  - WARNING: NaNs in {ds} Train")
        if test.isnull().values.any():
            print(f"  - WARNING: NaNs in {ds} Test")
            
    except Exception as e:
        print(f"  - FAILED {ds}: {e}")

print("Data Verification Complete.")
