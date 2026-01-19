from src.data_loader import CMAPSSDataLoader
import pandas as pd

try:
    loader = CMAPSSDataLoader()
    train_df, test_df = loader.process_data('FD001')

    print("Train NaNs:\n", train_df.isnull().sum()[train_df.isnull().sum() > 0])
    print("Test NaNs:\n", test_df.isnull().sum()[test_df.isnull().sum() > 0])

    # Check columns
    print("Train Columns:", train_df.columns)
    print("Train Shape:", train_df.shape)
    
except Exception as e:
    print(f"Error: {e}")
