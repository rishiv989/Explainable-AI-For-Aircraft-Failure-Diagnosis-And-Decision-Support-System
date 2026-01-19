from src.data_loader import CMAPSSDataLoader
from src.model import EngineHealthModel
import os

def train():
    print("Initializing Data Loader...")
    loader = CMAPSSDataLoader()
    # Process FD001
    train_df, test_df = loader.process_data('FD001')
    
    print("Initializing Model...")
    model = EngineHealthModel()
    
    print("Training Model with NEW Thresholds (>75 Normal)...")
    model.train(train_df)
    
    print("Saving Model...")
    model.save_model('model.pkl')
    print("Done.")

if __name__ == "__main__":
    train()
