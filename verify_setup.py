try:
    from src.data_loader import CMAPSSDataLoader
    from src.model import EngineHealthModel
    from src.xai_engine import XAIExplainer
    import pandas as pd
    import numpy as np

    print("Import successful.")

    # 1. Load Data
    loader = CMAPSSDataLoader()
    print("Loading data...")
    train_df, test_df = loader.process_data('FD001')
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # 2. Train Model
    model = EngineHealthModel()
    print("Training model...")
    model.train(train_df)
    
    # 3. Evaluate
    print("Evaluating...")
    metrics = model.evaluate(test_df)
    print("Metrics:", metrics)

    # 4. XAI Check
    print("Checking XAI...")
    # Get a sample instance from test
    sample_instance = test_df.sample(1)
    # Prepare features just like model does
    X_sample_df = model._prepare_features(sample_instance, fit_scaler=False)
    X_train_summary = model._prepare_features(train_df.sample(100), fit_scaler=False)
    
    # To check XAI, we need the underlying classifier
    print("Train Summary Stats:\n", X_train_summary.describe())
    print("Consant columns:", [col for col in X_train_summary.columns if X_train_summary[col].nunique() <= 1])
    
    explainer = XAIExplainer(
        model=model.abarth_model,
        train_data_sample=X_train_summary,
        feature_names=model.feature_columns
    )
    
    lime_exp = explainer.explain_instance_lime(X_sample_df.iloc[0].values)
    print("LIME explanation generated.")
    
    print("Verification complete!")
    
except Exception as e:
    print(f"Verification FAILED: {e}")
    import traceback
    traceback.print_exc()
