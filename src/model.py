import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import joblib
import os

class EngineHealthModel:
    def __init__(self):
        self.rul_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        self.abarth_model = RandomForestClassifier(n_estimators=100, random_state=42) # Health State Classifier
        self.scaler = MinMaxScaler()
        self.feature_columns = None
        self.drop_sensors = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19', 'setting_3'] # FD001 constant sensors

    def save_model(self, path='model_checkpoints.pkl'):
        joblib.dump({
            'rul_model': self.rul_model,
            'abarth_model': self.abarth_model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path='model_checkpoints.pkl'):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} not found.")
        
        data = joblib.load(path)
        self.rul_model = data['rul_model']
        self.abarth_model = data['abarth_model']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        print(f"Model loaded from {path}")

    def _prepare_features(self, df, fit_scaler=False):
        # Drop non-feature columns
        features = df.drop(['unit_number', 'time_in_cycles', 'RUL'], axis=1)
        
        # Drop constant sensors if known, or handle dynamically
        # For this implementation, we drop known useless sensors for FD001
        features = features.drop(columns=[col for col in self.drop_sensors if col in features.columns], errors='ignore')
        
        if fit_scaler:
            self.feature_columns = features.columns.tolist()
            scaled_features = self.scaler.fit_transform(features)
        else:
            # Ensure columns exact match
            features = features[self.feature_columns]
            scaled_features = self.scaler.transform(features)
            
        return pd.DataFrame(scaled_features, columns=self.feature_columns)

    def generate_labels(self, rul_series):
        # Adjusted for Demo Variety:
        # 0: Normal (>75), 1: Warning (30-75), 2: Critical (<30)
        labels = []
        for rul in rul_series:
            if rul > 75:
                labels.append(0) # Normal
            elif rul >= 30:
                labels.append(1) # Warning
            else:
                labels.append(2) # Critical
        return np.array(labels)
    
    def get_state_label(self, label_code):
        mapping = {0: "Normal", 1: "Warning", 2: "Critical"}
        return mapping.get(label_code, "Unknown")

    def train(self, train_df):
        X_train = self._prepare_features(train_df, fit_scaler=True)
        y_rul = train_df['RUL']
        y_state = self.generate_labels(y_rul)
        
        print("Training RUL Regressor...")
        self.rul_model.fit(X_train, y_rul)
        
        print("Training Health State Classifier...")
        self.abarth_model.fit(X_train, y_state)
        
        print("Training Complete.")
        
    def evaluate(self, test_df):
        X_test = self._prepare_features(test_df, fit_scaler=False)
        y_rul = test_df['RUL']
        y_state = self.generate_labels(y_rul)
        
        pred_rul = self.rul_model.predict(X_test)
        pred_state = self.abarth_model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_rul, pred_rul))
        acc = accuracy_score(y_state, pred_state)
        
        return {
            "RMSE": rmse,
            "Accuracy": acc,
            "Classification Report": classification_report(y_state, pred_state)
        }
        
    def predict(self, input_df):
        # input_df should have same raw columns as train_df
        X = self._prepare_features(input_df, fit_scaler=False)
        
        pred_rul = self.rul_model.predict(X)
        pred_state = self.abarth_model.predict(X)
        pred_probs = self.abarth_model.predict_proba(X) # Get probabilities
        
        pred_state_labels = [self.get_state_label(s) for s in pred_state]
        confidences = [max(p) for p in pred_probs] # Max probability is the confidence
        
        return pred_rul, pred_state_labels, confidences
