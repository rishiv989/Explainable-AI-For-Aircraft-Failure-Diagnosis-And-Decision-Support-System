import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class XAIExplainer:
    def __init__(self, model, train_data_sample, feature_names, class_names=['Normal', 'Warning', 'Critical']):
        self.model = model # The trained model object (e.g. classifier)
        self.feature_names = feature_names
        self.class_names = class_names
        self.train_data_sample = train_data_sample # Needs to be numpy array or dataframe
        
        # SHAP Explainer
        # Check if model is tree-based (XGBoost/RandomForest)
        # Using TreeExplainer is faster for trees
        try:
            self.shap_explainer = shap.TreeExplainer(self.model)
        except:
            self.shap_explainer = shap.KernelExplainer(self.model.predict, self.train_data_sample)

        # LIME Explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(self.train_data_sample),
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification'
        )

    def get_shap_values(self, X_instance):
        # Calculate SHAP values
        shap_values = self.shap_explainer.shap_values(X_instance)
        return shap_values

    def explain_instance_shap(self, X_instance):
        # Generate SHAP plot for single instance
        shap_values = self.get_shap_values(X_instance)
        # For multi-class, shap_values is a list. We usually care about the predicted class.
        # But for simplicity, let's just return the values and let the UI handle plotting or return a plot object.
        # Actually, beeswarm/force plots are complex to return as objects without executing plt.show().
        # We will return the values and also generate a textual summary.
        return shap_values

    def explain_instance_lime(self, X_instance_row, num_features=5):
        # X_instance_row should be 1D array
        exp = self.lime_explainer.explain_instance(
            X_instance_row, 
            self.model.predict_proba, 
            num_features=num_features
        )
        return exp

    def explain_prediction_text(self, shap_values, X_df, instance_index=0):
        # Generate simple text explanation
        # Identify top contributing features to the current prediction
        
        # If classifier, shap_values is list of arrays [class0, class1, class2]
        # We need to know which class was predicted or which one we are explaining.
        # Assuming we look at the 'Critical' class (index 2) contribution
        
        vals = shap_values[2][instance_index] if isinstance(shap_values, list) else shap_values[instance_index]
        
        # Create DF of feature importance
        imp = pd.DataFrame({'feature': self.feature_names, 'shap_value': vals})
        imp['abs_val'] = imp['shap_value'].abs()
        top_features = imp.sort_values('abs_val', ascending=False).head(3)
        
        explanation = f"Top factors pushing towards Critical status: "
        factors = []
        for _, row in top_features.iterrows():
            direction = "High" if row['shap_value'] > 0 else "Low" # Simplified
            factors.append(f"{row['feature']} ({direction} impact)")
        
        return explanation + ", ".join(factors)
