import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os
from sklearn.preprocessing import StandardScaler

class ModelExplainer:
    def __init__(self):
        self.ecom_data = None
        self.credit_data = None
        self.models = {}
        self.scaler = StandardScaler()
        
    def load_data_and_models(self):
        """Load the engineered datasets and trained models."""
        print("Loading data and models...")
        
        # Load data
        self.ecom_data = pd.read_csv('../data/ecom_features.csv')
        self.credit_data = pd.read_csv('../data/credit_features.csv')
        
        # Handle missing values
        self.ecom_data = self.ecom_data.fillna(0)
        self.credit_data = self.credit_data.fillna(0)
        
        # Load models
        model_dir = '../models'
        for model_file in os.listdir(model_dir):
            if model_file.endswith('.joblib'):
                model_name = model_file.replace('.joblib', '')
                self.models[model_name] = joblib.load(os.path.join(model_dir, model_file))
    
    def prepare_ecom_data(self):
        """Prepare e-commerce data for SHAP analysis."""
        print("\nPreparing e-commerce data for SHAP analysis...")
        
        # Select features and target
        X = self.ecom_data.drop(['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address', 'class'], 
                              axis=1, errors='ignore')
        
        # Identify and encode categorical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, X.columns
    
    def prepare_credit_data(self):
        """Prepare credit card data for SHAP analysis."""
        print("\nPreparing credit card data for SHAP analysis...")
        
        # Select features and target
        X = self.credit_data.drop(['Class', 'Time'], axis=1, errors='ignore')
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, X.columns
    
    def explain_model(self, model, X, feature_names, dataset_name, model_name):
        """Generate SHAP explanations for a model."""
        print(f"\nGenerating SHAP explanations for {dataset_name} - {model_name}...")
        
        # Create output directory if it doesn't exist
        os.makedirs('../plots/shap', exist_ok=True)
        
        try:
            # Use TreeExplainer for tree-based models, KernelExplainer for others
            if 'random_forest' in model_name:
                explainer = shap.TreeExplainer(model, X)
                # Calculate SHAP values (using a subset for performance)
                sample_size = min(1000, X.shape[0])
                X_sample = X[:sample_size]
                shap_values = explainer.shap_values(X_sample)
                
                # For binary classification, we take the second array (index 1) which is for class 1 (fraud)
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]
                
                # Summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
                plt.title(f"SHAP Summary - {dataset_name} - {model_name}")
                plt.tight_layout()
                plt.savefig(f'../plots/shap/{dataset_name}_{model_name}_summary.png')
                plt.close()
                
                # Bar plot of mean SHAP values
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                                plot_type='bar', show=False)
                plt.title(f"SHAP Feature Importance - {dataset_name} - {model_name}")
                plt.tight_layout()
                plt.savefig(f'../plots/shap/{dataset_name}_{model_name}_bar.png')
                plt.close()
                
                # For Random Forest, also show feature importance
                if hasattr(model, 'feature_importances_'):
                    plt.figure(figsize=(12, 8))
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1][:20]  # top 20 features
                    plt.title(f'Feature Importances - {dataset_name} - {model_name}')
                    plt.bar(range(len(indices)), importances[indices], align='center')
                    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
                    plt.tight_layout()
                    plt.savefig(f'../plots/shap/{dataset_name}_{model_name}_feature_importance.png')
                    plt.close()
            
            # For logistic regression or other models
            else:
                # Use KernelExplainer for non-tree models
                sample_size = min(100, X.shape[0])  # Smaller sample size for performance
                X_sample = X[:sample_size]
                
                # Define a prediction function for the model
                def predict_proba(x):
                    return model.predict_proba(x)[:, 1]  # Probability of class 1 (fraud)
                
                # Create explainer
                explainer = shap.KernelExplainer(predict_proba, X_sample)
                
                # Calculate SHAP values for a smaller sample
                shap_values = explainer.shap_values(X_sample, nsamples=100)
                
                # Summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
                plt.title(f"SHAP Summary - {dataset_name} - {model_name}")
                plt.tight_layout()
                plt.savefig(f'../plots/shap/{dataset_name}_{model_name}_summary.png')
                plt.close()
                
                # Bar plot of mean SHAP values
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                                plot_type='bar', show=False)
                plt.title(f"SHAP Feature Importance - {dataset_name} - {model_name}")
                plt.tight_layout()
                plt.savefig(f'../plots/shap/{dataset_name}_{model_name}_bar.png')
                plt.close()
            
            print(f"SHAP plots saved to ../plots/shap/{dataset_name}_{model_name}_*.png")
            
        except Exception as e:
            print(f"Error generating SHAP plots for {dataset_name} - {model_name}: {str(e)}")

def main():
    # Initialize the explainer
    explainer = ModelExplainer()
    
    # Load data and models
    explainer.load_data_and_models()
    
    # Process e-commerce data
    print("\n" + "="*50)
    print("E-COMMERCE FRAUD DETECTION EXPLANATIONS")
    print("="*50)
    
    X_ecom, ecom_features = explainer.prepare_ecom_data()
    
    # Generate explanations for e-commerce models
    for model_name, model in explainer.models.items():
        if 'ecom' in model_name:
            explainer.explain_model(model, X_ecom, ecom_features, 'ecom', model_name)
    
    # Process credit card data
    print("\n" + "="*50)
    print("CREDIT CARD FRAUD DETECTION EXPLANATIONS")
    print("="*50)
    
    X_credit, credit_features = explainer.prepare_credit_data()
    
    # Generate explanations for credit card models
    for model_name, model in explainer.models.items():
        if 'credit' in model_name:
            explainer.explain_model(model, X_credit, credit_features, 'credit', model_name)
    
    print("\nModel explainability analysis complete!")

if __name__ == "__main__":
    main()
