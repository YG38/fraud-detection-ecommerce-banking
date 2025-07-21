import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class FraudDetectionModel:
    def __init__(self):
        self.ecom_data = None
        self.credit_data = None
        self.models = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load the engineered datasets."""
        print("Loading engineered datasets...")
        self.ecom_data = pd.read_csv('../data/ecom_features.csv')
        self.credit_data = pd.read_csv('../data/credit_features.csv')
        
        # Handle any remaining missing values
        self.ecom_data = self.ecom_data.fillna(0)
        self.credit_data = self.credit_data.fillna(0)
        
        return self.ecom_data, self.credit_data
    
    def prepare_ecom_data(self):
        """Prepare e-commerce data for modeling."""
        print("\nPreparing e-commerce data for modeling...")
        
        # Select features and target
        X = self.ecom_data.drop(['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address', 'class'], axis=1, errors='ignore')
        y = self.ecom_data['class']
        
        # Identify categorical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # One-hot encode categorical columns
        if categorical_cols:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns
    
    def prepare_credit_data(self):
        """Prepare credit card data for modeling."""
        print("\nPreparing credit card data for modeling...")
        
        # Select features and target
        X = self.credit_data.drop(['Class', 'Time'], axis=1, errors='ignore')
        y = self.credit_data['Class']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns
    
    def handle_class_imbalance(self, X, y, method='smote'):
        """Handle class imbalance using different techniques."""
        print(f"\nHandling class imbalance using {method}...")
        
        if method == 'smote':
            # SMOTE for oversampling the minority class
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
        elif method == 'undersample':
            # Random undersampling of the majority class
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X, y)
        else:
            # No resampling
            X_resampled, y_resampled = X, y
        
        return X_resampled, y_resampled
    
    def train_models(self, X_train, y_train, dataset_name):
        """Train multiple models and store them."""
        print(f"\nTraining models on {dataset_name} data...")
        
        # Initialize models
        models = {
            'logistic_regression': LogisticRegression(
                class_weight='balanced', 
                random_state=42,
                max_iter=1000
            ),
            'random_forest': RandomForestClassifier(
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1
            )
        }
        
        # Train each model
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.models[f"{dataset_name}_{name}"] = model
    
    def evaluate_models(self, X_test, y_test, dataset_name):
        """Evaluate models and print metrics."""
        print(f"\nEvaluating models on {dataset_name} test data...")
        
        results = {}
        
        for model_name, model in self.models.items():
            if dataset_name in model_name:
                # Predict probabilities and classes
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                report = classification_report(y_test, y_pred, output_dict=True)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                # Store results
                results[model_name] = {
                    'precision_0': report['0']['precision'],
                    'recall_0': report['0']['recall'],
                    'f1_0': report['0']['f1-score'],
                    'precision_1': report['1']['precision'],
                    'recall_1': report['1']['recall'],
                    'f1_1': report['1']['f1-score'],
                    'roc_auc': roc_auc,
                    'accuracy': report['accuracy']
                }
                
                # Print results
                print(f"\n{model_name} Results:")
                print(f"ROC-AUC: {roc_auc:.4f}")
                print(f"Accuracy: {report['accuracy']:.4f}")
                print(f"Precision (Class 1): {report['1']['precision']:.4f}")
                print(f"Recall (Class 1): {report['1']['recall']:.4f}")
                print(f"F1-Score (Class 1): {report['1']['f1-score']:.4f}")
        
        return results
    
    def plot_feature_importance(self, model, feature_names, dataset_name, model_name):
        """Plot feature importance for tree-based models."""
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Plot
            plt.figure(figsize=(12, 8))
            plt.title(f"Feature Importances - {dataset_name} - {model_name}")
            plt.bar(range(min(20, len(feature_names))), 
                   importances[indices][:20],
                   align="center")
            plt.xticks(range(min(20, len(feature_names))), 
                      [feature_names[i] for i in indices][:20], 
                      rotation=90)
            plt.tight_layout()
            
            # Save the plot
            os.makedirs('../plots', exist_ok=True)
            plt.savefig(f'../plots/{dataset_name}_{model_name}_feature_importance.png')
            plt.close()
    
    def save_models(self):
        """Save trained models to disk."""
        os.makedirs('../models', exist_ok=True)
        for model_name, model in self.models.items():
            joblib.dump(model, f'../models/{model_name}.joblib')
        print("\nModels saved to disk.")

def main():
    # Initialize the fraud detection model
    fdm = FraudDetectionModel()
    
    # Load data
    fdm.load_data()
    
    # Process and model e-commerce data
    print("\n" + "="*50)
    print("E-COMMERCE FRAUD DETECTION")
    print("="*50)
    
    X_train_ecom, X_test_ecom, y_train_ecom, y_test_ecom, ecom_features = fdm.prepare_ecom_data()
    
    # Handle class imbalance for e-commerce data
    X_train_ecom_resampled, y_train_ecom_resampled = fdm.handle_class_imbalance(
        X_train_ecom, y_train_ecom, method='smote'
    )
    
    # Train and evaluate e-commerce models
    fdm.train_models(X_train_ecom_resampled, y_train_ecom_resampled, 'ecom')
    ecom_results = fdm.evaluate_models(X_test_ecom, y_test_ecom, 'ecom')
    
    # Process and model credit card data
    print("\n" + "="*50)
    print("CREDIT CARD FRAUD DETECTION")
    print("="*50)
    
    X_train_credit, X_test_credit, y_train_credit, y_test_credit, credit_features = fdm.prepare_credit_data()
    
    # Handle class imbalance for credit card data
    X_train_credit_resampled, y_train_credit_resampled = fdm.handle_class_imbalance(
        X_train_credit, y_train_credit, method='smote'
    )
    
    # Train and evaluate credit card models
    fdm.train_models(X_train_credit_resampled, y_train_credit_resampled, 'credit')
    credit_results = fdm.evaluate_models(X_test_credit, y_test_credit, 'credit')
    
    # Plot feature importance for tree-based models
    for model_name, model in fdm.models.items():
        if 'random_forest' in model_name:
            features = ecom_features if 'ecom' in model_name else credit_features
            fdm.plot_feature_importance(model, features, 
                                      'ecom' if 'ecom' in model_name else 'credit',
                                      'random_forest')
    
    # Save models
    fdm.save_models()
    
    print("\nModel training and evaluation complete!")

if __name__ == "__main__":
    main()
