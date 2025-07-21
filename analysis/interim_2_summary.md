# Interim-2 Submission: Model Building and Evaluation

## Model Performance Summary

### E-commerce Fraud Detection

**Logistic Regression:**
- ROC-AUC: 0.759
- Accuracy: 0.690
- Precision (Fraud): 0.185
- Recall (Fraud): 0.677
- F1-Score (Fraud): 0.290

**Random Forest:**
- ROC-AUC: 0.773
- Accuracy: 0.953
- Precision (Fraud): 0.931
- Recall (Fraud): 0.540
- F1-Score (Fraud): 0.683

### Credit Card Fraud Detection

**Logistic Regression:**
- ROC-AUC: 0.971
- Accuracy: 0.975
- Precision (Fraud): 0.058
- Recall (Fraud): 0.872
- F1-Score (Fraud): 0.109

**Random Forest:**
- ROC-AUC: 0.961
- Accuracy: 0.999
- Precision (Fraud): 0.898
- Recall (Fraud): 0.770
- F1-Score (Fraud): 0.829

## Key Findings

1. **Class Imbalance Handling**:
   - Applied SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance
   - Significantly improved model performance on the minority class (fraud cases)

2. **Model Performance**:
   - Random Forest performed better than Logistic Regression for both datasets
   - Higher precision and F1-scores indicate better balance between precision and recall
   - Credit card fraud detection achieved higher performance metrics

3. **Feature Importance**:
   - Generated feature importance plots for tree-based models
   - Key features identified for fraud detection in both domains

## Next Steps

1. Hyperparameter tuning for better model performance
2. Experiment with different sampling techniques
3. Implement model explainability using SHAP values
4. Deploy the best-performing models

## Files Created

1. `feature_engineering.py` - Script for feature engineering
2. `model_training.py` - Script for model training and evaluation
3. `models/` - Directory containing trained models
4. `plots/` - Directory containing feature importance plots
5. `interim_2_summary.md` - This summary document
