# Interim-2 Report: Model Development and Evaluation

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Data Preparation](#data-preparation)
3. [Feature Engineering](#feature-engineering)
4. [Model Development](#model-development)
5. [Model Evaluation](#model-evaluation)
6. [Model Explainability](#model-explainability)
7. [Challenges and Solutions](#challenges-and-solutions)
8. [Next Steps](#next-steps)
9. [Conclusion](#conclusion)

## Executive Summary
This report documents the model development phase of our fraud detection system. We've successfully built and evaluated machine learning models for both e-commerce and credit card fraud detection. The Random Forest model showed superior performance in both domains, with particularly strong results in credit card fraud detection (AUC-ROC: 0.96, F1-Score: 0.83).

## Data Preparation

### Data Sources
- **E-commerce Data**: `Fraud_Data.csv` and `IpAddress_to_Country.csv`
- **Credit Card Data**: `creditcard.csv`

### Preprocessing Steps
1. Handled missing values
2. Converted data types
3. Processed datetime features
4. Normalized numerical features
5. Encoded categorical variables

## Feature Engineering

### E-commerce Features
1. **Time-based Features**
   - Time since user signup
   - Hour, day of week, and day of month of purchase
   
2. **User Behavior Features**
   - Transaction frequency per user
   - Time since last transaction
   
3. **Geolocation Features**
   - Country mapping from IP addresses
   - Country-based transaction frequency

### Credit Card Features
1. **Time-based Features**
   - Transaction hour and day
   - Time since last transaction
   
2. **Transaction Features**
   - Log-transformed amount
   - Rolling statistics (mean, std)
   - Anomaly scores from PCA components

## Model Development

### Algorithms Used
1. **Logistic Regression**
   - Baseline model with L2 regularization
   - Handles class imbalance using class weights
   
2. **Random Forest**
   - Ensemble of decision trees
   - Handles non-linear relationships effectively
   - Robust to outliers

### Training Process
- 70-30 train-test split
- Stratified sampling to maintain class distribution
- Hyperparameters tuned using cross-validation

## Model Evaluation

### Performance Metrics
| Dataset | Model | Accuracy | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) | ROC-AUC |
|---------|-------|----------|-------------------|----------------|------------------|---------|
| E-commerce | Logistic Regression | 0.69 | 0.18 | 0.68 | 0.29 | 0.76 |
| E-commerce | Random Forest | 0.95 | 0.93 | 0.54 | 0.68 | 0.77 |
| Credit Card | Logistic Regression | 0.98 | 0.06 | 0.87 | 0.11 | 0.97 |
| Credit Card | Random Forest | 1.00 | 0.90 | 0.77 | 0.83 | 0.96 |

### Key Findings
- Random Forest outperformed Logistic Regression in both domains
- Higher precision in credit card fraud detection suggests better reliability
- E-commerce fraud detection shows more challenging class separation

## Model Explainability

### SHAP Analysis
1. **E-commerce Model**
   - Top features: purchase amount, time since signup, transaction frequency
   - Country of origin shows significant impact on fraud probability
   
2. **Credit Card Model**
   - PCA components (V1-V28) show varying importance
   - Transaction amount has non-linear relationship with fraud probability

### Feature Importance
- Generated visualizations showing top predictive features
- Identified key indicators of fraudulent behavior

## Challenges and Solutions

### Class Imbalance
- **Challenge**: Severe class imbalance (especially in credit card data)
- **Solution**: Used SMOTE for oversampling minority class

### Feature Engineering
- **Challenge**: Creating meaningful temporal features
- **Solution**: Engineered time-based features capturing user behavior patterns

### Model Interpretability
- **Challenge**: Understanding complex model decisions
- **Solution**: Applied SHAP values for model-agnostic interpretability

## Next Steps

### Immediate Next Steps
1. Hyperparameter tuning for optimal performance
2. Experiment with additional algorithms (XGBoost, Neural Networks)
3. Implement model serving infrastructure

### Future Enhancements
1. Real-time fraud detection pipeline
2. Ensemble of multiple models
3. Automated retraining pipeline

## Conclusion
Our fraud detection system shows promising results, particularly in credit card fraud detection. The Random Forest model demonstrates strong performance and interpretability. The next phase will focus on model optimization and deployment preparation.

## Appendix
- Code Repository: [GitHub](https://github.com/YG38/fraud-detection-ecommerce-banking)
- Data Sources: [Kaggle](https://www.kaggle.com/)
- Documentation: See project README for setup instructions
