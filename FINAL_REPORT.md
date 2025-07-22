# Fraud Detection System: E-commerce and Banking

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Data Description](#data-description)
4. [Methodology](#methodology)
5. [Results](#results)
6. [Deployment](#deployment)
7. [Conclusion](#conclusion)
8. [Future Work](#future-work)
9. [References](#references)

## Executive Summary
This project developed a fraud detection system for both e-commerce and banking transactions using machine learning. The system achieved high accuracy in identifying fraudulent transactions while maintaining interpretability through SHAP analysis.

## Project Overview
- **Goal**: Build a robust fraud detection system
- **Scope**: E-commerce and credit card transaction data
- **Technologies**: Python, Scikit-learn, Pandas, SHAP
- **Repository**: [GitHub](https://github.com/YG38/fraud-detection-ecommerce-banking)

## Data Description

### E-commerce Data
- Source: [Kaggle](https://www.kaggle.com/)
- Features: Transaction amount, user details, timestamps, device info, IP addresses
- Class Distribution: ~9.4% fraudulent transactions

### Credit Card Data
- Source: [Kaggle](https://www.kaggle.com/)
- Features: 28 anonymized features (V1-V28), transaction amount, time
- Class Distribution: ~0.17% fraudulent transactions

## Methodology

### Data Preprocessing
1. Handled missing values
2. Feature engineering
3. Normalization and scaling
4. Class imbalance handling (SMOTE)

### Models Developed
1. **Logistic Regression**
   - Baseline model with L2 regularization
   - Class weights for imbalance

2. **Random Forest**
   - 100 decision trees
   - Class-balanced subsampling

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrices
- SHAP values for interpretability

## Results

### E-commerce Fraud Detection
| Metric | Logistic Regression | Random Forest |
|--------|---------------------|---------------|
| Accuracy | 69.0% | 95.3% |
| Precision | 18.5% | 93.1% |
| Recall | 67.7% | 54.0% |
| F1-Score | 0.29 | 0.68 |
| ROC-AUC | 0.76 | 0.77 |

### Credit Card Fraud Detection
| Metric | Logistic Regression | Random Forest |
|--------|---------------------|---------------|
| Accuracy | 97.5% | 99.9% |
| Precision | 5.8% | 89.8% |
| Recall | 87.2% | 77.0% |
| F1-Score | 0.11 | 0.83 |
| ROC-AUC | 0.97 | 0.96 |

## Deployment

### Model Storage
- Models stored in `models/` directory
- Version controlled with Git
- Large files managed via .gitignore

### Usage
```python
# Load the trained model
import joblib
model = joblib.load('models/credit_random_forest.joblib')

# Make predictions
predictions = model.predict_proba(X_new)[:, 1]
```

## Conclusion
- Successfully developed fraud detection models for both domains
- Random Forest showed superior performance
- System is ready for deployment with proper monitoring

## Future Work
1. Real-time prediction API
2. Model retraining pipeline
3. Advanced deep learning approaches
4. Integration with transaction systems

## References
1. Scikit-learn Documentation
2. SHAP Documentation
3. Kaggle Datasets

---
Project by: [Your Name]  
Date: July 2024  
GitHub: [YG38/fraud-detection-ecommerce-banking](https://github.com/YG38/fraud-detection-ecommerce-banking)
