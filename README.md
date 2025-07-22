# Fraud Detection in E-commerce and Banking

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

A comprehensive fraud detection system for identifying fraudulent transactions in both e-commerce and banking domains using machine learning.

## 🚀 Features

- **Dual-Domain Detection**: Handles both e-commerce and credit card transactions
- **Multiple Models**: Implements Logistic Regression and Random Forest classifiers
- **Advanced Analytics**: Includes SHAP-based model interpretability
- **Production-Ready**: Clean, modular code with proper documentation

## 📊 Results Summary

| Dataset | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------|-------|----------|-----------|--------|----------|---------|
| E-commerce | Logistic Regression | 69.0% | 18.5% | 67.7% | 0.29 | 0.76 |
| E-commerce | Random Forest | 95.3% | 93.1% | 54.0% | 0.68 | 0.77 |
| Credit Card | Logistic Regression | 97.5% | 5.8% | 87.2% | 0.11 | 0.97 |
| Credit Card | Random Forest | 99.9% | 89.8% | 77.0% | 0.83 | 0.96 |

## 📂 Project Structure

```
fraud-detection-ecommerce-banking/
├── analysis/               # Analysis scripts and reports
│   ├── eda.py              # Exploratory data analysis
│   ├── feature_engineering.py  # Feature engineering pipeline
│   ├── model_training.py   # Model training and evaluation
│   ├── model_explainability.py  # SHAP analysis
│   ├── interim_1_report.md # Interim report 1
│   └── interim_2_report.md # Interim report 2
├── data/                   # Data files (not version controlled)
├── models/                 # Trained models (not version controlled)
├── plots/                  # Generated visualizations
├── .gitignore             # Git ignore file
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── FINAL_REPORT.md        # Comprehensive project report
```

## 🛠️ Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/YG38/fraud-detection-ecommerce-banking.git
   cd fraud-detection-ecommerce-banking
   ```

2. Install dependencies:
   ```bash
   cd analysis
   pip install -r requirements.txt
   ```

## Interim 1 Submission

### Files Submitted
- `analysis/eda.py`: Script for exploratory data analysis
- `analysis/interim_1_report.md`: Detailed report of findings
- Generated visualizations in the analysis directory

### Key Findings
- E-commerce data shows a 9.36% fraud rate
- Credit card data is highly imbalanced with only 0.17% fraud cases
- No missing values found in any dataset
- Comprehensive visualizations created for both datasets

## Next Steps
- Merge IP address data with transaction data
- Perform feature engineering
- Address class imbalance
- Develop and evaluate machine learning models

## Team
- [Your Name]

## Submission Dates
- Interim-1: July 20, 2025
- Interim-2: July 27, 2025
- Final Submission: July 29, 2025
