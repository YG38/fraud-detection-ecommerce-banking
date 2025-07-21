# Fraud Detection for E-commerce and Banking

This project focuses on detecting fraudulent transactions in both e-commerce and banking contexts using machine learning techniques.

## Project Structure

```
├── analysis/               # Analysis scripts and reports
│   ├── eda.py             # EDA and data visualization
│   ├── interim_1_report.md # Interim 1 report
│   └── requirements.txt    # Python dependencies for analysis
├── data/                   # Data files (not versioned)
│   ├── Fraud_Data.csv
│   ├── IpAddress_to_Country.csv
│   └── creditcard.csv
├── .gitignore             # Git ignore file
└── README.md              # This file
```

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/YG38/fraud-detection-ecommerce-banking.git
   cd fraud-detection-ecommerce-banking
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
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
