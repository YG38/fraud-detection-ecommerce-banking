# Fraud Detection Analysis - Interim 1

This directory contains the initial exploratory data analysis (EDA) and preprocessing for the fraud detection project.

## Project Structure

```
analysis/
├── eda.py             # Main script for exploratory data analysis
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Setup

1. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Analysis

To perform the initial EDA and generate visualizations:

```
python eda.py
```

This will:
1. Load and validate the datasets
2. Generate summary statistics
3. Create visualizations in the current directory

## Next Steps

After running the analysis, we will:
1. Merge the e-commerce data with IP address to country mapping
2. Perform feature engineering (time-based features, transaction patterns)
3. Handle class imbalance
4. Prepare the data for modeling
