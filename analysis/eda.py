import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for plots
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

def load_data():
    """Load and return the datasets."""
    print("Loading datasets...")
    
    # Load e-commerce data
    ecom_data = pd.read_csv('../data/Fraud_Data.csv')
    
    # Load IP to country mapping
    ip_data = pd.read_csv('../data/IpAddress_to_Country.csv')
    
    # Load credit card data
    credit_data = pd.read_csv('../data/creditcard.csv')
    
    return ecom_data, ip_data, credit_data

def basic_data_checks(df, name):
    """Perform basic data quality checks and print summary."""
    print(f"\n=== {name} Dataset Info ===")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nSummary statistics:")
    print(df.describe(include='all'))

def analyze_ecom_data(df):
    """Perform EDA on e-commerce data."""
    print("\n=== E-commerce Data Analysis ===")
    
    # Convert datetime columns
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    # 1. Class distribution
    print("\nClass distribution:")
    print(df['class'].value_counts(normalize=True) * 100)
    
    # 2. Basic visualizations
    plt.figure(figsize=(15, 10))
    
    # Purchase value distribution
    plt.subplot(2, 2, 1)
    sns.histplot(df['purchase_value'], bins=30, kde=True)
    plt.title('Distribution of Purchase Values')
    
    # Age distribution
    plt.subplot(2, 2, 2)
    sns.histplot(df['age'], bins=30, kde=True)
    plt.title('Age Distribution')
    
    # Browser distribution
    plt.subplot(2, 2, 3)
    df['browser'].value_counts().plot(kind='bar')
    plt.title('Browser Distribution')
    plt.xticks(rotation=45)
    
    # Source distribution
    plt.subplot(2, 2, 4)
    df['source'].value_counts().plot(kind='bar')
    plt.title('Traffic Source Distribution')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('ecommerce_eda.png')
    plt.close()

def analyze_credit_data(df):
    """Perform EDA on credit card data."""
    print("\n=== Credit Card Data Analysis ===")
    
    # 1. Class distribution
    print("\nClass distribution:")
    print(df['Class'].value_counts(normalize=True) * 100)
    
    # 2. Basic visualizations
    plt.figure(figsize=(15, 10))
    
    # Transaction amount distribution
    plt.subplot(2, 2, 1)
    sns.histplot(df['Amount'], bins=50, kde=True)
    plt.title('Distribution of Transaction Amounts')
    
    # Log transaction amount (to handle skewness)
    plt.subplot(2, 2, 2)
    sns.histplot(np.log1p(df['Amount']), bins=50, kde=True)
    plt.title('Log Distribution of Transaction Amounts')
    
    # Time vs Amount
    plt.subplot(2, 2, 3)
    sns.scatterplot(x='Time', y='Amount', hue='Class', data=df, alpha=0.5)
    plt.title('Time vs Amount by Class')
    
    # Correlation heatmap (first 10 features for clarity)
    plt.subplot(2, 2, 4)
    corr = df.corr()['Class'].sort_values(ascending=False)
    sns.barplot(x=corr.index[:10], y=corr.values[:10])
    plt.xticks(rotation=90)
    plt.title('Top 10 Features Correlated with Class')
    
    plt.tight_layout()
    plt.savefig('creditcard_eda.png')
    plt.close()

def main():
    # Load the data
    ecom_data, ip_data, credit_data = load_data()
    
    # Perform basic checks
    basic_data_checks(ecom_data, 'E-commerce')
    basic_data_checks(ip_data, 'IP to Country')
    basic_data_checks(credit_data, 'Credit Card')
    
    # Perform detailed analysis
    analyze_ecom_data(ecom_data)
    analyze_credit_data(credit_data)
    
    print("\nAnalysis complete. Check the generated plots in the analysis directory.")

if __name__ == "__main__":
    main()
