import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureEngineering:
    def __init__(self):
        self.ecom_data = None
        self.credit_data = None
        self.ip_data = None
        
    def load_data(self):
        """Load the datasets."""
        print("Loading datasets...")
        self.ecom_data = pd.read_csv('../data/Fraud_Data.csv')
        self.ip_data = pd.read_csv('../data/IpAddress_to_Country.csv')
        self.credit_data = pd.read_csv('../data/creditcard.csv')
        
        # Convert datetime columns
        self.ecom_data['signup_time'] = pd.to_datetime(self.ecom_data['signup_time'])
        self.ecom_data['purchase_time'] = pd.to_datetime(self.ecom_data['purchase_time'])
        
        return self.ecom_data, self.ip_data, self.credit_data
    
    def engineer_ecom_features(self):
        """Engineer features for e-commerce data."""
        print("Engineering e-commerce features...")
        df = self.ecom_data.copy()
        
        # 1. Time-based features
        df['purchase_hour'] = df['purchase_time'].dt.hour
        df['purchase_dayofweek'] = df['purchase_time'].dt.dayofweek
        df['purchase_dayofmonth'] = df['purchase_time'].dt.day
        df['signup_to_purchase_hours'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
        
        # 2. Transaction frequency features
        user_transaction_count = df['user_id'].value_counts().to_dict()
        df['user_transaction_count'] = df['user_id'].map(user_transaction_count)
        
        # 3. Browser and device features
        browser_counts = df['browser'].value_counts().to_dict()
        df['browser_frequency'] = df['browser'].map(browser_counts) / len(df)
        
        # 4. Time since last transaction
        df = df.sort_values(['user_id', 'purchase_time'])
        df['time_since_last_txn'] = df.groupby('user_id')['purchase_time'].diff().dt.total_seconds() / 3600
        
        # 5. Categorical encoding
        df = pd.get_dummies(df, columns=['source', 'browser', 'sex'])
        
        self.ecom_data = df
        return df
    
    def map_ip_to_country(self):
        """Map IP addresses to countries."""
        print("Mapping IP addresses to countries...")
        # Convert IP to integer if needed
        self.ip_data['lower_bound_ip_address'] = self.ip_data['lower_bound_ip_address'].astype(float)
        self.ip_data['upper_bound_ip_address'] = self.ip_data['upper_bound_ip_address'].astype(float)
        
        # Function to find country for an IP
        def find_country(ip):
            try:
                ip_float = float(ip)
                country = self.ip_data[
                    (self.ip_data['lower_bound_ip_address'] <= ip_float) & 
                    (self.ip_data['upper_bound_ip_address'] >= ip_float)
                ]['country'].values[0]
                return country
            except:
                return 'Unknown'
        
        # Apply to e-commerce data (sample for performance)
        sample_ips = self.ecom_data['ip_address'].sample(min(1000, len(self.ecom_data)))
        self.ecom_data['country'] = sample_ips.apply(find_country)
        
        # Add country features
        country_counts = self.ecom_data['country'].value_counts().to_dict()
        self.ecom_data['country_frequency'] = self.ecom_data['country'].map(
            lambda x: country_counts.get(x, 0) / len(self.ecom_data)
        )
        
        return self.ecom_data
    
    def engineer_credit_features(self):
        """Engineer features for credit card data."""
        print("Engineering credit card features...")
        df = self.credit_data.copy()
        
        # 1. Time-based features
        df['hour'] = (df['Time'] / 3600) % 24
        df['day'] = (df['Time'] / (3600 * 24)) % 7
        
        # 2. Transaction amount features
        df['log_amount'] = np.log1p(df['Amount'])
        
        # 3. Anomaly scores based on PCA components
        pca_components = [f'V{i}' for i in range(1, 29)]
        df['anomaly_score'] = df[pca_components].abs().sum(axis=1)
        
        # 4. Rolling statistics (using a small window for performance)
        window_size = 1000
        df['rolling_mean'] = df['Amount'].rolling(window=window_size, min_periods=1).mean()
        df['rolling_std'] = df['Amount'].rolling(window=window_size, min_periods=1).std()
        
        self.credit_data = df
        return df
    
    def save_features(self):
        """Save the engineered features."""
        print("Saving engineered features...")
        self.ecom_data.to_csv('../data/ecom_features.csv', index=False)
        self.credit_data.to_csv('../data/credit_features.csv', index=False)
        print("Features saved successfully!")

def main():
    # Initialize and run feature engineering
    fe = FeatureEngineering()
    fe.load_data()
    
    # Engineer features
    ecom_features = fe.engineer_ecom_features()
    ecom_with_country = fe.map_ip_to_country()
    credit_features = fe.engineer_credit_features()
    
    # Save the results
    fe.save_features()
    
    print("\nFeature engineering complete!")
    print(f"E-commerce features shape: {ecom_features.shape}")
    print(f"Credit card features shape: {credit_features.shape}")

if __name__ == "__main__":
    main()
