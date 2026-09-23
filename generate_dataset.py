"""
Generate a realistic synthetic fraud detection dataset
Saves to data/transactions.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def generate_dataset(n_samples=100000, fraud_rate=0.02, seed=42):
    """
    Generate a realistic synthetic banking transaction dataset for fraud detection.
    
    Features:
    - transaction_id: Unique ID per transaction
    - customer_id: Customer identifier
    - transaction_amount: Amount in INR
    - transaction_time: Seconds since midnight (0–86399)
    - location: Indian city
    - device_id: Device identifier
    - merchant_category: retail, grocery, restaurant, gas, online
    - account_age_days: Age of the account in days
    - transaction_count_24h: Number of transactions in the last 24 hours
    - avg_transaction_amount: Customer's average transaction amount
    - is_fraud: Target label (0 = legitimate, 1 = fraud)
    """
    np.random.seed(seed)

    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    cities = ['Mumbai', 'Delhi', 'Bengaluru', 'Hyderabad', 'Chennai',
              'Kolkata', 'Pune', 'Ahmedabad']
    merchants = ['retail', 'grocery', 'restaurant', 'gas', 'online']
    devices = [f'DEV{i:03d}' for i in range(200)]

    # ---- Legitimate transactions ----
    legit_amounts = np.clip(np.random.exponential(150, n_legit), 10, 5000).round(2)
    # Legitimate transactions lean towards daytime (bell curve around 2 PM)
    hour_weights = np.exp(-0.5 * ((np.arange(24) - 14) / 5) ** 2)
    hour_probs = hour_weights / hour_weights.sum()
    legit_hours = np.random.choice(24, n_legit, p=hour_probs)
    legit_times = legit_hours * 3600 + np.random.randint(0, 3600, n_legit)
    legit_dates = np.random.randint(0, 90, n_legit)
    legit_avg_amounts = np.clip(np.random.exponential(150, n_legit), 20, 3000).round(2)
    legit_acct_ages = np.random.randint(90, 3650, n_legit)
    legit_txn_counts = np.random.randint(1, 10, n_legit)

    legit_df = pd.DataFrame({
        'transaction_id': [f'TXN{i:08d}' for i in range(n_legit)],
        'customer_id': np.random.randint(1000, 10000, n_legit),
        'transaction_amount': legit_amounts,
        'transaction_time': legit_times,
        'transaction_timestamp': (
            pd.Timestamp('2025-01-01')
            + pd.to_timedelta(legit_dates, unit='D')
            + pd.to_timedelta(legit_times, unit='s')
        ),
        'location': np.random.choice(cities, n_legit, p=[0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.07, 0.05]),
        'device_id': np.random.choice(devices, n_legit),
        'merchant_category': np.random.choice(merchants, n_legit, p=[0.30, 0.25, 0.20, 0.10, 0.15]),
        'account_age_days': legit_acct_ages,
        'transaction_count_24h': legit_txn_counts,
        'avg_transaction_amount': legit_avg_amounts,
        'is_fraud': 0,
    })

    # ---- Fraudulent transactions ----
    # Fraud patterns: higher amounts, more night-time, newer accounts, higher frequency, more online
    fraud_amounts = np.clip(np.random.exponential(500, n_fraud), 50, 50000).round(2)
    # Fraudulent transactions skew towards night-time
    fraud_hour_weights = np.array([
        0.06, 0.07, 0.08, 0.08, 0.07, 0.05,  # 0-5 AM (high)
        0.03, 0.02, 0.02, 0.02, 0.02, 0.02,  # 6-11 AM (low)
        0.02, 0.02, 0.02, 0.02, 0.02, 0.02,  # 12-5 PM (low)
        0.03, 0.04, 0.05, 0.06, 0.07, 0.07   # 6-11 PM (rising)
    ])
    fraud_hour_probs = fraud_hour_weights / fraud_hour_weights.sum()
    fraud_hours = np.random.choice(24, n_fraud, p=fraud_hour_probs)
    fraud_times = fraud_hours * 3600 + np.random.randint(0, 3600, n_fraud)
    fraud_dates = np.random.randint(0, 90, n_fraud)
    fraud_avg_amounts = np.clip(np.random.exponential(120, n_fraud), 20, 2000).round(2)
    fraud_acct_ages = np.random.choice(
        np.arange(1, 3650), n_fraud,
        p=np.exp(-np.arange(1, 3650) / 300) / np.exp(-np.arange(1, 3650) / 300).sum()
    )  # Exponentially favor newer accounts
    fraud_txn_counts = np.clip(np.random.poisson(15, n_fraud), 1, 50)

    fraud_df = pd.DataFrame({
        'transaction_id': [f'TXN{i:08d}' for i in range(n_legit, n_samples)],
        'customer_id': np.random.randint(1000, 10000, n_fraud),
        'transaction_amount': fraud_amounts,
        'transaction_time': fraud_times,
        'transaction_timestamp': (
            pd.Timestamp('2025-01-01')
            + pd.to_timedelta(fraud_dates, unit='D')
            + pd.to_timedelta(fraud_times, unit='s')
        ),
        'location': np.random.choice(cities, n_fraud),
        'device_id': np.random.choice(devices, n_fraud),
        'merchant_category': np.random.choice(merchants, n_fraud, p=[0.10, 0.05, 0.05, 0.10, 0.70]),
        'account_age_days': fraud_acct_ages,
        'transaction_count_24h': fraud_txn_counts,
        'avg_transaction_amount': fraud_avg_amounts,
        'is_fraud': 1,
    })

    # Combine & shuffle
    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Inject a small number of missing values to make it realistic (~0.1%)
    n_missing = int(n_samples * 0.001)
    for col in ['transaction_amount', 'account_age_days', 'avg_transaction_amount']:
        missing_idx = np.random.choice(df.index, size=n_missing, replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


if __name__ == '__main__':
    print("Generating synthetic fraud detection dataset...")
    df = generate_dataset(n_samples=100000, fraud_rate=0.02)

    output_path = PROJECT_ROOT / 'data' / 'transactions.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✅ Dataset saved to: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n--- Class Distribution ---")
    fraud_counts = df['is_fraud'].value_counts()
    print(f"   Legitimate (0): {fraud_counts[0]:,}  ({fraud_counts[0]/len(df)*100:.2f}%)")
    print(f"   Fraud      (1): {fraud_counts[1]:,}  ({fraud_counts[1]/len(df)*100:.2f}%)")
    print(f"   Imbalance Ratio: {fraud_counts[0]/fraud_counts[1]:.0f}:1")
    print(f"\n--- Sample Rows ---")
    print(df.head(10).to_string(index=False))
    print(f"\n--- Missing Values ---")
    print(df.isnull().sum().to_string())
