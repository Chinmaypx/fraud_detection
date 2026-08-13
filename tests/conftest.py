"""
Pytest configuration and shared fixtures for fraud detection tests
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_transaction_data():
    """Sample transaction data for testing"""
    return {
        'transaction_amount': 250.00,
        'transaction_time': 36000,
        'location': 'Mumbai',
        'device_id': 'DEV001',
        'merchant_category': 'retail',
        'account_age_days': 365,
        'transaction_count_24h': 3,
        'avg_transaction_amount': 150.00,
    }


@pytest.fixture
def sample_transactions_df():
    """Sample DataFrame of transactions"""
    data = {
        'transaction_id': [f'TXN{i:08d}' for i in range(100)],
        'customer_id': np.random.randint(1000, 5000, 100),
        'transaction_amount': np.random.exponential(150, 100).round(2),
        'transaction_time': np.random.randint(0, 86400, 100),
        'location': np.random.choice(['Mumbai', 'Delhi', 'Bengaluru'], 100),
        'device_id': np.random.choice([f'DEV{i:03d}' for i in range(10)], 100),
        'merchant_category': np.random.choice(['retail', 'grocery', 'online'], 100),
        'account_age_days': np.random.randint(30, 3650, 100),
        'transaction_count_24h': np.random.randint(1, 20, 100),
        'avg_transaction_amount': np.random.exponential(150, 100).round(2),
        'is_fraud': np.random.choice([0, 1], 100, p=[0.95, 0.05])
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_preprocessed_data():
    """Sample preprocessed data for testing"""
    np.random.seed(42)
    return {
        'X_train': np.random.randn(1000, 10),
        'X_test': np.random.randn(200, 10),
        'y_train': np.random.choice([0, 1], 1000, p=[0.95, 0.05]),
        'y_test': np.random.choice([0, 1], 200, p=[0.95, 0.05]),
    }


@pytest.fixture
def sample_predictions():
    """Sample predictions and ground truth for evaluation"""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.9, 0.4, 0.15, 0.85, 0.1, 0.6, 0.95, 0.05])
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_proba': y_proba
    }


@pytest.fixture
def temp_model_dir(tmp_path):
    """Temporary directory for model artifacts"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir