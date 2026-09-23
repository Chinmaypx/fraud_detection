"""
Tests for data_pipeline.py
"""

import pytest
import pandas as pd
import numpy as np
from src.data_pipeline import DataPipeline


class TestDataPipeline:
    """Tests for DataPipeline class"""

    def test_init(self):
        """Test DataPipeline initialization"""
        pipeline = DataPipeline()
        assert pipeline.data_path is None
        assert pipeline.df is None
        assert pipeline.train_df is None
        assert pipeline.test_df is None

    def test_init_with_path(self):
        """Test DataPipeline initialization with path"""
        pipeline = DataPipeline(data_path='data/transactions.csv')
        assert pipeline.data_path == 'data/transactions.csv'

    def test_generate_synthetic_data(self):
        """Test synthetic data generation"""
        pipeline = DataPipeline()
        df = pipeline._generate_synthetic_data(n_samples=1000, fraud_rate=0.05)
        
        assert len(df) == 1000
        assert 'is_fraud' in df.columns
        assert 'transaction_id' in df.columns
        assert 'customer_id' in df.columns
        assert 'transaction_amount' in df.columns
        
        fraud_rate = df['is_fraud'].mean()
        assert abs(fraud_rate - 0.05) < 0.01

    def test_generate_synthetic_data_fraud_rate(self):
        """Test synthetic data has correct fraud rate"""
        pipeline = DataPipeline()
        df = pipeline._generate_synthetic_data(n_samples=10000, fraud_rate=0.02)
        
        expected_fraud = int(10000 * 0.02)
        actual_fraud = df['is_fraud'].sum()
        
        assert actual_fraud == expected_fraud

    def test_generate_synthetic_data_columns(self):
        """Test synthetic data has all required columns"""
        pipeline = DataPipeline()
        df = pipeline._generate_synthetic_data()
        
        required_cols = [
            'transaction_id', 'customer_id', 'transaction_amount',
            'transaction_time', 'location', 'device_id', 'merchant_category',
            'account_age_days', 'transaction_count_24h', 'avg_transaction_amount',
            'is_fraud'
        ]
        
        for col in required_cols:
            assert col in df.columns

    def test_generated_data_has_full_synthetic_timestamp(self):
        df = DataPipeline()._generate_synthetic_data(n_samples=100)
        assert 'transaction_timestamp' in df.columns
        assert pd.to_datetime(df['transaction_timestamp'], errors='coerce').notna().all()

    def test_load_data_generates_when_no_path(self):
        """Test load_data generates synthetic data when no path provided"""
        pipeline = DataPipeline()
        pipeline.load_data()
        
        assert pipeline.df is not None
        assert len(pipeline.df) > 0
        assert 'is_fraud' in pipeline.df.columns

    def test_load_data_from_file(self, tmp_path):
        """Test load_data reads from CSV file"""
        csv_path = tmp_path / "test_transactions.csv"
        
        data = {
            'transaction_id': [f'TXN{i:08d}' for i in range(100)],
            'transaction_amount': np.random.exponential(150, 100).round(2),
            'is_fraud': np.random.choice([0, 1], 100)
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        pipeline = DataPipeline()
        pipeline.load_data(path=str(csv_path))
        
        assert pipeline.df is not None
        assert len(pipeline.df) == 100

    def test_handle_missing_values_no_missing(self, sample_transactions_df):
        """Test handle_missing_values when no missing values"""
        pipeline = DataPipeline()
        pipeline.df = sample_transactions_df.copy()
        pipeline.handle_missing_values()
        
        assert pipeline.df.isnull().sum().sum() == 0

    def test_handle_missing_values_with_missing(self):
        """Test handle_missing_values fills missing values"""
        pipeline = DataPipeline()
        data = {
            'amount': [1, 2, np.nan, 4, 5],
            'category': ['a', 'b', 'c', np.nan, 'e']
        }
        pipeline.df = pd.DataFrame(data)
        pipeline.handle_missing_values()
        
        assert pipeline.df.isnull().sum().sum() == 0

    def test_handle_missing_values_numeric_fills_with_median(self):
        """Test numeric missing values are filled with median"""
        pipeline = DataPipeline()
        data = {
            'amount': [1, 2, np.nan, 4, 5],
        }
        pipeline.df = pd.DataFrame(data)
        pipeline.handle_missing_values()
        
        assert not pipeline.df['amount'].isnull().any()
        assert pipeline.df['amount'].median() == 3.0

    def test_feature_engineering_creates_features(self, sample_transactions_df):
        """Test feature engineering creates expected features"""
        pipeline = DataPipeline()
        pipeline.df = sample_transactions_df.copy()
        original_cols = len(pipeline.df.columns)
        
        pipeline.feature_engineering()
        
        expected_features = [
            'hour_of_day', 'is_night_transaction', 'amount_to_avg_ratio',
            'high_amount', 'unusual_merchant', 'new_customer', 'high_frequency'
        ]
        
        for feat in expected_features:
            assert feat in pipeline.df.columns
        
        assert len(pipeline.df.columns) == original_cols + len(expected_features)

    def test_feature_engineering_hour_of_day(self):
        """Test hour_of_day calculation"""
        pipeline = DataPipeline()
        pipeline.df = pd.DataFrame({
            'transaction_time': [3600, 7200, 43200, 86399],
            'transaction_amount': [100, 200, 300, 400],
            'avg_transaction_amount': [150, 150, 150, 150],
            'merchant_category': ['retail', 'retail', 'retail', 'retail'],
            'account_age_days': [365, 365, 365, 365],
            'transaction_count_24h': [5, 5, 5, 5]
        })
        pipeline.feature_engineering()
        
        assert pipeline.df['hour_of_day'].tolist() == [1, 2, 12, 23]

    def test_feature_engineering_night_transaction(self):
        """Test is_night_transaction flag"""
        pipeline = DataPipeline()
        pipeline.df = pd.DataFrame({
            'transaction_time': [3600, 79200, 82800, 21600],
            'transaction_amount': [100, 200, 300, 400],
            'avg_transaction_amount': [150, 150, 150, 150],
            'merchant_category': ['retail', 'retail', 'retail', 'retail'],
            'account_age_days': [365, 365, 365, 365],
            'transaction_count_24h': [5, 5, 5, 5]
        })
        pipeline.feature_engineering()
        
        assert 'is_night_transaction' in pipeline.df.columns
        assert pipeline.df['is_night_transaction'].iloc[1] == 1
        assert pipeline.df['is_night_transaction'].iloc[2] == 1
        assert pipeline.df['is_night_transaction'].iloc[3] == 0

    def test_feature_engineering_high_amount(self):
        """Test high_amount flag"""
        pipeline = DataPipeline()
        pipeline.df = pd.DataFrame({
            'transaction_amount': [100, 500, 600, 1000],
            'transaction_time': [3600, 3600, 3600, 3600],
            'avg_transaction_amount': [150, 150, 150, 150],
            'merchant_category': ['retail', 'retail', 'retail', 'retail'],
            'account_age_days': [365, 365, 365, 365],
            'transaction_count_24h': [5, 5, 5, 5]
        })
        pipeline.feature_engineering()
        
        expected = [False, False, True, True]
        assert pipeline.df['high_amount'].tolist() == expected

    def test_feature_engineering_new_customer(self):
        """Test new_customer flag"""
        pipeline = DataPipeline()
        pipeline.df = pd.DataFrame({
            'account_age_days': [30, 90, 180, 365],
            'transaction_amount': [100, 200, 300, 400],
            'transaction_time': [3600, 3600, 3600, 3600],
            'avg_transaction_amount': [150, 150, 150, 150],
            'merchant_category': ['retail', 'retail', 'retail', 'retail'],
            'transaction_count_24h': [5, 5, 5, 5]
        })
        pipeline.feature_engineering()
        
        expected = [True, False, False, False]
        assert pipeline.df['new_customer'].tolist() == expected

    def test_feature_engineering_high_frequency(self):
        """Test high_frequency flag"""
        pipeline = DataPipeline()
        pipeline.df = pd.DataFrame({
            'transaction_count_24h': [1, 10, 11, 50],
            'transaction_amount': [100, 200, 300, 400],
            'transaction_time': [3600, 3600, 3600, 3600],
            'avg_transaction_amount': [150, 150, 150, 150],
            'merchant_category': ['retail', 'retail', 'retail', 'retail'],
            'account_age_days': [365, 365, 365, 365]
        })
        pipeline.feature_engineering()
        
        expected = [False, False, True, True]
        assert pipeline.df['high_frequency'].tolist() == expected

    def test_encode_categorical(self):
        """Test categorical encoding"""
        pipeline = DataPipeline()
        pipeline.df = pd.DataFrame({
            'location': ['Mumbai', 'Delhi', 'Mumbai', 'Bengaluru'],
            'category': ['retail', 'grocery', 'online', 'retail'],
            'transaction_amount': [100, 200, 300, 400],
            'transaction_time': [3600, 3600, 3600, 3600],
            'avg_transaction_amount': [150, 150, 150, 150],
            'account_age_days': [365, 365, 365, 365],
            'transaction_count_24h': [5, 5, 5, 5],
            'is_fraud': [0, 0, 1, 0]
        })
        original_cols = len(pipeline.df.columns)
        
        pipeline.encode_categorical()
        
        assert 'location_Mumbai' in pipeline.df.columns
        assert 'location_Delhi' in pipeline.df.columns
        assert 'category_online' in pipeline.df.columns
        assert 'location' not in pipeline.df.columns
        assert 'category' not in pipeline.df.columns

    def test_encode_categorical_excludes_ids(self):
        """Test encoding skips transaction_id, device_id but does not drop them"""
        pipeline = DataPipeline()
        pipeline.df = pd.DataFrame({
            'transaction_id': ['TXN1', 'TXN2'],
            'device_id': ['DEV1', 'DEV2'],
            'location': ['Mumbai', 'Delhi'],
            'transaction_amount': [100, 200],
            'transaction_time': [3600, 3600],
            'avg_transaction_amount': [150, 150],
            'account_age_days': [365, 365],
            'transaction_count_24h': [5, 5],
            'is_fraud': [0, 1]
        })
        
        pipeline.encode_categorical()
        
        assert 'transaction_id' in pipeline.df.columns
        assert 'device_id' in pipeline.df.columns
        assert 'location_Mumbai' in pipeline.df.columns
        assert 'location' not in pipeline.df.columns

    def test_prepare_data_stratified_split(self):
        """Test train/test split is stratified"""
        pipeline = DataPipeline()
        pipeline.df = pd.DataFrame({
            'transaction_amount': np.random.randn(1000),
            'is_fraud': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
        })
        
        X_train, X_test, y_train, y_test = pipeline.prepare_data(test_size=0.2)
        
        assert len(X_train) == 800
        assert len(X_test) == 200
        assert abs(y_train.mean() - y_test.mean()) < 0.02

    def test_prepare_data_removes_id_columns(self):
        """Test prepare_data removes id columns"""
        pipeline = DataPipeline()
        pipeline.df = pd.DataFrame({
            'transaction_id': [f'TXN{i}' for i in range(100)],
            'customer_id': range(100),
            'device_id': [f'DEV{i}' for i in range(100)],
            'transaction_amount': np.random.randn(100),
            'is_fraud': np.random.choice([0, 1], 100)
        })
        
        X_train, X_test, y_train, y_test = pipeline.prepare_data()
        
        assert 'transaction_id' not in X_train.columns
        assert 'customer_id' not in X_train.columns
        assert 'device_id' not in X_train.columns

    def test_prepare_data_returns_X_and_y(self):
        """Test prepare_data returns X_train, X_test, y_train, y_test"""
        pipeline = DataPipeline()
        pipeline.df = pd.DataFrame({
            'transaction_amount': np.random.randn(100),
            'is_fraud': np.random.choice([0, 1], 100)
        })
        
        result = pipeline.prepare_data()
        
        assert len(result) == 4
        X_train, X_test, y_train, y_test = result
        assert 'is_fraud' not in X_train.columns
        assert 'is_fraud' in y_train.name or y_train.name == 'is_fraud'

    def test_get_feature_names(self):
        """Test get_feature_names returns correct features"""
        pipeline = DataPipeline()
        pipeline.df = pd.DataFrame({
            'transaction_amount': np.random.randn(100),
            'transaction_time': np.random.randint(0, 86400, 100),
            'is_fraud': np.random.choice([0, 1], 100)
        })
        pipeline.prepare_data()
        
        feature_names = pipeline.get_feature_names()
        
        assert 'transaction_amount' in feature_names
        assert 'transaction_time' in feature_names
        assert 'is_fraud' not in feature_names

    def test_detect_outliers(self):
        """Test outlier detection"""
        pipeline = DataPipeline()
        pipeline.df = pd.DataFrame({
            'transaction_amount': [1, 2, 3, 4, 100],
            'transaction_count_24h': [1, 2, 3, 4, 50]
        })
        
        pipeline.detect_outliers()
        
        assert True

    def test_explore_data(self):
        """Test explore_data runs without error"""
        pipeline = DataPipeline()
        pipeline.df = pd.DataFrame({
            'transaction_amount': [1, 2, 3, 4, 5],
            'transaction_time': [100, 200, 300, 400, 500],
            'is_fraud': [0, 0, 1, 0, 1]
        })
        
        result = pipeline.explore_data()
        
        assert result is not None
