"""
Tests for predict.py
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import torch


class TestFraudDetectorInit:
    """Tests for FraudDetector initialization"""

    def test_init_default(self):
        """Test FraudDetector initialization with defaults"""
        from src.predict import FraudDetector
        detector = FraudDetector()
        
        assert detector.model is None
        assert detector.scaler is None
        assert detector.model_path == 'models/'
        assert detector.feature_names is None

    def test_init_custom_path(self):
        """Test FraudDetector with custom model path"""
        from src.predict import FraudDetector
        detector = FraudDetector(model_path='/custom/path')
        
        assert detector.model_path == '/custom/path'

    def test_init_device_cpu(self):
        """Test FraudDetector device is set correctly"""
        from src.predict import FraudDetector
        detector = FraudDetector(device=torch.device('cpu'))
        
        assert detector.device == torch.device('cpu')


class TestFraudDetectorPreprocess:
    """Tests for FraudDetector preprocessing"""

    def test_preprocess_transaction_dict(self):
        """Test preprocess_transaction with dict input"""
        from src.predict import FraudDetector
        detector = FraudDetector()
        detector.feature_names = ['transaction_amount', 'hour_of_day', 'is_night_transaction']
        
        transaction = {
            'transaction_amount': 250.0,
            'transaction_time': 36000,
            'location': 'Mumbai',
            'device_id': 'DEV001',
            'merchant_category': 'retail'
        }
        
        result = detector.preprocess_transaction(transaction)
        
        assert isinstance(result, pd.DataFrame)
        assert 'transaction_amount' in result.columns

    def test_preprocess_transaction_dataframe(self):
        """Test preprocess_transaction with DataFrame input"""
        from src.predict import FraudDetector
        detector = FraudDetector()
        detector.feature_names = ['transaction_amount']
        
        df = pd.DataFrame([{'transaction_amount': 250.0}])
        result = detector.preprocess_transaction(df)
        
        assert isinstance(result, pd.DataFrame)

    def test_preprocess_creates_hour_of_day(self):
        """Test preprocessing creates hour_of_day feature"""
        from src.predict import FraudDetector
        detector = FraudDetector()
        detector.feature_names = ['hour_of_day']
        
        transaction = {'transaction_time': 43200}
        result = detector.preprocess_transaction(transaction)
        
        assert 'hour_of_day' in result.columns

    def test_preprocess_creates_night_transaction(self):
        """Test preprocessing creates is_night_transaction flag"""
        from src.predict import FraudDetector
        detector = FraudDetector()
        detector.feature_names = ['is_night_transaction']
        
        transaction = {'transaction_time': 79200}
        result = detector.preprocess_transaction(transaction)
        
        assert result['is_night_transaction'].iloc[0] == 1

    def test_preprocess_amount_to_avg_ratio(self):
        """Test preprocessing creates amount_to_avg_ratio"""
        from src.predict import FraudDetector
        detector = FraudDetector()
        detector.feature_names = ['amount_to_avg_ratio']
        
        transaction = {
            'transaction_amount': 300.0,
            'avg_transaction_amount': 150.0
        }
        result = detector.preprocess_transaction(transaction)
        
        assert 'amount_to_avg_ratio' in result.columns
        assert result['amount_to_avg_ratio'].iloc[0] == pytest.approx(2.0, rel=0.1)

    def test_preprocess_high_amount_flag(self):
        """Test preprocessing creates high_amount flag"""
        from src.predict import FraudDetector
        detector = FraudDetector()
        detector.feature_names = ['high_amount', 'amount_to_avg_ratio']
        
        transaction = {'transaction_amount': 600.0, 'avg_transaction_amount': 150.0}
        result = detector.preprocess_transaction(transaction)
        
        assert result['high_amount'].iloc[0] == 1

    def test_preprocess_unusual_merchant(self):
        """Test preprocessing creates unusual_merchant flag"""
        from src.predict import FraudDetector
        detector = FraudDetector()
        detector.feature_names = ['unusual_merchant']
        
        transaction = {'merchant_category': 'online'}
        result = detector.preprocess_transaction(transaction)
        
        assert result['unusual_merchant'].iloc[0] == 1

    def test_preprocess_new_customer_flag(self):
        """Test preprocessing creates new_customer flag"""
        from src.predict import FraudDetector
        detector = FraudDetector()
        detector.feature_names = ['new_customer']
        
        transaction = {'account_age_days': 30}
        result = detector.preprocess_transaction(transaction)
        
        assert result['new_customer'].iloc[0] == 1

    def test_preprocess_high_frequency_flag(self):
        """Test preprocessing creates high_frequency flag"""
        from src.predict import FraudDetector
        detector = FraudDetector()
        detector.feature_names = ['high_frequency']
        
        transaction = {'transaction_count_24h': 15}
        result = detector.preprocess_transaction(transaction)
        
        assert result['high_frequency'].iloc[0] == 1

    def test_preprocess_aligns_features(self):
        """Test preprocessing aligns with feature_names"""
        from src.predict import FraudDetector
        detector = FraudDetector()
        detector.feature_names = ['transaction_amount', 'hour_of_day', 'new_feature']
        
        transaction = {'transaction_amount': 250.0, 'transaction_time': 36000}
        result = detector.preprocess_transaction(transaction)
        
        assert list(result.columns) == ['transaction_amount', 'hour_of_day', 'new_feature']
        assert result['new_feature'].iloc[0] == 0


class TestFraudDetectorPredict:
    """Tests for FraudDetector prediction"""

    def test_predict_raises_error_without_model(self):
        """Test predict raises error when model not loaded"""
        from src.predict import FraudDetector
        detector = FraudDetector()
        
        with pytest.raises(ValueError, match="Model not loaded"):
            detector.predict({'transaction_amount': 250.0})

    def test_predict_without_scaler(self):
        """Test predict works without scaler (uses raw values)"""
        from src.predict import FraudDetector
        from src.pytorch_model import FraudDetectorNet
        
        detector = FraudDetector()
        detector.model = FraudDetectorNet(input_dim=1)
        detector.model.eval()
        detector.scaler = None
        
        with torch.no_grad():
            result = detector.predict({'transaction_amount': 250.0})
        
        assert 'is_fraud' in result
        assert 'fraud_probability' in result
        assert 'risk_level' in result

    def test_predict_returns_correct_structure(self):
        """Test predict returns correct result structure"""
        from src.predict import FraudDetector
        from src.pytorch_model import FraudDetectorNet
        
        detector = FraudDetector()
        detector.model = FraudDetectorNet(input_dim=1)
        detector.model.eval()
        detector.scaler = None
        
        result = detector.predict({'transaction_amount': 250.0})
        
        assert isinstance(result['is_fraud'], bool)
        assert isinstance(result['fraud_probability'], float)
        assert result['risk_level'] in ['LOW', 'MEDIUM', 'HIGH']

    def test_predict_threshold(self):
        """Test predict respects threshold parameter"""
        from src.predict import FraudDetector
        from src.pytorch_model import FraudDetectorNet
        
        detector = FraudDetector()
        detector.model = FraudDetectorNet(input_dim=1)
        detector.model.eval()
        detector.scaler = None
        
        result_high_threshold = detector.predict(
            {'transaction_amount': 250.0}, 
            threshold=0.99
        )
        
        assert result_high_threshold['threshold'] == 0.99

    def test_predict_batch(self):
        """Test predict_batch returns list of results"""
        from src.predict import FraudDetector
        from src.pytorch_model import FraudDetectorNet
        
        detector = FraudDetector()
        detector.model = FraudDetectorNet(input_dim=1)
        detector.model.eval()
        detector.scaler = None
        
        transactions = [
            {'transaction_amount': 100.0},
            {'transaction_amount': 200.0},
            {'transaction_amount': 300.0}
        ]
        
        results = detector.predict_batch(transactions)
        
        assert len(results) == 3
        assert all('is_fraud' in r for r in results)


class TestGetRiskLevel:
    """Tests for risk level categorization"""

    def test_risk_level_high(self):
        """Test HIGH risk for probability >= 0.7"""
        from src.predict import FraudDetector
        detector = FraudDetector()
        
        assert detector._get_risk_level(0.7) == 'HIGH'
        assert detector._get_risk_level(0.9) == 'HIGH'

    def test_risk_level_medium(self):
        """Test MEDIUM risk for probability >= 0.4 and < 0.7"""
        from src.predict import FraudDetector
        detector = FraudDetector()
        
        assert detector._get_risk_level(0.4) == 'MEDIUM'
        assert detector._get_risk_level(0.6) == 'MEDIUM'

    def test_risk_level_low(self):
        """Test LOW risk for probability < 0.4"""
        from src.predict import FraudDetector
        detector = FraudDetector()
        
        assert detector._get_risk_level(0.0) == 'LOW'
        assert detector._get_risk_level(0.3) == 'LOW'
        assert detector._get_risk_level(0.39) == 'LOW'


class TestLoadModel:
    """Tests for model loading"""

    def test_load_model_file_not_found(self):
        """Test load_model raises error when file not found"""
        from src.predict import FraudDetector
        detector = FraudDetector(model_path='/nonexistent/path')
        
        with pytest.raises(FileNotFoundError):
            detector.load_model()

    def test_load_scaler_file_not_found(self):
        """Test load_scaler handles missing file"""
        from src.predict import FraudDetector
        detector = FraudDetector(model_path='/nonexistent/path')
        
        result = detector.load_scaler()
        
        assert result is None
        assert detector.scaler is None

    def test_load_feature_names_file_not_found(self):
        """Test load_feature_names handles missing file"""
        from src.predict import FraudDetector
        detector = FraudDetector(model_path='/nonexistent/path')
        
        result = detector.load_feature_names()
        
        assert result is None
        assert detector.feature_names is None


class TestCreateSampleTransaction:
    """Tests for sample transaction creation"""

    def test_create_sample_transaction(self):
        """Test create_sample_transaction returns valid dict"""
        from src.predict import create_sample_transaction
        
        sample = create_sample_transaction()
        
        assert isinstance(sample, dict)
        assert 'transaction_amount' in sample
        assert 'transaction_time' in sample
        assert 'location' in sample
        assert 'merchant_category' in sample

    def test_sample_transaction_valid_values(self):
        """Test sample transaction has valid values"""
        from src.predict import create_sample_transaction
        
        sample = create_sample_transaction()
        
        assert sample['transaction_amount'] > 0
        assert 0 <= sample['transaction_time'] <= 86399
        assert sample['account_age_days'] >= 0
        assert sample['transaction_count_24h'] >= 0
        assert sample['avg_transaction_amount'] > 0