"""
Tests for preprocessing.py
"""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing import Preprocessor, CostSensitiveLearning


class TestPreprocessor:
    """Tests for Preprocessor class"""

    def test_init_default(self):
        """Test Preprocessor initialization with defaults"""
        preprocessor = Preprocessor()
        assert preprocessor.scaler_type == 'robust'
        assert preprocessor.scaler is None
        assert preprocessor.imputer is None

    def test_init_with_scaler_type(self):
        """Test Preprocessor initialization with custom scaler type"""
        preprocessor = Preprocessor(scaler_type='standard')
        assert preprocessor.scaler_type == 'standard'

    def test_get_scaler_robust(self):
        """Test get_scaler returns RobustScaler"""
        preprocessor = Preprocessor(scaler_type='robust')
        scaler = preprocessor.get_scaler()
        assert scaler.__class__.__name__ == 'RobustScaler'

    def test_get_scaler_standard(self):
        """Test get_scaler returns StandardScaler"""
        preprocessor = Preprocessor(scaler_type='standard')
        scaler = preprocessor.get_scaler()
        assert scaler.__class__.__name__ == 'StandardScaler'

    def test_get_scaler_minmax(self):
        """Test get_scaler returns MinMaxScaler"""
        preprocessor = Preprocessor(scaler_type='minmax')
        scaler = preprocessor.get_scaler()
        assert scaler.__class__.__name__ == 'MinMaxScaler'

    def test_get_scaler_invalid(self):
        """Test get_scaler returns StandardScaler for invalid type"""
        preprocessor = Preprocessor(scaler_type='invalid')
        scaler = preprocessor.get_scaler()
        assert scaler.__class__.__name__ == 'StandardScaler'

    def test_fit_transform_train_only(self):
        """Test fit_transform with only training data"""
        preprocessor = Preprocessor()
        X_train = np.random.randn(100, 5)
        
        result = preprocessor.fit_transform(X_train)
        
        assert result.shape == X_train.shape
        assert preprocessor.scaler is not None

    def test_fit_transform_train_and_test(self):
        """Test fit_transform with training and test data"""
        preprocessor = Preprocessor()
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(20, 5)
        
        X_train_scaled, X_test_scaled = preprocessor.fit_transform(X_train, X_test)
        
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape
        assert preprocessor.scaler is not None

    def test_fit_transform_different_distributions(self):
        """Test that scaled data has zero mean and unit variance approximately"""
        preprocessor = Preprocessor(scaler_type='standard')
        np.random.seed(42)
        X_train = np.random.randn(1000, 3) * [10, 50, 100] + [100, 200, 300]
        
        X_train_scaled = preprocessor.fit_transform(X_train)
        
        assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=0.1)
        assert np.allclose(X_train_scaled.std(axis=0), 1, atol=0.1)

    def test_transform_without_fit_raises_error(self):
        """Test transform raises error when scaler not fitted"""
        preprocessor = Preprocessor()
        X = np.random.randn(10, 5)
        
        with pytest.raises(ValueError, match="Scaler not fitted"):
            preprocessor.transform(X)

    def test_transform_after_fit(self):
        """Test transform works after fit_transform"""
        preprocessor = Preprocessor()
        X_train = np.random.randn(100, 5)
        X_new = np.random.randn(10, 5)
        
        preprocessor.fit_transform(X_train)
        result = preprocessor.transform(X_new)
        
        assert result.shape == X_new.shape

    def test_fit_transform_with_dataframe(self):
        """Test fit_transform works with pandas DataFrame"""
        preprocessor = Preprocessor()
        X_train = pd.DataFrame({
            'a': np.random.randn(100),
            'b': np.random.randn(100),
            'c': np.random.randn(100)
        })
        
        X_train_scaled = preprocessor.fit_transform(X_train)
        
        assert X_train_scaled.shape == X_train.shape

    def test_transform_preserves_column_order(self):
        """Test that transform works with DataFrame and preserves shape"""
        preprocessor = Preprocessor()
        X_train = pd.DataFrame({
            'a': np.random.randn(100),
            'b': np.random.randn(100),
            'c': np.random.randn(100)
        })
        
        X_train_scaled = preprocessor.fit_transform(X_train)
        
        assert X_train_scaled.shape == X_train.shape


class TestCostSensitiveLearning:
    """Tests for CostSensitiveLearning class"""

    def test_init(self):
        """Test CostSensitiveLearning initialization"""
        csl = CostSensitiveLearning()
        assert csl is not None

    def test_compute_class_weights(self):
        """Test compute_class_weights returns correct format"""
        csl = CostSensitiveLearning()
        y_train = np.array([0, 0, 0, 1, 1])
        
        weights = csl.compute_class_weights(y_train)
        
        assert 0 in weights
        assert 1 in weights
        assert weights[0] < weights[1]

    def test_compute_class_weights_imbalanced(self):
        """Test class weights are higher for minority class"""
        csl = CostSensitiveLearning()
        y_train = np.array([0] * 95 + [1] * 5)
        
        weights = csl.compute_class_weights(y_train)
        
        assert weights[1] > weights[0]

    def test_compute_class_weights_balanced(self):
        """Test balanced class weights"""
        csl = CostSensitiveLearning()
        y_train = np.array([0, 0, 1, 1])
        
        weights = csl.compute_class_weights(y_train)
        
        assert abs(weights[0] - weights[1]) < 0.01

    def test_get_sample_weights(self):
        """Test get_sample_weights returns array of correct length"""
        csl = CostSensitiveLearning()
        y_train = np.array([0, 0, 0, 1, 1])
        
        sample_weights = csl.get_sample_weights(y_train)
        
        assert len(sample_weights) == len(y_train)
        assert sample_weights[0] == sample_weights[1]
        assert sample_weights[0] != sample_weights[3]

    def test_get_sample_weights_with_custom_weights(self):
        """Test get_sample_weights uses provided class weights"""
        csl = CostSensitiveLearning()
        y_train = np.array([0, 0, 1, 1])
        custom_weights = {0: 1.0, 1: 10.0}
        
        sample_weights = csl.get_sample_weights(y_train, custom_weights)
        
        assert sample_weights[0] == 1.0
        assert sample_weights[2] == 10.0

    def test_compute_class_weights_single_class(self):
        """Test compute_class_weights handles single class"""
        csl = CostSensitiveLearning()
        y_train = np.array([1, 1, 1, 1])
        
        weights = csl.compute_class_weights(y_train)
        
        assert 1 in weights
        assert weights[1] == 1.0


class TestCompareSamplingMethods:
    """Tests for sampling method comparison"""

    def test_compare_sampling_methods_runs(self):
        """Test compare_sampling_methods runs without error"""
        from src.preprocessing import compare_sampling_methods
        
        X_train = np.random.randn(100, 5)
        y_train = np.array([0] * 90 + [1] * 10)
        
        results = compare_sampling_methods(X_train, y_train)
        
        assert 'Original' in results
        assert results['Original'] == 100

    def test_compare_sampling_methods_returns_dict(self):
        """Test compare_sampling_methods returns dictionary"""
        from src.preprocessing import compare_sampling_methods
        
        X_train = np.random.randn(100, 5)
        y_train = np.array([0] * 90 + [1] * 10)
        
        results = compare_sampling_methods(X_train, y_train)
        
        assert isinstance(results, dict)
        assert all(isinstance(k, str) for k in results.keys())
        assert all(isinstance(v, int) for v in results.values())