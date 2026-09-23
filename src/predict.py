"""
Prediction Module for Fraud Detection (PyTorch)
Handles loading PyTorch models and making predictions on new transactions
"""

import torch
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path

from .pytorch_model import FraudDetectorNet

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / 'models'


class FraudDetector:
    """
    Main prediction class for PyTorch-based fraud detection
    """
    
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.model_path = model_path if model_path is not None else str(DEFAULT_MODEL_DIR)
        self.feature_names = None
        
    def load_model(self, model_name='fraud_detector.pt'):
        """Load trained PyTorch model from disk"""
        model_file = Path(self.model_path) / model_name
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        checkpoint = torch.load(model_file, map_location=self.device, weights_only=True)
        input_dim = checkpoint['input_dim']
        
        self.model = FraudDetectorNet(input_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"PyTorch model loaded: {model_name} (input_dim={input_dim})")
        return self.model
    
    def load_scaler(self, scaler_name=None):
        """Load fitted scaler from disk"""
        scaler_name = scaler_name or 'scaler_mlp.pkl'
        scaler_file = Path(self.model_path) / scaler_name
        # Keep old checkpoints usable while preferring the model-specific artifact.
        if not scaler_file.exists() and scaler_name != 'scaler.pkl':
            scaler_file = Path(self.model_path) / 'scaler.pkl'
        
        if scaler_file.exists():
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Scaler loaded: {scaler_name}")
        
        return self.scaler
    
    def load_feature_names(self, filename='feature_names.json'):
        """Load feature names used during training"""
        feature_file = Path(self.model_path) / filename
        
        if feature_file.exists():
            with open(feature_file, 'r') as f:
                self.feature_names = json.load(f)
            print(f"Feature names loaded: {len(self.feature_names)} features")
        
        return self.feature_names
    
    def preprocess_transaction(self, transaction_data):
        """
        Preprocess a single transaction for prediction
        
        Args:
            transaction_data: dict with transaction features
            
        Returns:
            Preprocessed features ready for model
        """
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = transaction_data.copy()
        
        # Feature engineering (same as training pipeline)
        if 'transaction_time' in df.columns:
            df['hour_of_day'] = df['transaction_time'] // 3600
            df['is_night_transaction'] = ((df['hour_of_day'] >= 22) | 
                                           (df['hour_of_day'] <= 5)).astype(int)
        
        if 'transaction_amount' in df.columns and 'avg_transaction_amount' in df.columns:
            df['amount_to_avg_ratio'] = (
                df['transaction_amount'] / (df['avg_transaction_amount'] + 1)
            )
            df['high_amount'] = (df['transaction_amount'] > 500).astype(int)
        
        if 'merchant_category' in df.columns:
            df['unusual_merchant'] = (df['merchant_category'] == 'online').astype(int)
        
        if 'account_age_days' in df.columns:
            df['new_customer'] = (df['account_age_days'] < 90).astype(int)
        
        if 'transaction_count_24h' in df.columns:
            df['high_frequency'] = (df['transaction_count_24h'] > 10).astype(int)
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['transaction_id', 'customer_id', 'device_id']:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)
        
        # Remove non-feature columns
        drop_cols = ['transaction_id', 'customer_id', 'device_id', 'is_fraud']
        for col in drop_cols:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
        
        # Align with training features
        if self.feature_names is not None:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]
        
        return df
    
    def predict(self, transaction_data, threshold=0.5):
        """
        Predict if transaction is fraudulent using PyTorch model
        
        Args:
            transaction_data: dict or DataFrame with transaction features
            threshold: classification threshold (default 0.5)
            
        Returns:
            dict with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")
        
        features = self.preprocess_transaction(transaction_data)
        
        # Scale features
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features.values
        
        # Convert to tensor and predict
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(features_scaled).to(self.device)
            fraud_probability = self.model(X_tensor).cpu().numpy()
            
            if fraud_probability.ndim == 0:
                fraud_probability = float(fraud_probability)
            else:
                fraud_probability = float(fraud_probability[0])
        
        is_fraud = fraud_probability >= threshold
        
        result = {
            'is_fraud': bool(is_fraud),
            'fraud_probability': fraud_probability,
            'threshold': threshold,
            'risk_level': self._get_risk_level(fraud_probability)
        }
        
        return result
    
    def _get_risk_level(self, probability):
        """Categorize risk level based on fraud probability"""
        if probability >= 0.7:
            return 'HIGH'
        elif probability >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'

    def predict_batch(self, transactions, threshold=0.5):
        """Predict fraud for multiple transactions."""
        return [self.predict(transaction, threshold) for transaction in transactions]


class IEECISFraudDetector:
    """Inference wrapper for the IEEE-CIS MLP and isolated preprocessing."""

    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path if model_path is not None else str(DEFAULT_MODEL_DIR)
        self.model = None
        self.threshold = None
        from .ieee_cis_dataset import IEECISDataset
        self.adapter = IEECISDataset()

    def load_model(self, model_name='fraud_detector_ieee.pt'):
        from .ieee_cis_dataset import IEEE_FEATURES
        model_file = Path(self.model_path) / model_name
        if not model_file.is_file():
            raise FileNotFoundError(f"IEEE-CIS model file not found: {model_file}")
        checkpoint = torch.load(model_file, map_location=self.device, weights_only=True)
        if checkpoint.get('input_dim') != len(IEEE_FEATURES):
            raise ValueError('IEEE-CIS checkpoint has an incompatible feature count.')
        if checkpoint.get('feature_names') != IEEE_FEATURES:
            raise ValueError('IEEE-CIS checkpoint has an incompatible feature order.')
        threshold = checkpoint.get('threshold')
        if threshold is None or not np.isfinite(float(threshold)) or not 0 <= float(threshold) <= 1:
            raise ValueError('IEEE-CIS checkpoint has no valid decision threshold.')
        self.model = FraudDetectorNet(len(IEEE_FEATURES)).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.threshold = float(threshold)
        return self.model

    def load_preprocessing(self):
        return self.adapter.load_preprocessing(self.model_path)

    @property
    def feature_names(self):
        return self.adapter.feature_names

    @property
    def category_values(self):
        return self.adapter.category_values or {}

    def validate_categories(self, transaction_data):
        self.adapter.validate_categories(transaction_data)

    def predict(self, transaction_data):
        if self.model is None or self.threshold is None:
            raise ValueError('IEEE-CIS model not loaded.')
        frame = pd.DataFrame([transaction_data]) if isinstance(transaction_data, dict) else transaction_data.copy()
        scaled = self.adapter.transform(frame)
        tensor = torch.as_tensor(
            np.array(scaled.to_numpy(dtype=np.float32), copy=True), device=self.device
        )
        self.model.eval()
        with torch.no_grad():
            # FraudDetectorNet includes sigmoid, so its output is already a probability.
            probability = float(self.model(tensor).reshape(-1)[0].item())
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError('IEEE-CIS model returned an invalid fraud probability.')
        return {
            'model_name': 'fraud_detector_ieee.pt',
            'is_fraud': probability >= self.threshold,
            'predicted_class': int(probability >= self.threshold),
            'fraud_probability': probability,
            'threshold': self.threshold,
            'risk_level': 'HIGH' if probability >= self.threshold else 'LOW',
        }

def create_sample_transaction():
    """Create a sample transaction for testing"""
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


def test_predictions():
    """Test the fraud detection system"""
    detector = FraudDetector()
    
    try:
        detector.load_model()
        detector.load_scaler()
        detector.load_feature_names()
    except FileNotFoundError:
        print("No trained model found. Please train a model first.")
        return
    
    sample = create_sample_transaction()
    result = detector.predict(sample)
    
    print("\n--- Test Prediction ---")
    print(f"Transaction: {sample}")
    print(f"Result: {result}")


class LSTMFraudDetector:
    """
    Prediction class for the LSTM-based fraud detection model.

    For single-transaction prediction, builds a zero-padded sequence of
    length `seq_length` with the transaction as the last time step.
    """

    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.model_path = model_path if model_path is not None else str(DEFAULT_MODEL_DIR)
        self.feature_names = None
        self.seq_length = 10  # default, overridden on load

    def load_model(self, model_name='fraud_detector_lstm.pt'):
        """Load trained LSTM model from disk"""
        from .pytorch_model import FraudLSTMNet

        model_file = Path(self.model_path) / model_name

        if not model_file.exists():
            raise FileNotFoundError(f"LSTM model file not found: {model_file}")

        checkpoint = torch.load(model_file, map_location=self.device, weights_only=True)
        input_dim = checkpoint['input_dim']
        self.seq_length = checkpoint.get('seq_length', 10)

        self.model = FraudLSTMNet(input_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"LSTM model loaded: {model_name} (input_dim={input_dim}, seq_length={self.seq_length})")
        return self.model

    def load_scaler(self, scaler_name=None):
        """Load fitted scaler from disk"""
        scaler_name = scaler_name or 'scaler_lstm.pkl'
        scaler_file = Path(self.model_path) / scaler_name
        if not scaler_file.exists() and scaler_name != 'scaler.pkl':
            scaler_file = Path(self.model_path) / 'scaler.pkl'

        if scaler_file.exists():
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Scaler loaded: {scaler_name}")

        return self.scaler

    def load_feature_names(self, filename='feature_names_lstm.json'):
        """Load feature names used during training"""
        feature_file = Path(self.model_path) / filename
        if not feature_file.exists():
            feature_file = Path(self.model_path) / 'feature_names.json'

        if feature_file.exists():
            with open(feature_file, 'r') as f:
                self.feature_names = json.load(f)
            print(f"Feature names loaded: {len(self.feature_names)} features")

        return self.feature_names

    def preprocess_transaction(self, transaction_data):
        """
        Preprocess a single transaction for LSTM prediction.
        Reuses the same feature engineering as FraudDetector.
        """
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = transaction_data.copy()

        # Feature engineering (same as training pipeline)
        if 'transaction_time' in df.columns:
            df['hour_of_day'] = df['transaction_time'] // 3600
            df['is_night_transaction'] = ((df['hour_of_day'] >= 22) |
                                           (df['hour_of_day'] <= 5)).astype(int)

        if 'transaction_amount' in df.columns and 'avg_transaction_amount' in df.columns:
            df['amount_to_avg_ratio'] = (
                df['transaction_amount'] / (df['avg_transaction_amount'] + 1)
            )
            df['high_amount'] = (df['transaction_amount'] > 500).astype(int)

        if 'merchant_category' in df.columns:
            df['unusual_merchant'] = (df['merchant_category'] == 'online').astype(int)

        if 'account_age_days' in df.columns:
            df['new_customer'] = (df['account_age_days'] < 90).astype(int)

        if 'transaction_count_24h' in df.columns:
            df['high_frequency'] = (df['transaction_count_24h'] > 10).astype(int)

        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['transaction_id', 'customer_id', 'device_id']:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)

        # Remove non-feature columns
        drop_cols = ['transaction_id', 'customer_id', 'device_id', 'is_fraud']
        for col in drop_cols:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)

        # Align with training features
        if self.feature_names is not None:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]

        return df

    def predict(self, transaction_data, threshold=0.5):
        """
        Predict if a transaction is fraudulent using the LSTM model.

        For single-transaction inference, creates a padded sequence
        of shape (1, seq_length, n_features) with the transaction
        as the final time step.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")

        features = self.preprocess_transaction(transaction_data)

        # Scale features
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features.values

        # Build padded sequence: (1, seq_length, n_features)
        n_features = features_scaled.shape[1]
        seq = np.zeros((self.seq_length, n_features))
        seq[-1, :] = features_scaled[0]  # place transaction at last step

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
            logits = self.model(X_tensor)
            fraud_probability = torch.sigmoid(logits).cpu().numpy()

            if fraud_probability.ndim == 0:
                fraud_probability = float(fraud_probability)
            else:
                fraud_probability = float(fraud_probability[0])

        is_fraud = fraud_probability >= threshold

        result = {
            'is_fraud': bool(is_fraud),
            'fraud_probability': fraud_probability,
            'threshold': threshold,
            'risk_level': self._get_risk_level(fraud_probability),
        }

        return result

    def _get_risk_level(self, probability):
        """Categorize risk level based on fraud probability"""
        if probability >= 0.7:
            return 'HIGH'
        elif probability >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'

if __name__ == "__main__":
    test_predictions()
