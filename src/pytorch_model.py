"""
PyTorch Neural Network Models for Fraud Detection
Deep learning approach for detecting fraudulent banking transactions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class FraudDataset(Dataset):
    """
    Custom PyTorch Dataset for fraud detection data
    """
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X if isinstance(X, np.ndarray) else X.values)
        self.y = torch.FloatTensor(y if isinstance(y, np.ndarray) else y.values)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FraudDetectorNet(nn.Module):
    """
    Deep Neural Network for Fraud Detection
    
    Architecture:
    - Input → 128 (ReLU + BatchNorm + Dropout)
    - 128 → 256 (ReLU + BatchNorm + Dropout)  
    - 256 → 128 (ReLU + BatchNorm + Dropout)
    - 128 → 64 (ReLU + BatchNorm + Dropout)
    - 64 → 32 (ReLU + Dropout)
    - 32 → 1 (Sigmoid)
    
    Why this architecture:
    - Expanding then contracting layers help learn complex feature interactions
    - BatchNorm stabilizes training and allows higher learning rates
    - Dropout prevents overfitting on the majority class
    - Sigmoid output gives fraud probability directly
    """
    def __init__(self, input_dim, dropout_rate=0.3):
        super(FraudDetectorNet, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1: Input → 128
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 2: 128 → 256 (expand to learn more features)
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 3: 256 → 128 (contract)
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 4: 128 → 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            # Layer 5: 64 → 32
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            # Output: 32 → 1
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x).squeeze()


class AutoencoderDetector(nn.Module):
    """
    Autoencoder-based Anomaly Detector
    
    Trained on legitimate transactions only.
    High reconstruction error = potential fraud.
    
    This is a complementary approach:
    - Learns the 'normal' pattern of legitimate transactions
    - Fraud transactions deviate from normal → higher reconstruction error
    """
    def __init__(self, input_dim, encoding_dim=16):
        super(AutoencoderDetector, self).__init__()
        
        # Encoder: compress input to latent representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder: reconstruct from latent representation
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_reconstruction_error(self, x):
        """Get reconstruction error for anomaly detection"""
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error


def get_class_weights(y_train):
    """
    Compute class weights for imbalanced dataset

    In fraud detection, we want to heavily penalize missing fraud (false negatives)
    """
    n_samples = len(y_train)
    n_fraud = int(sum(y_train == 1)) if isinstance(y_train, np.ndarray) else int(y_train.sum())
    n_legit = n_samples - n_fraud

    # Weight inversely proportional to class frequency
    weight_legit = n_samples / (2.0 * n_legit)
    weight_fraud = n_samples / (2.0 * n_fraud)

    # Return scalar tensor for proper broadcasting with torch.where
    return torch.FloatTensor([weight_fraud / weight_legit])[0]


def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=512):
    """
    Create PyTorch DataLoaders for training and testing
    """
    train_dataset = FraudDataset(X_train, y_train)
    test_dataset = FraudDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, test_loader
