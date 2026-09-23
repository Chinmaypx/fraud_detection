"""
PyTorch Neural Network Models for Fraud Detection
Deep learning approach for detecting fraudulent banking transactions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


class FraudDataset(Dataset):
    """
    Custom PyTorch Dataset for fraud detection data
    """
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X if isinstance(X, np.ndarray) else X.values)
        self.y = torch.FloatTensor(y if isinstance(y, np.ndarray) else y.values)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]


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
        return self.network(x).squeeze(-1)


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


class FraudLSTMNet(nn.Module):
    """
    LSTM-based Fraud Detection Model for sequential transaction data

    Architecture:
    - Input → LSTM (2 layers, hidden_dim=64, bidirectional)
    - Last hidden state → 64 (ReLU + Dropout)
    - 64 → 32 (ReLU + Dropout)
    - 32 → 1 (Sigmoid)

    Why LSTM for fraud detection:
    - Captures temporal patterns in customer transaction history
    - Learns velocity spikes, location hopping, and spending progressions
    - Bidirectional processing sees both past context and future context
      within the sequence window
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout_rate=0.3):
        super(FraudLSTMNet, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer to capture temporal dependencies
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Dense classification head (hidden_dim * 2 for bidirectional)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),

            nn.Linear(32, 1),
            
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the output of the last time step
        # lstm_out shape: (batch, seq_len, hidden_dim * 2)
        last_step_out = lstm_out[:, -1, :]

        out = self.classifier(last_step_out)
        return out.squeeze(-1)


class SequenceFraudDataset(Dataset):
    """
    Dataset that builds per-customer sliding-window sequences from tabular data.

    For each transaction, it looks back at the previous `seq_length - 1`
    transactions from the same customer (sorted by time) to form a sequence.
    Sequences shorter than seq_length are zero-padded on the left.
    """
    def __init__(self, df, feature_cols, target_col='is_fraud', seq_length=10,
                 history_df=None):
        self.seq_length = seq_length
        self.sequences, self.labels = self._create_sequences(
            df, feature_cols, target_col, history_df
        )

    def _create_sequences(self, df, feature_cols, target_col, history_df=None):
        sequences = []
        labels = []
        timestamp_col = 'transaction_timestamp'
        if timestamp_col not in df.columns:
            raise ValueError(
                'LSTM sequences require transaction_timestamp with full date and time; '
                'transaction_time is only time-of-day and cannot establish chronology.'
            )

        targets = df.copy()
        targets['_sequence_target'] = True
        if history_df is not None and len(history_df):
            history = history_df.copy()
            history['_sequence_target'] = False
            combined = pd.concat([history, targets], ignore_index=True)
        else:
            combined = targets
        combined[timestamp_col] = pd.to_datetime(combined[timestamp_col], errors='raise')
        combined = combined.sort_values(
            ['customer_id', timestamp_col], kind='mergesort'
        )

        for _, group in combined.groupby('customer_id', sort=False):
            features = group[feature_cols].to_numpy()
            is_target = group['_sequence_target'].to_numpy()
            label_values = group[target_col].to_numpy()
            for i in np.flatnonzero(is_target):
                start_idx = max(0, i - self.seq_length + 1)
                seq = features[start_idx:i + 1]
                if len(seq) < self.seq_length:
                    pad = np.zeros((self.seq_length - len(seq), features.shape[1]))
                    seq = np.vstack([pad, seq])
                sequences.append(seq)
                labels.append(label_values[i])

        return (
            torch.FloatTensor(np.array(sequences)),
            torch.FloatTensor(np.array(labels)),
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]


def create_sequence_data_loaders(
    train_df, test_df, feature_cols, target_col='is_fraud',
    seq_length=10, batch_size=512
):
    """
    Create PyTorch DataLoaders for LSTM sequence training and testing.

    Args:
        train_df: Training DataFrame (must contain customer_id & transaction_time)
        test_df:  Test DataFrame
        feature_cols: List of feature column names (numeric, already scaled)
        target_col: Name of the target column
        seq_length: Number of past transactions per sequence
        batch_size: Batch size for DataLoaders

    Returns:
        (train_loader, test_loader, input_dim)
    """
    train_dataset = SequenceFraudDataset(
        train_df, feature_cols, target_col, seq_length
    )
    test_dataset = SequenceFraudDataset(
        test_df, feature_cols, target_col, seq_length
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    input_dim = len(feature_cols)
    return train_loader, test_loader, input_dim


def create_chronological_sequence_data_loaders(
    train_df, val_df, test_df, feature_cols, target_col='is_fraud',
    seq_length=10, batch_size=512
):
    """Build ordered sequence datasets; later partitions may use only past context."""
    train_dataset = SequenceFraudDataset(
        train_df, feature_cols, target_col, seq_length
    )
    val_dataset = SequenceFraudDataset(
        val_df, feature_cols, target_col, seq_length, history_df=train_df
    )
    test_history = pd.concat([train_df, val_df], ignore_index=True)
    test_dataset = SequenceFraudDataset(
        test_df, feature_cols, target_col, seq_length, history_df=test_history
    )

    def loader(dataset, shuffle):
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=0, pin_memory=True,
        )

    return (
        loader(train_dataset, True), loader(val_dataset, False),
        loader(test_dataset, False), len(feature_cols)
    )


def get_class_weights(y_train):
    """
    Compute class weights for imbalanced dataset
    
    In fraud detection, we want to heavily penalize missing fraud (false negatives)
    """
    n_samples = len(y_train)
    n_fraud = sum(y_train == 1) if isinstance(y_train, np.ndarray) else y_train.sum()
    n_legit = n_samples - n_fraud
    
    # Weight inversely proportional to class frequency
    weight_legit = n_samples / (2.0 * n_legit)
    weight_fraud = n_samples / (2.0 * n_fraud)
    
    return torch.FloatTensor([weight_fraud / weight_legit])


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
