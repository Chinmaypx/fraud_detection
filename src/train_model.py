"""
Model Training Module for Fraud Detection (PyTorch)
Trains deep neural network for fraud detection with training metrics tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import json
import pickle
from datetime import datetime

from src.pytorch_model import (
    FraudDetectorNet,
    get_class_weights, create_data_loaders
)


class PyTorchTrainer:
    """
    Train PyTorch fraud detection models with comprehensive metrics tracking
    """
    
    def __init__(self, input_dim, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.model = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_recall': [],
            'val_recall': [],
            'train_precision': [],
            'val_precision': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rates': [],
            'epoch_times': [],
        }
        self.best_model_state = None
        self.best_val_f1 = 0
        
    def build_model(self, dropout_rate=0.3):
        """Build the neural network model"""
        self.model = FraudDetectorNet(self.input_dim, dropout_rate).to(self.device)
        print(f"\nModel architecture:")
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Device: {self.device}")
        return self.model
    
    def _compute_metrics(self, y_true, y_pred):
        """Compute precision, recall, F1 for binary classification"""
        tp = ((y_pred == 1) & (y_true == 1)).sum().item()
        fp = ((y_pred == 1) & (y_true == 0)).sum().item()
        fn = ((y_pred == 0) & (y_true == 1)).sum().item()
        tn = ((y_pred == 0) & (y_true == 0)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        return accuracy, precision, recall, f1
    
    def train(self, train_loader, val_loader, y_train,
              epochs=50, learning_rate=0.001, weight_decay=1e-5,
              patience=10):
        """
        Train the model with early stopping and learning rate scheduling
        
        Args:
            train_loader: PyTorch DataLoader for training
            val_loader: PyTorch DataLoader for validation
            y_train: training labels (for class weight computation)
            epochs: maximum number of epochs
            learning_rate: initial learning rate
            weight_decay: L2 regularization
            patience: early stopping patience
        """
        if self.model is None:
            self.build_model()

        # Compute class weights for imbalanced data
        pos_weight = get_class_weights(
            y_train if isinstance(y_train, np.ndarray) else y_train.values
        ).to(self.device)

        criterion = nn.BCELoss(reduction='none')
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )

        print("\n" + "=" * 60)
        print("TRAINING PYTORCH FRAUD DETECTOR")
        print("=" * 60)
        print(f"Epochs: {epochs}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Device: {self.device}")
        print(f"Fraud class weight: {pos_weight.item():.2f}")
        
        best_val_f1 = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_start = datetime.now()
            
            # --- Training Phase ---
            self.model.train()
            train_loss = 0
            all_train_preds = []
            all_train_labels = []
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                # Apply class weights manually
                weights = torch.where(batch_y == 1, pos_weight, torch.ones_like(batch_y))
                loss = (criterion(outputs, batch_y) * weights).mean()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                preds = (outputs >= 0.5).float()
                all_train_preds.append(preds.cpu())
                all_train_labels.append(batch_y.cpu())
            
            train_loss /= len(train_loader.dataset)
            all_train_preds = torch.cat(all_train_preds)
            all_train_labels = torch.cat(all_train_labels)
            train_acc, train_prec, train_rec, train_f1 = self._compute_metrics(
                all_train_labels, all_train_preds
            )
            
            # --- Validation Phase ---
            self.model.eval()
            val_loss = 0
            all_val_preds = []
            all_val_labels = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    weights = torch.where(batch_y == 1, pos_weight, torch.ones_like(batch_y))
                    loss = (criterion(outputs, batch_y) * weights).mean()
                    
                    val_loss += loss.item() * batch_X.size(0)
                    preds = (outputs >= 0.5).float()
                    all_val_preds.append(preds.cpu())
                    all_val_labels.append(batch_y.cpu())
            
            val_loss /= len(val_loader.dataset)
            all_val_preds = torch.cat(all_val_preds)
            all_val_labels = torch.cat(all_val_labels)
            val_acc, val_prec, val_rec, val_f1 = self._compute_metrics(
                all_val_labels, all_val_preds
            )
            
            # Learning rate scheduling
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_f1)
            
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            
            # Store metrics
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['train_recall'].append(train_rec)
            self.training_history['val_recall'].append(val_rec)
            self.training_history['train_precision'].append(train_prec)
            self.training_history['val_precision'].append(val_prec)
            self.training_history['train_f1'].append(train_f1)
            self.training_history['val_f1'].append(val_f1)
            self.training_history['learning_rates'].append(current_lr)
            self.training_history['epoch_times'].append(epoch_time)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"\nEpoch [{epoch+1}/{epochs}] ({epoch_time:.1f}s)")
                print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"  Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
                print(f"  Train Recall: {train_rec:.4f} | Val Recall: {val_rec:.4f}")
                print(f"  LR: {current_lr:.6f}")
            
            # Early stopping + save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.best_val_f1 = val_f1
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}. Best Val F1: {best_val_f1:.4f}")
                break
        
        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        print(f"\nTraining complete! Best Val F1: {best_val_f1:.4f}")
        return self.training_history
    
    def save_model(self, filepath='models/'):
        """Save trained model and training history"""
        os.makedirs(filepath, exist_ok=True)
        
        # Save PyTorch model
        model_path = os.path.join(filepath, 'fraud_detector.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'architecture': 'FraudDetectorNet',
        }, model_path)
        print(f"Model saved to {model_path}")
        
        # Save training history
        history_path = os.path.join(filepath, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Training history saved to {history_path}")
        
        return model_path
    
    def load_model(self, filepath='models/fraud_detector.pt'):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.input_dim = checkpoint['input_dim']
        self.model = FraudDetectorNet(self.input_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded from {filepath}")
        return self.model
    
    def predict(self, X, threshold=0.5):
        """Get predictions and probabilities"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.FloatTensor(X).to(self.device)
            else:
                X_tensor = X.to(self.device)
            
            probabilities = self.model(X_tensor).cpu().numpy()
            predictions = (probabilities >= threshold).astype(int)
        
        return predictions, probabilities


def main():
    """Main training pipeline"""
    from src.data_pipeline import main as data_main
    from src.preprocessing import Preprocessor
    
    print("=" * 60)
    print("FRAUD DETECTION - PyTorch Training Pipeline")
    print("=" * 60)
    
    # Step 1: Data Pipeline
    pipeline, X_train, X_test, y_train, y_test = data_main()
    
    # Step 2: Preprocessing
    preprocessor = Preprocessor()
    X_train_scaled, X_test_scaled = preprocessor.fit_transform(X_train, X_test)
    
    # Save scaler
    os.makedirs('models', exist_ok=True)
    pickle.dump(preprocessor.scaler, open('models/scaler.pkl', 'wb'))
    
    # Save feature names
    feature_names = list(X_train.columns)
    with open('models/feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    
    # Step 3: Create data loaders
    input_dim = X_train_scaled.shape[1]
    train_loader, test_loader = create_data_loaders(
        X_train_scaled, y_train.values, 
        X_test_scaled, y_test.values,
        batch_size=512
    )
    
    # Step 4: Train model
    trainer = PyTorchTrainer(input_dim)
    trainer.build_model(dropout_rate=0.3)
    trainer.train(
        train_loader, test_loader, y_train,
        epochs=50, learning_rate=0.001, patience=10
    )
    
    # Step 5: Save model
    trainer.save_model('models/')
    
    # Step 6: Final evaluation
    predictions, probabilities = trainer.predict(X_test_scaled)
    
    from src.evaluate import ModelEvaluator
    evaluator = ModelEvaluator(y_test.values, predictions, probabilities, 'PyTorch Neural Network')
    evaluator.print_summary()
    
    # Save evaluation metrics
    metrics = evaluator.calculate_metrics()
    metrics_path = 'models/eval_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print(f"\nEvaluation metrics saved to {metrics_path}")
    
    print("\n✅ Training pipeline complete!")
    return trainer


if __name__ == "__main__":
    main()
