"""Explicit command-line training/evaluation for the ULB benchmark.

Example:
    python -m src.train_ulb --data-path C:\\Users\\chinm\\Downloads\\creditcard.csv

This path is local to the command line; no API endpoint accepts a filesystem path.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve

from .evaluate import ModelEvaluator
from .pytorch_model import create_data_loaders
from .train_model import PyTorchTrainer
from .ulb_dataset import ULBCreditCardDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / 'models'


def select_f1_threshold(y_true, probabilities):
    """Choose the threshold maximizing F1 on validation data only."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        raise ValueError('Validation data does not support threshold selection.')
    f1_values = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    best_index = int(np.argmax(f1_values))
    threshold = float(thresholds[best_index])
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return threshold, {
        'threshold': threshold,
        'threshold_method': 'maximum validation F1',
        'f1': float(f1_score(y_true, predictions, zero_division=0)),
        'fraud_count': int(np.sum(np.asarray(y_true) == 1)),
        'legitimate_count': int(np.sum(np.asarray(y_true) == 0)),
    }


def train_and_evaluate(data_path, model_dir=None, epochs=50,
                       batch_size=512, learning_rate=0.001, patience=10):
    """Train the existing MLP on chronological ULB partitions, then test once."""
    model_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
    adapter = ULBCreditCardDataset(deduplicate=True)
    adapter.load(data_path)
    splits = adapter.chronological_split()
    data = adapter.preprocess_splits(splits, model_dir=model_dir)

    train_loader, validation_loader = create_data_loaders(
        data['X_train'].to_numpy(), data['y_train'].to_numpy(),
        data['X_validation'].to_numpy(), data['y_validation'].to_numpy(),
        batch_size=batch_size,
    )
    trainer = PyTorchTrainer(input_dim=len(adapter.feature_names))
    trainer.build_model()
    history = trainer.train(
        train_loader, validation_loader, data['y_train'],
        epochs=epochs, learning_rate=learning_rate, patience=patience,
    )

    # Select the operating threshold from validation probabilities only.
    _, validation_probabilities = trainer.predict(data['X_validation'].to_numpy())
    threshold, validation_selection = select_f1_threshold(
        data['y_validation'].to_numpy(), validation_probabilities
    )
    validation_predictions = (validation_probabilities >= threshold).astype(int)
    validation_evaluator = ModelEvaluator(
        data['y_validation'].to_numpy(), validation_predictions,
        validation_probabilities, 'ULB validation',
    )
    validation_metrics = validation_evaluator.calculate_metrics()
    print(
        f'ULB validation F1 after threshold selection: '
        f"{validation_selection['f1']:.4f} "
        f'(selected probability threshold >= {threshold:.6f})'
    )
    print(
        f'ULB training-loop best validation F1 for early stopping: '
        f'{trainer.best_val_f1:.4f} (fixed probability threshold >= 0.50)'
    )

    # The held-out latest-time partition is evaluated only after model and
    # threshold selection are complete.
    test_predictions, test_probabilities = trainer.predict(
        data['X_test'].to_numpy(), threshold=threshold
    )
    test_evaluator = ModelEvaluator(
        data['y_test'].to_numpy(), test_predictions, test_probabilities,
        'ULB test',
    )
    test_metrics = test_evaluator.calculate_metrics()
    test_matrix = confusion_matrix(data['y_test'], test_predictions, labels=[0, 1])
    test_report = {
        'dataset': 'ULB/Worldline credit-card fraud benchmark',
        'deduplicated_exact_rows': adapter.duplicate_rows,
        'selected_threshold': threshold,
        'threshold_selection': 'maximum validation F1',
        'training_loop_validation': {
            'best_f1': float(trainer.best_val_f1),
            'threshold': 0.5,
            'prediction_rule': 'probability >= 0.50',
            'purpose': 'early stopping and checkpoint selection',
        },
        'validation': {
            'metrics': {key: float(value) for key, value in validation_metrics.items()},
            'selection': validation_selection,
            'confusion_matrix': confusion_matrix(
                data['y_validation'], validation_predictions, labels=[0, 1]
            ).tolist(),
        },
        'test': {
            'metrics': {key: float(value) for key, value in test_metrics.items()},
            'average_precision': float(test_metrics['PR-AUC']),
            'confusion_matrix': test_matrix.tolist(),
            'fraud_count': int(np.sum(data['y_test'] == 1)),
            'legitimate_count': int(np.sum(data['y_test'] == 0)),
        },
    }

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / 'fraud_detector_ulb.pt'
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'input_dim': trainer.input_dim,
        'architecture': 'FraudDetectorNet',
        'dataset': 'ULB/Worldline credit-card fraud benchmark',
        'threshold': threshold,
    }, checkpoint_path)
    with (output_dir / 'training_history_ulb.json').open('w', encoding='utf-8') as stream:
        json.dump(history, stream, indent=2)
    with (output_dir / 'eval_metrics_ulb.json').open('w', encoding='utf-8') as stream:
        json.dump(test_report, stream, indent=2)

    print(json.dumps(test_report, indent=2))
    print(f'ULB model checkpoint saved to {checkpoint_path}')
    return test_report


def main():
    parser = argparse.ArgumentParser(
        description='Train/evaluate the project MLP on the local ULB credit-card dataset.'
    )
    parser.add_argument(
        '--data-path', default=os.environ.get('ULB_CSV_PATH'),
        help='Local path to creditcard.csv (or set ULB_CSV_PATH).',
    )
    parser.add_argument('--model-dir', default=str(MODEL_DIR))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()
    if not args.data_path:
        parser.error('Provide --data-path or set ULB_CSV_PATH.')
    train_and_evaluate(
        args.data_path, args.model_dir, args.epochs, args.batch_size,
        args.learning_rate, args.patience,
    )


if __name__ == '__main__':
    main()
