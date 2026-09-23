"""Train and evaluate the IEEE-CIS MLP using local CSV inputs.

Example:
    python -m src.train_ieee_cis --transaction-path <train_transaction.csv> \
        --identity-path <train_identity.csv>

The test partition is held out until model selection and validation threshold
selection are complete. Existing MLP/LSTM artifacts are never written here.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve

from .evaluate import ModelEvaluator
from .ieee_cis_dataset import IEEE_FEATURES, TARGET, IEECISDataset
from .pytorch_model import create_data_loaders
from .train_model import PyTorchTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_NAME = "fraud_detector_ieee.pt"


def select_f1_threshold(labels, probabilities):
    """Select the operating threshold on validation data only."""
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if len(thresholds) == 0:
        raise ValueError("Validation data cannot support threshold selection.")
    f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    index = int(np.argmax(f1_values))
    threshold = float(thresholds[index])
    predictions = (np.asarray(probabilities) >= threshold).astype(np.int64)
    return threshold, float(f1_score(labels, predictions, zero_division=0))


def train_and_evaluate(transaction_path, identity_path, model_dir=None,
                       epochs=20, batch_size=512, learning_rate=0.001, patience=5):
    """Fit a separate IEEE-CIS MLP; evaluate the chronological test set once."""
    output = Path(model_dir) if model_dir is not None else MODEL_DIR
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    adapter = IEECISDataset()
    adapter.load(transaction_path, identity_path)
    splits = adapter.chronological_split()
    arrays = adapter.preprocess_splits(splits, model_dir=output)

    x_train = arrays["X_train"].to_numpy(dtype=np.float32, copy=True)
    y_train = arrays["y_train"].to_numpy(dtype=np.float32, copy=True)
    x_validation = arrays["X_validation"].to_numpy(dtype=np.float32, copy=True)
    y_validation = arrays["y_validation"].to_numpy(dtype=np.float32, copy=True)
    x_test = arrays["X_test"].to_numpy(dtype=np.float32, copy=True)
    y_test = arrays["y_test"].to_numpy(dtype=np.int64, copy=True)
    train_loader, validation_loader = create_data_loaders(
        x_train, y_train, x_validation, y_validation, batch_size=batch_size,
    )
    trainer = PyTorchTrainer(input_dim=len(IEEE_FEATURES))
    trainer.build_model()
    history = trainer.train(
        train_loader, validation_loader, y_train,
        epochs=epochs, learning_rate=learning_rate, patience=patience,
    )

    # Select model threshold using validation probabilities only.
    _, validation_probabilities = trainer.predict(x_validation)
    threshold, threshold_validation_f1 = select_f1_threshold(
        y_validation, validation_probabilities
    )
    validation_predictions = (validation_probabilities >= threshold).astype(np.int64)
    validation_metrics = ModelEvaluator(
        y_validation, validation_predictions,
        validation_probabilities, "IEEE-CIS validation",
    ).calculate_metrics()

    # The held-out latest-time partition is not used for any selection.
    test_probabilities = trainer.predict(x_test)[1]
    test_predictions = (test_probabilities >= threshold).astype(np.int64)
    test_metrics = ModelEvaluator(
        y_test, test_predictions,
        test_probabilities, "IEEE-CIS held-out test",
    ).calculate_metrics()
    test_confusion = confusion_matrix(y_test, test_predictions, labels=[0, 1])

    report = {
        "dataset": "IEEE-CIS Fraud Detection",
        "transaction_identity_join": adapter.join_report,
        "evaluation_strategy": "Chronological 70/15/15 partitions by unique TransactionDT; equal elapsed times stay together.",
        "time_feature_definition": "hour_of_day = floor(TransactionDT / 3600) modulo 24; this is an elapsed-time cycle bucket, not a wall-clock hour.",
        "features": list(IEEE_FEATURES),
        "class_counts": {
            split: {"legitimate": int((part[TARGET] == 0).sum()), "fraud": int((part[TARGET] == 1).sum())}
            for split, part in splits.items()
        },
        "threshold_selection": {
            "method": "maximum validation F1",
            "threshold": threshold,
            "validation_f1_at_selected_threshold": threshold_validation_f1,
            "training_loop_early_stopping_threshold": 0.5,
            "training_loop_best_validation_f1": float(trainer.best_val_f1),
        },
        "validation": {
            "metrics": {key: float(value) for key, value in validation_metrics.items()},
            "confusion_matrix": confusion_matrix(
                y_validation, validation_predictions, labels=[0, 1]
            ).tolist(),
        },
        "test": {
            "metrics": {key: float(value) for key, value in test_metrics.items()},
            "confusion_matrix": test_confusion.tolist(),
        },
    }

    checkpoint = {
        "model_state_dict": trainer.model.state_dict(),
        "input_dim": len(IEEE_FEATURES),
        "architecture": "FraudDetectorNet",
        "dataset": "IEEE-CIS Fraud Detection",
        "threshold": threshold,
        "feature_names": list(IEEE_FEATURES),
    }
    torch.save(checkpoint, output / MODEL_NAME)
    with (output / "training_history_ieee.json").open("w", encoding="utf-8") as stream:
        json.dump(history, stream, indent=2)
    with (output / "eval_metrics_ieee.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    print(json.dumps(report, indent=2))
    print(f"IEEE-CIS checkpoint saved to {output / MODEL_NAME}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Train the IEEE-CIS fraud detection MLP.")
    parser.add_argument("--transaction-path", default=os.environ.get("IEEE_CIS_TRANSACTION_CSV"))
    parser.add_argument("--identity-path", default=os.environ.get("IEEE_CIS_IDENTITY_CSV"))
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    if not args.transaction_path or not args.identity_path:
        parser.error("Pass --transaction-path and --identity-path (or their IEEE_CIS_*_CSV environment variables).")
    train_and_evaluate(
        args.transaction_path, args.identity_path, args.model_dir,
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, patience=args.patience,
    )


if __name__ == "__main__":
    main()
