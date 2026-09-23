"""Regression tests for ULB validation threshold selection."""

import numpy as np
from sklearn.metrics import f1_score

from src.train_ulb import select_f1_threshold
from src.train_model import snapshot_state_dict
import torch


def test_selected_validation_f1_is_distinct_from_fixed_half_threshold():
    labels = np.array([1, 1, 1, 0, 0, 0, 0])
    probabilities = np.array([0.90, 0.40, 0.30, 0.49, 0.20, 0.10, 0.05])

    threshold, selection = select_f1_threshold(labels, probabilities)
    fixed_threshold_f1 = f1_score(labels, probabilities >= 0.5)

    assert threshold == 0.3
    assert selection['threshold_method'] == 'maximum validation F1'
    assert selection['f1'] == f1_score(labels, probabilities >= threshold)
    assert selection['f1'] > fixed_threshold_f1


def test_training_best_state_snapshot_does_not_follow_later_updates():
    model = torch.nn.Linear(2, 1)
    snapshot = snapshot_state_dict(model)
    original = {name: tensor.clone() for name, tensor in snapshot.items()}

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)

    model.load_state_dict(snapshot)
    assert all(torch.equal(model.state_dict()[name], value)
               for name, value in original.items())
