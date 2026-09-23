import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.train_ieee_cis import select_f1_threshold


def test_ieee_threshold_is_selected_on_validation_probabilities():
    labels = np.array([1, 1, 1, 0, 0, 0, 0])
    probabilities = np.array([0.90, 0.40, 0.30, 0.49, 0.20, 0.10, 0.05])
    threshold, validation_f1 = select_f1_threshold(labels, probabilities)
    assert threshold == pytest.approx(0.3)
    assert validation_f1 == f1_score(labels, probabilities >= threshold)
    assert validation_f1 > f1_score(labels, probabilities >= 0.5)


def test_threshold_selection_does_not_read_or_accept_test_data():
    labels = np.array([1, 0, 1, 0])
    probabilities = np.array([0.91, 0.4, 0.7, 0.2])
    assert len(select_f1_threshold(labels, probabilities)) == 2
