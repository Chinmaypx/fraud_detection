"""Regression tests for held-out evaluation data."""

import numpy as np
import pandas as pd

from src.train_model import split_train_validation


def test_train_validation_split_keeps_rows_disjoint():
    features = pd.DataFrame({'row': np.arange(100)})
    labels = pd.Series([0, 1] * 50)

    train_x, val_x, train_y, val_y = split_train_validation(
        features, labels, validation_size=0.2
    )

    assert set(train_x.index).isdisjoint(val_x.index)
    assert len(train_x) + len(val_x) == len(features)
    assert set(train_y.unique()) == {0, 1}
    assert set(val_y.unique()) == {0, 1}
