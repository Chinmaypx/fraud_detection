"""Tests for the separate ULB credit-card dataset adapter."""

import numpy as np
import pandas as pd
import pytest

from src.ulb_dataset import ULBCreditCardDataset, ULB_COLUMNS, ULB_FEATURES
from src.pytorch_model import get_class_weights


def make_ulb_frame(rows=100):
    frame = pd.DataFrame({name: np.arange(rows, dtype=float) for name in ULB_FEATURES})
    frame['Class'] = np.array([0, 1] * (rows // 2) + ([0] if rows % 2 else []))
    return frame.loc[:, ULB_COLUMNS]


def test_missing_expected_column_is_rejected():
    frame = make_ulb_frame().drop(columns='V28')
    with pytest.raises(ValueError, match='V28'):
        ULBCreditCardDataset().load_frame(frame)


def test_missing_values_are_reported_and_rejected():
    frame = make_ulb_frame()
    frame.loc[0, 'Amount'] = np.nan
    adapter = ULBCreditCardDataset()
    with pytest.raises(ValueError, match='missing values'):
        adapter.load_frame(frame)
    assert adapter.missing_counts['Amount'] == 1


@pytest.mark.parametrize('bad_label', [2, -1, 'fraud'])
def test_class_requires_binary_labels(bad_label):
    frame = make_ulb_frame()
    if isinstance(bad_label, str):
        frame['Class'] = frame['Class'].astype(object)
    frame.loc[0, 'Class'] = bad_label
    with pytest.raises((ValueError, TypeError)):
        ULBCreditCardDataset().load_frame(frame)


def test_exact_duplicates_are_removed_without_mutating_input():
    frame = make_ulb_frame()
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    original_rows = len(duplicated)
    adapter = ULBCreditCardDataset()
    clean = adapter.load_frame(duplicated)
    assert adapter.duplicate_rows == 1
    assert len(clean) == original_rows - 1
    assert len(duplicated) == original_rows


def test_chronological_splits_are_disjoint_and_keep_equal_times_together():
    frame = make_ulb_frame(120)
    frame.loc[1, 'Time'] = frame.loc[0, 'Time']
    frame.loc[1, 'V1'] = 999.0
    adapter = ULBCreditCardDataset()
    adapter.load_frame(frame)
    splits = adapter.chronological_split()
    assert splits['train'].Time.max() < splits['validation'].Time.min()
    assert splits['validation'].Time.max() < splits['test'].Time.min()
    for part in splits.values():
        assert set(part.Class.unique()) == {0, 1}


def test_scaler_is_fitted_only_on_training_partition():
    frame = make_ulb_frame(100)
    frame['Time'] = np.arange(100) * 10.0
    adapter = ULBCreditCardDataset()
    adapter.load_frame(frame)
    splits = adapter.chronological_split()
    transformed = adapter.preprocess_splits(splits)
    expected_median = splits['train'][ULB_FEATURES].median().to_numpy()
    assert np.allclose(adapter.scaler.center_, expected_median)
    assert list(transformed['X_test'].columns) == ULB_FEATURES


def test_transform_enforces_deterministic_feature_order():
    frame = make_ulb_frame(100)
    adapter = ULBCreditCardDataset()
    adapter.load_frame(frame)
    adapter.preprocess_splits(adapter.chronological_split())
    one_row = frame.loc[[0], ULB_FEATURES]
    expected = adapter.transform(one_row)
    shuffled = adapter.transform(one_row.loc[:, list(reversed(ULB_FEATURES))])
    assert list(expected.columns) == ULB_FEATURES
    assert np.allclose(expected.to_numpy(), shuffled.to_numpy())


def test_class_weighting_increases_fraud_loss_weight():
    labels = np.array([0] * 999 + [1])
    positive_weight = get_class_weights(labels).item()
    assert positive_weight == pytest.approx(999.0)
    assert positive_weight > 1


def test_ulb_preprocessing_uses_separate_artifacts(tmp_path):
    for name in ('scaler.pkl', 'scaler_mlp.pkl', 'scaler_lstm.pkl'):
        (tmp_path / name).write_bytes(b'leave unchanged')
    adapter = ULBCreditCardDataset()
    adapter.load_frame(make_ulb_frame(100))
    adapter.preprocess_splits(adapter.chronological_split(), model_dir=tmp_path)

    assert (tmp_path / 'scaler_ulb.pkl').is_file()
    assert (tmp_path / 'feature_names_ulb.json').is_file()
    reloaded = ULBCreditCardDataset()
    reloaded.load_preprocessing(tmp_path)
    sample = make_ulb_frame(100).loc[[0], ULB_FEATURES]
    assert np.allclose(adapter.transform(sample), reloaded.transform(sample))
    for name in ('scaler.pkl', 'scaler_mlp.pkl', 'scaler_lstm.pkl'):
        assert (tmp_path / name).read_bytes() == b'leave unchanged'
