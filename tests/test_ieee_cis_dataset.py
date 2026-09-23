import json
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from src.ieee_cis_dataset import (
    CATEGORICAL_FEATURES, IEEE_FEATURES, IDENTITY_COLUMNS, TARGET,
    TRANSACTION_COLUMNS, IEECISDataset,
)


def make_frames(time_groups=100):
    rows = []
    for group in range(time_groups):
        for label in (0, 1):
            transaction_id = len(rows) + 1
            rows.append({
                "TransactionID": transaction_id,
                "TransactionDT": group * 3600,
                "isFraud": label,
                "TransactionAmt": float(20 + transaction_id),
                "ProductCD": "W" if label == 0 else "C",
                "card4": "visa",
                "card6": "debit",
                "addr1": 315.0,
                "addr2": 87.0,
                "dist1": float(transaction_id),
            })
    transactions = pd.DataFrame(rows, columns=TRANSACTION_COLUMNS)
    identities = pd.DataFrame({
        "TransactionID": [row["TransactionID"] for row in rows[::2]],
        "DeviceType": ["desktop"] * len(rows[::2]),
    }, columns=IDENTITY_COLUMNS)
    return transactions, identities


def loaded_adapter(time_groups=100):
    transactions, identities = make_frames(time_groups)
    adapter = IEECISDataset()
    adapter.load_frames(transactions, identities)
    return adapter, transactions, identities


def test_schema_validation_requires_transaction_identity_and_target_columns():
    transactions, identities = make_frames()
    with pytest.raises(ValueError, match="ProductCD"):
        IEECISDataset().load_frames(transactions.drop(columns="ProductCD"), identities)
    with pytest.raises(ValueError, match="isFraud"):
        IEECISDataset().load_frames(transactions.drop(columns="isFraud"), identities)
    with pytest.raises(ValueError, match="DeviceType"):
        IEECISDataset().load_frames(transactions, identities.drop(columns="DeviceType"))


@pytest.mark.parametrize("invalid_target", [2, -1, "fraud", np.nan])
def test_target_must_be_binary(invalid_target):
    transactions, identities = make_frames()
    transactions[TARGET] = transactions[TARGET].astype(object)
    transactions.loc[0, TARGET] = invalid_target
    with pytest.raises((ValueError, TypeError), match="isFraud"):
        IEECISDataset().load_frames(transactions, identities)


def test_left_join_keeps_transaction_population_and_reports_missing_identity():
    transactions, identities = make_frames(30)
    original = transactions.copy(deep=True)
    adapter = IEECISDataset()
    joined = adapter.load_frames(transactions, identities)
    assert len(joined) == len(transactions)
    assert adapter.join_report["transaction_rows_before_join"] == len(transactions)
    assert adapter.join_report["transaction_rows_after_join"] == len(transactions)
    assert adapter.join_report["transactions_without_identity_match"] == len(transactions) // 2
    assert joined["DeviceType"].isna().sum() == len(transactions) // 2
    pd.testing.assert_frame_equal(transactions, original)


def test_duplicate_identity_keys_are_rejected_instead_of_multiplying_rows():
    transactions, identities = make_frames()
    identities = pd.concat([identities, identities.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate TransactionID"):
        IEECISDataset().load_frames(transactions, identities)


def test_hour_of_day_is_elapsed_24_hour_bucket_not_timestamp():
    transactions, identities = make_frames(40)
    transactions.loc[:4, "TransactionDT"] = [0, 3599, 3600, 23 * 3600, 24 * 3600]
    adapter = IEECISDataset()
    joined = adapter.load_frames(transactions, identities)
    assert joined.loc[:4, "hour_of_day"].tolist() == [0, 0, 1, 23, 0]


def test_chronological_splits_keep_tied_times_together_and_are_disjoint():
    adapter, _, _ = loaded_adapter(100)
    splits = adapter.chronological_split()
    assert splits["train"].TransactionDT.max() < splits["validation"].TransactionDT.min()
    assert splits["validation"].TransactionDT.max() < splits["test"].TransactionDT.min()
    assert all(set(part[TARGET].unique()) == {0, 1} for part in splits.values())
    assert sum(len(part) for part in splits.values()) == len(adapter.data)


def test_feature_selection_contains_only_declared_user_facing_features():
    transactions, identities = make_frames()
    for excluded in ("V1", "C1", "D1", "M1", "id_01", "dist2", "R_emaildomain"):
        transactions[excluded] = 1
    adapter = IEECISDataset()
    adapter.load_frames(transactions, identities)
    assert adapter.feature_names == IEEE_FEATURES
    assert not any(re.match(r"^(?:V|C|D|M)\d+$|^id_\d+$", name) for name in adapter.feature_names)
    assert "TransactionID" not in adapter.feature_names
    assert "TransactionDT" not in adapter.feature_names
    assert "DeviceInfo" not in adapter.feature_names
    assert set(CATEGORICAL_FEATURES).issuperset({"ProductCD", "P_emaildomain", "DeviceType"})


def test_missing_values_are_imputed_and_preprocessing_is_fit_on_train_only():
    adapter, transactions, identities = loaded_adapter(100)
    transactions.loc[0, "TransactionAmt"] = np.nan
    transactions.loc[2, "dist1"] = np.nan
    adapter.load_frames(transactions, identities)
    splits = adapter.chronological_split()
    transformed = adapter.preprocess_splits(splits)
    expected = splits["train"]["TransactionAmt"].median()
    assert adapter.numeric_medians["TransactionAmt"] == pytest.approx(expected)
    assert adapter.numeric_center["TransactionAmt"] == pytest.approx(expected)
    assert np.isfinite(transformed["X_train"].to_numpy()).all()
    assert np.isfinite(transformed["X_test"].to_numpy()).all()


def test_category_encodings_are_fit_on_training_rows_only():
    adapter, _, _ = loaded_adapter()
    splits = adapter.chronological_split()
    adapter.preprocess_splits(splits)
    train_frequency = splits["train"].ProductCD.value_counts(normalize=True).to_dict()
    assert adapter.category_frequencies["ProductCD"] == train_frequency


def test_feature_order_and_preprocessing_persist_separately(tmp_path):
    adapter, _, _ = loaded_adapter()
    splits = adapter.chronological_split()
    transformed = adapter.preprocess_splits(splits, model_dir=tmp_path)
    loaded = IEECISDataset().load_preprocessing(tmp_path)
    assert json.loads((tmp_path / "feature_names_ieee.json").read_text()) == IEEE_FEATURES
    assert list(transformed["X_test"].columns) == IEEE_FEATURES
    assert list(loaded.transform(splits["test"].iloc[:4]).columns) == IEEE_FEATURES
    np.testing.assert_allclose(
        loaded.transform(splits["test"].iloc[:4]), transformed["X_test"].iloc[:4]
    )
    assert (tmp_path / "ieee_preprocessor.pkl").is_file()
    assert (tmp_path / "ieee_feature_metadata.json").is_file()


def test_unknown_categories_can_be_detected_for_api_validation():
    adapter, _, _ = loaded_adapter()
    adapter.preprocess_splits(adapter.chronological_split())
    valid = {name: None for name in IEEE_FEATURES}
    valid.update({"ProductCD": "W", "card4": "visa"})
    adapter.validate_categories(valid)
    valid["ProductCD"] = "unknown"
    with pytest.raises(ValueError, match="ProductCD"):
        adapter.validate_categories(valid)
