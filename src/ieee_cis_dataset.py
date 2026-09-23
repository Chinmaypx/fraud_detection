"""IEEE-CIS transaction adapter and leakage-safe preprocessing.

TransactionDT is elapsed time, not a wall-clock timestamp. ``hour_of_day`` is
the elapsed-time position within a 24-hour cycle: floor(TransactionDT / 3600)
modulo 24. Partitions are chronological and keep equal TransactionDT values
together. The transaction file is the base population; identity fields are
left-joined and missing identity records are retained.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


TRANSACTION_ID = "TransactionID"
TRANSACTION_TIME = "TransactionDT"
TARGET = "isFraud"
MISSING_CATEGORY = "__MISSING__"

IEEE_FEATURES = [
    "TransactionAmt", "ProductCD", "hour_of_day", "card4", "card6",
    "addr1", "addr2", "dist1", "P_emaildomain", "DeviceType",
]
NUMERIC_FEATURES = ["TransactionAmt", "hour_of_day", "dist1"]
CATEGORICAL_FEATURES = [name for name in IEEE_FEATURES if name not in NUMERIC_FEATURES]
TRANSACTION_COLUMNS = [
    TRANSACTION_ID, TRANSACTION_TIME, TARGET, "TransactionAmt", "ProductCD",
    "card4", "card6", "addr1", "addr2", "dist1", "P_emaildomain",
]
IDENTITY_COLUMNS = [TRANSACTION_ID, "DeviceType"]
PREPROCESSOR_FILE = "ieee_preprocessor.pkl"
FEATURES_FILE = "feature_names_ieee.json"
METADATA_FILE = "ieee_feature_metadata.json"


def _category_key(value):
    """Normalize CSV/API category values without assigning them meanings."""
    if value is None or pd.isna(value):
        return MISSING_CATEGORY
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    result = str(value).strip()
    return result if result else MISSING_CATEGORY


class IEECISDataset:
    """Validate/join IEEE-CIS inputs, split them chronologically, and encode."""

    def __init__(self):
        self.data = None
        self.feature_names = list(IEEE_FEATURES)
        self.numeric_medians = None
        self.numeric_center = None
        self.numeric_scale = None
        self.category_frequencies = None
        self.category_values = None
        self.join_report = None

    @staticmethod
    def _require_columns(frame, required, label):
        missing = [name for name in required if name not in frame.columns]
        if missing:
            raise ValueError(f"{label} is missing required columns: {missing}")

    def load(self, transaction_path, identity_path):
        """Read only supported columns; the original CSV files are never written."""
        transaction_path = Path(transaction_path).expanduser()
        identity_path = Path(identity_path).expanduser()
        for path, label in ((transaction_path, "transaction"), (identity_path, "identity")):
            if not path.is_file():
                raise FileNotFoundError(f"IEEE-CIS {label} CSV not found: {path}")

        transaction_header = pd.read_csv(transaction_path, nrows=0).columns
        identity_header = pd.read_csv(identity_path, nrows=0).columns
        self._require_columns(pd.DataFrame(columns=transaction_header), TRANSACTION_COLUMNS, "transaction CSV")
        self._require_columns(pd.DataFrame(columns=identity_header), IDENTITY_COLUMNS, "identity CSV")
        transactions = pd.read_csv(transaction_path, usecols=TRANSACTION_COLUMNS)
        identities = pd.read_csv(identity_path, usecols=IDENTITY_COLUMNS)
        return self.load_frames(transactions, identities)

    def load_frames(self, transactions, identities):
        """Join in memory while preserving every row in the transaction base."""
        self._require_columns(transactions, TRANSACTION_COLUMNS, "transaction data")
        self._require_columns(identities, IDENTITY_COLUMNS, "identity data")
        transaction_count = len(transactions)
        identity_count = len(identities)

        base = transactions.loc[:, TRANSACTION_COLUMNS].copy()
        identity = identities.loc[:, IDENTITY_COLUMNS].copy()
        if base[TRANSACTION_ID].isna().any() or identity[TRANSACTION_ID].isna().any():
            raise ValueError("TransactionID cannot be missing in either CSV.")
        if base[TRANSACTION_ID].duplicated().any():
            raise ValueError("Transaction CSV contains duplicate TransactionID values.")
        if identity[TRANSACTION_ID].duplicated().any():
            raise ValueError("Identity CSV contains duplicate TransactionID values.")

        base[TRANSACTION_TIME] = pd.to_numeric(base[TRANSACTION_TIME], errors="raise")
        time_values = base[TRANSACTION_TIME].to_numpy(dtype=np.float64)
        if not np.isfinite(time_values).all() or (time_values < 0).any():
            raise ValueError("TransactionDT must contain finite non-negative elapsed-time values.")
        for name in ("TransactionAmt", "dist1"):
            base[name] = pd.to_numeric(base[name], errors="raise")
            observed = base[name].dropna().to_numpy(dtype=np.float64)
            if not np.isfinite(observed).all():
                raise ValueError(f"{name} must contain finite values or missing values.")
        try:
            labels = pd.to_numeric(base[TARGET], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("isFraud must contain only binary values 0 and 1.") from exc
        if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
            raise ValueError("isFraud must contain only binary values 0 and 1.")
        if set(labels.unique()) != {0, 1}:
            raise ValueError("isFraud must contain both classes 0 and 1.")
        base[TARGET] = labels.astype("int64")
        base["hour_of_day"] = (np.floor(base[TRANSACTION_TIME] / 3600) % 24).astype("int64")

        merged = base.merge(
            identity, on=TRANSACTION_ID, how="left", sort=False,
            validate="one_to_one", indicator="_identity_join",
        )
        if len(merged) != transaction_count:
            raise RuntimeError("Identity join changed the transaction base population.")
        matched = int((merged["_identity_join"] == "both").sum())
        merged.drop(columns="_identity_join", inplace=True)
        merged.reset_index(drop=True, inplace=True)
        self.join_report = {
            "transaction_rows_before_join": transaction_count,
            "identity_rows": identity_count,
            "transaction_rows_after_join": len(merged),
            "transactions_with_identity_match": matched,
            "transactions_without_identity_match": transaction_count - matched,
        }
        print(
            "IEEE-CIS join: "
            f"transactions={transaction_count}, identity_rows={identity_count}, "
            f"joined_transactions={len(merged)}, matched={matched}, "
            f"without_identity={transaction_count - matched}"
        )
        self.data = merged
        return self.data

    def chronological_split(self, train_fraction=0.70, validation_fraction=0.15):
        """Create contiguous time-ordered partitions without splitting tied times."""
        if self.data is None:
            raise RuntimeError("Load IEEE-CIS data before splitting it.")
        if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
            raise ValueError("Split fractions must be between 0 and 1.")
        if train_fraction + validation_fraction >= 1:
            raise ValueError("Training and validation fractions must sum to less than 1.")

        ordered = self.data.sort_values(TRANSACTION_TIME, kind="mergesort")
        unique_times = ordered[TRANSACTION_TIME].drop_duplicates().to_numpy()
        if len(unique_times) < 3:
            raise ValueError("At least three distinct TransactionDT values are required.")
        val_boundary = unique_times[min(int(len(unique_times) * train_fraction), len(unique_times) - 2)]
        test_boundary = unique_times[
            min(int(len(unique_times) * (train_fraction + validation_fraction)), len(unique_times) - 1)
        ]
        result = {
            "train": ordered[ordered[TRANSACTION_TIME] < val_boundary].copy(),
            "validation": ordered[
                (ordered[TRANSACTION_TIME] >= val_boundary)
                & (ordered[TRANSACTION_TIME] < test_boundary)
            ].copy(),
            "test": ordered[ordered[TRANSACTION_TIME] >= test_boundary].copy(),
        }
        for name, part in result.items():
            if part.empty or set(part[TARGET].unique()) != {0, 1}:
                raise ValueError(f"{name} partition must be non-empty and contain both target classes.")
            print(
                f"{name.title()} partition: rows={len(part)}, "
                f"TransactionDT=[{part[TRANSACTION_TIME].min()}, {part[TRANSACTION_TIME].max()}], "
                f"fraud={int(part[TARGET].sum())}"
            )
        return result

    def _raw_features(self, frame):
        features = frame.loc[:, [name for name in IEEE_FEATURES if name != "hour_of_day"]].copy()
        if "hour_of_day" in frame:
            features["hour_of_day"] = frame["hour_of_day"]
        return features.loc[:, [name for name in IEEE_FEATURES if name != "hour_of_day"] + ["hour_of_day"]]

    def preprocess_splits(self, splits, model_dir=None):
        """Fit imputers, robust scaling, and category frequencies on train only."""
        train = self._raw_features(splits["train"])
        numeric = train.loc[:, NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
        self.numeric_medians = numeric.median().fillna(0.0).to_dict()
        numeric = numeric.fillna(self.numeric_medians)
        scaler = RobustScaler().fit(numeric)
        self.numeric_center = dict(zip(NUMERIC_FEATURES, scaler.center_.astype(float).tolist()))
        self.numeric_scale = dict(zip(NUMERIC_FEATURES, scaler.scale_.astype(float).tolist()))

        self.category_frequencies = {}
        self.category_values = {}
        for name in CATEGORICAL_FEATURES:
            keys = train[name].map(_category_key)
            counts = keys.value_counts(normalize=True, dropna=False)
            self.category_frequencies[name] = counts.to_dict()
            self.category_values[name] = sorted(
                key for key in counts.index.tolist() if key != MISSING_CATEGORY
            )

        transformed = {}
        for split_name, part in splits.items():
            raw = self._raw_features(part)
            numeric_part = raw.loc[:, NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
            numeric_part = numeric_part.fillna(self.numeric_medians)
            numeric_values = (
                (numeric_part.loc[:, NUMERIC_FEATURES].to_numpy(dtype=np.float64)
                 - np.array([self.numeric_center[name] for name in NUMERIC_FEATURES]))
                / np.array([self.numeric_scale[name] for name in NUMERIC_FEATURES])
            )
            encoded = pd.DataFrame(index=part.index)
            for index, name in enumerate(NUMERIC_FEATURES):
                encoded[name] = numeric_values[:, index]
            for name in CATEGORICAL_FEATURES:
                keys = raw[name].map(_category_key)
                encoded[name] = keys.map(self.category_frequencies[name]).fillna(0.0)
            encoded = encoded.loc[:, IEEE_FEATURES].astype(np.float32)
            transformed[f"X_{split_name}"] = encoded
            transformed[f"y_{split_name}"] = part[TARGET].astype("int64").copy()

        if model_dir is not None:
            self.save_preprocessing(model_dir)
        return transformed

    def transform(self, frame):
        """Transform API or source-schema rows in the persisted feature order."""
        if self.numeric_center is None or self.numeric_scale is None or self.category_frequencies is None:
            raise RuntimeError("Load fitted IEEE-CIS preprocessing before prediction.")
        missing = [name for name in IEEE_FEATURES if name not in frame.columns]
        if missing:
            raise ValueError(f"Missing IEEE-CIS prediction features: {missing}")
        raw = frame.loc[:, IEEE_FEATURES].copy()
        numeric = raw.loc[:, NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.fillna(self.numeric_medians)
        numeric_values = (
            (numeric.loc[:, NUMERIC_FEATURES].to_numpy(dtype=np.float64)
             - np.array([self.numeric_center[name] for name in NUMERIC_FEATURES]))
            / np.array([self.numeric_scale[name] for name in NUMERIC_FEATURES])
        )
        encoded = pd.DataFrame(index=raw.index)
        for index, name in enumerate(NUMERIC_FEATURES):
            encoded[name] = numeric_values[:, index]
        for name in CATEGORICAL_FEATURES:
            keys = raw[name].map(_category_key)
            encoded[name] = keys.map(self.category_frequencies[name]).fillna(0.0)
        return encoded.loc[:, IEEE_FEATURES].astype(np.float32)

    def validate_categories(self, row):
        """Reject unseen non-missing categories at the API boundary."""
        for name in CATEGORICAL_FEATURES:
            value = row.get(name)
            if value is None or pd.isna(value):
                continue
            key = _category_key(value)
            if key not in self.category_frequencies[name]:
                raise ValueError(f"Unsupported category for {name}.")

    def save_preprocessing(self, model_dir):
        if self.numeric_center is None or self.numeric_scale is None:
            raise RuntimeError("Fit preprocessing before saving IEEE-CIS artifacts.")
        output = Path(model_dir)
        output.mkdir(parents=True, exist_ok=True)
        state = {
            "numeric_medians": self.numeric_medians,
            "numeric_center": self.numeric_center,
            "numeric_scale": self.numeric_scale,
            "category_frequencies": self.category_frequencies,
            "category_values": self.category_values,
            "feature_names": self.feature_names,
        }
        with (output / PREPROCESSOR_FILE).open("wb") as stream:
            pickle.dump(state, stream)
        with (output / FEATURES_FILE).open("w", encoding="utf-8") as stream:
            json.dump(self.feature_names, stream, indent=2)
        with (output / METADATA_FILE).open("w", encoding="utf-8") as stream:
            json.dump({
                "feature_names": self.feature_names,
                "user_facing_features": self.feature_names,
                "category_values": self.category_values,
                "hour_of_day_definition": "floor(TransactionDT / 3600) modulo 24; elapsed-time cycle bucket, not a wall-clock hour",
            }, stream, indent=2)

    def load_preprocessing(self, model_dir):
        source = Path(model_dir)
        with (source / FEATURES_FILE).open("r", encoding="utf-8") as stream:
            names = json.load(stream)
        with (source / METADATA_FILE).open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        with (source / PREPROCESSOR_FILE).open("rb") as stream:
            state = pickle.load(stream)
        if names != IEEE_FEATURES or metadata.get("feature_names") != IEEE_FEATURES:
            raise ValueError("Saved IEEE-CIS feature order is incompatible.")
        if state.get("feature_names") != IEEE_FEATURES:
            raise ValueError("Saved IEEE-CIS preprocessor feature order is incompatible.")
        self.feature_names = names
        self.numeric_medians = state["numeric_medians"]
        self.numeric_center = state["numeric_center"]
        self.numeric_scale = state["numeric_scale"]
        self.category_frequencies = state["category_frequencies"]
        self.category_values = state["category_values"]
        return self
