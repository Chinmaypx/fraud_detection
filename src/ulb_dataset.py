"""Adapter and preprocessing for the ULB/Worldline credit-card benchmark.

This benchmark has no entity identifier or customer history. Its Time field is
used only to create chronological partitions; it is not treated as a customer
sequence. Exact duplicate rows are removed explicitly (keep the first row),
without changing the source CSV.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


ULB_FEATURES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
ULB_TARGET = 'Class'
ULB_COLUMNS = ULB_FEATURES + [ULB_TARGET]
PROTECTED_SCALERS = {'scaler.pkl', 'scaler_mlp.pkl', 'scaler_lstm.pkl'}


class ULBCreditCardDataset:
    """Load, validate, deduplicate, split, and scale the ULB dataset."""

    def __init__(self, deduplicate=True):
        self.deduplicate = deduplicate
        self.data = None
        self.missing_counts = None
        self.duplicate_rows = 0
        self.scaler = None
        self.feature_names = list(ULB_FEATURES)

    def load(self, csv_path):
        """Load a CSV from an explicit local path and validate its ULB schema."""
        path = Path(csv_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'ULB dataset CSV not found: {path}')
        return self.load_frame(pd.read_csv(path))

    def load_frame(self, frame):
        """Validate an in-memory frame; useful for tests and programmatic use."""
        missing_columns = [column for column in ULB_COLUMNS if column not in frame.columns]
        if missing_columns:
            raise ValueError(f'Missing required ULB columns: {missing_columns}')

        data = frame.loc[:, ULB_COLUMNS].copy()
        for column in ULB_COLUMNS:
            data[column] = pd.to_numeric(data[column], errors='raise')

        self.missing_counts = data.isna().sum()
        missing_values = int(self.missing_counts.sum())
        if missing_values:
            counts = self.missing_counts[self.missing_counts > 0].to_dict()
            raise ValueError(f'ULB dataset contains {missing_values} missing values: {counts}')

        labels = set(data[ULB_TARGET].unique().tolist())
        if not labels.issubset({0, 1}) or labels != {0, 1}:
            raise ValueError(f'Class must contain both binary labels 0 and 1; found {sorted(labels)}')

        # Do not silently resolve two otherwise-identical transactions with
        # contradictory labels.
        repeated_features = data.duplicated(subset=ULB_FEATURES, keep=False)
        if repeated_features.any():
            conflicting = data.loc[repeated_features].groupby(ULB_FEATURES)[ULB_TARGET].nunique()
            if (conflicting > 1).any():
                raise ValueError('Identical ULB feature rows have conflicting Class labels.')

        self.duplicate_rows = int(data.duplicated(subset=ULB_COLUMNS).sum())
        if self.deduplicate:
            data = data.drop_duplicates(subset=ULB_COLUMNS, keep='first')
        data[ULB_TARGET] = data[ULB_TARGET].astype('int64')
        self.data = data.reset_index(drop=True)
        self.missing_counts = self.missing_counts.astype('int64')

        class_counts = self.data[ULB_TARGET].value_counts().sort_index().to_dict()
        print(
            f'ULB data shape: {self.data.shape}; missing values: {missing_values}; '
            f'exact duplicate rows: {self.duplicate_rows} '
            f'({"removed" if self.deduplicate else "retained"}); '
            f'class counts after duplicate policy: {class_counts}'
        )
        return self.data

    def chronological_split(self, train_fraction=0.60, validation_fraction=0.20):
        """Split on distinct Time values, keeping equal timestamps together."""
        if self.data is None:
            raise RuntimeError('Load the dataset before splitting it.')
        if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
            raise ValueError('Split fractions must be between 0 and 1.')
        if train_fraction + validation_fraction >= 1:
            raise ValueError('Training and validation fractions must sum to less than 1.')

        ordered = self.data.sort_values('Time', kind='mergesort')
        unique_times = ordered['Time'].drop_duplicates().to_numpy()
        if len(unique_times) < 3:
            raise ValueError('At least three distinct Time values are required for splitting.')
        val_index = int(len(unique_times) * train_fraction)
        test_index = int(len(unique_times) * (train_fraction + validation_fraction))
        val_index = min(max(val_index, 1), len(unique_times) - 2)
        test_index = min(max(test_index, val_index + 1), len(unique_times) - 1)
        val_boundary, test_boundary = unique_times[val_index], unique_times[test_index]

        train = ordered[ordered['Time'] < val_boundary].copy()
        validation = ordered[
            (ordered['Time'] >= val_boundary) & (ordered['Time'] < test_boundary)
        ].copy()
        test = ordered[ordered['Time'] >= test_boundary].copy()
        result = {}
        for name, part in (('train', train), ('validation', validation), ('test', test)):
            labels = set(part[ULB_TARGET].unique().tolist())
            if labels != {0, 1}:
                raise ValueError(
                    f'{name} split must contain both classes for evaluation; '
                    f'found counts {part[ULB_TARGET].value_counts().sort_index().to_dict()}'
                )
            result[name] = part
            counts = part[ULB_TARGET].value_counts().sort_index().to_dict()
            print(
                f'{name.title()} split: rows={len(part)}, Time='
                f'[{part.Time.min()}, {part.Time.max()}], class counts={counts}'
            )
        return result

    def preprocess_splits(self, splits, model_dir=None):
        """Fit RobustScaler on training rows only and transform other partitions."""
        train_x = splits['train'].loc[:, self.feature_names]
        self.scaler = RobustScaler()
        train_values = self.scaler.fit_transform(train_x)
        transformed = {
            'X_train': pd.DataFrame(train_values, columns=self.feature_names, index=train_x.index),
            'y_train': splits['train'][ULB_TARGET].copy(),
        }
        for name in ('validation', 'test'):
            part_x = splits[name].loc[:, self.feature_names]
            values = self.scaler.transform(part_x)
            transformed[f'X_{name}'] = pd.DataFrame(
                values, columns=self.feature_names, index=part_x.index
            )
            transformed[f'y_{name}'] = splits[name][ULB_TARGET].copy()
        if model_dir is not None:
            self.save_preprocessing(model_dir)
        return transformed

    def transform(self, frame):
        """Apply the fitted scaler after enforcing the saved feature order."""
        if self.scaler is None:
            raise RuntimeError('Fit preprocessing before transforming transactions.')
        missing = [name for name in self.feature_names if name not in frame.columns]
        if missing:
            raise ValueError(f'Missing ULB features for transformation: {missing}')
        ordered = frame.loc[:, self.feature_names].apply(pd.to_numeric, errors='raise')
        if ordered.isna().any().any():
            raise ValueError('ULB features contain missing values.')
        return pd.DataFrame(
            self.scaler.transform(ordered),
            columns=self.feature_names,
            index=frame.index,
        )

    def save_preprocessing(self, model_dir):
        """Write only ULB-specific preprocessing files, never shared scalers."""
        if self.scaler is None:
            raise RuntimeError('Fit preprocessing before saving artifacts.')
        output = Path(model_dir)
        output.mkdir(parents=True, exist_ok=True)
        scaler_name = 'scaler_ulb.pkl'
        if scaler_name in PROTECTED_SCALERS:
            raise RuntimeError('Refusing to overwrite a shared/synthetic scaler.')
        with (output / scaler_name).open('wb') as stream:
            pickle.dump(self.scaler, stream)
        with (output / 'feature_names_ulb.json').open('w', encoding='utf-8') as stream:
            json.dump(self.feature_names, stream, indent=2)

    def load_preprocessing(self, model_dir):
        """Load the ULB-only scaler and reject incompatible feature metadata."""
        source = Path(model_dir)
        scaler_path = source / 'scaler_ulb.pkl'
        names_path = source / 'feature_names_ulb.json'
        if not scaler_path.is_file() or not names_path.is_file():
            raise FileNotFoundError(
                f'ULB preprocessing artifacts not found under {source}'
            )
        with names_path.open('r', encoding='utf-8') as stream:
            stored_names = json.load(stream)
        if stored_names != ULB_FEATURES:
            raise ValueError('Stored ULB feature order does not match the expected schema.')
        with scaler_path.open('rb') as stream:
            self.scaler = pickle.load(stream)
        if getattr(self.scaler, 'n_features_in_', None) != len(stored_names):
            self.scaler = None
            raise ValueError('Stored ULB scaler has an incompatible feature count.')
        self.feature_names = stored_names
        return self.scaler
