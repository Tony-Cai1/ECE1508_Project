"""Utilities for preparing financial time series data for LSTM training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


DEFAULT_FEATURE_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass
class SequenceSplit:
    """Container for one chronological data split."""

    X: np.ndarray
    y: np.ndarray
    baseline: Optional[np.ndarray] = None
    previous_close: Optional[np.ndarray] = None


class StockSequenceDataset(Dataset):
    """PyTorch dataset for pre-built time series windows."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def validate_columns(df: pd.DataFrame, feature_columns: Sequence[str], target_column: str) -> None:
    """Ensure the DataFrame contains the expected input and target columns."""

    required_columns = set(feature_columns) | {target_column}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_list}")


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame chronologically.

    Ratios operate on rows, preserving time order and avoiding random shuffling.
    """

    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    n_rows = len(df)
    if n_rows < 3:
        raise ValueError("Need at least 3 rows to create train, validation, and test splits.")

    train_end = int(n_rows * train_ratio)
    val_end = train_end + int(n_rows * val_ratio)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One of the chronological splits is empty. Increase dataset size or adjust ratios.")

    return train_df, val_df, test_df


def fit_feature_scaler(train_df: pd.DataFrame, feature_columns: Sequence[str]) -> StandardScaler:
    """Fit a feature scaler using training data only."""

    scaler = StandardScaler()
    scaler.fit(train_df[list(feature_columns)])
    return scaler


def fit_target_scaler(train_df: pd.DataFrame, target_column: str) -> StandardScaler:
    """Fit a target scaler using training targets only."""

    scaler = StandardScaler()
    scaler.fit(train_df[[target_column]])
    return scaler


def apply_scalers(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    feature_scaler: Optional[StandardScaler],
    target_scaler: Optional[StandardScaler],
) -> pd.DataFrame:
    """Return a transformed copy of the DataFrame."""

    transformed_df = df.copy()
    float_columns = set(feature_columns) | {target_column}
    for column in float_columns:
        transformed_df[column] = transformed_df[column].astype(np.float32)

    if feature_scaler is not None:
        scaled_features = feature_scaler.transform(df[list(feature_columns)])
        for idx, column in enumerate(feature_columns):
            transformed_df[column] = scaled_features[:, idx].astype(np.float32)

    if target_scaler is not None:
        transformed_df[target_column] = (
            target_scaler.transform(df[[target_column]]).reshape(-1).astype(np.float32)
        )

    return transformed_df


def create_sequences(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a time series table into sliding windows.

    X shape: (num_samples, lookback, num_features)
    y shape: (num_samples,)
    """

    if len(df) <= lookback:
        raise ValueError(
            f"Need more than lookback={lookback} rows in a split. Current rows: {len(df)}."
        )

    feature_values = df[list(feature_columns)].to_numpy(dtype=np.float32)
    target_values = df[target_column].to_numpy(dtype=np.float32)

    X_sequences: List[np.ndarray] = []
    y_targets: List[float] = []

    for end_idx in range(lookback, len(df)):
        start_idx = end_idx - lookback
        X_sequences.append(feature_values[start_idx:end_idx])
        y_targets.append(target_values[end_idx])

    return np.stack(X_sequences), np.array(y_targets, dtype=np.float32)


def create_baseline_sequence(
    df: pd.DataFrame,
    baseline_column: str,
    lookback: int,
) -> np.ndarray:
    """Create a baseline array aligned with sequence targets."""

    if len(df) <= lookback:
        raise ValueError(
            f"Need more than lookback={lookback} rows in a split. Current rows: {len(df)}."
        )

    baseline_values = df[baseline_column].to_numpy(dtype=np.float32)
    baselines: List[float] = []

    for end_idx in range(lookback, len(df)):
        baselines.append(baseline_values[end_idx])

    return np.array(baselines, dtype=np.float32)


def create_previous_close_sequence(
    df: pd.DataFrame,
    lookback: int,
) -> np.ndarray:
    """Create the previous close aligned with each prediction target."""

    if len(df) <= lookback:
        raise ValueError(
            f"Need more than lookback={lookback} rows in a split. Current rows: {len(df)}."
        )

    close_values = df["close"].to_numpy(dtype=np.float32)
    previous_closes: List[float] = []

    for end_idx in range(lookback, len(df)):
        previous_closes.append(close_values[end_idx - 1])

    return np.array(previous_closes, dtype=np.float32)


def build_split_sequences(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    lookback: int,
    normalize: bool = True,
    baseline_column: Optional[str] = None,
) -> Dict[str, object]:
    """
    Build train/validation/test sequence sets with train-only normalization.

    To avoid leakage:
    - scalers are fit only on the training rows
    - validation and test windows are made only from their own split rows
    """

    feature_scaler: Optional[StandardScaler] = None
    target_scaler: Optional[StandardScaler] = None

    if normalize:
        feature_scaler = fit_feature_scaler(train_df, feature_columns)
        target_scaler = fit_target_scaler(train_df, target_column)

    scaled_train = apply_scalers(train_df, feature_columns, target_column, feature_scaler, target_scaler)
    scaled_val = apply_scalers(val_df, feature_columns, target_column, feature_scaler, target_scaler)
    scaled_test = apply_scalers(test_df, feature_columns, target_column, feature_scaler, target_scaler)

    train_X, train_y = create_sequences(scaled_train, feature_columns, target_column, lookback)
    val_X, val_y = create_sequences(scaled_val, feature_columns, target_column, lookback)
    test_X, test_y = create_sequences(scaled_test, feature_columns, target_column, lookback)
    train_baseline = create_baseline_sequence(train_df, baseline_column, lookback) if baseline_column else None
    val_baseline = create_baseline_sequence(val_df, baseline_column, lookback) if baseline_column else None
    test_baseline = create_baseline_sequence(test_df, baseline_column, lookback) if baseline_column else None
    train_previous_close = create_previous_close_sequence(train_df, lookback)
    val_previous_close = create_previous_close_sequence(val_df, lookback)
    test_previous_close = create_previous_close_sequence(test_df, lookback)

    return {
        "train": SequenceSplit(
            X=train_X,
            y=train_y,
            baseline=train_baseline,
            previous_close=train_previous_close,
        ),
        "val": SequenceSplit(
            X=val_X,
            y=val_y,
            baseline=val_baseline,
            previous_close=val_previous_close,
        ),
        "test": SequenceSplit(
            X=test_X,
            y=test_y,
            baseline=test_baseline,
            previous_close=test_previous_close,
        ),
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_columns": list(feature_columns),
        "target_column": target_column,
        "lookback": lookback,
        "baseline_column": baseline_column,
    }


def create_dataloaders(
    splits: Dict[str, object],
    batch_size: int = 32,
) -> Dict[str, DataLoader]:
    """Create DataLoaders from pre-built split arrays."""

    train_split: SequenceSplit = splits["train"]  # type: ignore[assignment]
    val_split: SequenceSplit = splits["val"]  # type: ignore[assignment]
    test_split: SequenceSplit = splits["test"]  # type: ignore[assignment]

    train_dataset = StockSequenceDataset(train_split.X, train_split.y)
    val_dataset = StockSequenceDataset(val_split.X, val_split.y)
    test_dataset = StockSequenceDataset(test_split.X, test_split.y)

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    }


def prepare_datasets(
    df: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    target_column: str = "close",
    lookback: int = 20,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 32,
    normalize: bool = True,
    baseline_column: Optional[str] = None,
) -> Dict[str, object]:
    """Full beginner-friendly pipeline from DataFrame to DataLoaders."""

    feature_columns = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    validate_columns(df, feature_columns, target_column)

    ordered_df = df.reset_index(drop=True).copy()
    train_df, val_df, test_df = chronological_split(ordered_df, train_ratio, val_ratio, test_ratio)
    splits = build_split_sequences(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_columns=feature_columns,
        target_column=target_column,
        lookback=lookback,
        normalize=normalize,
        baseline_column=baseline_column,
    )
    dataloaders = create_dataloaders(splits, batch_size=batch_size)

    return {
        **splits,
        "dataloaders": dataloaders,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }
