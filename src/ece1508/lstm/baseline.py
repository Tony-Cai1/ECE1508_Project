"""Shared helpers for LSTM baseline scripts: download, detrending, and debugging prints."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from .data_preparation import DEFAULT_FEATURE_COLUMNS


def download_stock_data(
    ticker: str = "AAPL",
    period: str = "5y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download OHLCV data with yfinance and reshape it for the training pipeline.

    Returned columns:
    open, high, low, close, volume
    """

    data = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for ticker={ticker}, period={period}, interval={interval}.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    cleaned = data[DEFAULT_FEATURE_COLUMNS].copy()
    cleaned = cleaned.dropna().reset_index(drop=True)

    if len(cleaned) <= 20:
        raise ValueError("Downloaded dataset is too small for lookback-based training.")

    return cleaned


def add_macro_trend_features(df: pd.DataFrame, trend_window: int) -> pd.DataFrame:
    """Estimate a trailing macro trend and express OHLC prices as deviations from it."""

    transformed = df.copy()
    transformed["macro_trend"] = transformed["close"].rolling(window=trend_window, min_periods=1).mean()
    transformed["detrended_open"] = transformed["open"] - transformed["macro_trend"]
    transformed["detrended_high"] = transformed["high"] - transformed["macro_trend"]
    transformed["detrended_low"] = transformed["low"] - transformed["macro_trend"]
    transformed["detrended_close"] = transformed["close"] - transformed["macro_trend"]
    return transformed


def explain_tensor_shapes(lookback: int, num_features: int, batch_size: int) -> None:
    """Print the expected tensor shapes for beginners."""

    print("Tensor shape guide")
    print(f"Input batch X shape : ({batch_size}, {lookback}, {num_features})")
    print("Meaning            : (batch_size, sequence_length, input_size)")
    print(f"Target batch y shape: ({batch_size}, 1)")
    print("Model output shape : (batch_size, 1)")
