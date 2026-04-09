"""Short end-to-end demo: download data, train a few epochs, print sample metrics.

Run: python demo.py

Requires network access for yfinance. Outputs are printed to the console; plots are
optional (disabled by default for a fast, headless-friendly run).
"""

from __future__ import annotations

import pandas as pd
import torch
import yfinance as yf

from data_preparation import prepare_datasets
from evaluate import evaluate_model, print_metrics
from main import add_macro_trend_features, explain_tensor_shapes
from model import LSTMForecaster
from train import train_model


def demo_download(ticker: str = "AAPL", period: str = "1mo", interval: str = "1h") -> pd.DataFrame:
    """Fetch a small OHLCV window for a quick local run."""

    data = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for ticker={ticker}.")

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
    cleaned = data[["open", "high", "low", "close", "volume"]].copy()
    cleaned = cleaned.dropna().reset_index(drop=True)
    # Chronological split + sliding windows need enough rows (train share must exceed lookback).
    if len(cleaned) < 120:
        raise ValueError(
            f"Downloaded dataset is too small ({len(cleaned)} rows). "
            "Try a longer period or coarser interval."
        )
    return cleaned


def main() -> None:
    ticker = "AAPL"
    # Keep lookback small so 15% validation slice still has enough rows for sliding windows.
    lookback = 16
    trend_window = 12
    hidden_size = 32
    num_layers = 1
    dropout = 0.1
    batch_size = 16
    learning_rate = 1e-3
    epochs = 3
    feature_columns = [
        "detrended_open",
        "detrended_high",
        "detrended_low",
        "detrended_close",
        "volume",
    ]
    target_column = "detrended_close"

    print("=== Sample input: market data (first 3 rows, selected columns) ===")
    df = demo_download(ticker=ticker)
    df = add_macro_trend_features(df, trend_window=trend_window)
    print(df[["close", "macro_trend", "detrended_close"]].head(3).to_string())
    print()

    prepared = prepare_datasets(
        df=df,
        feature_columns=feature_columns,
        target_column=target_column,
        lookback=lookback,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=batch_size,
        normalize=True,
        baseline_column="macro_trend",
    )
    dataloaders = prepared["dataloaders"]
    target_scaler = prepared["target_scaler"]
    num_features = len(prepared["feature_columns"])

    sample_X, sample_y = next(iter(dataloaders["train"]))
    print("=== Sample model input / target tensors (one batch) ===")
    print(f"X batch shape: {tuple(sample_X.shape)}  (batch, lookback, features)")
    print(f"y batch shape: {tuple(sample_y.shape)}  (batch, 1)")
    print(f"Feature columns: {prepared['feature_columns']}")
    print()

    explain_tensor_shapes(lookback=lookback, num_features=num_features, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = LSTMForecaster(
        input_size=num_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    model, history = train_model(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        patience=5,
        checkpoint_path="demo_lstm_model.pt",
    )

    print("=== Sample output: last epoch losses ===")
    print(f"train_loss: {history['train_loss'][-1]:.6f}")
    print(f"val_loss:   {history['val_loss'][-1]:.6f}")
    print()

    test_baseline = prepared["test"].baseline
    test_previous_close = prepared["test"].previous_close
    test_results = evaluate_model(
        model=model,
        dataloader=dataloaders["test"],
        device=device,
        target_scaler=target_scaler,
        reconstruction_baseline=test_baseline,
        previous_close=test_previous_close,
    )
    print("=== Sample output: test metrics (price space) ===")
    print_metrics(test_results["metrics"], split_name="Demo test")
    print("\nDemo finished. Checkpoint saved as demo_lstm_model.pt (overwritten each run).")


if __name__ == "__main__":
    main()
