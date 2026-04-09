"""Short end-to-end demo: one dataset, LSTM + Transformer regressors, sample metrics.

From the repository root (after `uv sync` or `pip install -e .`):

    uv run python demo.py

Requires network access for yfinance. Checkpoints are written to the current working directory.
"""

from __future__ import annotations

import pandas as pd
import torch
import yfinance as yf

from ece1508.lstm.baseline import add_macro_trend_features, explain_tensor_shapes
from ece1508.lstm.data_preparation import prepare_datasets
from ece1508.lstm.evaluate import evaluate_model, print_metrics
from ece1508.lstm.lstm_forecaster import LSTMForecaster
from ece1508.lstm.train import train_model
from ece1508.transformer import TimeSeriesTransformer


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
    if len(cleaned) < 120:
        raise ValueError(
            f"Downloaded dataset is too small ({len(cleaned)} rows). "
            "Try a longer period or coarser interval."
        )
    return cleaned


def main() -> None:
    ticker = "AAPL"
    lookback = 16
    trend_window = 12
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

    test_baseline = prepared["test"].baseline
    test_previous_close = prepared["test"].previous_close

    # --- LSTM regression (sequence-to-one) ---
    lstm = LSTMForecaster(
        input_size=num_features,
        hidden_size=32,
        num_layers=1,
        dropout=0.1,
    )
    lstm, lstm_history = train_model(
        model=lstm,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        patience=5,
        checkpoint_path="demo_lstm_model.pt",
    )
    print("=== LSTM: last epoch losses ===")
    print(f"train_loss: {lstm_history['train_loss'][-1]:.6f}")
    print(f"val_loss:   {lstm_history['val_loss'][-1]:.6f}\n")

    lstm_results = evaluate_model(
        model=lstm,
        dataloader=dataloaders["test"],
        device=device,
        target_scaler=target_scaler,
        reconstruction_baseline=test_baseline,
        previous_close=test_previous_close,
    )
    print_metrics(lstm_results["metrics"], split_name="LSTM — test (price space)")

    # --- Transformer regression (same tensors; horizon=1 matches y shape) ---
    transformer = TimeSeriesTransformer(
        input_dim=num_features,
        d_model=48,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        horizon=1,
        dim_feedforward=128,
        max_pos_len=max(lookback * 2, 64),
    )
    transformer, tf_history = train_model(
        model=transformer,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        patience=5,
        checkpoint_path="demo_transformer_model.pt",
    )
    print("\n=== Transformer: last epoch losses ===")
    print(f"train_loss: {tf_history['train_loss'][-1]:.6f}")
    print(f"val_loss:   {tf_history['val_loss'][-1]:.6f}\n")

    tf_results = evaluate_model(
        model=transformer,
        dataloader=dataloaders["test"],
        device=device,
        target_scaler=target_scaler,
        reconstruction_baseline=test_baseline,
        previous_close=test_previous_close,
    )
    print_metrics(tf_results["metrics"], split_name="Transformer — test (price space)")

    print("\nDemo finished.")
    print("Checkpoints: demo_lstm_model.pt, demo_transformer_model.pt (overwritten each run).")


if __name__ == "__main__":
    main()
