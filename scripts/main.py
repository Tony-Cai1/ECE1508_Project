"""Example entry point for the stock LSTM forecasting baseline."""

from __future__ import annotations

import torch

from ece1508.baseline import (
    add_macro_trend_features,
    download_stock_data,
    explain_tensor_shapes,
)
from ece1508.data_preparation import prepare_datasets
from ece1508.evaluate import evaluate_model, plot_predictions, plot_training_history, print_metrics
from ece1508.lstm_forecaster import LSTMForecaster
from ece1508.train import (
    print_convergence_summary,
    print_training_procedure_summary,
    print_validation_strategy_summary,
    train_model,
)


def main() -> None:
    ticker = "TSLA"
    period = "7d"
    interval = "1m"
    lookback = 60
    trend_window = 30
    hidden_size = 64
    num_layers = 2
    dropout = 0.2
    batch_size = 32
    learning_rate = 1e-3
    epochs = 50
    feature_columns = ["detrended_open", "detrended_high", "detrended_low", "detrended_close", "volume"]
    target_column = "detrended_close"
    test_plot_path = f"{ticker.lower()}_actual_vs_predicted_test_set.png"
    training_plot_path = f"{ticker.lower()}_training_history.png"

    df = download_stock_data(ticker=ticker, period=period, interval=interval)
    df = add_macro_trend_features(df, trend_window=trend_window)
    print(f"Downloaded {len(df)} rows for {ticker} with interval={interval} over period={period}.")
    print(
        "Target decomposition: actual price = macro trend + short-term deviation, "
        f"using a trailing {trend_window}-day moving-average trend."
    )

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

    explain_tensor_shapes(lookback=lookback, num_features=num_features, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
        patience=10,
        checkpoint_path="best_lstm_model.pt",
    )

    print_validation_strategy_summary(
        train_size=len(prepared["train_df"]),
        val_size=len(prepared["val_df"]),
        test_size=len(prepared["test_df"]),
        lookback=lookback,
    )
    print_training_procedure_summary(
        history=history,
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=10,
    )
    print_convergence_summary(history)
    plot_training_history(
        history,
        title=f"LSTM Training Curve: {ticker}",
        save_path=training_plot_path,
    )
    print(f"Saved training history plot to: {training_plot_path}")

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

    print_metrics(test_results["metrics"], split_name=f"Test (lookback={lookback})")
    plot_predictions(
        actuals=test_results["actuals"],
        predictions=test_results["predictions"],
        title=f"{ticker} Test Set: Actual vs Predicted Close Price",
        save_path=test_plot_path,
    )
    print(f"Saved test-set prediction plot to: {test_plot_path}")

    print("\nHow to extend later")
    print("- Multi-stock: add a stock identifier column and group windows by ticker before batching.")
    print("- Transformer baseline: replace the LSTM encoder with a transformer encoder using the same dataset pipeline.")
    print("- Hybrid models: add CNN or attention blocks before or after the LSTM for richer sequence representations.")


if __name__ == "__main__":
    main()
