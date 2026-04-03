"""Evaluation helpers for forecasting results."""

from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from train import regression_metrics


def directional_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    previous_close: np.ndarray,
) -> Dict[str, float]:
    """Measure whether predicted price direction matches the actual direction."""

    if len(previous_close) != len(y_true):
        raise ValueError("previous_close length must match target length.")

    true_direction = np.sign(y_true - previous_close)
    pred_direction = np.sign(y_pred - previous_close)
    matches = pred_direction == true_direction

    up_mask = true_direction > 0
    down_mask = true_direction < 0

    up_accuracy = float(matches[up_mask].mean()) if np.any(up_mask) else float("nan")
    down_accuracy = float(matches[down_mask].mean()) if np.any(down_mask) else float("nan")

    return {
        "directional_accuracy": float(matches.mean()),
        "up_accuracy": up_accuracy,
        "down_accuracy": down_accuracy,
        "up_samples": float(np.sum(up_mask)),
        "down_samples": float(np.sum(down_mask)),
    }


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference and return predictions and targets."""

    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.append(outputs.cpu().numpy())
            targets.append(y_batch.cpu().numpy())

    y_pred = np.vstack(predictions).reshape(-1)
    y_true = np.vstack(targets).reshape(-1)
    return y_pred, y_true


def inverse_transform_series(
    values: np.ndarray,
    scaler: StandardScaler | None,
) -> np.ndarray:
    """Inverse transform a 1D array if a scaler is available."""

    if scaler is None:
        return values
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(-1)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    target_scaler: StandardScaler | None = None,
    reconstruction_baseline: np.ndarray | None = None,
    previous_close: np.ndarray | None = None,
) -> Dict[str, np.ndarray | Dict[str, float]]:
    """Predict and compute metrics, optionally reconstructing final prices."""

    y_pred_scaled, y_true_scaled = predict(model, dataloader, device)
    y_pred = inverse_transform_series(y_pred_scaled, target_scaler)
    y_true = inverse_transform_series(y_true_scaled, target_scaler)

    if reconstruction_baseline is not None:
        if len(reconstruction_baseline) != len(y_pred):
            raise ValueError("Reconstruction baseline length must match prediction length.")
        y_pred = y_pred + reconstruction_baseline
        y_true = y_true + reconstruction_baseline

    metrics = regression_metrics(y_true, y_pred)
    if previous_close is not None:
        metrics.update(directional_metrics(y_true, y_pred, previous_close))

    return {
        "predictions": y_pred,
        "actuals": y_true,
        "metrics": metrics,
    }


def print_metrics(metrics: Dict[str, float], split_name: str = "Test") -> None:
    """Print regression metrics clearly."""

    print(f"{split_name} Metrics")
    print(f"MSE : {metrics['mse']:.4f}")
    print(f"MAE : {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    if "directional_accuracy" in metrics:
        print(f"Direction Accuracy: {metrics['directional_accuracy']:.4%}")
        print(f"Up Accuracy       : {metrics['up_accuracy']:.4%}")
        print(f"Down Accuracy     : {metrics['down_accuracy']:.4%}")


def plot_predictions(
    actuals: np.ndarray,
    predictions: np.ndarray,
    title: str = "Actual vs Predicted Close Price",
    save_path: str | None = None,
) -> None:
    """Plot actual and predicted values on the same figure."""

    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label="Actual", linewidth=2)
    plt.plot(predictions, label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if "agg" in plt.get_backend().lower():
        plt.close()
    else:
        plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training and Validation Loss",
    save_path: str | None = None,
) -> None:
    """Plot train and validation loss across epochs."""

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Validation Loss", linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if "agg" in plt.get_backend().lower():
        plt.close()
    else:
        plt.show()
