"""Training utilities for the LSTM forecasting baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EarlyStopping:
    """Simple early stopping helper based on validation loss."""

    patience: int = 10
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics."""

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    return {"mse": mse, "mae": mae, "rmse": rmse}


def run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    """
    Run one training or evaluation epoch.

    If optimizer is None, the model runs in evaluation mode.
    """

    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    all_targets: List[np.ndarray] = []
    all_predictions: List[np.ndarray] = []

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        all_targets.append(y_batch.detach().cpu().numpy())
        all_predictions.append(predictions.detach().cpu().numpy())

    y_true = np.vstack(all_targets).reshape(-1)
    y_pred = np.vstack(all_predictions).reshape(-1)

    metrics = regression_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(dataloader.dataset)
    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    patience: int = 10,
    checkpoint_path: str = "best_lstm_model.pt",
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train the LSTM model and save the best checkpoint by validation loss."""

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stopper = EarlyStopping(patience=patience)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_mae": [],
        "train_rmse": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
    }

    checkpoint = Path(checkpoint_path)

    best_val_loss = float("inf")
    model.to(device)

    for epoch in range(1, epochs + 1):
        train_metrics = run_one_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = run_one_epoch(model, val_loader, criterion, device, optimizer=None)

        history["train_loss"].append(train_metrics["loss"])
        history["train_mae"].append(train_metrics["mae"])
        history["train_rmse"].append(train_metrics["rmse"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"Val RMSE: {val_metrics['rmse']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), checkpoint)

        if early_stopper.step(val_metrics["loss"]):
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model, history


def print_training_procedure_summary(
    history: Dict[str, List[float]],
    learning_rate: float,
    batch_size: int,
    patience: int,
) -> None:
    """Print a concise explanation of how training was run."""

    trained_epochs = len(history["train_loss"])
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_val_loss = history["val_loss"][best_epoch - 1]

    print("\nTraining Procedure")
    print("- Optimizer: Adam")
    print("- Loss function: Mean Squared Error (MSE)")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Batch size: {batch_size}")
    print(f"- Epochs completed: {trained_epochs}")
    print(f"- Early stopping patience: {patience} epochs without validation-loss improvement")
    print(f"- Best model checkpoint selected by validation loss at epoch {best_epoch}")
    print(f"- Best validation loss: {best_val_loss:.4f}")


def print_validation_strategy_summary(
    train_size: int,
    val_size: int,
    test_size: int,
    lookback: int,
) -> None:
    """Print the validation strategy used by the pipeline."""

    print("\nValidation Strategy")
    print("- Data split: chronological train/validation/test split")
    print(f"- Sequence lookback window: {lookback} time steps")
    print(f"- Training rows: {train_size}")
    print(f"- Validation rows: {val_size}")
    print(f"- Test rows: {test_size}")
    print("- Validation data is not shuffled during evaluation")
    print("- Feature and target scalers are fit on training data only to avoid leakage")
    print("- Early stopping monitors validation loss")


def print_convergence_summary(history: Dict[str, List[float]]) -> None:
    """Describe whether the loss decreased during training."""

    trained_epochs = len(history["train_loss"])
    initial_train_loss = history["train_loss"][0]
    final_train_loss = history["train_loss"][-1]
    initial_val_loss = history["val_loss"][0]
    final_val_loss = history["val_loss"][-1]
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_val_loss = history["val_loss"][best_epoch - 1]

    print("\nConvergence Behavior")
    print(f"- Initial train loss: {initial_train_loss:.4f}")
    print(f"- Final train loss: {final_train_loss:.4f}")
    print(f"- Initial validation loss: {initial_val_loss:.4f}")
    print(f"- Final validation loss: {final_val_loss:.4f}")
    print(f"- Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    if final_train_loss < initial_train_loss and best_val_loss <= initial_val_loss:
        print("- Observation: loss decreased during training, indicating convergence.")
    elif final_train_loss < initial_train_loss:
        print("- Observation: training loss decreased, but validation loss should be checked for overfitting.")
    else:
        print("- Observation: loss did not clearly decrease; hyperparameters or data setup may need adjustment.")

    print(f"- Total trained epochs: {trained_epochs}")
