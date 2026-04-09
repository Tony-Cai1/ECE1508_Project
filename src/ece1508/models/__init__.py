"""
models — LSTM and Transformer variants for stock return forecasting.

Each architecture comes in three modes:
  Regression  – continuous return prediction, trained with StockReturnLoss
  Classifier  – binary direction prediction, trained with BCE
  MultiTask   – both heads jointly, trained with a weighted combination of both losses
"""

from .lstm import LSTMClassifier, LSTMMultiTask, LSTMRegression
from .transformer import (
    TransformerClassifier,
    TransformerMultiTask,
    TransformerRegression,
)

__all__ = [
    "LSTMRegression",
    "LSTMClassifier",
    "LSTMMultiTask",
    "TransformerRegression",
    "TransformerClassifier",
    "TransformerMultiTask",
]
