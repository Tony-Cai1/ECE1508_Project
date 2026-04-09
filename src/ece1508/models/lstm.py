"""
LSTM model variants for stock return forecasting.

Three modes sharing the same encoder backbone:
  LSTMRegression  – predicts a continuous close return  (batch, 1)
  LSTMClassifier  – predicts P(return > 0) via sigmoid  (batch, 1)
  LSTMMultiTask   – both heads simultaneously           tuple((batch,1), (batch,1))
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _LSTMBackbone(nn.Module):
    """Shared LSTM encoder. Subclasses add prediction heads on top."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final-layer hidden state: shape (batch, hidden_size)."""
        _, (h, _) = self.lstm(x)
        return h[-1]


class LSTMRegression(_LSTMBackbone):
    """
    LSTM with a single regression head.

    Input:  (batch, seq, input_size)
    Output: (batch, 1)  – predicted close return
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__(input_size, hidden_size, num_layers, dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._encode(x))


class LSTMClassifier(_LSTMBackbone):
    """
    LSTM with a binary classification head predicting P(return > 0).

    Input:  (batch, seq, input_size)
    Output: (batch, 1)  – probability in [0, 1]
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__(input_size, hidden_size, num_layers, dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self._encode(x)))


class LSTMMultiTask(_LSTMBackbone):
    """
    LSTM with both regression and classification heads.

    Input:  (batch, seq, input_size)
    Output: tuple of
              reg_out – (batch, 1) continuous return prediction
              cls_out – (batch, 1) P(return > 0) in [0, 1]
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__(input_size, hidden_size, num_layers, dropout)
        self.reg_head = nn.Linear(hidden_size, 1)
        self.cls_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._encode(x)
        return self.reg_head(h), torch.sigmoid(self.cls_head(h))
