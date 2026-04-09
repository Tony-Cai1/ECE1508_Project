"""PyTorch LSTM model for one-step-ahead stock forecasting."""

from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """
    LSTM baseline for sequence-to-one forecasting.

    Expected input shape:
    (batch_size, sequence_length, input_size)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        x shape: (batch_size, sequence_length, input_size)
        output shape: (batch_size, 1)
        """

        _, (hidden_state, _) = self.lstm(x)
        final_hidden = hidden_state[-1]
        prediction = self.fc(final_hidden)
        return prediction
