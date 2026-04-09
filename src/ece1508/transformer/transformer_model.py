"""PyTorch Transformer model for time series stock return forecasting."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to the token embeddings."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TimeSeriesTransformer(nn.Module):
    """
    Transformer encoder model for sequence-to-one stock return forecasting.

    Expected input shape:  (batch_size, sequence_length, input_dim)
    Output shape:          (batch_size, horizon)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        horizon: int = 1,
        dim_feedforward: int = 256,
        max_pos_len: int = 5000,
    ) -> None:
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.horizon = horizon

        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_pos_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=dim_feedforward,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, horizon)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        src shape:    (batch_size, sequence_length, input_dim)
        output shape: (batch_size, horizon)
        """
        src = self.input_linear(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return self.decoder(output[:, -1, :])
