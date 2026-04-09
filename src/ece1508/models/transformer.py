"""
Transformer model variants for stock return forecasting.

Three modes sharing the same encoder backbone:
  TransformerRegression  – predicts a continuous close return  (batch, 1)
  TransformerClassifier  – predicts P(return > 0) via sigmoid  (batch, 1)
  TransformerMultiTask   – both heads simultaneously           tuple((batch,1), (batch,1))

PositionalEncoding is self-contained here so this package has no cross-directory imports.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to token embeddings."""

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


class _TransformerBackbone(nn.Module):
    """Shared Transformer encoder. Subclasses add prediction heads on top."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        dim_feedforward: int = 256,
        max_pos_len: int = 5000,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = _PositionalEncoding(d_model, max_len=max_pos_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=dim_feedforward,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _encode(self, src: torch.Tensor) -> torch.Tensor:
        """Return the last-position representation: shape (batch, d_model)."""
        src = self.input_linear(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        return self.transformer_encoder(src)[:, -1, :]


class TransformerRegression(_TransformerBackbone):
    """
    Transformer with a single regression head.

    Input:  (batch, seq, input_dim)
    Output: (batch, 1)  – predicted close return
    """

    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 3, dropout: float = 0.2,
                 dim_feedforward: int = 256, max_pos_len: int = 5000) -> None:
        super().__init__(input_dim, d_model, nhead, num_layers, dropout, dim_feedforward, max_pos_len)
        self.head = nn.Linear(d_model, 1)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        return self.head(self._encode(src))


class TransformerClassifier(_TransformerBackbone):
    """
    Transformer with a binary classification head predicting P(return > 0).

    Input:  (batch, seq, input_dim)
    Output: (batch, 1)  – probability in [0, 1]
    """

    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 3, dropout: float = 0.2,
                 dim_feedforward: int = 256, max_pos_len: int = 5000) -> None:
        super().__init__(input_dim, d_model, nhead, num_layers, dropout, dim_feedforward, max_pos_len)
        self.head = nn.Linear(d_model, 1)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self._encode(src)))


class TransformerMultiTask(_TransformerBackbone):
    """
    Transformer with both regression and classification heads.

    Input:  (batch, seq, input_dim)
    Output: tuple of
              reg_out – (batch, 1) continuous return prediction
              cls_out – (batch, 1) P(return > 0) in [0, 1]
    """

    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 3, dropout: float = 0.2,
                 dim_feedforward: int = 256, max_pos_len: int = 5000) -> None:
        super().__init__(input_dim, d_model, nhead, num_layers, dropout, dim_feedforward, max_pos_len)
        self.reg_head = nn.Linear(d_model, 1)
        self.cls_head = nn.Linear(d_model, 1)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._encode(src)
        return self.reg_head(h), torch.sigmoid(self.cls_head(h))
