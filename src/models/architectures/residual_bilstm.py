"""
Residual Bidirectional LSTM for image classification.

Adds skip (residual) connections between stacked BiLSTM layers to allow
gradient flow through deeper networks.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class ResidualBiLSTMBlock(nn.Module):
    """Single BiLSTM layer with a residual connection."""
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        # projection if input size != output size
        self.proj = (
            nn.Linear(input_size, hidden_size * 2)
            if input_size != hidden_size * 2
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        out, _ = self.lstm(x)
        out = self.layer_norm(out + residual)
        out = self.dropout(out)
        return out


class ResidualBiLSTMModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        image_size: Tuple[int, int] = (48, 48),
        in_channels: int = 1,
        hidden_size: int = 256,
        num_layers: int = 4,
        dropout: float = 0.3,
        fc_dims: List[int] = None,
        **kwargs,
    ):
        super().__init__()
        fc_dims = fc_dims or [128]
        input_size = image_size[1] * in_channels
        bi_hidden = hidden_size * 2

        blocks: list = []
        for i in range(num_layers):
            in_sz = input_size if i == 0 else bi_hidden
            blocks.append(ResidualBiLSTMBlock(in_sz, hidden_size, dropout))
        self.lstm_blocks = nn.ModuleList(blocks)

        layers: list = []
        prev = bi_hidden
        for dim in fc_dims:
            layers += [nn.Linear(prev, dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = dim
        layers.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, H, W * C)
        for block in self.lstm_blocks:
            x = block(x)
        x = x[:, -1, :]  # last time step
        return self.classifier(x)
