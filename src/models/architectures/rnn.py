"""
Vanilla RNN model for image classification.

Treats each row of the image as a time step so the input is reshaped
from ``(B, C, H, W)`` → ``(B, H, W*C)``.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class RNNModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        image_size: Tuple[int, int] = (48, 48),
        in_channels: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        nonlinearity: str = "tanh",
        fc_dims: List[int] = None,
        **kwargs,
    ):
        super().__init__()
        fc_dims = fc_dims or [128]
        self.image_size = image_size
        self.in_channels = in_channels

        input_size = image_size[1] * in_channels  # W * C

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            nonlinearity=nonlinearity,
        )

        rnn_out = hidden_size * (2 if bidirectional else 1)

        layers: list = []
        prev = rnn_out
        for dim in fc_dims:
            layers += [nn.Linear(prev, dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = dim
        layers.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, H, W * C)  # (B, H, W*C)
        out, _ = self.rnn(x)
        x = out[:, -1, :]  # last time step
        return self.classifier(x)
