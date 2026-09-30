"""
LSTM with Additive (Bahdanau-style) Attention for image classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class Attention(nn.Module):
    """Additive attention over LSTM hidden states."""
    def __init__(self, hidden_size: int, attention_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: (B, T, H)
        Returns:
            context: (B, H)
            weights: (B, T)
        """
        energy = torch.tanh(self.W(lstm_output))   # (B, T, attn_dim)
        scores = self.v(energy).squeeze(-1)         # (B, T)
        weights = F.softmax(scores, dim=-1)         # (B, T)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)  # (B, H)
        return context, weights


class LSTMAttentionModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        image_size: Tuple[int, int] = (48, 48),
        in_channels: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        attention_dim: int = 128,
        fc_dims: List[int] = None,
        **kwargs,
    ):
        super().__init__()
        fc_dims = fc_dims or [128]
        input_size = image_size[1] * in_channels

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = Attention(hidden_size, attention_dim)

        layers: list = []
        prev = hidden_size
        for dim in fc_dims:
            layers += [nn.Linear(prev, dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = dim
        layers.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, H, W * C)
        out, _ = self.lstm(x)
        context, _ = self.attention(out)
        return self.classifier(context)
