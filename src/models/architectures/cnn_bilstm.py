"""
CNN-BiLSTM hybrid: CNN feature extractor followed by a Bidirectional LSTM.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class CNNBiLSTMModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        image_size: Tuple[int, int] = (48, 48),
        in_channels: int = 1,
        cnn_channels: List[int] = None,
        cnn_kernel_sizes: List[int] = None,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
        fc_dims: List[int] = None,
        dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        cnn_channels = cnn_channels or [32, 64, 128]
        cnn_kernel_sizes = cnn_kernel_sizes or [3] * len(cnn_channels)
        fc_dims = fc_dims or [128]

        # --- 1D CNN over width dimension ---
        conv_layers: list = []
        prev_ch = in_channels
        for ch, ks in zip(cnn_channels, cnn_kernel_sizes):
            conv_layers += [
                nn.Conv2d(prev_ch, ch, kernel_size=(1, ks), padding=(0, ks // 2)),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ]
            prev_ch = ch
        self.cnn = nn.Sequential(*conv_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *image_size)
            feat = self.cnn(dummy)
            _, C_out, H_out, W_out = feat.shape
            self.cnn_out_features = C_out * W_out

        self.lstm = nn.LSTM(
            input_size=self.cnn_out_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=True,
        )

        bi_hidden = lstm_hidden_size * 2

        layers: list = []
        prev = bi_hidden
        for dim in fc_dims:
            layers += [nn.Linear(prev, dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = dim
        layers.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)
        B, C, H, W = feat.shape
        feat = feat.permute(0, 2, 1, 3).reshape(B, H, C * W)
        out, _ = self.lstm(feat)
        # concat last forward + first backward
        fwd = out[:, -1, : out.size(2) // 2]
        bwd = out[:, 0, out.size(2) // 2 :]
        x = torch.cat([fwd, bwd], dim=-1)
        return self.classifier(x)
