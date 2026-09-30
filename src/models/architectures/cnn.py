"""
Configurable CNN architecture for image classification.

Supports:
  - Variable number of convolutional blocks
  - Configurable channels, kernel sizes, pooling
  - Batch normalisation toggle
  - Multiple activation functions
  - Configurable fully-connected head
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class CNN(nn.Module):
    """
    A flexible convolutional neural network.

    Args:
        num_classes:  Number of output classes.
        image_size:   ``(H, W)`` of input images.
        in_channels:  Number of input channels (1=grayscale, 3=RGB).
        channels:     List of output channels per conv block.
        kernel_sizes: List of kernel sizes per conv block.
        pool_sizes:   List of max-pool sizes per conv block.
        fc_dims:      List of hidden FC layer dimensions.
        dropout:      Dropout probability.
        batch_norm:   Whether to use batch normalisation.
        activation:   Activation function name.
    """

    def __init__(
        self,
        num_classes: int = 7,
        image_size: Tuple[int, int] = (48, 48),
        in_channels: int = 1,
        channels: List[int] = None,
        kernel_sizes: List[int] = None,
        pool_sizes: List[int] = None,
        fc_dims: List[int] = None,
        dropout: float = 0.5,
        batch_norm: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        channels = channels or [32, 64, 128, 256]
        kernel_sizes = kernel_sizes or [3] * len(channels)
        pool_sizes = pool_sizes or [2] * len(channels)
        fc_dims = fc_dims or [512]

        act_fn = _get_activation(activation)

        # --- convolutional blocks ---
        conv_layers: list = []
        prev_ch = in_channels
        for ch, ks, ps in zip(channels, kernel_sizes, pool_sizes):
            pad = ks // 2
            conv_layers.append(nn.Conv2d(prev_ch, ch, kernel_size=ks, padding=pad))
            if batch_norm:
                conv_layers.append(nn.BatchNorm2d(ch))
            conv_layers.append(act_fn)
            conv_layers.append(nn.MaxPool2d(kernel_size=ps))
            prev_ch = ch

        self.features = nn.Sequential(*conv_layers)

        # --- compute flattened size dynamically ---
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *image_size)
            flat_size = self.features(dummy).view(1, -1).size(1)

        # --- classifier head ---
        fc_layers: list = []
        prev_dim = flat_size
        for dim in fc_dims:
            fc_layers.append(nn.Linear(prev_dim, dim))
            fc_layers.append(act_fn)
            fc_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        fc_layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def _get_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
        "elu": nn.ELU(inplace=True),
        "gelu": nn.GELU(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(activations)}")
    return activations[name]
