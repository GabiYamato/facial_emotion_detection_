"""
Vision Transformer (ViT) for image classification.

Splits the image into fixed-size patches, linearly embeds them, adds
positional embeddings, and processes with a standard Transformer encoder.

Reference: Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class PatchEmbedding(nn.Module):
    """Split image into patches and project to embedding dimension."""
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int, image_size: Tuple[int, int]):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, num_patches, embed_dim)
        x = self.proj(x)                     # (B, embed_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)     # (B, num_patches, embed_dim)
        return x


class TransformerEncoderBlock(nn.Module):
    """Standard Transformer encoder block with pre-norm."""
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-norm
        x_ln = self.ln1(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT).

    Args:
        num_classes:   Number of output classes.
        image_size:    ``(H, W)`` of input images.
        in_channels:   Input channels.
        patch_size:    Size of each square patch.
        embed_dim:     Embedding / hidden dimension.
        num_heads:     Number of attention heads.
        num_layers:    Number of Transformer encoder layers.
        mlp_dim:       Hidden dim of the feed-forward network.
        dropout:       Dropout rate inside Transformer blocks.
        emb_dropout:   Dropout on patch + positional embeddings.
    """
    def __init__(
        self,
        num_classes: int = 7,
        image_size: Tuple[int, int] = (48, 48),
        in_channels: int = 1,
        patch_size: int = 8,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim, image_size)
        num_patches = self.patch_embed.num_patches

        # CLS token + positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.emb_dropout = nn.Dropout(emb_dropout)

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)                           # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)             # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                     # (B, N+1, D)
        x = self.emb_dropout(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.ln(x[:, 0])  # CLS token output
        return self.head(x)
