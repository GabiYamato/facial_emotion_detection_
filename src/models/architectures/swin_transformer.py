"""
Swin Transformer for image classification.

A hierarchical vision transformer using shifted windows for efficient
self-attention. Simplified implementation suitable for small image sizes
(e.g., 48×48).

Reference: Liu et al., "Swin Transformer: Hierarchical Vision Transformer
using Shifted Windows", ICCV 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Optional


# =========================================================================
# Window utilities
# =========================================================================

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition into non-overlapping windows of size ``window_size``."""
    B, H, W, C = x.shape
    nH = H // window_size
    nW = W // window_size
    x = x.view(B, nH, window_size, nW, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return x  # (B*nH*nW, ws, ws, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window_partition."""
    nH = H // window_size
    nW = W // window_size
    B = int(windows.shape[0] / (nH * nW))
    x = windows.view(B, nH, nW, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# =========================================================================
# Swin blocks
# =========================================================================

class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""
    def __init__(self, dim: int, window_size: int, num_heads: int, attn_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        # relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # compute relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, ws, ws)
        coords_flatten = coords.view(2, -1)  # (2, ws*ws)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # add relative position bias
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1
        )
        bias = bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn + bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer block with optional shifted window."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        x_res = x
        x = self.norm1(x).view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # partition windows
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # window attention
        attn_out = self.attn(x_windows)

        # merge windows
        attn_out = attn_out.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_out, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, C)
        x = x_res + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    """Patch merging for downsampling spatial resolution by 2×."""
    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, H // 2, W // 2


class SwinStage(nn.Module):
    """One stage of the Swin Transformer (multiple blocks + optional merging)."""
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        downsample: bool = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else window_size // 2
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                )
            )
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x: torch.Tensor, H: int, W: int):
        for block in self.blocks:
            x = block(x, H, W)
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        return x, H, W


# =========================================================================
# Swin Transformer
# =========================================================================

class SwinTransformer(nn.Module):
    """
    Swin Transformer for image classification.

    Args:
        num_classes:       Output classes.
        image_size:        ``(H, W)`` of input.
        in_channels:       Input channels.
        patch_size:        Initial patch embedding size.
        embed_dim:         Base embedding dimension (doubles each stage).
        depths:            Number of Transformer blocks per stage.
        num_heads:         Attention heads per stage.
        window_size:       Window size for local attention.
        mlp_ratio:         MLP hidden ratio.
        dropout:           Dropout rate.
        attention_dropout: Attention dropout rate.
    """
    def __init__(
        self,
        num_classes: int = 7,
        image_size: Tuple[int, int] = (48, 48),
        in_channels: int = 1,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: List[int] = None,
        num_heads: List[int] = None,
        window_size: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        depths = depths or [2, 2, 6, 2]
        num_heads = num_heads or [3, 6, 12, 24]

        # Adjust depths/heads to fit spatial dimensions
        H, W = image_size[0] // patch_size, image_size[1] // patch_size
        num_stages = 0
        h, w = H, W
        for _ in range(len(depths)):
            if h < window_size or w < window_size:
                break
            num_stages += 1
            h, w = h // 2, w // 2
        # Use at most num_stages stages (last stage has no downsampling)
        num_stages = max(num_stages, 1)
        depths = depths[:num_stages]
        num_heads = num_heads[:num_stages]

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.H = H
        self.W = W

        # Build stages
        self.stages = nn.ModuleList()
        dim = embed_dim
        for i, (d, nh) in enumerate(zip(depths, num_heads)):
            # Adjust window size if spatial resolution is smaller
            ws = min(window_size, H // (2 ** i), W // (2 ** i))
            ws = max(ws, 1)
            downsample = i < len(depths) - 1
            self.stages.append(
                SwinStage(
                    dim=dim,
                    depth=d,
                    num_heads=min(nh, dim),  # heads can't exceed dim
                    window_size=ws,
                    mlp_ratio=mlp_ratio,
                    drop=dropout,
                    attn_drop=attention_dropout,
                    downsample=downsample,
                )
            )
            if downsample:
                dim *= 2

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)  # (B, embed_dim, H/p, W/p)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.patch_norm(x)

        for stage in self.stages:
            x, H, W = stage(x, H, W)

        x = self.norm(x)
        x = x.mean(dim=1)  # global average pooling
        return self.head(x)
