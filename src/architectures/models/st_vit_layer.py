from typing import Dict, Any, List

import torch
from torch import nn
from torch.nn import functional as F


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(
        self,
        x,
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(
        self,
        x,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) * self.scaling
        attn_scores = attn_scores.softmax(dim=-1)
        attn_scores = self.attn_drop(attn_scores)

        out = (attn_scores @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class FeedForward(nn.Module):
    def __init__(
        self,
        embed_dim,
        ff_dim,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        x,
    ) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        x,
    ) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        return x


class TemporalDifferenceConvolution(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
    ) -> None:
        super().__init__()
        self.tdc = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(
        self,
        x,
    ) -> torch.Tensor:
        B, C, T, H, W = x.shape
        x_diff = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        x_diff = F.pad(x_diff, (0, 0, 0, 0, 0, 0, 1, 0))
        x_diff = self.tdc(x_diff)
        return x_diff


class PhysFormerModel(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        num_layers,
        num_heads,
        ff_dim,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(embed_dim, num_heads, ff_dim)
                for _ in range(num_layers)
            ]
        )
        self.tdc = TemporalDifferenceConvolution(
            embed_dim, embed_dim, kernel_size=3, padding=1
        )
        self.ffn = FeedForward(embed_dim, ff_dim)
        self.output_layer = nn.Linear(embed_dim, 1)  # rPPG prediction

    def forward(
        self,
        x,
    ) -> torch.Tensor:
        x = self.patch_embed(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.permute(0, 2, 1).unsqueeze(2)  # B, C, 1, T, P -> B, C, T, 1, P
        x = self.tdc(x).squeeze(2)
        x = self.ffn(x)
        x = self.output_layer(x)
        return x
