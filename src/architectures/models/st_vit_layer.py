import math
import copy

from einops import reduce, rearrange

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import ModuleList


class TemporalCenterDifferenceConvolution(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        theta: float = 0.6,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.theta = theta
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < self.eps:
            return out_normal
        else:
            depth = self.conv.weight.shape[2]
            if depth > 1:
                kernel_0 = reduce(
                    self.conv.weight[
                        :,
                        :,
                        0,
                        :,
                        :,
                    ],
                    "out_channels in_channels_per_group height_kernel_size width_kernel_size -> out_channels in_channels_per_group",
                    "sum",
                )
                kernel_2 = reduce(
                    self.conv.weight[
                        :,
                        :,
                        -1,
                        :,
                        :,
                    ],
                    "out_channels in_channels_per_group height_kernel_size width_kernel_size -> out_channels in_channels_per_group",
                    "sum",
                )

                kernel_diff = kernel_0 + kernel_2
                kernel_diff = kernel_diff[
                    :,
                    :,
                    None,
                    None,
                    None,
                ]

                out_diff = F.conv3d(
                    input=x,
                    weight=kernel_diff,
                    bias=self.conv.bias,
                    stride=self.conv.stride,
                    padding=0,
                    dilation=self.conv.dilation,
                    groups=self.conv.groups,
                )
                return out_normal - self.theta * out_diff
            else:
                return out_normal


class MultiHeadTCDCSelfSGAttention(nn.Module):
    def __init__(
        self,
        feature_size: int,
        num_heads: int,
        model_dims: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
        bias: bool,
        theta: float,
        eps: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.feature_size = feature_size  # 4
        projected_out = not (num_heads == 1)
        self.num_heads = num_heads

        self.query_projection = nn.Sequential(
            TemporalCenterDifferenceConvolution(
                in_channels=model_dims,
                out_channels=model_dims,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                theta=theta,
                eps=eps,
            ),
            nn.BatchNorm3d(model_dims),
        )
        self.key_projection = nn.Sequential(
            TemporalCenterDifferenceConvolution(
                in_channels=model_dims,
                out_channels=model_dims,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                theta=theta,
                eps=eps,
            ),
            nn.BatchNorm3d(model_dims),
        )
        self.value_projection = nn.Sequential(
            nn.Conv3d(
                in_channels=model_dims,
                out_channels=model_dims,
                kernel_size=1,
                stride=stride,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
        )
        self.out_projection = (
            nn.Sequential(
                nn.Linear(
                    model_dims,
                    model_dims,
                ),
                nn.Dropout(attention_dropout),
            )
            if projected_out
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        sharp_gradient: float,
    ) -> torch.Tensor:
        patch_size = x.shape[1]
        depth_size = patch_size // self.feature_size**2
        x = rearrange(
            x,
            "batch_size (depth height width) channels -> batch_size channels depth height width",
            D=depth_size,
            H=self.feature_size,
            W=self.feature_size,
        )
        query, key, value = (
            self.query_projection(x),
            self.key_projection(x),
            self.value_projection(x),
        )
        query, key, value = (
            rearrange(
                i,
                "batch_size channels depth height width -> batch_size (depth height width) channels",
            )
            for i in [query, key, value]
        )
        query, key, value = (
            rearrange(
                i,
                "batch_size patches (num_heads head_dims) -> batch_size num_heads patches head_dims",
                num_heads=self.num_heads,
            )
            for i in [query, key, value]
        )

        attention_score = (
            torch.einsum(
                "batch_size num_heads patches head_dims, batch_size num_heads head_dims patches -> batch_size num_heads patches patches",
                query,
                key,
            )
            / sharp_gradient
        )
        attention_score = F.softmax(
            attention_score,
            dim=-1,
        )

        attention = torch.einsum(
            "batch_size num_heads patches patches, batch_size num_heads patches head_dims -> batch_size num_heads patches head_dims",
            attention_score,
            value,
        )
        attention = rearrange(
            attention,
            "batch_size num_heads patches head_dims -> batch_size patches (num_heads head_dims)",
        )
        return self.out_projection(attention)


class SpatioTemporalFeedForward(nn.Module):
    def __init__(
        self,
        feature_size: int,
        model_dims: int,
        feed_forward_dims: int,
        feed_forward_dropout: float,
    ) -> None:
        super().__init__()
        self.feature_size = feature_size  # 4

        self.feed_forward1 = nn.Sequential(
            nn.Conv3d(
                in_channels=model_dims,
                out_channels=feed_forward_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
            ),
            nn.BatchNorm3d(feed_forward_dims),
        )
        self.spatio_temporal_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=feed_forward_dims,
                out_channels=feed_forward_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=feed_forward_dims,
                bias=False,
            ),
            nn.BatchNorm3d(feed_forward_dims),
        )
        self.feed_forward2 = nn.Sequential(
            nn.Conv3d(
                in_channels=feed_forward_dims,
                out_channels=model_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
            ),
            nn.BatchNorm3d(model_dims),
        )

        self.feed_forward = nn.Sequential(
            self.feed_forward1,
            nn.ELU(),
            nn.Dropout(feed_forward_dropout),
            self.spatio_temporal_conv,
            nn.ELU(),
            nn.Dropout(feed_forward_dropout),
            self.feed_forward2,
            nn.Dropout(feed_forward_dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        patch_size = x.shape[1]
        depth_size = patch_size // self.feature_size**2
        x = rearrange(
            x,
            "batch_size (depth height width) channels -> batch_size channels depth height width",
            D=depth_size,
            H=self.feature_size,
            W=self.feature_size,
        )
        forwarded = self.feed_forward2(x)
        return forwarded


class EncoderBlock(nn.Module):
    def __init__(
        self,
        feature_size: int,
        num_heads: int,
        model_dims: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
        bias: bool,
        theta: float,
        eps: float,
        attention_dropout: float,
        feed_forward_dims: int,
        feed_forward_dropout: float,
    ) -> None:
        super().__init__()
        self.pre_attention_norm = nn.LayerNorm(model_dims)
        self.attention = MultiHeadTCDCSelfSGAttention(
            feature_size=feature_size,
            num_heads=num_heads,
            model_dims=model_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            theta=theta,
            eps=eps,
            attention_dropout=attention_dropout,
        )

        self.pre_feed_forward_norm = nn.LayerNorm(model_dims)
        self.feed_forward = SpatioTemporalFeedForward(
            feature_size=feature_size,
            model_dims=model_dims,
            feed_forward_dims=feed_forward_dims,
            feed_forward_dropout=feed_forward_dropout,
        )

        self.norm = nn.LayerNorm(model_dims)

    def forward(
        self,
        x: torch.Tensor,
        sharp_gradient: float,
    ) -> torch.Tensor:
        x = (
            self.attention(
                x=self.pre_attention_norm(x),
                sharp_gradient=sharp_gradient,
            )
            + x
        )
        x = self.feed_forward(self.pre_feed_forward_norm(x)) + x
        return self.norm(x)


class PhysFormerEncoder(nn.Module):
    def __init__(
        self,
        feature_size: int,
        num_heads: int,
        model_dims: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
        bias: bool,
        theta: float,
        eps: float,
        attention_dropout: float,
        feed_forward_dims: int,
        feed_forward_dropout: float,
        num_layers: int,
    ) -> None:
        super().__init__()
        layer = EncoderBlock(
            feature_size=feature_size,
            num_heads=num_heads,
            model_dims=model_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            theta=theta,
            eps=eps,
            attention_dropout=attention_dropout,
            feed_forward_dims=feed_forward_dims,
            feed_forward_dropout=feed_forward_dropout,
        )
        self.layers = self.get_clone(
            layer=layer,
            num_layers=num_layers,
        )

    def forward(
        self,
        x: torch.Tensor,
        sharp_gradient: float,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x=x,
                sharp_gradient=sharp_gradient,
            )
        return x

    @staticmethod
    def get_clone(
        layer: nn.Module,
        num_layers: int,
    ) -> ModuleList:
        return ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
