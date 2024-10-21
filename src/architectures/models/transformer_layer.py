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


class MultiHeadSelfTDCSGAttention(nn.Module):
    def __init__(
        self,
        input_size: int,
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
        attn_dropout: float,
        residual_dropout: float,
    ) -> None:
        super().__init__()
        self.input_size = input_size  # 4
        self.num_heads = num_heads

        self.layer_norm = nn.LayerNorm(model_dims)

        self.q_proj = nn.Sequential(
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
        self.k_proj = nn.Sequential(
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
        self.v_proj = nn.Sequential(
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

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(
        self,
        x: torch.Tensor,
        sharp_gradient: float,
    ) -> torch.Tensor:
        x = self.layer_norm(x)
        patch_size = x.shape[1]
        depth_size = patch_size // self.input_size**2
        x = rearrange(
            x,
            "batch_size (depth height width) channels -> batch_size channels depth height width",
            D=depth_size,
            H=self.input_size,
            W=self.input_size,
        )
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q, k, v = (
            rearrange(
                i,
                "batch_size channels depth height width -> batch_size (depth height width) channels",
            )
            for i in [q, k, v]
        )
        q, k, v = (
            rearrange(
                i,
                "batch_size patches (num_heads head_dims) -> batch_size num_heads patches head_dims",
                num_heads=self.num_heads,
            )
            for i in [q, k, v]
        )

        attn_score = (
            torch.einsum(
                "batch_size num_heads patches head_dims, batch_size num_heads head_dims patches -> batch_size num_heads patches patches",
                q,
                k,
            )
            / sharp_gradient
        )
        attn_score = self.attn_dropout(
            F.softmax(
                attn_score,
                dim=-1,
            )
        )

        attn = torch.einsum(
            "batch_size num_heads patches patches, batch_size num_heads patches head_dims -> batch_size num_heads patches head_dims",
            attn_score,
            v,
        )

        attn = rearrange(
            attn,
            "batch_size num_heads patches head_dims -> batch_size patches (num_heads head_dims)",
        )
        return q + self.residual_dropout(attn)


class SpatioTemporalFeedForward(nn.Module):
    def __init__(
        self,
        input_size: int,
        model_dims: int,
        feed_forward_dims: int,
        feed_forward_dropout: float,
        residual_dropout: float,
    ) -> None:
        super().__init__()
        self.input_size = input_size  # 4

        self.layer_norm = nn.LayerNorm(model_dims)
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
            nn.ELU(),
        )
        self.feed_forward_dropout = nn.Dropout(feed_forward_dropout)
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
            nn.ELU(),
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
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.layer_norm(x)
        patch_size = x.shape[1]
        depth_size = patch_size // self.input_size**2
        x = rearrange(
            x,
            "batch_size (depth height width) channels -> batch_size channels depth height width",
            D=depth_size,
            H=self.input_size,
            W=self.input_size,
        )
        forwarded = self.feed_forward2(
            self.spatio_temporal_conv(self.feed_forward_dropout(self.feed_forward1(x)))
        )
        residual = x + self.residual_dropout(forwarded)
        residual = rearrange(
            residual,
            "batch_size channels depth height width -> batch_size (depth height width) channels",
        )
        return residual


class EncoderBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
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
        attn_dropout: float,
        feed_forward_dims: int,
        feed_forward_dropout: float,
        residual_dropout: float,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadSelfTDCSGAttention(
            input_size=input_size,
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
            attn_dropout=attn_dropout,
        )
        self.feed_forward = SpatioTemporalFeedForward(
            input_size=input_size,
            model_dims=model_dims,
            feed_forward_dims=feed_forward_dims,
            feed_forward_dropout=feed_forward_dropout,
            residual_dropout=residual_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        sharp_gradient: float,
    ) -> torch.Tensor:
        x = self.attn(
            x=x,
            sharp_gradient=sharp_gradient,
        )
        x = self.feed_forward(x)
        return x


class PhysFormerEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
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
        attn_dropout: float,
        feed_forward_dims: int,
        feed_forward_dropout: float,
        residual_dropout: float,
        num_layers: int,
    ) -> None:
        super().__init__()
        layer = EncoderBlock(
            input_size=input_size,
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
            attn_dropout=attn_dropout,
            feed_forward_dims=feed_forward_dims,
            feed_forward_dropout=feed_forward_dropout,
            residual_dropout=residual_dropout,
        )
        self.layers = self.get_clone(
            module=layer,
            iteration=num_layers,
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
        module: nn.Module,
        iteration: int,
    ) -> ModuleList:
        return ModuleList([copy.deepcopy(module) for _ in range(iteration)])
