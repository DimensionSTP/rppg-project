from typing import Dict, Tuple
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
        tcdc_kernel_size: int,
        tcdc_stride: int,
        tcdc_padding: int,
        tcdc_dilation: int,
        tcdc_groups: int,
        tcdc_bias: bool,
        tcdc_theta: float,
        tcdc_eps: float,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=tcdc_kernel_size,
            stride=tcdc_stride,
            padding=tcdc_padding,
            dilation=tcdc_dilation,
            groups=tcdc_groups,
            bias=tcdc_bias,
        )
        self.tcdc_theta = tcdc_theta
        self.tcdc_eps = tcdc_eps

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        out_normal = self.conv(x)

        if math.fabs(self.tcdc_theta - 0.0) < self.tcdc_eps:
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
                return out_normal - self.tcdc_theta * out_diff
            else:
                return out_normal


class MultiHeadTCDCSelfSGAttention(nn.Module):
    def __init__(
        self,
        feature_size: int,
        sharp_gradient: float,
        num_heads: int,
        model_dims: int,
        tcdc_kernel_size: int,
        tcdc_stride: int,
        tcdc_padding: int,
        tcdc_dilation: int,
        tcdc_groups: int,
        tcdc_bias: bool,
        tcdc_theta: float,
        tcdc_eps: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.sharp_gradient = sharp_gradient
        self.num_heads = num_heads
        self.projected_out = not (self.num_heads == 1)

        self.query_projection = nn.Sequential(
            TemporalCenterDifferenceConvolution(
                in_channels=model_dims,
                out_channels=model_dims,
                tcdc_kernel_size=tcdc_kernel_size,
                tcdc_stride=tcdc_stride,
                tcdc_padding=tcdc_padding,
                tcdc_dilation=tcdc_dilation,
                tcdc_groups=tcdc_groups,
                tcdc_bias=tcdc_bias,
                tcdc_theta=tcdc_theta,
                tcdc_eps=tcdc_eps,
            ),
            nn.BatchNorm3d(model_dims),
        )
        self.key_projection = nn.Sequential(
            TemporalCenterDifferenceConvolution(
                in_channels=model_dims,
                out_channels=model_dims,
                tcdc_kernel_size=tcdc_kernel_size,
                tcdc_stride=tcdc_stride,
                tcdc_padding=tcdc_padding,
                tcdc_dilation=tcdc_dilation,
                tcdc_groups=tcdc_groups,
                tcdc_bias=tcdc_bias,
                tcdc_theta=tcdc_theta,
                tcdc_eps=tcdc_eps,
            ),
            nn.BatchNorm3d(model_dims),
        )
        self.value_projection = nn.Sequential(
            nn.Conv3d(
                in_channels=model_dims,
                out_channels=model_dims,
                kernel_size=1,
                stride=tcdc_stride,
                padding=0,
                dilation=tcdc_dilation,
                groups=tcdc_groups,
                bias=tcdc_bias,
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
            if self.projected_out
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.get_qkv(x=x)
        query, key, value = (
            qkv["query"],
            qkv["key"],
            qkv["value"],
        )

        attention = self.get_attention(
            query=query,
            key=key,
            value=value,
        )
        return self.out_projection(attention)

    def get_qkv(
        self,
        x: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        num_patches = x.shape[1]
        depth_size = num_patches // self.feature_size**2
        x = rearrange(
            x,
            "batch_size (depth height width) channels -> batch_size channels depth height width",
            depth=depth_size,
            height=self.feature_size,
            width=self.feature_size,
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
        return {
            "query": query,
            "key": key,
            "value": value,
        }

    def get_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        attention_score = (
            torch.einsum(
                "B H P D, B H K D -> B H P K",
                query,
                key,
            )
            / self.sharp_gradient
        )
        attention_score = F.softmax(
            attention_score,
            dim=-1,
        )

        attention = torch.einsum(
            "B H P K, B H K D -> B H P D",
            attention_score,
            value,
        )
        attention = rearrange(
            attention,
            "batch_size num_heads patches head_dims -> batch_size patches (num_heads head_dims)",
        )
        return attention


class MultiHeadTCDCCrossSGAttention(nn.Module):
    def __init__(
        self,
        feature_size: int,
        sharp_gradient: float,
        num_heads: int,
        model_dims: int,
        tcdc_kernel_size: int,
        tcdc_stride: int,
        tcdc_padding: int,
        tcdc_dilation: int,
        tcdc_groups: int,
        tcdc_bias: bool,
        tcdc_theta: float,
        tcdc_eps: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.sharp_gradient = sharp_gradient

        self.slow_num_heads = num_heads
        self.slow_projected_out = not (self.slow_num_heads == 1)
        self.fast_num_heads = num_heads // 2
        self.fast_projected_out = not (self.fast_num_heads == 1)

        self.slow_model_dims = model_dims
        self.fast_model_dims = model_dims // 2

        self.slow_query_projection = nn.Sequential(
            TemporalCenterDifferenceConvolution(
                in_channels=self.slow_model_dims,
                out_channels=self.slow_model_dims,
                tcdc_kernel_size=tcdc_kernel_size,
                tcdc_stride=tcdc_stride,
                tcdc_padding=tcdc_padding,
                tcdc_dilation=tcdc_dilation,
                tcdc_groups=tcdc_groups,
                tcdc_bias=tcdc_bias,
                tcdc_theta=tcdc_theta,
                tcdc_eps=tcdc_eps,
            ),
            nn.BatchNorm3d(self.slow_model_dims),
        )
        self.slow_key_projection = nn.Sequential(
            TemporalCenterDifferenceConvolution(
                in_channels=self.slow_model_dims,
                out_channels=self.slow_model_dims,
                tcdc_kernel_size=tcdc_kernel_size,
                tcdc_stride=tcdc_stride,
                tcdc_padding=tcdc_padding,
                tcdc_dilation=tcdc_dilation,
                tcdc_groups=tcdc_groups,
                tcdc_bias=tcdc_bias,
                tcdc_theta=tcdc_theta,
                tcdc_eps=tcdc_eps,
            ),
            nn.BatchNorm3d(self.slow_model_dims),
        )
        self.slow_value_projection = nn.Sequential(
            nn.Conv3d(
                in_channels=self.slow_model_dims,
                out_channels=self.slow_model_dims,
                kernel_size=1,
                stride=tcdc_stride,
                padding=0,
                dilation=tcdc_dilation,
                groups=tcdc_groups,
                bias=tcdc_bias,
            ),
        )
        self.slow_out_projection = (
            nn.Sequential(
                nn.Linear(
                    self.slow_model_dims,
                    self.slow_model_dims,
                ),
                nn.Dropout(attention_dropout),
            )
            if self.slow_projected_out
            else nn.Identity()
        )

        self.fast_query_projection = nn.Sequential(
            TemporalCenterDifferenceConvolution(
                in_channels=self.fast_model_dims,
                out_channels=self.fast_model_dims,
                tcdc_kernel_size=tcdc_kernel_size,
                tcdc_stride=tcdc_stride,
                tcdc_padding=tcdc_padding,
                tcdc_dilation=tcdc_dilation,
                tcdc_groups=tcdc_groups,
                tcdc_bias=tcdc_bias,
                tcdc_theta=tcdc_theta,
                tcdc_eps=tcdc_eps,
            ),
            nn.BatchNorm3d(self.fast_model_dims),
        )
        self.fast_key_projection = nn.Sequential(
            TemporalCenterDifferenceConvolution(
                in_channels=self.fast_model_dims,
                out_channels=self.fast_model_dims,
                tcdc_kernel_size=tcdc_kernel_size,
                tcdc_stride=tcdc_stride,
                tcdc_padding=tcdc_padding,
                tcdc_dilation=tcdc_dilation,
                tcdc_groups=tcdc_groups,
                tcdc_bias=tcdc_bias,
                tcdc_theta=tcdc_theta,
                tcdc_eps=tcdc_eps,
            ),
            nn.BatchNorm3d(self.fast_model_dims),
        )
        self.fast_value_projection = nn.Sequential(
            nn.Conv3d(
                in_channels=self.fast_model_dims,
                out_channels=self.fast_model_dims,
                kernel_size=1,
                stride=tcdc_stride,
                padding=0,
                dilation=tcdc_dilation,
                groups=tcdc_groups,
                bias=tcdc_bias,
            ),
        )
        self.fast_out_projection = (
            nn.Sequential(
                nn.Linear(
                    self.fast_model_dims,
                    self.fast_model_dims,
                ),
                nn.Dropout(attention_dropout),
            )
            if self.fast_projected_out
            else nn.Identity()
        )

    def forward(
        self,
        slow_x: torch.Tensor,
        fast_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        slow_qkv = self.get_qkv(
            x=slow_x,
            speed="slow",
        )
        fast_qkv = self.get_qkv(
            x=fast_x,
            speed="fast",
        )
        slow_query, slow_key, slow_value = (
            slow_qkv["query"],
            slow_qkv["key"],
            slow_qkv["value"],
        )
        fast_query, fast_key, fast_value = (
            fast_qkv["query"],
            fast_qkv["key"],
            fast_qkv["value"],
        )

        slow_attention = self.get_attention(
            query=slow_query,
            key=slow_key,
            value=slow_value,
        )
        fast_attention = self.get_attention(
            query=fast_query,
            key=fast_key,
            value=fast_value,
        )
        cross_attention = self.get_attention(
            query=fast_query,
            key=slow_key,
            value=slow_value,
        )
        slow_output = self.slow_out_projection(slow_attention)
        fast_output = self.fast_out_projection(fast_attention + cross_attention)
        return (
            slow_output,
            fast_output,
        )

    def get_qkv(
        self,
        x: Dict[str, torch.Tensor],
        speed: str,
    ) -> Dict[str, torch.Tensor]:
        num_patches = x.shape[1]
        depth_size = num_patches // self.feature_size**2
        x = rearrange(
            x,
            "batch_size (depth height width) channels -> batch_size channels depth height width",
            depth=depth_size,
            height=self.feature_size,
            width=self.feature_size,
        )
        if speed == "slow":
            query, key, value = (
                self.slow_query_projection(x),
                self.slow_key_projection(x),
                self.slow_value_projection(x),
            )
        elif speed == "fast":
            query, key, value = (
                self.fast_query_projection(x),
                self.fast_key_projection(x),
                self.fast_value_projection(x),
            )
        else:
            raise ValueError("Invalid speed. Choose 'slow' or 'fast'.")
        query, key, value = (
            rearrange(
                i,
                "batch_size channels depth height width -> batch_size (depth height width) channels",
            )
            for i in [query, key, value]
        )
        if speed == "slow":
            query, key, value = (
                rearrange(
                    i,
                    "batch_size patches (num_heads head_dims) -> batch_size num_heads patches head_dims",
                    num_heads=self.slow_num_heads,
                )
                for i in [query, key, value]
            )
        else:
            query, key, value = (
                rearrange(
                    i,
                    "batch_size patches (num_heads head_dims) -> batch_size num_heads patches head_dims",
                    num_heads=self.fast_num_heads,
                )
                for i in [query, key, value]
            )
        return {
            "query": query,
            "key": key,
            "value": value,
        }

    def get_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        attention_score = (
            torch.einsum(
                "B H P D, B H K D -> B H P K",
                query,
                key,
            )
            / self.sharp_gradient
        )
        attention_score = F.softmax(
            attention_score,
            dim=-1,
        )

        attention = torch.einsum(
            "B H P K, B H K D -> B H P D",
            attention_score,
            value,
        )
        attention = rearrange(
            attention,
            "batch_size num_heads patches head_dims -> batch_size patches (num_heads head_dims)",
        )
        return attention


class SpatioTemporalFeedForward(nn.Module):
    def __init__(
        self,
        feature_size: int,
        model_dims: int,
        feed_forward_dims: int,
        feed_forward_dropout: float,
    ) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.model_dims = model_dims
        self.feed_forward_dims = feed_forward_dims
        self.feed_forward_dropout = feed_forward_dropout

        self.feed_forward = self.get_ff_net()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        forwarded = self.get_forwarded(x=x)
        return forwarded

    def get_ff_net(self) -> nn.Sequential:
        feed_forward1 = nn.Sequential(
            nn.Conv3d(
                in_channels=self.model_dims,
                out_channels=self.feed_forward_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
            ),
            nn.BatchNorm3d(self.feed_forward_dims),
        )
        spatio_temporal_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=self.feed_forward_dims,
                out_channels=self.feed_forward_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=self.feed_forward_dims,
                bias=False,
            ),
            nn.BatchNorm3d(self.feed_forward_dims),
        )
        feed_forward2 = nn.Sequential(
            nn.Conv3d(
                in_channels=self.feed_forward_dims,
                out_channels=self.model_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
            ),
            nn.BatchNorm3d(self.model_dims),
        )

        feed_forward = nn.Sequential(
            feed_forward1,
            nn.ELU(inplace=True),
            nn.Dropout(self.feed_forward_dropout),
            spatio_temporal_conv,
            nn.ELU(inplace=True),
            nn.Dropout(self.feed_forward_dropout),
            feed_forward2,
            nn.Dropout(self.feed_forward_dropout),
        )
        return feed_forward

    def get_forwarded(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        num_patches = x.shape[1]
        depth_size = num_patches // self.feature_size**2
        x = rearrange(
            x,
            "batch_size (depth height width) channels -> batch_size channels depth height width",
            depth=depth_size,
            height=self.feature_size,
            width=self.feature_size,
        )

        forwarded = self.feed_forward(x)

        forwarded = rearrange(
            forwarded,
            "batch_size channels depth height width -> batch_size (depth height width) channels",
        )
        return forwarded


class SpatioTemporalCrossFeedForward(nn.Module):
    def __init__(
        self,
        feature_size: int,
        model_dims: int,
        feed_forward_dims: int,
        feed_forward_dropout: float,
    ) -> None:
        super().__init__()
        self.feature_size = feature_size

        self.slow_model_dims = model_dims
        self.fast_model_dims = model_dims // 2

        self.slow_feed_forward_dims = feed_forward_dims
        self.fast_feed_forward_dims = feed_forward_dims // 2

        self.feed_forward_dropout = feed_forward_dropout

        self.slow_feed_forward = self.get_ff_net(speed="slow")
        self.fast_feed_forward = self.get_ff_net(speed="fast")

    def forward(
        self,
        slow_x: torch.Tensor,
        fast_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        slow_output = self.get_forwarded(
            x=slow_x,
            speed="slow",
        )
        fast_output = self.get_forwarded(
            x=fast_x,
            speed="fast",
        )
        return (
            slow_output,
            fast_output,
        )

    def get_ff_net(
        self,
        speed: str,
    ) -> nn.Sequential:
        if speed == "slow":
            model_dims = self.slow_model_dims
            feed_forward_dims = self.slow_feed_forward_dims
        elif speed == "fast":
            model_dims = self.fast_model_dims
            feed_forward_dims = self.fast_feed_forward_dims
        else:
            raise ValueError("Invalid speed. Choose 'slow' or 'fast'.")

        feed_forward1 = nn.Sequential(
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
        spatio_temporal_conv = nn.Sequential(
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
        feed_forward2 = nn.Sequential(
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

        feed_forward = nn.Sequential(
            feed_forward1,
            nn.ELU(inplace=True),
            nn.Dropout(self.feed_forward_dropout),
            spatio_temporal_conv,
            nn.ELU(inplace=True),
            nn.Dropout(self.feed_forward_dropout),
            feed_forward2,
            nn.Dropout(self.feed_forward_dropout),
        )
        return feed_forward

    def get_forwarded(
        self,
        x: torch.Tensor,
        speed: str,
    ) -> torch.Tensor:
        num_patches = x.shape[1]
        depth_size = num_patches // self.feature_size**2
        x = rearrange(
            x,
            "batch_size (depth height width) channels -> batch_size channels depth height width",
            depth=depth_size,
            height=self.feature_size,
            width=self.feature_size,
        )

        if speed == "slow":
            forwarded = self.slow_feed_forward(x)
        elif speed == "fast":
            forwarded = self.fast_feed_forward(x)
        else:
            raise ValueError("Invalid speed. Choose 'slow' or 'fast'.")

        forwarded = rearrange(
            forwarded,
            "batch_size channels depth height width -> batch_size (depth height width) channels",
        )
        return forwarded


class EncoderBlock(nn.Module):
    def __init__(
        self,
        feature_size: int,
        sharp_gradient: float,
        num_heads: int,
        model_dims: int,
        tcdc_kernel_size: int,
        tcdc_stride: int,
        tcdc_padding: int,
        tcdc_dilation: int,
        tcdc_groups: int,
        tcdc_bias: bool,
        tcdc_theta: float,
        tcdc_eps: float,
        attention_dropout: float,
        feed_forward_dims: int,
        feed_forward_dropout: float,
    ) -> None:
        super().__init__()
        self.pre_attention_norm = nn.LayerNorm(model_dims)
        self.attention = MultiHeadTCDCSelfSGAttention(
            feature_size=feature_size,
            sharp_gradient=sharp_gradient,
            num_heads=num_heads,
            model_dims=model_dims,
            tcdc_kernel_size=tcdc_kernel_size,
            tcdc_stride=tcdc_stride,
            tcdc_padding=tcdc_padding,
            tcdc_dilation=tcdc_dilation,
            tcdc_groups=tcdc_groups,
            tcdc_bias=tcdc_bias,
            tcdc_theta=tcdc_theta,
            tcdc_eps=tcdc_eps,
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
    ) -> torch.Tensor:
        x = self.attention(x=self.pre_attention_norm(x)) + x
        x = self.feed_forward(self.pre_feed_forward_norm(x)) + x
        return self.norm(x)


class PPEncoderBlock(nn.Module):
    def __init__(
        self,
        feature_size: int,
        sharp_gradient: float,
        num_heads: int,
        model_dims: int,
        tcdc_kernel_size: int,
        tcdc_stride: int,
        tcdc_padding: int,
        tcdc_dilation: int,
        tcdc_groups: int,
        tcdc_bias: bool,
        tcdc_theta: float,
        tcdc_eps: float,
        attention_dropout: float,
        feed_forward_dims: int,
        feed_forward_dropout: float,
    ) -> None:
        super().__init__()
        self.slow_pre_attention_norm = nn.LayerNorm(model_dims)
        self.fast_pre_attention_norm = nn.LayerNorm(model_dims // 2)
        self.attention = MultiHeadTCDCCrossSGAttention(
            feature_size=feature_size,
            sharp_gradient=sharp_gradient,
            num_heads=num_heads,
            model_dims=model_dims,
            tcdc_kernel_size=tcdc_kernel_size,
            tcdc_stride=tcdc_stride,
            tcdc_padding=tcdc_padding,
            tcdc_dilation=tcdc_dilation,
            tcdc_groups=tcdc_groups,
            tcdc_bias=tcdc_bias,
            tcdc_theta=tcdc_theta,
            tcdc_eps=tcdc_eps,
            attention_dropout=attention_dropout,
        )

        self.slow_pre_feed_forward_norm = nn.LayerNorm(model_dims)
        self.fast_pre_feed_forward_norm = nn.LayerNorm(model_dims // 2)
        self.feed_forward = SpatioTemporalCrossFeedForward(
            feature_size=feature_size,
            model_dims=model_dims,
            feed_forward_dims=feed_forward_dims,
            feed_forward_dropout=feed_forward_dropout,
        )

        self.slow_norm = nn.LayerNorm(model_dims)
        self.fast_norm = nn.LayerNorm(model_dims // 2)

    def forward(
        self,
        slow_x: torch.Tensor,
        fast_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        slow_x, fast_x = self.attention(
            slow_x=self.slow_pre_attention_norm(slow_x),
            fast_x=self.fast_pre_attention_norm(fast_x),
        ) + (slow_x + fast_x)
        slow_x, fast_x = self.feed_forward(
            slow_x=self.slow_pre_feed_forward_norm(slow_x),
            fast_x=self.fast_pre_feed_forward_norm(fast_x),
        ) + (slow_x + fast_x)
        return (
            self.slow_norm(slow_x),
            self.fast_norm(fast_x),
        )


class PhysFormerEncoder(nn.Module):
    def __init__(
        self,
        feature_size: int,
        sharp_gradient: float,
        num_heads: int,
        model_dims: int,
        tcdc_kernel_size: int,
        tcdc_stride: int,
        tcdc_padding: int,
        tcdc_dilation: int,
        tcdc_groups: int,
        tcdc_bias: bool,
        tcdc_theta: float,
        tcdc_eps: float,
        attention_dropout: float,
        feed_forward_dims: int,
        feed_forward_dropout: float,
        num_layers: int,
    ) -> None:
        super().__init__()
        layer = EncoderBlock(
            feature_size=feature_size,
            sharp_gradient=sharp_gradient,
            num_heads=num_heads,
            model_dims=model_dims,
            tcdc_kernel_size=tcdc_kernel_size,
            tcdc_stride=tcdc_stride,
            tcdc_padding=tcdc_padding,
            tcdc_dilation=tcdc_dilation,
            tcdc_groups=tcdc_groups,
            tcdc_bias=tcdc_bias,
            tcdc_theta=tcdc_theta,
            tcdc_eps=tcdc_eps,
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
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x=x)
        return x

    @staticmethod
    def get_clone(
        layer: nn.Module,
        num_layers: int,
    ) -> ModuleList:
        return ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])


class PhysFormerPPEncoder(nn.Module):
    def __init__(
        self,
        feature_size: int,
        sharp_gradient: float,
        num_heads: int,
        model_dims: int,
        tcdc_kernel_size: int,
        tcdc_stride: int,
        tcdc_padding: int,
        tcdc_dilation: int,
        tcdc_groups: int,
        tcdc_bias: bool,
        tcdc_theta: float,
        tcdc_eps: float,
        attention_dropout: float,
        feed_forward_dims: int,
        feed_forward_dropout: float,
        num_layers: int,
    ) -> None:
        super().__init__()
        layer = PPEncoderBlock(
            feature_size=feature_size,
            sharp_gradient=sharp_gradient,
            num_heads=num_heads,
            model_dims=model_dims,
            tcdc_kernel_size=tcdc_kernel_size,
            tcdc_stride=tcdc_stride,
            tcdc_padding=tcdc_padding,
            tcdc_dilation=tcdc_dilation,
            tcdc_groups=tcdc_groups,
            tcdc_bias=tcdc_bias,
            tcdc_theta=tcdc_theta,
            tcdc_eps=tcdc_eps,
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
        slow_x: torch.Tensor,
        fast_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            slow_x, fast_x = layer(
                slow_x=slow_x,
                fast_x=fast_x,
            )
        return (
            slow_x,
            fast_x,
        )

    @staticmethod
    def get_clone(
        layer: nn.Module,
        num_layers: int,
    ) -> ModuleList:
        return ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
