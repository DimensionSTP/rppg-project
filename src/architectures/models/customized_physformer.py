from typing import Dict

from einops import rearrange

import torch
from torch import nn

from .st_vit_layer import PhysFormerEncoder


class CustomizedPhysFormer(nn.Module):
    def __init__(
        self,
        is_pretrained: bool,
        patch_size: int,
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
        self.is_pretrained = is_pretrained
        self.feature_size = feature_size

        self.stem0 = nn.Sequential(
            nn.Conv3d(
                in_channels=3,
                out_channels=model_dims // 4,
                kernel_size=(
                    1,
                    5,
                    5,
                ),
                stride=1,
                padding=(
                    0,
                    2,
                    2,
                ),
                dilation=1,
                groups=1,
                bias=True,
            ),
            nn.BatchNorm3d(model_dims // 4),
        )
        self.stem1 = nn.Sequential(
            nn.Conv3d(
                in_channels=model_dims // 4,
                out_channels=model_dims // 2,
                kernel_size=(
                    3,
                    3,
                    3,
                ),
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                bias=True,
            ),
            nn.BatchNorm3d(model_dims // 2),
        )
        self.stem2 = nn.Sequential(
            nn.Conv3d(
                in_channels=model_dims // 2,
                out_channels=model_dims,
                kernel_size=(
                    3,
                    3,
                    3,
                ),
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                bias=True,
            ),
            nn.BatchNorm3d(model_dims),
        )

        self.stem = nn.Sequential(
            self.stem0,
            nn.ReLU(inplace=True),
            nn.MaxPool3d(
                kernel_size=(
                    1,
                    2,
                    2,
                ),
                stride=(
                    1,
                    2,
                    2,
                ),
            ),
            self.stem1,
            nn.ReLU(inplace=True),
            nn.MaxPool3d(
                kernel_size=(
                    1,
                    2,
                    2,
                ),
                stride=(
                    1,
                    2,
                    2,
                ),
            ),
            self.stem2,
            nn.ReLU(inplace=True),
            nn.MaxPool3d(
                kernel_size=(
                    1,
                    2,
                    2,
                ),
                stride=(
                    1,
                    2,
                    2,
                ),
            ),
        )

        self.patch_embedding = nn.Conv3d(
            in_channels=model_dims,
            out_channels=model_dims,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        self.physformer_encoder = PhysFormerEncoder(
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
            num_layers=num_layers,
        )

        self.upsample0 = nn.Sequential(
            nn.Upsample(
                scale_factor=(
                    2,
                    1,
                    1,
                ),
            ),
            nn.Conv3d(
                in_channels=model_dims,
                out_channels=model_dims,
                kernel_size=(
                    3,
                    1,
                    1,
                ),
                stride=1,
                padding=(
                    1,
                    0,
                    0,
                ),
                dilation=1,
                groups=1,
                bias=True,
            ),
            nn.BatchNorm3d(model_dims),
        )
        self.upsample1 = nn.Sequential(
            nn.Upsample(
                scale_factor=(
                    2,
                    1,
                    1,
                ),
            ),
            nn.Conv3d(
                in_channels=model_dims,
                out_channels=model_dims // 2,
                kernel_size=(
                    3,
                    1,
                    1,
                ),
                stride=1,
                padding=(
                    1,
                    0,
                    0,
                ),
                dilation=1,
                groups=1,
                bias=True,
            ),
            nn.BatchNorm3d(model_dims // 2),
        )

        self.upsample = nn.Sequential(
            self.upsample0,
            nn.ELU(inplace=True),
            self.upsample1,
            nn.ELU(inplace=True),
        )

        self.output_layer = nn.Conv1d(
            in_channels=model_dims // 2,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
        )

        if not is_pretrained:
            self.init_weights()

    def forward(
        self,
        encoded: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.stem(encoded)

        encoded = self.patch_embedding(encoded)
        encoded = rearrange(
            encoded,
            "batch_size channels depth height width -> batch_size (depth height width) channels",
        )

        encoded = self.physformer_encoder(encoded)

        num_patches = encoded.shape[1]
        depth_size = num_patches // self.feature_size**2
        encoded = rearrange(
            encoded,
            "batch_size (depth height width) channels -> batch_size channels depth height width",
            depth=depth_size,
            height=self.feature_size,
            width=self.feature_size,
        )

        encoded = self.upsample(encoded)
        encoded = encoded.mean(dim=-1).mean(dim=-1)

        rppg = self.output_layer(encoded)
        rppg = rearrange(
            rppg,
            "batch_size 1 signals -> batch_size signals",
        )
        return {
            "rppg": rppg,
        }

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.normal_(
                        m.bias,
                        std=1e-6,
                    )

        self.apply(_init)
