from typing import Dict
import math

import torch
from torch import nn
from torch.nn import functional as F


class CombinedLabelDistributionLoss(nn.Module):
    def __init__(
        self,
        min_bpm: int,
        max_bpm: int,
    ) -> None:
        super().__init__()
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.num_bpm_classes = max_bpm - min_bpm + 1

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        frame_rate: torch.Tensor,
        alpha: float,
        beta: float,
    ) -> Dict[str, torch.Tensor]:
        pred = (
            pred
            - pred.mean(
                dim=-1,
                keepdim=True,
            )
        ) / pred.std(
            dim=-1,
            keepdim=True,
        )

        rppg_loss = self.negative_pearson_loss(
            pred=pred,
            target=target,
        )

        pred_distribution = self.signal_to_bpm_distribution(
            signal=pred,
            frame_rate=frame_rate,
        )
        target_distribution = self.signal_to_bpm_distribution(
            signal=target,
            frame_rate=frame_rate,
        )

        kl_div_loss = F.kl_div(
            input=F.log_softmax(
                pred_distribution,
                dim=-1,
            ),
            target=target_distribution,
            reduction="mean",
            log_target=False,
        )

        freq_ce_loss = F.cross_entropy(
            input=pred_distribution,
            target=target_distribution.argmax(dim=-1),
        )

        freq_mse_loss = F.mse_loss(
            input=pred_distribution,
            target=target_distribution,
        )

        total_loss = alpha * rppg_loss + beta * (
            kl_div_loss + freq_ce_loss + freq_mse_loss
        )

        bpm_mae = F.l1_loss(
            input=pred_distribution.argmax(dim=-1).float(),
            target=target_distribution.argmax(dim=-1).float(),
        )

        return {
            "total_loss": total_loss,
            "bpm_mae": bpm_mae,
        }

    def negative_pearson_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred_mean = pred.mean(
            dim=-1,
            keepdim=True,
        )
        target_mean = target.mean(
            dim=-1,
            keepdim=True,
        )

        pred_centered = pred - pred_mean
        target_centered = target - target_mean

        pred_std = pred_centered.pow(2).sum(dim=-1).sqrt()
        target_std = target_centered.pow(2).sum(dim=-1).sqrt()

        covariance = (pred_centered * target_centered).sum(dim=-1)

        pearson_correlation = covariance / (pred_std * target_std)

        loss = 1 - pearson_correlation
        return loss.mean()

    def signal_to_bpm_distribution(
        self,
        signal: torch.Tensor,
        frame_rate: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        fft_result = torch.fft.fft(
            signal,
            dim=-1,
        )

        frequency = [torch.fft.fftfreq(signal.shape[-1], d=1.0 / i) for i in frame_rate]
        frequency = torch.stack(frequency).to(signal.device)

        bpm = torch.abs(frequency * 60)
        bpm_indices = (bpm >= self.min_bpm) & (bpm <= self.max_bpm)

        masked_fft_result = torch.where(
            bpm_indices,
            fft_result,
            torch.tensor(
                0.0,
                device=fft_result.device,
            ),
        )
        fft_magnitude = torch.abs(masked_fft_result)
        bpm_distribution = fft_magnitude / fft_magnitude.sum(
            dim=-1,
            keepdim=True,
        )
        return torch.clamp(
            bpm_distribution,
            min=1e-15,
        )
