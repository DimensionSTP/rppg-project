from typing import Dict
import math

import torch
from torch import nn
from torch.nn import functional as F


class CombinedLabelDistributionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        bpm: torch.Tensor,
        min_bpm: int,
        max_bpm: int,
        std: float,
        frame_rate: torch.Tensor,
        alpha: float,
        beta: float,
    ) -> Dict[str, torch.Tensor]:
        bpm = torch.clamp(
            bpm,
            min=min_bpm,
            max=max_bpm,
        )

        pred = (
            pred
            - torch.mean(
                pred,
                dim=-1,
                keepdim=True,
            )
        ) / torch.std(
            pred,
            dim=-1,
            keepdim=True,
        )

        rppg_loss = self.negative_pearson_loss(
            pred=pred,
            target=target,
        )

        combined_loss = self.combined_loss(
            pred=pred,
            bpm=bpm,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            std=std,
            frame_rate=frame_rate,
        )
        kl_div_loss = combined_loss["kl_div_loss"]
        freq_ce_loss = combined_loss["freq_ce_loss"]
        bpm_mae = combined_loss["bpm_mae"]

        total_loss = alpha * rppg_loss + beta * (freq_ce_loss + kl_div_loss)

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

    @staticmethod
    def normal_distribution(
        scaled_bpm: torch.Tensor,
        min_bpm: int,
        max_bpm: int,
        std: float,
    ) -> Dict[str, torch.Tensor]:
        scaled_bpm = scaled_bpm.unsqueeze(1)

        range_size = max_bpm - min_bpm + 1
        bpm_range = torch.arange(
            0,
            range_size,
            dtype=scaled_bpm.dtype,
            device=scaled_bpm.device,
        ).unsqueeze(0)

        diff = bpm_range - scaled_bpm
        exponent = -(diff**2) / (2 * std**2)
        normal_distribution = torch.exp(exponent) / (
            torch.sqrt(2 * torch.tensor(math.pi)) * std
        )
        normal_distribution = torch.clamp(
            normal_distribution,
            min=1e-15,
        )
        return {
            "normal_distribution": normal_distribution,
            "bpm_range": bpm_range,
        }

    def pred_frequency_distribution(
        self,
        pred: torch.Tensor,
        absolute_beat: torch.Tensor,
        num_signals: int,
    ) -> torch.Tensor:
        angular_frequencies = (
            2
            * torch.pi
            * torch.arange(
                0,
                num_signals,
                dtype=pred.dtype,
                device=pred.device,
            )
        ) / num_signals
        angular_frequencies = angular_frequencies.unsqueeze(0).unsqueeze(0)
        hanning_window = torch.hann_window(
            num_signals,
            dtype=pred.dtype,
            device=pred.device,
        ).unsqueeze(0)
        pred = pred * hanning_window
        absolute_beat = absolute_beat.unsqueeze(-1).to(
            pred.device,
            dtype=pred.dtype,
        )
        pred_frequency_distribution = (
            torch.einsum(
                "B S, B F S -> B F",
                pred,
                torch.sin(absolute_beat * angular_frequencies),
            )
            ** 2
            + torch.einsum(
                "B S, B F S -> B F",
                pred,
                torch.cos(absolute_beat * angular_frequencies),
            )
            ** 2
        )
        return pred_frequency_distribution

    def normalized_pred_frequency_distribution(
        self,
        pred: torch.Tensor,
        frame_rate: torch.Tensor,
        bpm_range: torch.Tensor,
    ) -> torch.Tensor:
        num_signals = pred.size(-1)
        frequency = frame_rate / num_signals
        bps_range = bpm_range / 60.0
        absolute_beat = bps_range / frequency.unsqueeze(1)
        pred_frequency_distribution = self.pred_frequency_distribution(
            pred=pred,
            absolute_beat=absolute_beat,
            num_signals=num_signals,
        )
        normalized_pred_frequency_distribution = (
            pred_frequency_distribution / pred_frequency_distribution.sum()
        )
        return normalized_pred_frequency_distribution

    def combined_loss(
        self,
        pred: torch.Tensor,
        bpm: torch.Tensor,
        min_bpm: int,
        max_bpm: int,
        std: float,
        frame_rate: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        scaled_bpm = bpm - min_bpm

        distribution_outputs = self.normal_distribution(
            scaled_bpm=scaled_bpm,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            std=std,
        )
        bpm_distribution = distribution_outputs["normal_distribution"]
        bpm_range = distribution_outputs["bpm_range"]

        normalized_pred_frequency_distribution = (
            self.normalized_pred_frequency_distribution(
                pred=pred,
                frame_rate=frame_rate,
                bpm_range=bpm_range,
            )
        )

        kl_div_loss = F.kl_div(
            input=F.log_softmax(
                normalized_pred_frequency_distribution,
                dim=-1,
            ),
            target=bpm_distribution,
            reduction="sum",
            log_target=False,
        )

        freq_ce_loss = F.cross_entropy(
            input=normalized_pred_frequency_distribution,
            target=scaled_bpm.long(),
        )
        bpm_mae = F.l1_loss(
            input=torch.max(
                normalized_pred_frequency_distribution,
                dim=-1,
            )[0],
            target=scaled_bpm,
        )
        return {
            "kl_div_loss": kl_div_loss,
            "freq_ce_loss": freq_ce_loss,
            "bpm_mae": bpm_mae,
        }
