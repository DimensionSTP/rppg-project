from typing import Dict
import math

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class CombinedLabelDistributionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def normal_distribution(
        bpm: torch.Tensor,
        min_bpm: int,
        max_bpm: int,
        std: float,
    ) -> Dict[str, torch.Tensor]:
        scaled_bpm = bpm - min_bpm
        range_size = max_bpm - min_bpm
        bpm_range = torch.arange(0, range_size).float().unsqueeze(0)
        diff = bpm_range - scaled_bpm
        exponent = -(diff**2) / (2 * std**2)
        normal_distribution = torch.exp(exponent) / (
            torch.sqrt(2 * torch.tensor(math.pi)) * std
        )
        normal_distribution = torch.clamp(normal_distribution, min=1e-15)
        return {
            "normal_distribution": normal_distribution,
            "bpm_range": bpm_range,
        }

    @staticmethod
    def kl_divergence_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        criterion = nn.KLDivLoss(reduction="batchmean")
        log_pred = torch.log(pred)
        loss = criterion(log_pred, target)
        return loss

    def negative_pearson_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred_mean = pred.mean(
            dim=1,
            keepdim=True,
        )
        target_mean = target.mean(
            dim=1,
            keepdim=True,
        )

        pred_centered = pred - pred_mean
        target_centered = target - target_mean

        pred_std = pred_centered.pow(2).sum(dim=1).sqrt()
        target_std = target_centered.pow(2).sum(dim=1).sqrt()

        covariance = (pred_centered * target_centered).sum(dim=1)

        pearson_correlation = covariance / (pred_std * target_std)

        loss = 1 - pearson_correlation
        return loss.mean()

    def complex_absolute_given_k(self, output, k, N):
        two_pi_n_over_N = (
            Variable(
                2 * math.pi * torch.arange(0, N, dtype=torch.float), requires_grad=True
            )
            / N
        )
        hanning_window = Variable(
            torch.from_numpy(np.hanning(N)).type(torch.FloatTensor), requires_grad=True
        ).view(1, -1)

        k = k.type(torch.FloatTensor).cuda()
        two_pi_n_over_N = two_pi_n_over_N.cuda()
        hanning_window = hanning_window.cuda()

        output = output.view(1, -1) * hanning_window
        output = output.view(1, 1, -1).type(torch.cuda.FloatTensor)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)

        complex_absolute = (
            torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2
            + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2
        )

        return complex_absolute

    def complex_absolute(self, output, sampling_rate, bpm_range):
        output = output.view(1, -1)
        N = output.size()[1]
        unit_per_hz = sampling_rate / N
        feasible_bpm = bpm_range / 60.0
        k = feasible_bpm / unit_per_hz
        complex_absolute = self.complex_absolute_given_k(output, k, N)
        return (1.0 / complex_absolute.sum()) * complex_absolute

    def combined_loss(
        self,
        pred: torch.Tensor,
        bpm: torch.Tensor,
        min_bpm: int,
        max_bpm: int,
        std: float,
        frame_rate: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        distribution_outputs = self.normal_distribution(
            bpm,
            min_bpm,
            max_bpm,
            std,
        )
        bpm_distribution = distribution_outputs["normal_distribution"]
        bpm_range = distribution_outputs["bpm_range"]

        pred = pred.view(1, -1)
        bpm = bpm.view(1, -1)

        complex_absolute = self.complex_absolute(pred, frame_rate, bpm_range)
        frequency_distribution = F.softmax(
            complex_absolute.view(-1),
            dim=0,
        )
        kl_div_loss = self.kl_divergence_loss(
            pred=frequency_distribution,
            target=bpm_distribution,
        )

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)

        return (
            kl_div_loss,
            F.cross_entropy(complex_absolute, bpm.view((1)).type(torch.long)),
            torch.abs(bpm[0] - whole_max_idx),
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        bpm: torch.Tensor,
        min_bpm: int = 40,
        max_bpm: int = 179,
        std: float = 1.0,
        frame_rate: torch.Tensor = 30.0,
        alpha: float = 0.05,
        beta: float = 5.0,
    ) -> Dict[str, torch.Tensor]:
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

        kl_div_loss, freq_ce_loss, bpm_mae = self.combined_loss(
            pred=pred,
            bpm=bpm,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            std=std,
            frame_rate=frame_rate,
        )

        total_loss = alpha * rppg_loss + beta * (freq_ce_loss + kl_div_loss)

        return {
            "total_loss": total_loss,
            "rppg_loss": rppg_loss,
            "kl_div_loss": kl_div_loss,
            "freq_ce_loss": freq_ce_loss,
            "bpm_mae": bpm_mae,
        }
