import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class CombinedLabelDistributionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def normal_distribution(mean, label_k, std):
        return math.exp(-((label_k - mean) ** 2) / (2 * std**2)) / (
            math.sqrt(2 * math.pi) * std
        )

    @staticmethod
    def kl_divergence_loss(predictions, targets):
        criterion = nn.KLDivLoss(reduction="batchmean")
        log_predictions = torch.log(predictions)
        loss = criterion(log_predictions, targets)
        return loss

    def negative_pearson_loss(self, predictions, targets):
        loss = 0
        for i in range(predictions.shape[0]):
            sum_x = torch.sum(predictions[i])
            sum_y = torch.sum(targets[i])
            sum_xy = torch.sum(predictions[i] * targets[i])
            sum_x2 = torch.sum(torch.pow(predictions[i], 2))
            sum_y2 = torch.sum(torch.pow(targets[i], 2))
            N = predictions.shape[1]
            pearson_correlation = (N * sum_xy - sum_x * sum_y) / (
                torch.sqrt(
                    (N * sum_x2 - torch.pow(sum_x, 2))
                    * (N * sum_y2 - torch.pow(sum_y, 2))
                )
            )
            loss += 1 - pearson_correlation
        return loss / predictions.shape[0]

    def compute_complex_absolute_given_k(self, output, k, N):
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

    def compute_complex_absolute(self, output, sampling_rate, bpm_range):
        output = output.view(1, -1)
        N = output.size()[1]
        unit_per_hz = sampling_rate / N
        feasible_bpm = bpm_range / 60.0
        k = feasible_bpm / unit_per_hz
        complex_absolute = self.compute_complex_absolute_given_k(output, k, N)
        return (1.0 / complex_absolute.sum()) * complex_absolute

    def compute_combined_loss(self, predictions, target, sampling_rate, std):
        target_distribution = [
            self.normal_distribution(int(target), i, std) for i in range(140)
        ]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.Tensor(target_distribution).cuda()

        predictions = predictions.view(1, -1)
        target = target.view(1, -1)

        bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
        complex_absolute = self.compute_complex_absolute(
            predictions, sampling_rate, bpm_range
        )
        frequency_distribution = F.softmax(complex_absolute.view(-1), dim=0)
        loss_kl = self.kl_divergence_loss(frequency_distribution, target_distribution)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)

        return (
            loss_kl,
            F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)),
            torch.abs(target[0] - whole_max_idx),
        )

    def forward(self, predictions, targets, avg_hr, frame_rate, a, b, std=1.0):
        # Normalizing predictions
        predictions = (
            predictions - torch.mean(predictions, dim=-1, keepdim=True)
        ) / torch.std(predictions, dim=-1, keepdim=True)

        # Calculate Negative Pearson Loss
        loss_rPPG = self.negative_pearson_loss(predictions, targets)

        # Adjust avg_hr
        avg_hr = avg_hr - 40

        # Calculate combined loss
        loss_kl, loss_freq_ce, mae_hr = self.compute_combined_loss(
            predictions, avg_hr, frame_rate, std
        )

        # Calculate total loss
        total_loss = a * loss_rPPG + b * (loss_freq_ce + loss_kl)

        return total_loss, loss_rPPG, loss_kl, loss_freq_ce, mae_hr
