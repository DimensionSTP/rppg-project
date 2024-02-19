import os
import math
from typing import Tuple, Dict, Any

import pandas as pd

import torch
from torch import nn, optim
from torch.nn import functional as F

from lightning.pytorch import LightningModule

from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam


class RythmArchitecture(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        strategy: str,
        lr: float,
        t_max: int,
        eta_min: float,
        interval: str,
        connected_dir: str,
    ) -> None:
        super().__init__()
        self.model = model
        self.strategy = strategy
        self.lr = lr
        self.t_max = t_max
        self.eta_min = eta_min
        self.interval = interval
        self.connected_dir = connected_dir

    def forward(
        self,
        stmap: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, output = self.model(stmap)
        return output

    def step(
        self,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stmap, label = batch
        pred = self(stmap)
        loss = F.mse_loss(pred, label)
        visual_loss = F.l1_loss(pred, label)
        return (loss, pred, label, visual_loss)

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.strategy == "deepspeed_stage_3":
            optimizer = FusedAdam(self.parameters(), lr=self.lr)
        elif (
            self.strategy == "deepspeed_stage_2_offload"
            or self.strategy == "deepspeed_stage_3_offload"
        ):
            optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.lr)
        else:
            optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.t_max, eta_min=self.eta_min
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": self.interval},
        }

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, pred, label, visual_loss = self.step(batch)
        self.log(
            "train_rmse_loss",
            math.sqrt(loss),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_mae_loss",
            visual_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, pred, label, visual_loss = self.step(batch)
        self.log(
            "val_rmse_loss",
            math.sqrt(loss),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_mae_loss",
            visual_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, pred, label, visual_loss = self.step(batch)
        self.log(
            "test_rmse_loss",
            math.sqrt(loss),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "test_mae_loss",
            visual_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def predict_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        loss, pred, label, visual_loss = self.step(batch)
        pred = pred.view(-1)
        label = label.view(-1)
        pred = pred.tolist()
        label = label.tolist()
        table = {"pred": pred, "label": label}
        df = pd.DataFrame(table)
        if not os.path.exists(f"{self.connected_dir}/records"):
            os.makedirs(f"{self.connected_dir}/records")
        df.to_csv(f"{self.connected_dir}/records/{batch_idx}.csv", index=False)

    def train_epoch_end(
        self,
        train_step_outputs: torch.Tensor,
    ) -> None:
        pass

    def validation_epoch_end(
        self,
        validation_step_outputs: torch.Tensor,
    ) -> None:
        pass

    def test_epoch_end(
        self,
        test_step_outputs: torch.Tensor,
    ) -> None:
        pass
