from typing import Dict, Any
import math

import torch
from torch import optim, nn
from torch.nn import functional as F

from lightning.pytorch import LightningModule

from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam


class RythmArchitecture(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        strategy: str,
        lr: float,
        weight_decay: float,
        warmup_rate: float,
        eta_min_rate: float,
        interval: str,
        connected_dir: str,
    ) -> None:
        super().__init__()
        self.model = model
        self.strategy = strategy
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_rate = warmup_rate
        self.eta_min_rate = eta_min_rate
        self.interval = interval
        self.connected_dir = connected_dir

    def forward(
        self,
        stmap: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(stmap)
        rnn_output_seq = output["rnn_output_sequence"]
        return rnn_output_seq

    def step(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        stmap = batch["stmap"]
        label = batch["label"]
        index = batch["index"]
        pred = self(stmap)
        loss = F.mse_loss(
            pred,
            label,
        )
        visual_loss = F.l1_loss(
            pred,
            label,
        )
        return {
            "loss": loss,
            "visual_loss": visual_loss,
            "pred": pred,
            "label": label,
            "index": index,
        }

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.strategy == "deepspeed_stage_3":
            optimizer = FusedAdam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif (
            self.strategy == "deepspeed_stage_2_offload"
            or self.strategy == "deepspeed_stage_3_offload"
        ):
            optimizer = DeepSpeedCPUAdam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        total_steps = self.trainer.estimated_stepping_batches * self.trainer.max_epochs
        warmup_steps = int(total_steps * self.warmup_rate)
        t_max = total_steps - warmup_steps
        eta_min = self.lr * self.eta_min_rate

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        warmup_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda,
        )
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                warmup_scheduler,
                main_scheduler,
            ],
            milestones=[
                warmup_steps,
            ],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.interval,
            },
        }

    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        output = self.step(batch)
        loss = output["loss"]
        visual_loss = output["visual_loss"]
        pred = output["pred"]
        label = output["label"]
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
        return {
            "loss": loss,
            "pred": pred,
            "label": label,
        }

    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        output = self.step(batch)
        loss = output["loss"]
        visual_loss = output["visual_loss"]
        pred = output["pred"]
        label = output["label"]
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
        return {
            "loss": loss,
            "pred": pred,
            "label": label,
        }

    def test_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        output = self.step(batch)
        loss = output["loss"]
        visual_loss = output["visual_loss"]
        pred = output["pred"]
        label = output["label"]
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
        return {
            "loss": loss,
            "pred": pred,
            "label": label,
        }

    def predict_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        output = self.step(batch)
        pred = output["pred"]
        index = output["index"]
        index = index.unsqueeze(-1).float()
        output = torch.cat(
            (
                pred,
                index,
            ),
            dim=-1,
        )
        gathered_output = self.all_gather(output)
        return gathered_output

    def on_train_epoch_end(self) -> None:
        pass

    def on_validation_epoch_end(self) -> None:
        pass

    def on_test_epoch_end(self) -> None:
        pass
