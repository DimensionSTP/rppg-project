from typing import Dict, Any
import math

import torch
from torch import optim, nn

from lightning.pytorch import LightningModule

from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam


class PhysFormerArchitecture(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        frame_rate_column_name: str,
        first_alpha: float,
        first_beta: float,
        alpha_factor: float,
        beta_factor: float,
        strategy: str,
        lr: float,
        weight_decay: float,
        warmup_ratio: float,
        eta_min_ratio: float,
        interval: str,
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.frame_rate_column_name = frame_rate_column_name
        self.first_alpha = first_alpha
        self.first_beta = first_beta
        self.alpha_factor = alpha_factor
        self.beta_factor = beta_factor
        self.strategy = strategy
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.eta_min_ratio = eta_min_ratio
        self.interval = interval

    def forward(
        self,
        encoded: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(encoded)
        return output

    def step(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        encoded = batch["encoded"]
        frame_rate = batch[self.frame_rate_column_name]
        label = batch["label"]
        index = batch["index"]

        output = self(encoded)
        pred = output["rppg"]

        total_epochs = self.trainer.max_epochs
        current_epoch = self.current_epoch
        epoch_scalar = current_epoch / total_epochs
        alpha = self.first_alpha * math.pow(
            self.alpha_factor,
            epoch_scalar,
        )
        beta = self.first_beta * math.pow(
            self.beta_factor,
            epoch_scalar,
        )

        losses = self.criterion(
            pred=pred,
            target=label,
            frame_rate=frame_rate,
            alpha=alpha,
            beta=beta,
        )

        loss = losses["total_loss"]
        visual_loss = losses["bpm_mae"]
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
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.warmup_ratio)
        t_max = total_steps - warmup_steps
        eta_min = self.lr * self.eta_min_ratio

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
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train_visual_loss",
            visual_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
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
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "val_visual_loss",
            visual_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
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
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "test_visual_loss",
            visual_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
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
