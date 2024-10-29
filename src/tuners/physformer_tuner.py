from typing import Dict, Any
import os
import json

from torch.utils.data import DataLoader

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

from ..architectures.models.customized_physformerpp import CustomizedPhysFormerPP
from ..architectures.models.customized_physformer import CustomizedPhysFormer
from ..architectures.physformer_architecture import PhysFormerArchitecture


class PhysFormerTuner:
    def __init__(
        self,
        hparams: Dict[str, Any],
        module_params: Dict[str, Any],
        direction: str,
        seed: int,
        num_trials: int,
        hparams_save_path: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: WandbLogger,
    ) -> None:
        self.hparams = hparams
        self.module_params = module_params
        self.direction = direction
        self.seed = seed
        self.num_trials = num_trials
        self.hparams_save_path = hparams_save_path
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger

    def __call__(self) -> None:
        study = optuna.create_study(
            direction=self.direction,
            sampler=TPESampler(seed=self.seed),
            pruner=HyperbandPruner(),
        )
        study.optimize(
            self.optuna_objective,
            n_trials=self.num_trials,
        )
        trial = study.best_trial
        best_score = trial.value
        best_params = trial.params
        print(f"Best score : {best_score}")
        print(f"Parameters : {best_params}")

        if not os.path.exists(self.hparams_save_path):
            os.makedirs(
                self.hparams_save_path,
                exist_ok=True,
            )

        with open(f"{self.hparams_save_path}/best_params.json", "w") as json_file:
            json.dump(
                best_params,
                json_file,
            )

    def optuna_objective(
        self,
        trial: optuna.trial.Trial,
    ) -> float:
        seed_everything(self.seed)

        params = dict()
        params["seed"] = self.seed
        if self.hparams.sharp_gradient:
            params["sharp_gradient"] = trial.suggest_float(
                name="sharp_gradient",
                low=self.hparams.sharp_gradient.low,
                high=self.hparams.sharp_gradient.high,
                log=self.hparams.sharp_gradient.log,
            )
        if self.hparams.num_heads:
            params["num_heads"] = trial.suggest_categorical(
                name="num_heads",
                choices=self.hparams.num_heads,
            )
        if self.hparams.num_layers:
            params["num_layers"] = trial.suggest_categorical(
                name="num_layers",
                choices=self.hparams.num_layers,
            )
        if self.hparams.tcdc_theta:
            params["tcdc_theta"] = trial.suggest_float(
                name="tcdc_theta",
                low=self.hparams.tcdc_theta.low,
                high=self.hparams.tcdc_theta.high,
                log=self.hparams.tcdc_theta.log,
            )
        if self.hparams.attention_dropout:
            params["attention_dropout"] = trial.suggest_float(
                name="attention_dropout",
                low=self.hparams.attention_dropout.low,
                high=self.hparams.attention_dropout.high,
                log=self.hparams.attention_dropout.log,
            )
        if self.hparams.feed_forward_dropout:
            params["feed_forward_dropout"] = trial.suggest_float(
                name="feed_forward_dropout",
                low=self.hparams.feed_forward_dropout.low,
                high=self.hparams.feed_forward_dropout.high,
                log=self.hparams.feed_forward_dropout.log,
            )
        if self.hparams.std:
            params["std"] = trial.suggest_float(
                name="std",
                low=self.hparams.std.low,
                high=self.hparams.std.high,
                log=self.hparams.std.log,
            )
        if self.hparams.first_alpha:
            params["first_alpha"] = trial.suggest_float(
                name="first_alpha",
                low=self.hparams.first_alpha.low,
                high=self.hparams.first_alpha.high,
                log=self.hparams.first_alpha.log,
            )
        if self.hparams.first_beta:
            params["first_beta"] = trial.suggest_float(
                name="first_beta",
                low=self.hparams.first_beta.low,
                high=self.hparams.first_beta.high,
                log=self.hparams.first_beta.log,
            )
        if self.hparams.alpha_factor:
            params["alpha_factor"] = trial.suggest_float(
                name="alpha_factor",
                low=self.hparams.alpha_factor.low,
                high=self.hparams.alpha_factor.high,
                log=self.hparams.alpha_factor.log,
            )
        if self.hparams.beta_factor:
            params["beta_factor"] = trial.suggest_float(
                name="beta_factor",
                low=self.hparams.beta_factor.low,
                high=self.hparams.beta_factor.high,
                log=self.hparams.beta_factor.log,
            )
        if self.hparams.lr:
            params["lr"] = trial.suggest_float(
                name="lr",
                low=self.hparams.lr.low,
                high=self.hparams.lr.high,
                log=self.hparams.lr.log,
            )
        if self.hparams.weight_decay:
            params["weight_decay"] = trial.suggest_float(
                name="weight_decay",
                low=self.hparams.weight_decay.low,
                high=self.hparams.weight_decay.high,
                log=self.hparams.weight_decay.log,
            )
        if self.hparams.warmup_ratio:
            params["warmup_ratio"] = trial.suggest_float(
                name="warmup_ratio",
                low=self.hparams.warmup_ratio.low,
                high=self.hparams.warmup_ratio.high,
                log=self.hparams.warmup_ratio.log,
            )
        if self.hparams.eta_min_ratio:
            params["eta_min_ratio"] = trial.suggest_float(
                name="eta_min_ratio",
                low=self.hparams.eta_min_ratio.low,
                high=self.hparams.eta_min_ratio.high,
                log=self.hparams.eta_min_ratio.log,
            )

        if self.module_params.is_pp:
            model = CustomizedPhysFormerPP(
                is_pretrained=self.module_params.is_pretrained,
                patch_size=self.module_params.patch_size,
                feature_size=self.module_params.feature_size,
                sharp_gradient=params["sharp_gradient"],
                num_heads=params["num_heads"],
                model_dims=self.module_params.model_dims,
                tcdc_kernel_size=self.module_params.tcdc_kernel_size,
                tcdc_stride=self.module_params.tcdc_stride,
                tcdc_padding=self.module_params.tcdc_padding,
                tcdc_dilation=self.module_params.tcdc_dilation,
                tcdc_groups=self.module_params.tcdc_groups,
                tcdc_bias=self.module_params.tcdc_bias,
                tcdc_theta=params["tcdc_theta"],
                tcdc_eps=self.module_params.tcdc_eps,
                attention_dropout=params["attention_dropout"],
                feed_forward_dims=self.module_params.feed_forward_dims,
                feed_forward_dropout=params["feed_forward_dropout"],
                num_layers=params["num_layers"],
            )
        else:
            model = CustomizedPhysFormer(
                is_pretrained=self.module_params.is_pretrained,
                patch_size=self.module_params.patch_size,
                feature_size=self.module_params.feature_size,
                sharp_gradient=params["sharp_gradient"],
                num_heads=params["num_heads"],
                model_dims=self.module_params.model_dims,
                tcdc_kernel_size=self.module_params.tcdc_kernel_size,
                tcdc_stride=self.module_params.tcdc_stride,
                tcdc_padding=self.module_params.tcdc_padding,
                tcdc_dilation=self.module_params.tcdc_dilation,
                tcdc_groups=self.module_params.tcdc_groups,
                tcdc_bias=self.module_params.tcdc_bias,
                tcdc_theta=params["tcdc_theta"],
                tcdc_eps=self.module_params.tcdc_eps,
                attention_dropout=params["attention_dropout"],
                feed_forward_dims=self.module_params.feed_forward_dims,
                feed_forward_dropout=params["feed_forward_dropout"],
                num_layers=params["num_layers"],
            )
        architecture = PhysFormerArchitecture(
            model=model,
            frame_rate_column_name=self.module_params.frame_rate_column_name,
            bpm_column_name=self.module_params.bpm_column_name,
            min_bpm=self.module_params.min_bpm,
            max_bpm=self.module_params.max_bpm,
            std=params["std"],
            first_alpha=params["first_alpha"],
            first_beta=params["first_beta"],
            alpha_factor=params["alpha_factor"],
            beta_factor=params["beta_factor"],
            strategy=self.module_params.strategy,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            warmup_ratio=params["warmup_ratio"],
            eta_min_ratio=params["eta_min_ratio"],
            interval=self.module_params.interval,
        )

        self.logger.log_hyperparams(params)
        callbacks = EarlyStopping(
            monitor=self.module_params.monitor,
            mode=self.module_params.mode,
            patience=self.module_params.patience,
            min_delta=self.module_params.min_delta,
        )

        trainer = Trainer(
            devices=self.module_params.devices,
            accelerator=self.module_params.accelerator,
            strategy=self.module_params.strategy,
            log_every_n_steps=self.module_params.log_every_n_steps,
            precision=self.module_params.precision,
            accumulate_grad_batches=self.module_params.accumulate_grad_batches,
            gradient_clip_val=self.module_params.gradient_clip_val,
            gradient_clip_algorithm=self.module_params.gradient_clip_algorithm,
            max_epochs=self.module_params.max_epochs,
            enable_checkpointing=False,
            callbacks=callbacks,
            logger=self.logger,
        )

        try:
            trainer.fit(
                model=architecture,
                train_dataloaders=self.train_loader,
                val_dataloaders=self.val_loader,
            )
            self.logger.experiment.alert(
                title="Tuning Complete",
                text="Tuning process has successfully finished.",
                level="INFO",
            )
        except Exception as e:
            self.logger.experiment.alert(
                title="Tuning Error",
                text="An error occurred during tuning",
                level="ERROR",
            )
            raise e

        return trainer.callback_metrics[self.module_params.monitor].item()
