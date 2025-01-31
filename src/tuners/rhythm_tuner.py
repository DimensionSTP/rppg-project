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

from ..architectures.rhythm_architecture import RythmArchitecture
from ..architectures.models.customized_rhythmnet import CustomizedRhythmNet


class RhythmTuner:
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
        print(f"Best score: {best_score}")
        print(f"Parameters: {best_params}")

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
        if self.hparams.backbone:
            params["backbone"] = trial.suggest_categorical(
                name="backbone",
                choices=self.hparams.backbone,
            )
        if self.hparams.backbone_pretrained:
            params["backbone_pretrained"] = trial.suggest_categorical(
                name="backbone_pretrained",
                choices=self.hparams.backbone_pretrained,
            )
        if self.hparams.rnn_type:
            params["rnn_type"] = trial.suggest_categorical(
                name="rnn_type",
                choices=self.hparams.rnn_type,
            )
        if self.hparams.rnn_num_layers:
            params["rnn_num_layers"] = trial.suggest_int(
                name="rnn_num_layers",
                low=self.hparams.rnn_num_layers.low,
                high=self.hparams.rnn_num_layers.high,
                log=self.hparams.rnn_num_layers.log,
            )
        if self.hparams.direction:
            params["direction"] = trial.suggest_categorical(
                name="direction",
                choices=self.hparams.direction,
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

        model = CustomizedRhythmNet(
            backbone=params["backbone"],
            backbone_pretrained=params["backbone_pretrained"],
            rnn_type=params["rnn_type"],
            rnn_num_layers=params["rnn_num_layers"],
            direction=params["direction"],
        )
        architecture = RythmArchitecture(
            model=model,
            strategy=self.module_params.strategy,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            warmup_ratio=params["warmup_ratio"],
            eta_min_ratio=params["eta_min_ratio"],
            interval=self.module_params.interval,
            connected_dir=self.module_params.connected_dir,
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
