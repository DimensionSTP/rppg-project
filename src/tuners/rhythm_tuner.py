import os
from typing import Dict, Any
import json
import warnings

warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.wandb import WandbLogger

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

from ..architectures.rhythm_architecture import RythmArchitecture
from ..architectures.models.customized_rhythmnet import CustomizedRhythmNet


class RhythmTuner:
    def __init__(
        self,
        hparams: Dict[str, Any],
        module_params: Dict[str, Any],
        num_trials: int,
        seed: int,
        hparams_save_path: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: WandbLogger,
    ) -> None:
        self.hparams = hparams
        self.module_params = module_params
        self.num_trials = num_trials
        self.hparams_save_path = hparams_save_path
        self.seed = seed
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger

    def __call__(self) -> None:
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=self.seed),
            pruner=HyperbandPruner(),
        )
        study.optimize(self.optuna_objective, n_trials=self.num_trials)
        trial = study.best_trial
        best_score = trial.value
        best_params = trial.params
        print(f"Best score : {best_score}")
        print(f"Parameters : {best_params}")

        if not os.path.exists(self.hparams_save_path):
            os.makedirs(self.hparams_save_path, exist_ok=True)

        with open(f"{self.hparams_save_path}/best_params.json", "w") as json_file:
            json.dump(best_params, json_file)

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
        if self.hparams.t_max:
            params["t_max"] = trial.suggest_int(
                name="t_max",
                low=self.hparams.t_max.low,
                high=self.hparams.t_max.high,
                log=self.hparams.t_max.log,
            )
        if self.hparams.eta_min:
            params["eta_min"] = trial.suggest_float(
                name="eta_min",
                low=self.hparams.eta_min.low,
                high=self.hparams.eta_min.high,
                log=self.hparams.eta_min.log,
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
            lr=params["lr"],
            t_max=params["t_max"],
            eta_min=params["eta_min"],
            interval=self.module_params.interval,
            project_dir=self.module_params.project_dir,
        )

        pruning_callback = PyTorchLightningPruningCallback(
            trial, monitor="val_rmse_loss"
        )
        self.logger.log_hyperparams(params)

        trainer = Trainer(
            devices=1,
            accelerator=self.module_params.accelerator,
            log_every_n_steps=self.module_params.log_every_n_steps,
            precision=self.module_params.precision,
            max_epochs=self.module_params.max_epochs,
            enable_checkpointing=False,
            callbacks=pruning_callback,
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

        return trainer.callback_metrics["val_rmse_loss"].item()
