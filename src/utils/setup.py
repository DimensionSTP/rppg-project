from typing import List

from omegaconf import DictConfig
from hydra.utils import instantiate

from torch.utils.data import Dataset, DataLoader

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger


class SetUp:
    def __init__(self, config: DictConfig):
        self.config = config
        self.train_split = config.split.train
        self.val_split = config.split.val
        self.test_split = config.split.test

    def get_train_loader(self) -> DataLoader:
        train_dataset: Dataset = instantiate(
            self.config.dataset_module, split=self.train_split
        )
        return DataLoader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def get_val_loader(self) -> DataLoader:
        val_dataset: Dataset = instantiate(
            self.config.dataset_module, split=self.val_split
        )
        return DataLoader(
            dataset=val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def get_test_loader(self) -> DataLoader:
        test_dataset: Dataset = instantiate(
            self.config.dataset_module, split=self.test_split
        )
        return DataLoader(
            dataset=test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def get_dataset_module(self) -> LightningDataModule:
        dataset_module: LightningDataModule = instantiate(self.config.dataset_module)
        return dataset_module

    def get_architecture_module(self) -> LightningModule:
        architecture_module: LightningModule = instantiate(
            self.config.architecture_module
        )
        return architecture_module

    def get_callbacks(self) -> List:
        model_checkpotint: ModelCheckpoint = instantiate(
            self.config.callbacks.model_checkpoint
        )
        early_stopping: EarlyStopping = instantiate(
            self.config.callbacks.early_stopping
        )
        return [model_checkpotint, early_stopping]

    def get_wandb_logger(self) -> WandbLogger:
        wandb_logger: WandbLogger = instantiate(self.config.logger.wandb)
        return wandb_logger
