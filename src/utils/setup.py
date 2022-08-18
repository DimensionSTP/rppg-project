from typing import List, Any

from omegaconf import DictConfig
from hydra.utils import instantiate

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

class SetUp():
    def __init__(
        self,
        config: DictConfig
    ):
        self.config = config
        
    def get_dataset_module(self) -> LightningDataModule:
        dataset_module: LightningDataModule = instantiate(self.config.dataset_module)
        return dataset_module
    
    def get_architecture_module(self) -> LightningModule:
        architecture_module: LightningModule = instantiate(self.config.architecture_module)
        return architecture_module

    def get_callbacks(self) -> List:
        model_checkpotint: ModelCheckpoint = instantiate(self.config.callbacks.model_checkpoint) 
        early_stopping: EarlyStopping = instantiate(self.config.callbacks.early_stopping)
        return [model_checkpotint, early_stopping]
        
    def get_wandb_logger(self) -> WandbLogger:
        wandb_logger: WandbLogger =  instantiate(self.config.logger.wandb)
        return wandb_logger