import os
from typing import Optional

from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule

from .datasets.dataset import STMapDataset


class STMapDataModule(LightningDataModule):
    def __init__(self, data_path: str, batch_size: int):
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = STMapDataset(data_path=self.hparams.data_path, split="train")
            self.val = STMapDataset(data_path=self.hparams.data_path, split="val")

        if stage == "test" or stage is None:
            self.test = STMapDataset(data_path=self.hparams.data_path, split="test")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=os.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=os.cpu_count(),
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=os.cpu_count(),
        )
