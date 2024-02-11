from typing import Optional

from torch.utils.data import DataLoader

from lightning.pytorch import LightningDataModule

from .stmap_dataset import CustomDataset


class CustomDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(
        self,
        stage: Optional[str] = None,
    ) -> None:
        if stage == "fit" or stage is None:
            self.train = CustomDataset(data_path=self.hparams.data_path, split="train")
            self.val = CustomDataset(data_path=self.hparams.data_path, split="val")

        if stage == "test" or stage is None:
            self.test = CustomDataset(data_path=self.hparams.data_path, split="test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=True,
        )
