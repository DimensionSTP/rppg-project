from typing import Tuple, List
import glob

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.data_list = self.load_data(data_path, split)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        stmap_path = self.data_path[:-1]
        stmap_name = self.data_list[idx].split("/")[-1][:-4] + ".npy"
        stmap = torch.tensor(
            np.load(f"{stmap_path}/{self.split}/stmaps/{stmap_name}"),
            dtype=torch.float32,
        )
        label_df = pd.read_csv(self.data_list[idx])
        label = torch.tensor(
            list(label_df["BPM"]),
            dtype=torch.float32,
        )
        return (stmap, label)

    @staticmethod
    def load_data(
        data_path: str,
        split: str,
    ) -> List[str]:
        data_list = glob.glob(data_path + split + "/hrvs/*.csv")
        data_list = [i.replace("\\", "/", 10) for i in data_list]
        return data_list
