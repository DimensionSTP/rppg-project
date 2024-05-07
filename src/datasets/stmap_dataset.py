from typing import Dict, Any, List
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
        target_column_name: str,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.target_column_name = target_column_name
        self.split = split
        self.data_list = self.get_dataset()

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        stmap_path = self.data_path[:-1]
        stmap_name = self.data_list[idx].split("/")[-1][:-4] + ".npy"
        if self.split == "predict":
            stmap = torch.tensor(
                np.load(f"{stmap_path}/test/stmaps/{stmap_name}"),
                dtype=torch.float32,
            )
        else:
            stmap = torch.tensor(
                np.load(f"{stmap_path}/{self.split}/stmaps/{stmap_name}"),
                dtype=torch.float32,
            )
        label_df = pd.read_csv(self.data_list[idx])
        label = torch.tensor(
            list(label_df[self.target_column_name]),
            dtype=torch.float32,
        )
        return {
            "stmap": stmap,
            "label": label,
            "index": idx,
        }

    def get_dataset(self) -> List[str]:
        if self.split == "predict":
            data_list = glob.glob(self.data_path + "test" + "/hrvs/*.csv")
        else:
            data_list = glob.glob(self.data_path + self.split + "/hrvs/*.csv")
        data_list = [i.replace("\\", "/", 10) for i in data_list]
        return data_list
