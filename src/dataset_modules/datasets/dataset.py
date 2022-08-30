from typing import List, Any
import glob

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class STMapDataset(Dataset):
    def __init__(self, data_path: str, split: str):
        super().__init__()
        self.data_list = self.load_data(data_path, split)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        stmap = torch.tensor(
            np.load(f"{self.data_list[idx][:-4]}.npy"), dtype=torch.float32
        )
        label_df = pd.read_csv(self.data_list[idx])
        label = torch.tensor(list(label_df["BPM"]), dtype=torch.float32)
        return stmap, label

    @staticmethod
    def load_data(data_path: str, split: str) -> List:
        data_list = glob.glob(data_path + split + "/*.csv")
        return data_list
