import os
from typing import Tuple

import numpy as np
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset


class VIPLDataset(Dataset):
    def __init__(
        self,
        clip_frame_size: int,
        preprocessed_dataset: str,
        data_path: str,
        transform: torch,
    ) -> None:
        super().__init__()
        self.clip_frame_size = clip_frame_size
        self.preprocessed_dataset = pd.read_csv(
            preprocessed_dataset,
            delimiter=" ",
            header=None,
        )
        self.data_path = data_path
        self.transform = transform

    def __len__(self) -> int:
        pass

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[np.ndarray, float, float, np.ndarray]:
        video_path = os.path.join(
            self.root_dir,
            str(self.preprocessed_dataset.iloc[idx, 0]),
        )
        start_frame = self.preprocessed_dataset.iloc[idx, 1]

        tube_token = self.get_single_tube_token(
            video_path,
            start_frame,
        )
        frame_rate = self.preprocessed_dataset.iloc[idx, 2]
        avg_hr = self.preprocessed_dataset.iloc[idx, 3]
        ecg_label = self.preprocessed_dataset.iloc[idx, 5 : 5 + 160].values
        return (tube_token, frame_rate, avg_hr, ecg_label)

    def get_single_tube_token(
        self,
        video_path: str,
        start_frame: int,
    ) -> np.ndarray:
        tube_token = np.zeros(
            self.clip_frame_size,
            128,
            128,
            3,
        )
        crop_range = np.random.randint(16)

        for i in range(self.clip_frame_size):
            frame = start_frame + i
            image_name = f"image_{frame:05d}.png"
            image_path = os.path.join(
                video_path,
                image_name,
            )
            image = cv2.imread(image_path)
            if image is None:  # It seems some frames missing
                image = cv2.imread(self.root_dir + "p30/v1/source2/image_00737.png")
            image = cv2.resize(
                image,
                (128 + crop_range, 128 + crop_range),
                interpolation=cv2.INTER_CUBIC,
            )[
                (crop_range // 2) : (128 + crop_range // 2),
                (crop_range // 2) : (128 + crop_range // 2),
                :,
            ]
            tube_token[i, :, :, :] = image
        return tube_token
