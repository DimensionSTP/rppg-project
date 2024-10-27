from typing import Dict, Any, List
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from einops import rearrange

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


class VIPLDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        metadata_path: str,
        aug_interval: int,
        split: str,
        split_ratio: float,
        seed: int,
        file_path_column_name: str,
        frame_index_column_name: str,
        tube_index_column_name: str,
        frame_rate_column_name: str,
        bpm_column_name: str,
        ecg_column_name: str,
        num_devices: int,
        batch_size: int,
        clip_frame_size: int,
        image_size: int,
        augmentation_probability: float,
        augmentations: List[str],
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.aug_interval = aug_interval
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.file_path_column_name = file_path_column_name
        self.frame_index_column_name = frame_index_column_name
        self.tube_index_column_name = tube_index_column_name
        self.frame_rate_column_name = frame_rate_column_name
        self.bpm_column_name = bpm_column_name
        self.ecg_column_name = ecg_column_name
        self.num_devices = num_devices
        self.batch_size = batch_size
        self.augmentation_probability = augmentation_probability
        self.augmentations = augmentations
        metadata = self.get_metadata()
        self.file_paths = metadata["file_paths"]
        self.frame_indices = metadata["frame_indices"]
        self.tube_indices = metadata["tube_indices"]
        self.frame_rates = metadata["frame_rates"]
        self.bpms = metadata["bpms"]
        self.labels = metadata["labels"]
        self.clip_frame_size = clip_frame_size
        self.image_size = image_size
        self.transform = self.get_transform()

    def __len__(self) -> int:
        return len(self.bpms)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        images_path = os.path.join(
            self.data_path,
            "vipl_tube",
            str(self.file_paths[idx]),
            "mp_rgb_full",
        )

        tube_token = self.get_single_tube_token(
            images_path=images_path,
            frame_index=self.frame_indices[idx],
            tube_index=self.tube_indices[idx],
        )
        tube_token = tube_token / 255.0

        first_slice = tube_token[
            0,
            :,
            :,
            :,
        ]
        transformed = self.transform(image=first_slice)
        replay = transformed["replay"]
        transformed_slices = [
            A.ReplayCompose.replay(
                replay,
                image=tube_token[
                    depth,
                    :,
                    :,
                    :,
                ],
            )["image"]
            for depth in range(self.clip_frame_size)
        ]
        transformed_tube_token = torch.stack(
            transformed_slices,
            dim=0,
        )
        transformed_tube_token = rearrange(
            transformed_tube_token,
            "depth channel height width -> channel depth height width",
        )

        frame_rate = torch.tensor(
            self.frame_rates[idx],
            dtype=torch.float32,
        )
        bpm = torch.tensor(
            self.bpms[idx],
            dtype=torch.float32,
        )
        ecg_label = torch.tensor(
            self.labels[idx][: self.clip_frame_size],
            dtype=torch.float32,
        )
        return {
            "encoded": transformed_tube_token,
            self.frame_rate_column_name: frame_rate,
            self.bpm_column_name: bpm,
            "label": ecg_label,
            "index": idx,
        }

    def get_metadata(self) -> Dict[str, List[Any]]:
        if self.split in ["train", "val"]:
            pickle_path = (
                f"{self.metadata_path}/vipl/aug_interval={self.aug_interval}_train.pkl"
            )
            data = pd.read_pickle(pickle_path)
            data = data.fillna("_")
            train_data, val_data = train_test_split(
                data,
                test_size=self.split_ratio,
                random_state=self.seed,
                shuffle=True,
            )
            if self.split == "train":
                data = train_data
            else:
                data = val_data
        elif self.split == "test":
            pickle_path = f"{self.metadata_path}/vipl/aug_interval={self.aug_interval}_{self.split}.pkl"
            data = pd.read_pickle(pickle_path)
            data = data.fillna("_")
        elif self.split == "predict":
            pickle_path = (
                f"{self.metadata_path}/vipl/aug_interval={self.aug_interval}_test.pkl"
            )
            data = pd.read_pickle(pickle_path)
            data = data.fillna("_")
            if self.num_devices > 1:
                last_row = data.iloc[-1]
                total_batch_size = self.num_devices * self.batch_size
                remainder = (len(data) % total_batch_size) % self.num_devices
                if remainder != 0:
                    num_dummies = self.num_devices - remainder
                    repeated_rows = pd.DataFrame([last_row] * num_dummies)
                    repeated_rows.reset_index(
                        drop=True,
                        inplace=True,
                    )
                    data = pd.concat(
                        [
                            data,
                            repeated_rows,
                        ],
                        ignore_index=True,
                    )
        else:
            raise ValueError(f"Inavalid split: {self.split}")
        file_paths = data[self.file_path_column_name].tolist()
        frame_indices = data[self.frame_index_column_name].tolist()
        tube_indices = data[self.tube_index_column_name].tolist()
        frame_rates = data[self.frame_rate_column_name].tolist()
        bpms = data[self.bpm_column_name].tolist()
        labels = data[self.ecg_column_name].tolist()
        return {
            "file_paths": file_paths,
            "frame_indices": frame_indices,
            "tube_indices": tube_indices,
            "frame_rates": frame_rates,
            "bpms": bpms,
            "labels": labels,
        }

    def get_single_tube_token(
        self,
        images_path: str,
        frame_index: int,
        tube_index: int,
    ) -> np.ndarray:
        tube_token = np.zeros(
            (
                self.clip_frame_size,
                self.image_size,
                self.image_size,
                3,
            ),
        )
        crop_range = np.random.randint(self.image_size // 8)

        for i in range(self.clip_frame_size):
            frame = frame_index + tube_index + i
            image_name = f"image_{frame:05d}.png"
            image_path = os.path.join(
                images_path,
                image_name,
            )

            before_frame = frame - 1
            before_image_name = f"image_{before_frame:05d}.png"
            before_image_path = os.path.join(
                images_path,
                before_image_name,
            )

            after_frame = frame + 1
            after_image_name = f"image_{after_frame:05d}.png"
            after_image_path = os.path.join(
                images_path,
                after_image_name,
            )

            if os.path.exists(image_path):
                image = cv2.imread(image_path)
            elif os.path.exists(before_image_path):
                image = cv2.imread(before_image_path)
            elif os.path.exists(after_image_path):
                image = cv2.imread(after_image_path)
            else:
                raise ValueError(
                    f"There are no images in both successive front and back frames: {image_path}"
                )

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(
                image,
                (
                    self.image_size + crop_range,
                    self.image_size + crop_range,
                ),
                interpolation=cv2.INTER_CUBIC,
            )[
                (crop_range // 2) : (self.image_size + crop_range // 2),
                (crop_range // 2) : (self.image_size + crop_range // 2),
                :,
            ]
            tube_token[
                i,
                :,
                :,
                :,
            ] = image
        return tube_token

    def get_transform(self) -> A.ReplayCompose:
        transforms = [A.Resize(self.image_size, self.image_size)]
        if self.split == "train":
            for aug in self.augmentations:
                if aug == "hflip":
                    transforms.append(
                        A.HorizontalFlip(
                            p=self.augmentation_probability,
                        )
                    )
                elif aug == "vflip":
                    transforms.append(
                        A.VerticalFlip(
                            p=self.augmentation_probability,
                        )
                    )
            transforms.append(
                A.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                )
            )
            transforms.append(ToTensorV2())
            return A.ReplayCompose(transforms)
        else:
            transforms.append(
                A.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                )
            )
            transforms.append(ToTensorV2())
            return A.ReplayCompose(transforms)
