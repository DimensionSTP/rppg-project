import dotenv

dotenv.load_dotenv(
    override=True,
)

import os

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../../configs/",
    config_name="physca.yaml",
)
def save_tube(
    config: DictConfig,
) -> None:
    train_df = pd.read_pickle(
        f"{config.connected_dir}/metadata/{config.data_type}/aug_interval={config.aug_interval}_train.pkl"
    )
    test_df = pd.read_pickle(
        f"{config.connected_dir}/metadata/{config.data_type}/aug_interval={config.aug_interval}_test.pkl"
    )

    for metadata_df in [train_df, test_df]:
        metadata_df = metadata_df.fillna("_")
        for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
            file_path = row[config.file_path_column_name]
            frame_index = row[config.frame_index_column_name]
            tube_index = row[config.tube_index_column_name]
            images_path = os.path.join(
                f"{config.connected_dir}/data",
                f"{config.data_type}_image",
                str(file_path),
                "mp_rgb_full",
            )
            tube_token = get_single_tube_token(
                clip_frame_size=config.clip_frame_size,
                image_size=config.image_size,
                images_path=images_path,
                frame_index=frame_index,
                tube_index=tube_index,
            )
            save_path = os.path.join(
                f"{config.connected_dir}/data",
                f"{config.data_type}_tube",
                str(file_path),
                "mp_tube",
            )
            save_name = f"clip_frame_size={config.clip_frame_size}-aug_interval={config.aug_interval}-frame_index={frame_index}-tube_index={tube_index}.npy"
            os.makedirs(
                save_path,
                exist_ok=True,
            )
            np.save(
                f"{save_path}/{save_name}",
                tube_token,
            )


def get_single_tube_token(
    clip_frame_size: int,
    image_size: int,
    images_path: str,
    frame_index: int,
    tube_index: int,
) -> np.ndarray:
    tube_token = np.zeros(
        (
            clip_frame_size,
            image_size,
            image_size,
            3,
        ),
    )

    before_image = None
    for i in range(clip_frame_size):
        frame = frame_index + tube_index + i
        image_name = f"image_{frame:05d}.png"
        image_path = os.path.join(
            images_path,
            image_name,
        )

        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            before_image = image
        else:
            image = before_image

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )
        image = cv2.resize(
            image,
            (
                image_size,
                image_size,
            ),
            interpolation=cv2.INTER_CUBIC,
        )[
            0:image_size,
            0:image_size,
            :,
        ]
        tube_token[
            i,
            :,
            :,
            :,
        ] = image
    return tube_token


if __name__ == "__main__":
    save_tube()
