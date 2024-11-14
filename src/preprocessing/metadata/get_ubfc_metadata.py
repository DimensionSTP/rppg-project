import dotenv

dotenv.load_dotenv(
    override=True,
)

import os

import pandas as pd
import cv2
from tqdm import tqdm

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../../configs/",
    config_name="physca.yaml",
)
def get_ubfc_metadata(
    config: DictConfig,
) -> None:
    if config.data_type != "ubfc":
        raise ValueError(
            f"Invalid data type: {config.data_type}. Only you can choose 'ubfc'."
        )

    dataset_path = f"{config.connected_dir}/data/UBFC-rPPG/DATASET_2"
    all_data = []

    for subject_folder in tqdm(sorted(os.listdir(dataset_path))):
        subject_path = os.path.join(
            dataset_path,
            subject_folder,
        )
        if not os.path.isdir(subject_path):
            continue
        ground_truth_path = os.path.join(
            subject_path,
            "ground_truth.txt",
        )
        video_path = os.path.join(
            subject_path,
            "vid.avi",
        )

        if not os.path.isfile(ground_truth_path) or not os.path.isfile(video_path):
            print(f"Missing ground_truth.txt or vid.avi in {subject_folder}")
            continue

        with open(ground_truth_path, "r") as f:
            lines = f.readlines()

        ecg_values = lines[0].split()
        bpm_values = lines[1].split()

        assert len(ecg_values) == len(
            bpm_values
        ), "Mismatch between ECG and BPM lengths."

        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        for index in range(len(ecg_values)):
            image_name = f"image_{index:05d}.png"
            data_entry = {
                "dir_path": subject_folder,
                "image_name": image_name,
                config.frame_rate_column_name: frame_rate,
                config.ecg_column_name: ecg_values[index],
                config.bpm_column_name: bpm_values[index],
            }
            all_data.append(data_entry)

    df = pd.DataFrame(all_data)
    save_path = f"{config.connected_dir}/metadata/{config.data_type}"
    os.makedirs(
        save_path,
        exist_ok=True,
    )
    df.to_csv(
        f"{save_path}/metadata.csv",
        index=False,
    )


if __name__ == "__main__":
    get_ubfc_metadata()
