import dotenv

dotenv.load_dotenv(
    override=True,
)

import os
import warnings
from glob import glob
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pandas as pd
import mne
import cv2
from tqdm import tqdm

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../../configs/",
    config_name="physca.yaml",
)
def get_mahnob_metadata(
    config: DictConfig,
) -> None:
    if config.data_type != "mahnob":
        raise ValueError(
            f"Invalid data type: {config.data_type}. Only you can choose 'mahnob'."
        )

    dataset_path = f"{config.connected_dir}/data/MAHNOB-HCI-HR-Estimation-V1/Sessions"
    session_folders = sorted([f for f in os.listdir(dataset_path) if f.isdigit()])
    original_timestamp_column = "DateTimeStampStartOffset"
    sessions_dfs = []
    for session in tqdm(session_folders):
        dir_path = os.path.join(
            dataset_path,
            session,
        )

        bdf_files = glob(
            os.path.join(
                dir_path,
                "*.bdf",
            )
        )
        if not bdf_files:
            continue
        bdf_file = bdf_files[0]

        tsv_files = glob(
            os.path.join(
                dir_path,
                "*All-Data*.tsv",
            )
        )
        if not tsv_files:
            continue
        tsv_file = tsv_files[0]

        video_files = glob(
            os.path.join(
                dir_path,
                "*C1*.avi",
            )
        )
        if not video_files:
            continue
        video_file = video_files[0]

        with redirect_stdout(StringIO()):
            raw_data = mne.io.read_raw_bdf(
                bdf_file,
                preload=True,
            )
        ecg_data, times = raw_data["EXG1"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            all_data_df = pd.read_csv(
                tsv_file,
                sep="\t",
                skiprows=23,
                error_bad_lines=False,
                warn_bad_lines=False,
            )
        all_data_df = all_data_df.dropna(
            subset=[original_timestamp_column]
        ).reset_index(drop=True)
        frame_timestamps = pd.to_datetime(
            all_data_df.loc[:, original_timestamp_column],
            format="%H:%M:%S.%f",
        ) - pd.to_datetime("00:00:00.000")
        frame_timestamps = frame_timestamps.dt.total_seconds()
        initial_offset = frame_timestamps.iloc[0]
        frame_timestamps = frame_timestamps - initial_offset

        cap = cv2.VideoCapture(video_file)
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        mapped_ecgs = []
        for frame_time in frame_timestamps:
            idx = np.searchsorted(
                times,
                frame_time,
            )
            if idx == 0:
                avg_ecg = ecg_data[0][idx]
            elif idx >= len(times):
                avg_ecg = ecg_data[0][-1]
            else:
                avg_ecg = (ecg_data[0][idx - 1] + ecg_data[0][idx]) / 2
            mapped_ecgs.append(avg_ecg)

        session_df = pd.DataFrame(
            {
                "dir_path": int(session),
                config.frame_index_column_name: range(len(frame_timestamps)),
                "timestamp": frame_timestamps,
                config.frame_rate_column_name: float(frame_rate),
                config.ecg_column_name: mapped_ecgs,
            }
        )
        sessions_dfs.append(session_df)

    combined_df = pd.concat(
        sessions_dfs,
        ignore_index=True,
    )
    save_path = f"{config.connected_dir}/metadata/{config.data_type}"
    os.makedirs(
        save_path,
        exist_ok=True,
    )
    combined_df.to_csv(
        f"{save_path}/metadata.csv",
        index=False,
    )


if __name__ == "__main__":
    get_mahnob_metadata()
