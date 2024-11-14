import dotenv

dotenv.load_dotenv(
    override=True,
)

from typing import List
import os

import numpy as np
import pandas as pd
import json
from tqdm import tqdm

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../../configs/",
    config_name="physca.yaml",
)
def get_pure_metadata(
    config: DictConfig,
) -> None:
    if config.data_type != "pure":
        raise ValueError(
            f"Invalid data type: {config.data_type}. Only you can choose 'pure'."
        )

    dataset_path = f"{config.connected_dir}/data/PURE"
    combined_df = pd.DataFrame()
    for dir in tqdm(sorted(os.listdir(dataset_path))):
        dir_path = os.path.join(
            dataset_path,
            dir,
        )
        if os.path.isdir(dir_path):
            processed_df = process_directory(
                dir_path=dir_path,
                config=config,
            )
            if processed_df is not None:
                combined_df = pd.concat(
                    [
                        combined_df,
                        processed_df,
                    ],
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


def interpolate_values(
    df: pd.DataFrame,
    timestamps: List[int],
    config: DictConfig,
) -> pd.DataFrame:
    results = []
    timestamps = np.array(
        timestamps,
        dtype=np.int64,
    )
    for ts in timestamps:
        idx = np.searchsorted(
            df.index,
            ts,
        )
        previous_ts = df.index[idx - 1] if idx > 0 else None
        next_ts = df.index[idx] if idx < len(df.index) else None

        previous_bpm = (
            df.loc[previous_ts, "pulseRate"] if previous_ts is not None else None
        )
        next_bpm = df.loc[next_ts, "pulseRate"] if next_ts is not None else None
        previous_ecg = (
            df.loc[previous_ts, "waveform"] if previous_ts is not None else None
        )
        next_ecg = df.loc[next_ts, "waveform"] if next_ts is not None else None

        if previous_ts is not None and next_ts is not None:
            dist_prev = ts - previous_ts
            dist_next = next_ts - ts
            ecg = (previous_ecg * dist_next + next_ecg * dist_prev) / (
                dist_prev + dist_next
            )
            bpm = (previous_bpm * dist_next + next_bpm * dist_prev) / (
                dist_prev + dist_next
            )
        elif previous_ts is not None:
            ecg = previous_ecg
            bpm = previous_bpm
        elif next_ts is not None:
            ecg = next_ecg
            bpm = next_bpm
        else:
            continue

        results.append(
            {
                "timestamp": ts,
                config.ecg_column_name: ecg,
                config.bpm_column_name: bpm,
                "previous_timestamp": previous_ts,
                "previous_ECG": previous_ecg,
                "previous_BPM": previous_bpm,
                "next_timestamp": next_ts,
                "next_ECG": next_ecg,
                "next_BPM": next_bpm,
            }
        )
    metadata_df = (
        pd.DataFrame(results).sort_values(by="timestamp").reset_index(drop=True)
    )
    metadata_df["image_name"] = [f"image{idx:05d}.png" for idx in metadata_df.index]
    return metadata_df


def process_directory(
    dir_path: str,
    config: DictConfig,
) -> pd.DataFrame:
    json_path = os.path.join(dir_path, f"{os.path.basename(dir_path)}.json")

    if not os.path.isfile(json_path):
        return None

    with open(json_path, "r") as f:
        data = json.load(f)

    timestamps = [entry["Timestamp"] for entry in data["/FullPackage"]]
    values = [
        {"Timestamp": entry["Timestamp"], **entry["Value"]}
        for entry in data["/FullPackage"]
    ]

    df = pd.DataFrame(values).set_index("Timestamp")
    processed_df = interpolate_values(
        df=df,
        timestamps=timestamps,
        config=config,
    )
    processed_df["dir_path"] = os.path.basename(dir_path)
    processed_df[config.frame_rate_column_name] = 30.0

    processed_df = processed_df[
        [
            "dir_path",
            "image_name",
            "timestamp",
            config.frame_rate_column_name,
            config.ecg_column_name,
            config.bpm_column_name,
            "previous_timestamp",
            "previous_ECG",
            "previous_BPM",
            "next_timestamp",
            "next_ECG",
            "next_BPM",
        ]
    ]
    return processed_df


if __name__ == "__main__":
    get_pure_metadata()
