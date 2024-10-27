import dotenv

dotenv.load_dotenv(
    override=True,
)

import os

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="physformer.yaml",
)
def preprocess_vipl_metadata(
    config: DictConfig,
) -> None:
    df = pd.read_csv(
        f"{config.connected_dir}/metadata/vipl/fold1_train.txt",
        sep="\s+",
        header=None,
    )

    df.columns = [
        config.file_path_column_name,
        config.frame_index_column_name,
        config.frame_rate_column_name,
        config.bpm_column_name,
        "std",
    ] + [f"col_{i}" for i in range(5, df.shape[1])]

    df[config.ecg_column_name] = df.apply(
        lambda row: row[5:].tolist(),
        axis=1,
    )

    df = df[
        [
            config.file_path_column_name,
            config.frame_index_column_name,
            config.frame_rate_column_name,
            config.bpm_column_name,
            "std",
            config.ecg_column_name,
        ]
    ]

    df[config.frame_index_column_name] = df[config.frame_index_column_name] - 1

    ecg_length = len(df.iloc[0, -1])
    end_index = ecg_length - config.clip_frame_size
    aug_times = end_index // config.aug_interval
    tube_indices = [i * config.aug_interval for i in range(aug_times)]

    df_list = []
    for tube_index in tube_indices:
        copied_df = df.copy()
        copied_df.insert(
            2,
            config.tube_index_column_name,
            tube_index,
        )
        df_list.append(copied_df)
    augmented_df = pd.concat(
        df_list,
        ignore_index=True,
    )

    missing_file_paths = set()
    checked_paths = {}

    for _, row in tqdm(augmented_df.iterrows(), total=len(augmented_df)):
        file_path = row[config.file_path_column_name]
        tube_index = row[config.tube_index_column_name]
        frame_index = row[config.frame_index_column_name]

        checked_key = (
            file_path,
            tube_index,
            frame_index,
        )

        if file_path in missing_file_paths:
            continue

        if checked_key in checked_paths:
            if checked_paths[checked_key] is True:
                continue
        else:
            all_files_exist = True
            for i in range(config.clip_frame_size):
                frame_index = row[config.frame_index_column_name]
                tube_index = row[config.tube_index_column_name]
                frame = frame_index + tube_index + i
                image_name = f"image_{frame:05d}.png"
                image_path = os.path.join(
                    f"{config.connected_dir}/data/vipl_tube",
                    file_path,
                    "mp_rgb_full",
                    image_name,
                )

                if not os.path.exists(image_path):
                    all_files_exist = False
                    missing_file_paths.add(file_path)
                    break

            checked_paths[file_path] = all_files_exist
    augmented_df = augmented_df[
        ~augmented_df[config.file_path_column_name].isin(missing_file_paths)
    ]

    train_df, test_df = train_test_split(
        augmented_df,
        test_size=config.split_ratio,
        random_state=config.seed,
    )

    train_df.to_pickle(
        f"{config.connected_dir}/metadata/vipl/aug_interval={config.aug_interval}_train.pkl"
    )
    test_df.to_pickle(
        f"{config.connected_dir}/metadata/vipl/aug_interval={config.aug_interval}_test.pkl"
    )


if __name__ == "__main__":
    preprocess_vipl_metadata()
