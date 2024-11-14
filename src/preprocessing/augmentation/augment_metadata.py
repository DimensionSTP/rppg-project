import dotenv

dotenv.load_dotenv(
    override=True,
)

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../../configs/",
    config_name="physca.yaml",
)
def augment_metadata(
    config: DictConfig,
) -> None:
    if config.data_type not in ["ubfc", "pure"]:
        raise ValueError(
            f"Invalid data type: {config.data_type}. Choose in ['ubfc', 'pure']."
        )

    df = pd.read_csv(f"{config.connected_dir}/metadata/{config.data_type}/metadata.csv")

    df_columns = [
        "dir_path",
        "image_name",
        config.ecg_column_name,
        config.bpm_column_name,
    ]
    df = df[df_columns]

    df[config.ecg_column_name] = df[config.ecg_column_name].astype(float)
    df[config.bpm_column_name] = df[config.bpm_column_name].astype(float)

    df[config.frame_index_column_name] = (
        df["image_name"].str.extract(r"(\d+)").astype(int)
    )

    df_list = []
    dir_groups = df.groupby("dir_path")
    for dir_path, group in tqdm(dir_groups):
        for start_index in range(
            0,
            group[config.frame_index_column_name].max() + 1,
            config.aug_interval,
        ):
            subset_df = group[
                group[config.frame_index_column_name].between(
                    start_index,
                    start_index + config.clip_frame_size - 1,
                )
            ]

            if len(subset_df) < config.clip_frame_size:
                continue

            subset_df = subset_df.iloc[: config.clip_frame_size]

            row = {
                config.file_path_column_name: str(dir_path),
                config.frame_index_column_name: start_index,
                config.tube_index_column_name: 0,
                config.frame_rate_column_name: 30.0,
                config.ecg_column_name: subset_df[config.ecg_column_name].tolist(),
                config.bpm_column_name: subset_df[config.bpm_column_name].tolist(),
            }
            df_list.append(row)

    augmented_df = pd.DataFrame(df_list)

    train_df, test_df = train_test_split(
        augmented_df,
        test_size=config.split_ratio,
        random_state=config.seed,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    augmented_df.to_pickle(
        f"{config.connected_dir}/metadata/{config.data_type}/aug_interval={config.aug_interval}_all.pkl"
    )
    train_df.to_pickle(
        f"{config.connected_dir}/metadata/{config.data_type}/aug_interval={config.aug_interval}_train.pkl"
    )
    test_df.to_pickle(
        f"{config.connected_dir}/metadata/{config.data_type}/aug_interval={config.aug_interval}_test.pkl"
    )


if __name__ == "__main__":
    augment_metadata()
