import os
from typing import List

import pandas as pd
import cv2

from hrv import HRV
import maxima_peak_detection as mpd

INTERVAL_SIZE = 0.5
SAMPLING_RATE = 500
DATA_PATH = "./preprocessing/hrv/data"
VIDEO_PATH = "./preprocessing/raw/cam_under300/"
FILE_NAME = os.listdir(VIDEO_PATH)
SAVE_PATH = "./preprocessing/hrv/result"
hrv = HRV()


def main(
    interval_size: int,
    sampling_rate: int,
    data_path: str,
    video_path: str,
    file_name: List,
    save_path: str,
):
    for file in file_name:
        cap = cv2.VideoCapture(os.path.join(video_path, file))
        frame_rate = cap.get(5)
        cap.release()
        window_size = int(300 / frame_rate)

        ppg_df = pd.read_csv(
            f"{data_path}/{file[:-4]}.txt", header=None, delimiter="\t", skiprows=23
        )
        print("-" * 200)
        print(f"{file[:-4]}.txt + 분석 시작!")

        ppg_value = ppg_df[1].values
        # Sliding window
        hrv_list = []

        for i in range(
            0,
            len(ppg_value) - window_size * sampling_rate,
            int(interval_size * sampling_rate),
        ):
            sliding_ppg = ppg_value[i : i + window_size * sampling_rate]
            sliding_ppg = sliding_ppg.astype("float")
            # Peak detection
            peak = mpd.detect_peaks(
                sliding_ppg, fs=sampling_rate, detrend_factor=1, is_display=False
            )
            # Calcualte HRV
            hrv_list.append(hrv(peak=peak, sampling_rate=sampling_rate))
        # file write (.csv)
        df = pd.DataFrame(hrv_list)
        df.columns = ["BPM"]
        df.to_csv(
            f"{save_path}/{file[:-4]}_300.csv", encoding="utf-8-sig", index=False
        )  # 결과 저장
        print(f"{file[:-4]}.csv + 분석 결과 저장 완료!")
        print("-" * 200)


if __name__ == "__main__":
    main(
        interval_size=INTERVAL_SIZE,
        sampling_rate=SAMPLING_RATE,
        data_path=DATA_PATH,
        video_path=VIDEO_PATH,
        file_name=FILE_NAME,
        save_path=SAVE_PATH,
    )
