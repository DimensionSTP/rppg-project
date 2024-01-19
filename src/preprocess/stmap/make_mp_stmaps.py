import os
from typing import Tuple, List

from tqdm import tqdm
import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import MinMaxScaler

import pandas as pd

# Chunks the ROI into blocks of size 5x5
def chunkify(img, block_width: int = 5, block_height: int = 5,) -> List[np.ndarray]:
    shape = img.shape
    x_len = shape[1] // block_width
    y_len = shape[0] // block_height

    chunks = []
    x_indices = [i for i in range(0, shape[1] + 1, x_len)]
    y_indices = [i for i in range(0, shape[0] + 1, y_len)]

    for i in range(len(x_indices) - 1):
        start_x = x_indices[i]
        end_x = x_indices[i + 1]
        for j in range(len(y_indices) - 1):
            start_y = y_indices[j]
            end_y = y_indices[j + 1]
            chunks.append(img[start_x:end_x, start_y:end_y])

    return chunks


# Function to read the the video data as an array of frames and additionally return metadata like FPS, Dims etc.
def get_frames_and_video_meta_data(video_path: str, meta_data_only: bool = False,) -> Tuple[np.ndarray, float, int]:
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(5)  # frame rate

    # Frame dimensions: WxH
    frame_dims = (int(cap.get(3)), int(cap.get(4)))
    # Paper mentions a stride of 0.5 seconds = 15 frames
    sliding_window_stride = int(frame_rate / 2)
    num_frames = int(cap.get(7))
    if meta_data_only:
        return {
            "frame_rate": frame_rate,
            "sliding_window_stride": sliding_window_stride,
            "num_frames": num_frames,
        }

    # Frames from the video have shape num_frames x H x W x C
    frames = np.zeros((num_frames, frame_dims[1], frame_dims[0], 3), dtype="uint8")

    with tqdm(total=num_frames, desc="get frames") as pbar:
        frame_counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            pbar.update(1)
            if not ret:
                break

            frames[frame_counter, :, :, :] = frame
            frame_counter += 1
            if frame_counter == num_frames:
                break

    cap.release()
    return (frames, frame_rate, sliding_window_stride)


# Optimized function for converting videos to Spatio-temporal maps
def preprocess_video_to_st_maps(video_path: str,) -> Tuple[np.ndarray, int, List[int]]:
    frames, frame_rate, sliding_window_stride = get_frames_and_video_meta_data(
        video_path
    )

    num_frames = frames.shape[0]
    x_size = frames.shape[2]
    y_size = frames.shape[1]
    # clip_size = int(frame_rate * 10)
    clip_size = 300
    num_maps = int((num_frames - clip_size) / sliding_window_stride + 1)
    if num_maps < 0:
        print(video_path)
        return None

    # stacked_maps is the all the st maps for a given video (=num_maps) stacked.
    stacked_maps = np.zeros((num_maps, 3, clip_size, 25))
    # stacked_maps = np.zeros((num_maps, 3, clip_size, 255))
    # processed_maps will contain all the data after processing each frame, but not yet converted into maps
    processed_frames = []
    stacked_coordinates = []
    map_index = 0

    # Init scaler and detector
    scaler = MinMaxScaler()
    detector = mp.solutions.face_detection.FaceDetection(
        min_detection_confidence=0.8, model_selection=0
    )

    # First we process all the frames and then work with sliding window to save repeated processing for the same frame index
    for idx, frame in enumerate(tqdm(frames, desc="detect faces")):
        """
        Preprocess the Image
        Step 1: Use cv2 face detector based on Haar cascades
        Step 2: Crop the frame based on the face co-ordinates (we need to do 160%)
        Step 3: Downsample the face cropped frame to output_shape = 36x36
        """
        # frame = cv2.flip(frame, 0)
        faces = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # face detected
        if isinstance(faces.detections, list):
            coordinates = faces.detections[0].location_data.relative_bounding_box
            xmin = coordinates.xmin
            ymin = coordinates.ymin
            width = coordinates.width
            height = coordinates.height

            # coordinates of detected faces are out of bounds
            if (
                xmin >= 1.0
                or xmin <= 0.0
                or ymin >= 1.0
                or ymin <= 0.0
                or width >= 1.0
                or width <= 0.0
                or height >= 1.0
                or height <= 0.0
            ):
                frame_cropped = frame
                frame_masked = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                coordinate = [0, 0, 0, 0]
            else:
                x = int(xmin * x_size)
                y = int(ymin * y_size)
                w = int(width * x_size)
                h = int(height * y_size)

                frame_cropped = frame[y : (y + h), x : (x + w)]
                frame_masked = frame_cropped
                coordinate = [x, y, w, h]
        # face undetected
        else:
            frame_cropped = frame
            frame_masked = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            coordinate = [0, 0, 0, 0]

        try:
            frame_yuv = cv2.cvtColor(frame_masked, cv2.COLOR_RGB2YUV)

        except:
            print("\n--------- ERROR! -----------\nUsual cv empty error")
            print(f"Shape of img1: {frame.shape}")
            print(f"This is at idx: {idx}")
            exit(666)

        processed_frames.append(frame_yuv)
        stacked_coordinates.append(coordinate)

        if not os.path.exists(f"{video_path[:-10]}/mp_rgb_full/"):
            os.makedirs(f"{video_path[:-10]}/mp_rgb_full/")
        if not os.path.exists(f"{video_path[:-10]}/mp_yuv_full/"):
            os.makedirs(f"{video_path[:-10]}/mp_yuv_full/")

        cv2.imwrite(
            f"{video_path[:-10]}/mp_rgb_full/image_{idx:05d}.png", frame_cropped
        )
        cv2.imwrite(f"{video_path[:-10]}/mp_yuv_full/image_{idx:05d}.png", frame_yuv)

    # At this point we have the processed maps from all the frames in a video and now we do the sliding window part.
    for start_frame_index in tqdm(
        range(0, num_frames, sliding_window_stride), desc="make STMaps", total=num_maps
    ):
        end_frame_index = start_frame_index + clip_size
        if end_frame_index > num_frames:
            break
        spatio_temporal_map = np.zeros((3, clip_size, 25))
        # spatio_temporal_map = np.zeros((3, clip_size, 255))

        for idx, frame in enumerate(
            processed_frames[start_frame_index:end_frame_index]
        ):
            roi_blocks = chunkify(frame)
            # roi_blocks_5 = chunkify(frame, 5, 5)
            # roi_blocks_6 = chunkify(frame, 6, 6)
            # roi_blocks_7 = chunkify(frame, 7, 7)
            # roi_blocks_8 = chunkify(frame, 8, 8)
            # roi_blocks_9 = chunkify(frame, 9, 9)
            # roi_blocks = roi_blocks_5 + roi_blocks_6 + roi_blocks_7 + roi_blocks_8 + roi_blocks_9
            for block_idx, block in enumerate(roi_blocks):
                avg_pixels = cv2.mean(block)
                spatio_temporal_map[0, idx, block_idx] = avg_pixels[0]
                spatio_temporal_map[1, idx, block_idx] = avg_pixels[1]
                spatio_temporal_map[2, idx, block_idx] = avg_pixels[2]

        for block_idx in range(spatio_temporal_map.shape[2]):
            # Not sure about uint8
            fn_scale_0_255 = lambda x: (x * 255.0).astype(np.uint8)
            scaled_channel_0 = scaler.fit_transform(
                spatio_temporal_map[0, :, block_idx].reshape(-1, 1)
            )
            spatio_temporal_map[0, :, block_idx] = fn_scale_0_255(
                scaled_channel_0.flatten()
            )
            scaled_channel_1 = scaler.fit_transform(
                spatio_temporal_map[1, :, block_idx].reshape(-1, 1)
            )
            spatio_temporal_map[1, :, block_idx] = fn_scale_0_255(
                scaled_channel_1.flatten()
            )
            scaled_channel_2 = scaler.fit_transform(
                spatio_temporal_map[2, :, block_idx].reshape(-1, 1)
            )
            spatio_temporal_map[2, :, block_idx] = fn_scale_0_255(
                scaled_channel_2.flatten()
            )

        stacked_maps[map_index, :, :, :] = spatio_temporal_map
        map_index += 1

    # return stacked_maps, clip_size
    return (stacked_maps, clip_size, stacked_coordinates)


if __name__ == "__main__":
    videos = []
    for (path, dir, files) in os.walk("/data/VIPL-HR-V1/data/"):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == ".avi":
                # path = path.replace("\\", "/", 10)
                videos.append(f"{path}/{filename}")
                print("%s/%s" % (path, filename))
    videos = [i for i in videos if "source4" not in i]
    videos = sorted(videos)
    for idx, video in enumerate(videos):
        split = video.split("/")
        print("-" * 200)
        print(f"Order {idx+1}, {split[-4]}_{split[-3]}_{split[-2]} processing start!")
        stmap, clip_size, coordinates = preprocess_video_to_st_maps(video)
        # preprocess_video_to_st_maps(video)
        print(f"Order {idx+1}, {split[-4]}_{split[-3]}_{split[-2]} processing done!")
        coordinates_df = pd.DataFrame(coordinates, columns=["x", "y", "w", "h"])
        np.save(
            f"D:/project_Han/rppg-project/preprocessing/vipl_mp_stmaps/{split[-4]}_{split[-3]}_{split[-2]}_{clip_size}.npy",
            stmap,
        )
        coordinates_df.to_csv(
            f"D:/project_Han/rppg-project/preprocessing/vipl_coordinates/{split[-4]}_{split[-3]}_{split[-2]}_{clip_size}.csv",
            index=False,
        )
        print(
            f"Order {idx+1}, {split[-4]}_{split[-3]}_{split[-2]}_{clip_size}.npy saved!"
        )
        print(
            f"Order {idx+1}, {split[-4]}_{split[-3]}_{split[-2]}_{clip_size}.csv saved!"
        )
        print("-" * 200)
