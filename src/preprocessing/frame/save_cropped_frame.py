import dotenv

dotenv.load_dotenv(
    override=True,
)

import os
import warnings

warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
from tqdm import tqdm

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../../configs/",
    config_name="physca.yaml",
)
def save_cropped_frame(
    config: DictConfig,
) -> None:
    if config.data_type not in ["vipl", "ubfc", "pure", "mahnob"]:
        raise ValueError(
            f"Invalid data type: {config.data_type}. Choose in ['vipl', 'ubfc', 'pure', 'mahnob']."
        )

    if config.data_type == "vipl":
        dataset_path = os.path.join(
            f"{config.connected_dir}/data",
            "VIPL-HR-V1",
            "data",
        )
    elif config.data_type == "ubfc":
        dataset_path = os.path.join(
            f"{config.connected_dir}/data",
            "UBFC-rPPG",
            "DATASET_2",
        )
    elif config.data_type == "pure":
        dataset_path = os.path.join(
            f"{config.connected_dir}/data",
            "PURE",
        )
    else:
        dataset_path = os.path.join(
            f"{config.connected_dir}/data",
            "MAHNOB-HCI-HR-Estimation-V1",
            "Sessions",
        )

    detector = mp.solutions.face_detection.FaceDetection(
        min_detection_confidence=0.8,
        model_selection=0,
    )

    if config.data_type == "pure":
        for path, _, files in os.walk(dataset_path):
            images = []
            for filename in files:
                if filename.endswith(".png"):
                    image_path = os.path.join(
                        path,
                        filename,
                    )
                    images.append(image_path)

            for index, image_path in enumerate(
                tqdm(sorted(images), desc=f"Processing folder {path}")
            ):
                relative_path = os.path.relpath(
                    os.path.dirname(image_path),
                    dataset_path,
                )
                save_path = os.path.join(
                    f"{config.connected_dir}/data/{config.data_type}_image",
                    relative_path,
                    "mp_rgb_full",
                )
                preprocess_one_image(
                    detector=detector,
                    image_path=image_path,
                    save_path=save_path,
                    index=index,
                )

    else:
        videos = []
        for path, _, files in os.walk(dataset_path):
            for filename in files:
                if config.data_type == "mahnob":
                    if "C1" in filename and filename.endswith(".avi"):
                        video_path = os.path.join(
                            path,
                            filename,
                        )
                        videos.append(video_path)
                else:
                    if filename.endswith(".avi"):
                        video_path = os.path.join(
                            path,
                            filename,
                        )
                        videos.append(video_path)

        for index, video_path in enumerate(sorted(videos)):
            print(f"Processing video {index + 1}/{len(videos)}: {video_path}")
            relative_path = os.path.relpath(
                os.path.dirname(video_path),
                dataset_path,
            )
            save_path = os.path.join(
                f"{config.connected_dir}/data/{config.data_type}_image",
                relative_path,
                "mp_rgb_full",
            )
            preprocess_one_video(
                detector=detector,
                video_path=video_path,
                save_path=save_path,
            )
            print(f"Completed processing {video_path}")


def preprocess_one_image(
    detector: mp.solutions.face_detection.FaceDetection,
    image_path: str,
    save_path: str,
    index: int,
) -> None:
    image = cv2.imread(image_path)
    if image is None:
        return

    os.makedirs(
        save_path,
        exist_ok=True,
    )

    faces = detector.process(
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )
    )

    if isinstance(faces.detections, list):
        coordinates = faces.detections[0].location_data.relative_bounding_box
        xmin = coordinates.xmin
        ymin = coordinates.ymin
        width = coordinates.width
        height = coordinates.height

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
            cropped_image = image
        else:
            x, y, w, h = (
                int(xmin * image.shape[1]),
                int(ymin * image.shape[0]),
                int(width * image.shape[1]),
                int(height * image.shape[0]),
            )
            cropped_image = image[y : y + h, x : x + w]
    else:
        cropped_image = image

    cv2.imwrite(
        f"{save_path}/image_{index:05d}.png",
        cropped_image,
    )


def preprocess_one_video(
    detector: mp.solutions.face_detection.FaceDetection,
    video_path: str,
    save_path: str,
) -> None:
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_dims = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    os.makedirs(
        save_path,
        exist_ok=True,
    )

    with tqdm(total=frame_count, desc="Processing frames") as pbar:
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            faces = detector.process(
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )
            )

            if isinstance(faces.detections, list):
                coordinates = faces.detections[0].location_data.relative_bounding_box
                xmin = coordinates.xmin
                ymin = coordinates.ymin
                width = coordinates.width
                height = coordinates.height

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
                    cropped_frame = frame
                else:
                    x, y, w, h = (
                        int(xmin * frame_dims[0]),
                        int(ymin * frame_dims[1]),
                        int(width * frame_dims[0]),
                        int(height * frame_dims[1]),
                    )
                    cropped_frame = frame[y : y + h, x : x + w]
            else:
                cropped_frame = frame

            cv2.imwrite(
                f"{save_path}/image_{frame_index:05d}.png",
                cropped_frame,
            )
            frame_index += 1
            pbar.update(1)

    cap.release()


if __name__ == "__main__":
    save_cropped_frame()
