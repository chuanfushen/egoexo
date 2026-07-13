"""Minimal YOLO + ego projection demo in the raw fisheye image space.

This demo deliberately does NOT undistort video frames. YOLO boxes and the
Kannala-Brandt ego projection therefore both live in raw-video pixel space.

Example:
    uv run python demo_raw_yolo_ego_projection.py \
        --take cmu_soccer12_2 --exo-cam cam01 \
        --model yolo26n.pt --batch-size 64 --max-frames 300 \
        --output outputs/raw_ego_projection.json \
        --video-output outputs/raw_ego_projection.mp4
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.embodiedgait.camera import (
    gopro_calib_to_K_D,
    gopro_calib_to_world_camera,
    world_to_pixel_fisheye,
)
from src.embodiedgait.loader import (
    iter_video_frames,
    list_exo_videos,
    load_gopro_calibs,
    load_trajectory,
)


def nearest_pose_index(timestamps_us: np.ndarray, target_us: float) -> int:
    """Return the trajectory sample nearest to a video-frame timestamp."""
    right = int(np.searchsorted(timestamps_us, target_us))
    if right <= 0:
        return 0
    if right >= len(timestamps_us):
        return len(timestamps_us) - 1
    left = right - 1
    if target_us - timestamps_us[left] <= timestamps_us[right] - target_us:
        return left
    return right


def point_is_in_front(
    point_world: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> bool:
    """Check positive camera-space depth before accepting a projection."""
    rotation, _ = cv2.Rodrigues(rvec)
    point_camera = rotation @ point_world.reshape(3, 1) + tvec.reshape(3, 1)
    return bool(point_camera[2, 0] > 0)


def select_person(boxes, projection: np.ndarray | None) -> int | None:
    """Pick the highest confidence*area person box containing projection."""
    if projection is None or boxes is None or len(boxes) == 0:
        return None

    px, py = projection.tolist()
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidence = boxes.conf.detach().cpu().numpy()
    best_index = None
    best_score = -1.0
    for index, (bbox, conf) in enumerate(zip(xyxy, confidence)):
        x1, y1, x2, y2 = bbox.tolist()
        if x1 <= px <= x2 and y1 <= py <= y2:
            score = float(conf) * max(0.0, (x2 - x1) * (y2 - y1))
            if score > best_score:
                best_index = index
                best_score = score
    return best_index


def run(args: argparse.Namespace) -> None:
    videos = list_exo_videos(args.take)
    video = next((item for item in videos if item["cam_id"] == args.exo_cam), None)
    if video is None:
        raise FileNotFoundError(
            f"No video for {args.exo_cam}; available: "
            f"{[item['cam_id'] for item in videos]}"
        )

    calibs = load_gopro_calibs(args.take)
    if args.exo_cam not in calibs:
        raise KeyError(
            f"No calibration for {args.exo_cam}; available: {list(calibs)}"
        )

    trajectory = load_trajectory(args.take)
    trajectory_timestamps = trajectory["tracking_timestamp_us"].to_numpy(
        dtype=np.float64
    )
    rvec, tvec = gopro_calib_to_world_camera(calibs[args.exo_cam])
    model = YOLO(args.model)

    frame_iterator = iter_video_frames(video["path"], max_frames=args.max_frames)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_output_path = Path(args.video_output)
    video_output_path.parent.mkdir(parents=True, exist_ok=True)

    K_raw = D = None
    video_writer = None
    processed = selected = 0
    first_record = True

    try:
        with output_path.open("w", encoding="utf-8") as output_file:
            output_file.write(
                json.dumps(
                    {
                        "take_name": args.take,
                        "exo_cam": args.exo_cam,
                        "coordinate_space": "raw_distorted",
                        "fps": args.fps,
                    }
                )[:-1]
            )
            output_file.write(', "frames": [\n')

            while True:
                batch = []
                for _ in range(args.batch_size):
                    try:
                        batch.append(next(frame_iterator))
                    except StopIteration:
                        break
                if not batch:
                    break

                if K_raw is None:
                    height, width = batch[0][1].shape[:2]
                    K_raw, D = gopro_calib_to_K_D(
                        calibs[args.exo_cam], width, height
                    )
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(video_output_path), fourcc, args.fps, (width, height)
                    )
                    if not video_writer.isOpened():
                        raise RuntimeError(
                            f"Cannot create output video: {video_output_path}"
                        )

                results = model.predict(
                    source=[frame for _, frame in batch],
                    classes=[0],
                    conf=args.conf,
                    imgsz=args.imgsz,
                    device=args.device,
                    verbose=False,
                )

                for (frame_index, frame), result in zip(batch, results):
                    frame_time_us = (
                        trajectory_timestamps[0]
                        + frame_index / args.fps * 1_000_000.0
                    )
                    pose_index = nearest_pose_index(
                        trajectory_timestamps, frame_time_us
                    )
                    pose = trajectory.iloc[pose_index]
                    point_world = np.array(
                        [
                            float(pose["tx_world_device"]),
                            float(pose["ty_world_device"]),
                            float(pose["tz_world_device"]),
                        ],
                        dtype=np.float64,
                    )

                    projection = None
                    if point_is_in_front(point_world, rvec, tvec):
                        candidate = world_to_pixel_fisheye(
                            point_world[None, :], rvec, tvec, K_raw, D
                        )[0]
                        height, width = frame.shape[:2]
                        if (
                            np.all(np.isfinite(candidate))
                            and 0 <= candidate[0] < width
                            and 0 <= candidate[1] < height
                        ):
                            projection = candidate

                    boxes = result.boxes
                    best_index = select_person(boxes, projection)
                    detections = []
                    annotated = frame.copy()
                    if boxes is not None and len(boxes):
                        xyxy = boxes.xyxy.detach().cpu().tolist()
                        confidences = boxes.conf.detach().cpu().tolist()
                        for index, (bbox, confidence) in enumerate(
                            zip(xyxy, confidences)
                        ):
                            detections.append(
                                {
                                    "bbox_xyxy": [round(value, 2) for value in bbox],
                                    "confidence": round(confidence, 6),
                                    "contains_ego_projection": index == best_index,
                                }
                            )
                            x1, y1, x2, y2 = (int(value) for value in bbox)
                            is_selected = index == best_index
                            color = (0, 255, 0) if is_selected else (255, 160, 0)
                            thickness = 4 if is_selected else 2
                            cv2.rectangle(
                                annotated, (x1, y1), (x2, y2), color, thickness
                            )
                            label = (
                                f"EGO {confidence:.2f}"
                                if is_selected
                                else f"person {confidence:.2f}"
                            )
                            cv2.putText(
                                annotated,
                                label,
                                (x1, max(25, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                color,
                                2,
                                cv2.LINE_AA,
                            )

                    if projection is not None:
                        px, py = (int(round(value)) for value in projection)
                        cv2.circle(annotated, (px, py), 12, (0, 0, 255), 3)
                        cv2.drawMarker(
                            annotated,
                            (px, py),
                            (0, 0, 255),
                            cv2.MARKER_CROSS,
                            30,
                            3,
                        )
                        cv2.putText(
                            annotated,
                            "ego projection",
                            (px + 15, max(25, py - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    else:
                        cv2.putText(
                            annotated,
                            "ego projection outside image",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    video_writer.write(annotated)

                    record = {
                        "frame_index": frame_index,
                        "trajectory_timestamp_us": int(
                            trajectory_timestamps[pose_index]
                        ),
                        "ego_world_xyz": point_world.tolist(),
                        "ego_projection_xy": (
                            [round(float(value), 2) for value in projection]
                            if projection is not None
                            else None
                        ),
                        "selected_detection_index": best_index,
                        "detections": detections,
                    }
                    if not first_record:
                        output_file.write(",\n")
                    json.dump(record, output_file, ensure_ascii=False)
                    first_record = False
                    processed += 1
                    selected += best_index is not None

                print(f"Processed {processed} frames; selected {selected}")

            output_file.write("\n]}\n")
    finally:
        if video_writer is not None:
            video_writer.release()

    print(f"Saved JSON:  {output_path}")
    print(f"Saved video: {video_output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--take", required=True)
    parser.add_argument("--exo-cam", default="cam01")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--output", default="outputs/raw_ego_projection.json")
    parser.add_argument(
        "--video-output", default="outputs/raw_ego_projection.mp4"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=6400)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
