"""Render raw and undistorted EgoExo4D ego-trajectory projections side by side.

No detector or tracker is used. The left view uses Kannala-Brandt fisheye
projection on the raw Exo frame. The right view undistorts the frame and uses
a pinhole projection with the matching new camera matrix.

Example:
    uv run python demo_ego_projection_comparison.py \
        --take cmu_soccer12_2 --exo-cam cam01 \
        --max-frames 600 --output outputs/ego_projection_compare.mp4
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.embodiedgait.camera import (
    build_undistort_maps_gopro,
    gopro_calib_to_K_D,
    gopro_calib_to_world_camera,
    undistort_frame,
    world_to_pixel_fisheye,
)
from src.embodiedgait.loader import (
    iter_video_frames,
    list_exo_videos,
    load_gopro_calibs,
    load_trajectory,
)


def nearest_index(timestamps_us: np.ndarray, target_us: float) -> int:
    """Find the trajectory sample closest to target_us."""
    right = int(np.searchsorted(timestamps_us, target_us))
    if right <= 0:
        return 0
    if right >= len(timestamps_us):
        return len(timestamps_us) - 1
    left = right - 1
    if target_us - timestamps_us[left] <= timestamps_us[right] - target_us:
        return left
    return right


def pinhole_project(
    points_world: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Project world points into an undistorted pinhole image."""
    points_2d, _ = cv2.projectPoints(
        points_world.reshape(-1, 1, 3).astype(np.float32),
        rvec.astype(np.float32),
        tvec.astype(np.float32),
        K.astype(np.float32),
        distCoeffs=None,
    )
    return points_2d.reshape(-1, 2)


def valid_projection_mask(
    points_world: np.ndarray,
    points_2d: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Keep finite, in-front-of-camera, in-image projected points."""
    rotation, _ = cv2.Rodrigues(rvec)
    camera_points = (rotation @ points_world.T + tvec.reshape(3, 1)).T
    return (
        (camera_points[:, 2] > 0)
        & np.isfinite(points_2d).all(axis=1)
        & (points_2d[:, 0] >= 0)
        & (points_2d[:, 0] < width)
        & (points_2d[:, 1] >= 0)
        & (points_2d[:, 1] < height)
    )


def make_trajectory_overlay(
    height: int,
    width: int,
    points_2d: np.ndarray,
    valid: np.ndarray,
    stride: int,
) -> np.ndarray:
    """Pre-render the complete trajectory once for efficient video output."""
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    indices = np.arange(0, len(points_2d), stride)
    for first, second in zip(indices[:-1], indices[1:]):
        if valid[first] and valid[second]:
            p1 = tuple(np.rint(points_2d[first]).astype(int))
            p2 = tuple(np.rint(points_2d[second]).astype(int))
            cv2.line(overlay, p1, p2, (0, 180, 0), 2, cv2.LINE_AA)
    for index in indices:
        if valid[index]:
            point = tuple(np.rint(points_2d[index]).astype(int))
            cv2.circle(overlay, point, 3, (0, 220, 0), -1, cv2.LINE_AA)
    return overlay


def draw_current_ego(
    image: np.ndarray,
    point_2d: np.ndarray,
    valid: bool,
) -> None:
    if not valid:
        cv2.putText(
            image,
            "current ego outside image",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return
    point = tuple(np.rint(point_2d).astype(int))
    cv2.circle(image, point, 13, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.drawMarker(
        image, point, (0, 0, 255), cv2.MARKER_CROSS, 32, 3, cv2.LINE_AA
    )
    cv2.putText(
        image,
        "current ego",
        (point[0] + 16, max(28, point[1] - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


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
    calibration = calibs[args.exo_cam]

    trajectory = load_trajectory(args.take)
    timestamps_us = trajectory["tracking_timestamp_us"].to_numpy(np.float64)
    positions_world = trajectory[
        ["tx_world_device", "ty_world_device", "tz_world_device"]
    ].to_numpy(np.float64)
    rvec, tvec = gopro_calib_to_world_camera(calibration)

    frames = iter_video_frames(video["path"], max_frames=args.max_frames)
    try:
        first_index, first_frame = next(frames)
    except StopIteration as exc:
        raise RuntimeError("Video contains no frames") from exc

    height, width = first_frame.shape[:2]
    K_raw, D = gopro_calib_to_K_D(calibration, width, height)
    map1, map2, new_K = build_undistort_maps_gopro(
        calibration, width, height, balance=args.balance
    )

    raw_points = world_to_pixel_fisheye(
        positions_world, rvec, tvec, K_raw, D
    )
    undistorted_points = pinhole_project(
        positions_world, rvec, tvec, new_K
    )
    raw_valid = valid_projection_mask(
        positions_world, raw_points, rvec, tvec, width, height
    )
    undistorted_valid = valid_projection_mask(
        positions_world, undistorted_points, rvec, tvec, width, height
    )
    raw_overlay = make_trajectory_overlay(
        height, width, raw_points, raw_valid, args.trajectory_stride
    )
    undistorted_overlay = make_trajectory_overlay(
        height,
        width,
        undistorted_points,
        undistorted_valid,
        args.trajectory_stride,
    )

    output_width = int(round(width * args.display_scale)) * 2
    output_height = int(round(height * args.display_scale))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")

    def process_frame(frame_index: int, raw_frame: np.ndarray) -> None:
        frame_time_us = timestamps_us[0] + frame_index / args.fps * 1_000_000.0
        pose_index = nearest_index(timestamps_us, frame_time_us)

        raw_view = cv2.add(raw_frame, raw_overlay)
        undistorted_view = undistort_frame(raw_frame, map1, map2)
        undistorted_view = cv2.add(undistorted_view, undistorted_overlay)

        draw_current_ego(raw_view, raw_points[pose_index], bool(raw_valid[pose_index]))
        draw_current_ego(
            undistorted_view,
            undistorted_points[pose_index],
            bool(undistorted_valid[pose_index]),
        )
        cv2.putText(
            raw_view,
            "RAW FISHEYE (K_raw + D)",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            undistorted_view,
            "UNDISTORTED (new_K, pinhole)",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        combined = np.concatenate([raw_view, undistorted_view], axis=1)
        if args.display_scale != 1.0:
            combined = cv2.resize(
                combined,
                (output_width, output_height),
                interpolation=cv2.INTER_AREA,
            )
        writer.write(combined)

    processed = 0
    try:
        process_frame(first_index, first_frame)
        processed = 1
        for frame_index, frame in frames:
            process_frame(frame_index, frame)
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed} frames")
    finally:
        writer.release()

    print(f"Saved {processed} frames: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--take", required=True)
    parser.add_argument("--exo-cam", default="cam01")
    parser.add_argument("--output", default="outputs/ego_projection_compare.mp4")
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--balance", type=float, default=0.8)
    parser.add_argument("--trajectory-stride", type=int, default=10)
    parser.add_argument(
        "--display-scale",
        type=float,
        default=0.5,
        help="resize each view before concatenation (default: 0.5)",
    )
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if not 0 <= args.balance <= 1:
        parser.error("--balance must be between 0 and 1")
    if args.trajectory_stride < 1:
        parser.error("--trajectory-stride must be at least 1")
    if args.display_scale <= 0:
        parser.error("--display-scale must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
