"""Pipeline: project ego device trajectory onto exo (GoPro) video frames.

Usage:
    from src.embodiedgait.pipeline import run
    run("cmu_bike01_2", "output.mp4", max_frames=300)
"""

import logging

import numpy as np

from src.embodiedgait.camera import (
    device_axes_endpoints,
    gopro_calib_to_K_D,
    gopro_calib_to_world_camera,
    build_undistort_maps_gopro,
    undistort_frame,
    world_to_pixel,
    world_to_pixel_fisheye,
)
from src.embodiedgait.loader import (
    iter_video_frames,
    list_exo_videos,
    load_gopro_calibs,
    load_trajectory,
)
from src.embodiedgait.visualizer import (
    add_text_overlay,
    create_output_writer,
    draw_axes_at_pose,
    draw_trajectory_on_frame,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def run(
    take_name: str,
    output_path: str = "output.mp4",
    max_frames: int = 300,
    exo_cam: str = "cam01",
    fps: float | None = None,
    trajectory_subsample: int = 50,
    axis_length: float = 0.5,
) -> None:
    """Project the Aria (ego) device trajectory onto an exo (GoPro) video.

    Uses gopro_calibs.csv for correct KANNALABRANDTK3 fisheye calibration
    and cam-in-world extrinsics (inverted to world→camera for projection).

    Args:
        take_name:   e.g. 'cmu_bike01_2'.
        output_path: Output MP4 video.
        max_frames:  Max frames to process.
        exo_cam:     Exo camera ID: 'cam01'-'cam04'.
        fps:         Output FPS (default: keep source FPS).
        trajectory_subsample: Every Nth trajectory point for the trail.
        axis_length: Length of coordinate axes in meters.
    """
    # ── 1. Ego device trajectory ───────────────────────────────────
    log.info("Loading ego trajectory for %s...", take_name)
    traj_df = load_trajectory(take_name)
    log.info("  %d pose entries", len(traj_df))

    traj_sampled = traj_df.iloc[::trajectory_subsample]
    ego_world_positions = traj_sampled[
        ["tx_world_device", "ty_world_device", "tz_world_device"]
    ].to_numpy(dtype=np.float64)
    log.info("  Subsampled: %d waypoints (every %dth)",
             len(ego_world_positions), trajectory_subsample)

    traj_ts = traj_df["tracking_timestamp_us"].to_numpy(dtype=np.float64)
    traj_pos = traj_df[
        ["tx_world_device", "ty_world_device", "tz_world_device"]
    ].to_numpy(dtype=np.float64)

    # ── 2. GoPro calibration (from gopro_calibs.csv) ───────────────
    log.info("Loading gopro_calibs for %s...", take_name)
    calibs = load_gopro_calibs(take_name)
    log.info("  Cameras: %s", list(calibs.keys()))

    if exo_cam not in calibs:
        raise KeyError(f"Exo camera '{exo_cam}' not in gopro_calibs. "
                       f"Available: {list(calibs.keys())}")

    calib_row = calibs[exo_cam]

    # ── 3. Exo video ───────────────────────────────────────────────
    videos = list_exo_videos(take_name)
    log.info("  Exo videos: %s", [v["cam_id"] for v in videos])
    video = next((v for v in videos if v["cam_id"] == exo_cam), None)
    if video is None:
        raise FileNotFoundError(
            f"Exo camera '{exo_cam}' not found for {take_name}."
        )
    video_path = video["path"]
    log.info("  Video: %s  (%.1f MB)", exo_cam, video["size_bytes"] / 1e6)

    # ── 4. Per-frame processing ────────────────────────────────────
    log.info("Processing up to %d frames...", max_frames)

    writer = None
    # Deferred: built on first frame after we know actual dimensions
    maps = None     # (map1, map2)
    K_proj = None   # projection intrinsics (for undistorted space)
    rvec = None     # world→camera rotation (Rodrigues)
    tvec = None     # world→camera translation
    D = None        # KB distortion coeffs

    for frame_idx, frame in iter_video_frames(video_path, max_frames=max_frames):
        h, w = frame.shape[:2]

        # ── Build calibration + undistortion maps on first frame ──
        if maps is None:
            K_raw, D = gopro_calib_to_K_D(calib_row, w, h)
            rvec, tvec = gopro_calib_to_world_camera(calib_row)
            # Build undistortion maps (fisheye → pinhole)
            map1, map2, new_K = build_undistort_maps_gopro(calib_row, w, h, balance=0.8)
            maps = (map1, map2)
            K_proj = new_K
            log.info("  gopro_calibs: K_raw fx=%.1f fy=%.1f  →  "
                     "new_K fx=%.1f fy=%.1f  k1=%.4f k2=%.4f",
                     K_raw[0,0], K_raw[1,1], new_K[0,0], new_K[1,1],
                     D[0], D[1])
            log.info("  cam-in-world pos: (%.2f, %.2f, %.2f)",
                     float(calib_row["tx_world_cam"]),
                     float(calib_row["ty_world_cam"]),
                     float(calib_row["tz_world_cam"]))

        # ── Current ego device pose ──────────────────────────────
        frame_time_us = traj_ts[0] + frame_idx / 30.0 * 1_000_000
        closest = int(np.searchsorted(traj_ts, frame_time_us))
        closest = max(0, min(closest, len(traj_ts) - 1))
        row = traj_df.iloc[closest]

        # ── Project + draw on RAW (distorted) frame ─────────────
        # We draw on the distorted frame so that fisheye-projected
        # pixels match the raw video geometry.  The undistort remap
        # warps everything (frame + drawings) together.
        result_raw = frame.copy()

        # Trajectory trail
        pts_2d_traj = world_to_pixel_fisheye(ego_world_positions, rvec, tvec, K_raw, D)
        result_raw = draw_trajectory_on_frame(
            result_raw, pts_2d_traj, trail_length=len(pts_2d_traj))

        # Coordinate axes
        origin_w, x_w, y_w, z_w = device_axes_endpoints(
            float(row["tx_world_device"]),
            float(row["ty_world_device"]),
            float(row["tz_world_device"]),
            float(row["qx_world_device"]),
            float(row["qy_world_device"]),
            float(row["qz_world_device"]),
            float(row["qw_world_device"]),
            axis_length=axis_length,
        )
        fwd_w = origin_w - (z_w - origin_w)  # -Z = forward
        pts_world = np.stack([origin_w, x_w, y_w, fwd_w])
        pts_2d_axes = world_to_pixel_fisheye(pts_world, rvec, tvec, K_raw, D)

        result_raw = draw_axes_at_pose(
            result_raw,
            tuple(pts_2d_axes[0].tolist()),
            pts_2d_axes[1] - pts_2d_axes[0],
            pts_2d_axes[2] - pts_2d_axes[0],
            pts_2d_axes[3] - pts_2d_axes[0],
            x_label="X(R)", y_label="Y(Up)", z_label="Fwd",
        )

        # ── Undistort the composed frame ────────────────────────
        map1, map2 = maps
        result = undistort_frame(result_raw, map1, map2)

        if writer is None:
            h_u, w_u = result.shape[:2]
            fps_val = fps if fps is not None else 30.0
            writer = create_output_writer(output_path, fps_val, w_u, h_u)
            log.info("  Output: %dx%d @ %.1f fps", w_u, h_u, fps_val)

        # ── Text overlay ─────────────────────────────────────────
        info = [
            f"Take: {take_name}  |  Exo: {exo_cam}  |  Frame: {frame_idx}",
            f"Ego pos: ({row['tx_world_device']:.2f}, {row['ty_world_device']:.2f}, {row['tz_world_device']:.2f})",
        ]
        for i, text in enumerate(info):
            add_text_overlay(result, text, position=(10, 30 + i * 28))

        writer.write(result)

        if frame_idx % 50 == 0:
            log.info("  Frame %d / %d", frame_idx, max_frames)

    if writer is not None:
        writer.release()

    log.info("Done!  %d frames → %s", frame_idx + 1, output_path)
