"""Single-video pipeline: undistort → detect/track → pick person → crop JPG.

Supports two modes:
  --yolo: YOLO + ByteTrack tracking (fast, bbox only)
  default: SAM3 text-prompted detection (slow, bbox + mask)

Usage:
    uv run python -m src.embodiedgait.tracking_pipeline \
        --take cmu_bike01_2 --exo-cam cam01 --max-frames 100

    uv run python -m src.embodiedgait.tracking_pipeline \
        --take cmu_bike01_2 --exo-cam cam01 --max-frames 100 --yolo
"""

import argparse
import base64
import json
import logging
import os
from collections import defaultdict
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from src.embodiedgait.camera import (
    build_undistort_maps_gopro,
    gopro_calib_to_K_D,
    gopro_calib_to_world_camera,
    undistort_frame,
)
from src.embodiedgait.detection import Detection, PersonDetector
from src.embodiedgait.loader import (
    iter_video_frames,
    list_exo_videos,
    load_gopro_calibs,
    load_trajectory,
)
from src.embodiedgait.yolo_tracker import TrackedPerson, YOLOTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


def project_trajectory_point(
    tx: float,
    ty: float,
    tz: float,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Project a single 3D world point to 2D using pinhole camera model.

    Args:
        tx, ty, tz: World coordinates of the point.
        rvec: World→camera Rodrigues rotation vector.
        tvec: World→camera translation vector.
        K: Camera intrinsics (3×3 pinhole, e.g. new_K from undistort).

    Returns:
        (2,) array of pixel coordinates (u, v).
    """
    pts3d = np.array([[tx, ty, tz]], dtype=np.float32)
    pts2d, _ = cv2.projectPoints(
        pts3d.reshape(-1, 1, 3),
        rvec.astype(np.float32),
        tvec.astype(np.float32),
        K.astype(np.float32),
        distCoeffs=None,
    )
    return pts2d.reshape(2)


def _point_in_bbox(px: float, py: float, bbox: tuple[float, float, float, float]) -> bool:
    """Check if a point (px, py) falls inside a bbox (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


def pick_best_detection(
    detections: list[Detection],
    proj_2d: np.ndarray,
) -> tuple[Detection | None, int, float]:
    """Pick the best detection: must contain the projection point, then
    prefer highest confidence × largest area.

    Returns (best_detection, index, distance) — best_detection is None if
    no detection contains the projection point.
    """
    if not detections:
        return None, -1, float("inf")

    pu, pv = float(proj_2d[0]), float(proj_2d[1])
    best_idx = -1
    best_score = -1.0

    for i, d in enumerate(detections):
        if not _point_in_bbox(pu, pv, d.bbox):
            continue
        # Combined score: confidence × area
        score = d.score * d.area
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx < 0:
        return None, -1, float("inf")

    best_det = detections[best_idx]
    center = np.array(best_det.center, dtype=np.float64)
    best_dist = float(np.linalg.norm(center - proj_2d))
    return best_det, best_idx, best_dist


def process_video(
    take_name: str,
    exo_cam: str = "cam01",
    model_path: str = "/LargeModelDev/users/chuanfu.shen/ckpts/sam3/sam3.pt",
    output_dir: str = "outputs",
    max_frames: int | None = None,
    device: str | None = None,
    crop_tos_base: str | None = None,
    crop_padding: int = 20,
    prompt: str = "person",
    detector: PersonDetector | None = None,
) -> dict[str, Any]:
    """Run SAM3 detection + closest-bbox cropping on a single video.

    Pipeline:
        1. Load GoPro calibration → build undistortion maps
        2. Load ego trajectory → align timestamps
        3. For each frame:
            a. Undistort fisheye → pinhole
            b. SAM3 text-prompted detection ("person")
            c. Project ego pose → 2D pixel
            d. Pick detection closest to projection point
            e. Crop best bbox → save JPG (local or TOS)
        4. Save results as JSON

    Args:
        take_name: e.g. 'cmu_bike01_2'.
        exo_cam: Exo camera ID: 'cam01'-'cam04'.
        model_path: Path to SAM3 checkpoint.
        output_dir: Directory for output JSON and cropped JPGs.
        max_frames: Maximum frames to process (None = all).
        device: 'cuda' or 'cpu' (None = auto).
        crop_tos_base: TOS base path for cropped JPGs (None = save locally).
        crop_padding: Extra pixels around bbox when cropping.
        prompt: SAM3 text prompt (default: "person").

    Returns:
        Dict with keys: take_name, exo_cam, frames, stats.
    """
    log.info("=" * 60)
    log.info("Processing: %s / %s", take_name, exo_cam)

    # ── 1. Load calibration & build undistortion maps ──────────────
    log.info("Loading GoPro calibration...")
    calibs = load_gopro_calibs(take_name)
    if exo_cam not in calibs:
        raise KeyError(f"Camera '{exo_cam}' not found. Available: {list(calibs.keys())}")
    calib_row = calibs[exo_cam]

    videos = list_exo_videos(take_name)
    video = next((v for v in videos if v["cam_id"] == exo_cam), None)
    if video is None:
        raise FileNotFoundError(f"No video for {exo_cam} in {take_name}")
    video_path = video["path"]

    # Read first frame for dimensions
    first_frame = None
    for _, frame in iter_video_frames(video_path, max_frames=1):
        first_frame = frame
        break
    if first_frame is None:
        raise RuntimeError("Could not read first frame")
    h, w = first_frame.shape[:2]
    log.info("Video resolution: %dx%d", w, h)

    K_raw, D = gopro_calib_to_K_D(calib_row, w, h)
    rvec, tvec = gopro_calib_to_world_camera(calib_row)
    map1, map2, new_K = build_undistort_maps_gopro(calib_row, w, h, balance=0.8)
    log.info("Undistort: K_raw fx=%.1f → new_K fx=%.1f", K_raw[0, 0], new_K[0, 0])
    log.info(
        "Cam-in-world position: (%.2f, %.2f, %.2f)",
        float(calib_row["tx_world_cam"]),
        float(calib_row["ty_world_cam"]),
        float(calib_row["tz_world_cam"]),
    )

    # ── 2. Load trajectory ─────────────────────────────────────────
    log.info("Loading ego trajectory...")
    traj_df = load_trajectory(take_name)
    traj_ts = traj_df["tracking_timestamp_us"].to_numpy(dtype=np.float64)
    log.info("  %d pose entries, duration=%.1fs", len(traj_df),
             (traj_ts[-1] - traj_ts[0]) / 1e6)

    # ── 3. Initialize SAM3 detector (or reuse existing) ────────────
    if detector is None:
        log.info("Initializing SAM3 detector: %s", model_path)
        kwargs = {"ckpt_path": model_path}
        if device:
            kwargs["device"] = device
        detector = PersonDetector(**kwargs)

    # ── 4. Output directories ──────────────────────────────────────
    seq_dir = os.path.join(output_dir, "sequences", take_name, exo_cam)
    os.makedirs(seq_dir, exist_ok=True)
    if crop_tos_base:
        crop_dir = f"{crop_tos_base}/{take_name}/{exo_cam}"
    else:
        crop_dir = seq_dir
    log.info("Output: %s", seq_dir)

    # ── 5. Per-frame processing ────────────────────────────────────
    log.info("Processing frames (max=%s)...", max_frames or "all")

    frame_records: list[dict] = []
    n_cropped = 0
    h_u = w_u = 0

    for frame_idx, raw_frame in iter_video_frames(video_path, max_frames=max_frames):
        # a. Undistort
        undistorted = undistort_frame(raw_frame, map1, map2)
        if frame_idx == 0:
            h_u, w_u = undistorted.shape[:2]

        # b. SAM3 detect
        detections = detector.detect(undistorted, prompt=prompt)

        # c. Project current ego pose → 2D
        frame_time_us = traj_ts[0] + frame_idx / 30.0 * 1_000_000
        closest = int(np.searchsorted(traj_ts, frame_time_us))
        closest = max(0, min(closest, len(traj_ts) - 1))
        row = traj_df.iloc[closest]

        proj_2d = project_trajectory_point(
            float(row["tx_world_device"]),
            float(row["ty_world_device"]),
            float(row["tz_world_device"]),
            rvec, tvec, new_K,
        )

        # d. Pick detection closest to projection point
        best_det, best_idx, best_dist = pick_best_detection(detections, proj_2d)

        # e. Crop best bbox → save JPG
        crop_path = None
        if best_det is not None:
            x1 = max(0, int(best_det.bbox[0]) - crop_padding)
            y1 = max(0, int(best_det.bbox[1]) - crop_padding)
            x2 = min(w_u, int(best_det.bbox[2]) + crop_padding)
            y2 = min(h_u, int(best_det.bbox[3]) + crop_padding)

            if x2 > x1 and y2 > y1:
                crop = undistorted[y1:y2, x1:x2]
                jpg_name = f"{frame_idx:06d}.jpg"
                jpg_path = os.path.join(seq_dir, jpg_name)
                cv2.imwrite(jpg_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 90])

                if crop_tos_base:
                    _upload_jpg_to_tos(jpg_path, crop_tos_base, take_name, exo_cam, frame_idx)
                    os.remove(jpg_path)  # cleanup local after upload

                crop_path = jpg_name
                n_cropped += 1

        # f. Record per-frame data
        frame_records.append({
            "frame_idx": frame_idx,
            "proj_2d": proj_2d.tolist(),
            "traj_pos_world": [
                float(row["tx_world_device"]),
                float(row["ty_world_device"]),
                float(row["tz_world_device"]),
            ],
            "detections": [
                {
                    "bbox": list(d.bbox),
                    "center": list(d.center),
                    "score": d.score,
                    "dist_to_proj": float(np.linalg.norm(
                        np.array(d.center) - np.array(proj_2d)
                    )),
                }
                for d in detections
            ],
            "best_detection": (
                _serialize_detection(best_det, best_dist)
                if best_det else None
            ),
            "crop_path": crop_path,
        })

        if frame_idx % 50 == 0:
            log.info("  Frame %d: %d detections, best_dist=%.0fpx",
                     frame_idx, len(detections),
                     best_dist if best_det else -1)

    # ── 6. Build output ───────────────────────────────────────────
    n_with_det = sum(1 for fr in frame_records if fr["best_detection"] is not None)

    output = {
        "take_name": take_name,
        "exo_cam": exo_cam,
        "config": {
            "model_path": model_path,
            "prompt": prompt,
            "crop_padding": crop_padding,
        },
        "frames": frame_records,
        "stats": {
            "n_frames_processed": len(frame_records),
            "n_frames_with_detections": n_with_det,
            "n_detections_total": sum(len(fr["detections"]) for fr in frame_records),
            "n_cropped": n_cropped,
        },
    }

    # ── 7. Save JSONs locally ───────────────────────────────────────
    output_path = os.path.join(seq_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved to: %s", output_path)

    coco = _build_coco_json(frame_records, take_name, exo_cam, h_u, w_u)
    coco_path = os.path.join(seq_dir, "coco.json")
    with open(coco_path, "w") as f:
        json.dump(coco, f, indent=2)
    log.info("COCO masks saved to: %s", coco_path)

    # ── 8. Upload to TOS if requested ───────────────────────────────
    if crop_tos_base:
        tos_seq = f"{crop_tos_base}/sequences/{take_name}/{exo_cam}"
        _upload_json_to_tos(output_path, tos_seq, "results.json")
        _upload_json_to_tos(coco_path, tos_seq, "coco.json")
        log.info("Uploaded to TOS: %s", tos_seq)

    log.info("Stats: %d frames, %d with detections, %d cropped",
             len(frame_records), n_with_det, n_cropped)
    log.info("=" * 60)

    return output


# ── YOLO Tracking Pipeline ────────────────────────────────────────────


def _expand_bbox(bbox: tuple, scale: float = 1.5) -> tuple:
    """Expand a bbox (x1,y1,x2,y2) by the given scale factor around its center."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def _score_tracks(
    tracks: dict[int, list[dict]],
    proj_by_frame: dict[int, tuple[float, float]],
) -> list[tuple[int, float, float, int]]:
    """Score each track by overlap with ego projection.

    Args:
        tracks: {track_id: [{frame_idx, bbox}, ...]}
        proj_by_frame: {frame_idx: (px, py)}

    Returns:
        List of (track_id, overlap_ratio, avg_dist, n_frames) sorted by score desc.
    """
    scored = []
    for track_id, entries in tracks.items():
        n_total = len(entries)
        n_inside = 0
        total_dist = 0.0
        n_with_proj = 0

        for e in entries:
            fi = e["frame_idx"]
            if fi in proj_by_frame:
                px, py = proj_by_frame[fi]
                bbox = e["bbox"]
                if _point_in_bbox(px, py, bbox):
                    n_inside += 1
                # Distance to bbox center (use original bbox for distance)
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                total_dist += np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                n_with_proj += 1

        overlap_ratio = n_inside / max(n_total, 1)
        avg_dist = total_dist / max(n_with_proj, 1) if n_with_proj > 0 else float("inf")
        scored.append((track_id, overlap_ratio, avg_dist, n_total))

    # Sort by overlap_ratio desc, then by n_total desc (prefer longer tracks)
    scored.sort(key=lambda x: (x[1], x[3]), reverse=True)
    return scored


def process_video_yolo(
    take_name: str,
    exo_cam: str = "cam01",
    yolo_model_path: str = "yolo26n.pt",
    output_dir: str = "outputs",
    max_frames: int | None = None,
    device: str | None = None,
    crop_tos_base: str | None = None,
    crop_padding: int = 20,
    yolo_conf: float = 0.25,
    track_overlap_threshold: float = 0.2,
) -> dict[str, Any]:
    """YOLO + ByteTrack tracking pipeline: track all persons, select ego-person track.

    Single-pass pipeline:
        1. Load GoPro calibration → build undistortion maps
        2. Load ego trajectory → align timestamps
        3. Single pass: YOLO tracking + ego projection + save crops (if proj in bbox)
        4. Track scoring: overlap ratio with ego projection
        5. Select ego-person track, delete non-ego crops
        6. Save results as JSON

    Returns:
        Dict with keys: take_name, exo_cam, frames, stats.
    """
    log.info("=" * 60)
    log.info("[YOLO] Processing: %s / %s", take_name, exo_cam)

    # ── 1. Load calibration & build undistortion maps ──────────────────
    log.info("Loading GoPro calibration...")
    calibs = load_gopro_calibs(take_name)
    if exo_cam not in calibs:
        raise KeyError(f"Camera '{exo_cam}' not found. Available: {list(calibs.keys())}")
    calib_row = calibs[exo_cam]

    videos = list_exo_videos(take_name)
    video = next((v for v in videos if v["cam_id"] == exo_cam), None)
    if video is None:
        raise FileNotFoundError(f"No video for {exo_cam} in {take_name}")
    video_path = video["path"]

    # Read first frame for dimensions
    first_frame = None
    for _, frame in iter_video_frames(video_path, max_frames=1):
        first_frame = frame
        break
    if first_frame is None:
        raise RuntimeError("Could not read first frame")
    h, w = first_frame.shape[:2]
    log.info("Video resolution: %dx%d", w, h)

    K_raw = gopro_calib_to_K_D(calib_row, w, h)[0]
    rvec, tvec = gopro_calib_to_world_camera(calib_row)
    map1, map2, new_K = build_undistort_maps_gopro(calib_row, w, h, balance=0.8)
    log.info("Undistort: K_raw fx=%.1f → new_K fx=%.1f", K_raw[0, 0], new_K[0, 0])

    # ── 2. Load trajectory ─────────────────────────────────────────────
    log.info("Loading ego trajectory...")
    traj_df = load_trajectory(take_name)
    traj_ts = traj_df["tracking_timestamp_us"].to_numpy(dtype=np.float64)
    log.info("  %d pose entries, duration=%.1fs", len(traj_df),
             (traj_ts[-1] - traj_ts[0]) / 1e6)

    # ── 3. Initialize YOLO tracker ─────────────────────────────────────
    log.info("Initializing YOLO tracker: %s", yolo_model_path)
    tracker = YOLOTracker(model_path=yolo_model_path, device=device or "cuda", conf=yolo_conf)

    # ── 4. Single pass: YOLO tracking + ego projection + crop ──────────
    log.info("Processing frames (max_frames=%s)...", max_frames or "all")

    # Output directory
    seq_dir = os.path.join(output_dir, "sequences", take_name, exo_cam)
    os.makedirs(seq_dir, exist_ok=True)

    # Per-frame records
    frame_records: list[dict] = []
    # Track accumulation: {track_id: [(frame_idx, bbox), ...]}
    tracks: dict[int, list[dict]] = defaultdict(list)
    # Ego projection by frame
    proj_by_frame: dict[int, tuple[float, float]] = {}
    # Which track_id was used to crop each frame (for cleanup after scoring)
    crop_track: dict[int, int] = {}
    ego_track_ids: set[int] = set()
    h_u = w_u = 0
    n_cropped = 0

    for frame_idx, raw_frame in iter_video_frames(video_path, max_frames=max_frames):
        # a. Undistort
        undistorted = undistort_frame(raw_frame, map1, map2)
        if frame_idx == 0:
            h_u, w_u = undistorted.shape[:2]

        # b. YOLO track
        persons = tracker.track_frame(undistorted)

        # c. Project ego pose → 2D
        frame_time_us = traj_ts[0] + frame_idx / 30.0 * 1_000_000
        closest = int(np.searchsorted(traj_ts, frame_time_us))
        closest = max(0, min(closest, len(traj_ts) - 1))
        row = traj_df.iloc[closest]

        proj_2d = project_trajectory_point(
            float(row["tx_world_device"]),
            float(row["ty_world_device"]),
            float(row["tz_world_device"]),
            rvec, tvec, new_K,
        )
        pu, pv = float(proj_2d[0]), float(proj_2d[1])
        proj_by_frame[frame_idx] = (pu, pv)

        # d. Record per-frame data + accumulate tracks
        detections_list = []
        for p in persons:
            detections_list.append({
                "track_id": p.track_id,
                "bbox": list(p.bbox),
                "center": list(p.center),
                "score": p.conf,
                "dist_to_proj": float(np.sqrt(
                    (p.center[0] - pu) ** 2 + (p.center[1] - pv) ** 2
                )),
            })
            tracks[p.track_id].append({
                "frame_idx": frame_idx,
                "bbox": p.bbox,
                "conf": p.conf,
            })

        # e. Save crop if ego proj is inside any tracked person's bbox
        crop_path = None
        for p in persons:
            if _point_in_bbox(pu, pv, p.bbox):
                x1_c = max(0, int(p.bbox[0]) - crop_padding)
                y1_c = max(0, int(p.bbox[1]) - crop_padding)
                x2_c = min(w_u, int(p.bbox[2]) + crop_padding)
                y2_c = min(h_u, int(p.bbox[3]) + crop_padding)
                if x2_c > x1_c and y2_c > y1_c:
                    crop = undistorted[y1_c:y2_c, x1_c:x2_c]
                    jpg_name = f"{frame_idx:06d}.jpg"
                    jpg_path = os.path.join(seq_dir, jpg_name)
                    cv2.imwrite(jpg_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if crop_tos_base:
                        _upload_jpg_to_tos(jpg_path, crop_tos_base, take_name, exo_cam, frame_idx)
                        os.remove(jpg_path)
                    crop_path = jpg_name
                    crop_track[frame_idx] = p.track_id
                    n_cropped += 1
                break  # one crop per frame is enough

        frame_records.append({
            "frame_idx": frame_idx,
            "proj_2d": [pu, pv],
            "traj_pos_world": [
                float(row["tx_world_device"]),
                float(row["ty_world_device"]),
                float(row["tz_world_device"]),
            ],
            "detections": detections_list,
            "best_detection": None,   # filled after track scoring
            "crop_path": crop_path,
        })

        if frame_idx % 200 == 0:
            log.info("  Frame %d: %d persons, %d tracks, %d crops so far",
                     frame_idx, len(persons), len(tracks), n_cropped)

    # ── 5. Track scoring ───────────────────────────────────────────────
    log.info("Scoring %d tracks against ego projection...", len(tracks))
    scored = _score_tracks(tracks, proj_by_frame)

    if scored:
        qualifying = [(tid, ratio, dist, n) for tid, ratio, dist, n in scored
                      if ratio >= track_overlap_threshold and n >= 5]

        if qualifying:
            ego_track_ids = {tid for tid, _, _, _ in qualifying}
            log.info("  Ego-person tracks (overlap >= %.2f): %d",
                     track_overlap_threshold, len(ego_track_ids))
            for tid, ratio, dist, n in qualifying[:5]:
                log.info("    track %d: overlap=%.3f avg_dist=%.0fpx frames=%d",
                         tid, ratio, dist, n)
        else:
            best_tid, best_ratio, best_dist, best_n = scored[0]
            ego_track_ids = {best_tid}
            log.info("  No track above threshold. Fallback: track %d "
                     "(overlap=%.3f, dist=%.0fpx, frames=%d)",
                     best_tid, best_ratio, best_dist, best_n)
    else:
        log.warning("  No tracked persons found at all!")

    # ── 6. Update frame_records + cleanup non-ego crops ───────────────
    best_by_frame: dict[int, dict] = {}
    for tid in ego_track_ids:
        for entry in tracks[tid]:
            fi = entry["frame_idx"]
            px, py = proj_by_frame.get(fi, (0, 0))
            dist = float(np.sqrt(
                ((entry["bbox"][0] + entry["bbox"][2]) / 2.0 - px) ** 2 +
                ((entry["bbox"][1] + entry["bbox"][3]) / 2.0 - py) ** 2
            ))
            if fi not in best_by_frame or entry["conf"] > best_by_frame[fi].get("score", 0):
                best_by_frame[fi] = {
                    "track_id": tid,
                    "bbox": list(entry["bbox"]),
                    "center": [
                        (entry["bbox"][0] + entry["bbox"][2]) / 2.0,
                        (entry["bbox"][1] + entry["bbox"][3]) / 2.0,
                    ],
                    "score": entry["conf"],
                    "dist_to_proj": dist,
                }

    n_deleted = 0
    for fr in frame_records:
        fi = fr["frame_idx"]
        if fi in best_by_frame:
            fr["best_detection"] = best_by_frame[fi]
        # Remove crops from non-ego tracks
        if fr["crop_path"] and crop_track.get(fi) not in ego_track_ids:
            jpg_path = os.path.join(seq_dir, fr["crop_path"])
            if os.path.exists(jpg_path):
                os.remove(jpg_path)
            fr["crop_path"] = None
            n_deleted += 1

    n_cropped -= n_deleted
    log.info("  Frames with best_detection: %d / %d (deleted %d non-ego crops)",
             sum(1 for fr in frame_records if fr["best_detection"]),
             len(frame_records), n_deleted)

    # ── 7. Build output ────────────────────────────────────────────────
    n_with_det = sum(1 for fr in frame_records if fr["best_detection"] is not None)

    output = {
        "take_name": take_name,
        "exo_cam": exo_cam,
        "pipeline": "yolo_tracking",
        "config": {
            "yolo_model_path": yolo_model_path,
            "yolo_conf": yolo_conf,
            "crop_padding": crop_padding,
            "track_overlap_threshold": track_overlap_threshold,
        },
        "ego_track_ids": sorted(list(ego_track_ids)),
        "frames": frame_records,
        "stats": {
            "n_frames_processed": len(frame_records),
            "n_frames_with_detections": n_with_det,
            "n_detections_total": sum(len(fr["detections"]) for fr in frame_records),
            "n_tracks_total": len(tracks),
            "n_ego_tracks": len(ego_track_ids),
            "n_cropped": n_cropped,
        },
    }

    # ── 8. Save JSONs ───────────────────────────────────────────────────
    output_path = os.path.join(seq_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved to: %s", output_path)

    coco = _build_coco_json(frame_records, take_name, exo_cam, h_u, w_u)
    coco_path = os.path.join(seq_dir, "coco.json")
    with open(coco_path, "w") as f:
        json.dump(coco, f, indent=2)
    log.info("COCO saved to: %s", coco_path)

    # ── 9. Upload to TOS if requested ──────────────────────────────────
    if crop_tos_base:
        tos_seq = f"{crop_tos_base}/sequences/{take_name}/{exo_cam}"
        _upload_json_to_tos(output_path, tos_seq, "results.json")
        _upload_json_to_tos(coco_path, tos_seq, "coco.json")
        log.info("Uploaded to TOS: %s", tos_seq)

    log.info("[YOLO] Stats: %d frames, %d with ego-track, %d cropped, %d total tracks",
             len(frame_records), n_with_det, n_cropped, len(tracks))
    log.info("=" * 60)

    return output


def _upload_json_to_tos(local_path: str, tos_dir: str, filename: str) -> None:
    """Upload a local JSON file to TOS."""
    from utils.fsspec_util import get_tosfs
    fs = get_tosfs()
    tos_path = f"{tos_dir}/{filename}"
    with open(local_path, "rb") as f:
        data = f.read()
    try:
        with fs.open(tos_path, "wb") as f:
            f.write(data)
    except Exception as e:
        log.error("  [upload] failed %s: %s", tos_path, e)


def _mask_to_rle(mask) -> dict:
    """Encode a binary mask as COCO RLE dict with base64 string counts."""
    import torch
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    # Ensure 2D binary mask (H, W)
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        log.warning("_mask_to_rle: expected 2D mask, got shape %s, taking first slice", mask.shape)
        mask = mask[0] if mask.shape[0] < mask.shape[-1] else mask[..., 0]
    mask = (mask > 0.5).astype(np.uint8)
    if mask.sum() == 0:
        return {"size": list(mask.shape), "counts": ""}
    rle = mask_utils.encode(np.asfortranarray(mask))
    if isinstance(rle, list):
        rle = rle[0] if rle else {"size": list(mask.shape), "counts": ""}
    rle["counts"] = base64.b64encode(rle["counts"]).decode("ascii")
    return rle


def _serialize_detection(det, dist_to_proj: float) -> dict:
    """Serialize a Detection to JSON-safe dict, including RLE mask."""
    out: dict = {
        "bbox": list(det.bbox),
        "center": list(det.center),
        "score": det.score,
        "dist_to_proj": dist_to_proj,
    }
    if det.mask is not None:
        out["mask_rle"] = _mask_to_rle(det.mask)
    return out


def _build_coco_json(
    frame_records: list[dict],
    take_name: str,
    exo_cam: str,
    img_h: int,
    img_w: int,
) -> dict:
    """Build COCO-format JSON from frame records with RLE masks.

    Returns dict with keys: images, annotations, categories.
    """
    images = []
    annotations = []
    ann_id = 0

    for fr in frame_records:
        frame_idx = fr["frame_idx"]
        images.append({
            "id": frame_idx,
            "file_name": f"{frame_idx:06d}.jpg",
            "width": img_w,
            "height": img_h,
        })
        best = fr.get("best_detection")
        if best and "mask_rle" in best:
            x1, y1, x2, y2 = best["bbox"]
            bbox_w = max(0, x2 - x1)
            bbox_h = max(0, y2 - y1)
            annotations.append({
                "id": ann_id,
                "image_id": frame_idx,
                "category_id": 1,
                "bbox": [x1, y1, bbox_w, bbox_h],
                "area": bbox_w * bbox_h,
                "segmentation": best["mask_rle"],
                "score": best["score"],
                "dist_to_proj": best["dist_to_proj"],
            })
            ann_id += 1

    return {
        "info": {
            "take_name": take_name,
            "exo_cam": exo_cam,
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}],
    }


def _upload_jpg_to_tos(
    local_path: str,
    tos_base: str,
    take_name: str,
    exo_cam: str,
    frame_idx: int,
) -> None:
    """Upload a crop JPG to TOS under sequences/ structure."""
    from utils.fsspec_util import get_tosfs
    fs = get_tosfs()
    tos_path = f"{tos_base}/sequences/{take_name}/{exo_cam}/{frame_idx:06d}.jpg"
    with open(local_path, "rb") as f:
        data = f.read()
    try:
        with fs.open(tos_path, "wb") as f:
            f.write(data)
    except Exception as e:
        log.error("  [upload] failed %s: %s", tos_path, e)


# ── Result loading utilities ────────────────────────────────────────


def load_results(path: str) -> dict:
    """Load a pipeline output JSON file."""
    with open(path) as f:
        return json.load(f)


def get_best_detection_bboxes(
    data: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract best detection bboxes as numpy arrays from pipeline output.

    Returns:
        (bboxes, centers, proj_2d) — bboxes (N,4), centers (N,2), proj_2d (N,2).
        NaN where no detection for that frame.
    """
    n = len(data["frames"])
    bboxes = np.full((n, 4), np.nan, dtype=np.float32)
    centers = np.full((n, 2), np.nan, dtype=np.float32)
    proj_2d = np.full((n, 2), np.nan, dtype=np.float32)

    for i, fr in enumerate(data["frames"]):
        proj_2d[i] = fr["proj_2d"]
        best = fr.get("best_detection")
        if best:
            bboxes[i] = best["bbox"]
            centers[i] = best["center"]

    return bboxes, centers, proj_2d


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Video person detection/tracking + ego-projection cropping"
    )
    parser.add_argument("--take", type=str, default="cmu_bike01_2")
    parser.add_argument("--exo-cam", type=str, default="cam01",
                        help="Exo camera ID, or 'auto' to discover all cams for this take")
    parser.add_argument("--model", type=str,
                        default="/LargeModelDev/users/chuanfu.shen/ckpts/sam3/sam3.pt",
                        help="SAM3 model path (ignored if --yolo)")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--crop", type=str, default=None,
                        help="TOS base path for cropped JPGs")
    parser.add_argument("--crop-padding", type=int, default=20)
    parser.add_argument("--prompt", type=str, default="person",
                        help="SAM3 text prompt (ignored if --yolo)")

    # YOLO tracking options
    parser.add_argument("--yolo", action="store_true",
                        help="Use YOLO + ByteTrack tracking (fast, bbox only)")
    parser.add_argument("--yolo-model", type=str, default="yolo26n.pt",
                        help="YOLO model path (default: yolo26n.pt)")
    parser.add_argument("--yolo-conf", type=float, default=0.25,
                        help="YOLO confidence threshold (default: 0.25)")
    parser.add_argument("--track-overlap", type=float, default=0.2,
                        help="Min fraction of track frames where ego proj must be "
                             "inside bbox to select as ego-person track (default: 0.2)")

    args = parser.parse_args()

    # Resolve exo cameras
    if args.exo_cam == "auto":
        from src.embodiedgait.loader import load_gopro_calibs
        calibs = load_gopro_calibs(args.take)
        exo_cams = list(calibs.keys())
        log.info("Auto-discovered %d exo cameras for %s: %s",
                 len(exo_cams), args.take, exo_cams)
    else:
        exo_cams = [args.exo_cam]

    for exo_cam in exo_cams:
        log.info("Processing cam: %s", exo_cam)
        if args.yolo:
            process_video_yolo(
                take_name=args.take,
                exo_cam=exo_cam,
                yolo_model_path=args.yolo_model,
                output_dir=args.output_dir,
                max_frames=args.max_frames,
                device=args.device,
                crop_tos_base=args.crop,
                crop_padding=args.crop_padding,
                yolo_conf=args.yolo_conf,
                track_overlap_threshold=args.track_overlap,
            )
        else:
            process_video(
                take_name=args.take,
                exo_cam=exo_cam,
                model_path=args.model,
                output_dir=args.output_dir,
                max_frames=args.max_frames,
                device=args.device,
                crop_tos_base=args.crop,
                crop_padding=args.crop_padding,
                prompt=args.prompt,
            )


if __name__ == "__main__":
    main()
