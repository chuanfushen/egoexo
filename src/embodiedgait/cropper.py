"""Crop person from video frames using best-track bboxes and upload to TOS.

Reads pipeline output JSONs, extracts best-track bboxes, crops frames,
and saves as JPG to TOS with directory structure:
    egoexogait/{take_name}/{camera}/{frame_idx:06d}.jpg

Usage:
    # Process all available results:
    uv run python -m src.embodiedgait.cropper --input-dir outputs

    # Single result:
    uv run python -m src.embodiedgait.cropper \
        --input outputs/cmu_bike01_2_cam01.json
"""

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.fsspec_util import get_tosfs

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

TOS_BASE = "tos://drobotics-ailab/users/chuanfu.shen/data/egoexogait"


def crop_and_upload_one(
    result_path: str,
    padding: int = 20,
    jpg_quality: int = 90,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Crop person from one video using its pipeline result JSON.

    Args:
        result_path: Path to JSON output from tracking_pipeline.
        padding: Extra pixels around bbox for context.
        jpg_quality: JPEG compression quality (0-100).
        skip_existing: Skip frames already on TOS.

    Returns:
        Dict with stats: {take_name, camera, n_frames, n_cropped, n_skipped, n_missing}.
    """
    with open(result_path) as f:
        data = json.load(f)

    take_name = data["take_name"]
    exo_cam = data["exo_cam"]
    frames = data.get("frames") or []
    if not frames:
        log.warning("No frames in %s, skipping", result_path)
        return {"take_name": take_name, "camera": exo_cam, "n_cropped": 0}

    # Build frame_idx → bbox lookup from best_detection per frame
    n_total = len(frames)
    bbox_data = []
    for fr in frames:
        best = fr.get("best_detection")
        if best:
            bbox_data.append((fr["frame_idx"], best["bbox"]))
    valid_frames = bbox_data
    log.info("[%s/%s] %d/%d frames have bbox", take_name, exo_cam,
             len(valid_frames), n_total)

    if not valid_frames:
        return {"take_name": take_name, "camera": exo_cam,
                "n_frames": n_total, "n_cropped": 0, "n_missing": n_total}

    # ── Load calibration & build undistort maps ──────────────────
    from src.embodiedgait.camera import (
        build_undistort_maps_gopro, gopro_calib_to_K_D, gopro_calib_to_world_camera, undistort_frame,
    )
    from src.embodiedgait.loader import iter_video_frames, list_exo_videos, load_gopro_calibs

    calibs = load_gopro_calibs(take_name)
    calib_row = calibs[exo_cam]
    videos = list_exo_videos(take_name)
    video = next(v for v in videos if v["cam_id"] == exo_cam)
    video_path = video["path"]

    for _, frame in iter_video_frames(video_path, max_frames=1):
        h, w = frame.shape[:2]
        break
    map1, map2, _ = build_undistort_maps_gopro(calib_row, w, h)

    # ── TOS filesystem ───────────────────────────────────────────
    fs = get_tosfs()
    tos_dir = f"{TOS_BASE}/{take_name}/{exo_cam}"

    # ── Process frames ───────────────────────────────────────────
    n_cropped = 0
    n_skipped = 0
    n_missing = 0
    t_start = time.time()

    # Build a lookup: frame_idx -> bbox
    bbox_lookup = {idx: bbox for idx, bbox in valid_frames}
    target_frames = set(bbox_lookup.keys())

    for frame_idx, raw_frame in iter_video_frames(video_path):
        if frame_idx not in target_frames:
            n_missing += 1
            continue

        tos_path = f"{tos_dir}/{frame_idx:06d}.jpg"

        # Skip existing
        if skip_existing:
            try:
                if fs.exists(tos_path):
                    n_skipped += 1
                    continue
            except Exception:
                pass

        # Undistort
        undistorted = undistort_frame(raw_frame, map1, map2)
        h_u, w_u = undistorted.shape[:2]

        # Crop
        x1, y1, x2, y2 = bbox_lookup[frame_idx]
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(w_u, int(x2) + padding)
        y2 = min(h_u, int(y2) + padding)

        if x2 <= x1 or y2 <= y1:
            n_missing += 1
            continue

        crop = undistorted[y1:y2, x1:x2]

        # Encode to JPEG
        _, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])

        # Upload to TOS
        try:
            with fs.open(tos_path, "wb") as f:
                f.write(jpg.tobytes())
            n_cropped += 1
        except Exception as e:
            log.error("  Failed to upload %s: %s", tos_path, e)
            n_missing += 1

        if (n_cropped + n_skipped) % 200 == 0:
            elapsed = time.time() - t_start
            rate = (n_cropped + n_skipped) / max(elapsed, 1)
            log.info("  [%s/%s] %d uploaded, %d skipped, %d missing (%.1f fps)",
                     take_name, exo_cam, n_cropped, n_skipped, n_missing, rate)

    elapsed = time.time() - t_start
    log.info("[%s/%s] done: %d cropped, %d skipped, %d missing in %.0fs",
             take_name, exo_cam, n_cropped, n_skipped, n_missing, elapsed)

    return {
        "take_name": take_name,
        "camera": exo_cam,
        "n_frames": n_total,
        "n_cropped": n_cropped,
        "n_skipped": n_skipped,
        "n_missing": n_missing,
    }


# ── Multi-process worker (module-level, picklable) ─────────────────


def _crop_worker(
    paths: list[str],
    padding: int,
    jpg_quality: int,
    skip_existing: bool,
    queue,
) -> None:
    """Process a subset of result JSONs in one process."""
    for rp in paths:
        r = crop_and_upload_one(rp, padding, jpg_quality, skip_existing)
        queue.put(r)


# ── Batch processing ───────────────────────────────────────────────


def crop_batch(
    input_dir: str = "outputs",
    result_paths: list[str] | None = None,
    padding: int = 20,
    jpg_quality: int = 90,
    num_workers: int = 4,
    skip_existing: bool = True,
) -> list[dict]:
    """Process multiple result JSONs in parallel.

    Args:
        input_dir: Directory containing result JSON files.
        result_paths: Explicit list of result JSON paths (overrides input_dir).
        padding: Extra pixels around bbox.
        jpg_quality: JPEG quality (0-100).
        num_workers: Number of parallel workers.
        skip_existing: Skip frames already on TOS.
    """
    if result_paths is None:
        result_paths = sorted(Path(input_dir).glob("*_cam*.json"))
        result_paths = [str(p) for p in result_paths if "batch_summary" not in p.name
                        and "node_" not in p.name]

    log.info("Found %d result files to process", len(result_paths))

    if num_workers <= 1:
        results = []
        for i, rp in enumerate(result_paths):
            log.info("[%d/%d] %s", i + 1, len(result_paths), rp)
            r = crop_and_upload_one(rp, padding, jpg_quality, skip_existing)
            results.append(r)
        return results

    # Multi-process
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    workers = []
    for w in range(num_workers):
        my_paths = result_paths[w::num_workers]
        p = ctx.Process(
            target=_crop_worker,
            args=(my_paths, padding, jpg_quality, skip_existing, result_queue),
        )
        p.start()
        workers.append(p)

    results = []
    for _ in range(len(result_paths)):
        results.append(result_queue.get())

    for p in workers:
        p.join()

    # Summary
    total_cropped = sum(r["n_cropped"] for r in results)
    total_skipped = sum(r["n_skipped"] for r in results)
    total_missing = sum(r["n_missing"] for r in results)
    log.info("=" * 60)
    log.info("Batch complete: %d cropped, %d skipped, %d missing",
             total_cropped, total_skipped, total_missing)

    return results


# ── CLI ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Crop person from video frames using tracker results"
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Single result JSON or directory of results")
    parser.add_argument("--input-dir", type=str, default="outputs",
                        help="Directory containing result JSONs")
    parser.add_argument("--padding", type=int, default=20,
                        help="Extra pixels around bbox (default: 20)")
    parser.add_argument("--jpg-quality", type=int, default=90,
                        help="JPEG quality 0-100 (default: 90)")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Parallel workers (default: 1)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-upload even if file already exists on TOS")

    args = parser.parse_args()

    if args.input:
        # Single file
        result = crop_and_upload_one(
            args.input, args.padding, args.jpg_quality,
            skip_existing=not args.no_skip,
        )
        print(json.dumps(result, indent=2))
    else:
        crop_batch(
            input_dir=args.input_dir,
            padding=args.padding,
            jpg_quality=args.jpg_quality,
            num_workers=args.num_workers,
            skip_existing=not args.no_skip,
        )


if __name__ == "__main__":
    main()
