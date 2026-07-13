"""Data loading: trajectory CSV, ego_pose JSON, video frames from TOS."""

import io
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.fsspec_util import get_tosfs, open_with_fs, list_tos_directory

# TOS base path for the dataset
BASE_PATH = "tos://drobotics-ailab/users/chuanfu.shen/data/egoexo4d-defaults"
# Local cache directory
CACHE_DIR = Path(os.environ.get("EGOEXO_CACHE", Path.home() / ".cache" / "egoexo4d"))


def _ensure_fs(fs=None):
    """Return a TosFileSystem instance, creating one if needed."""
    if fs is not None:
        return fs
    fs = get_tosfs()
    if fs is None:
        raise RuntimeError(
            "TOS filesystem not available. Set VOLC_ACCESSKEY and VOLC_SECRETKEY."
        )
    return fs


def load_trajectory(take_name: str, fs=None) -> pd.DataFrame:
    """Load closed_loop_trajectory.csv for a take as a DataFrame.

    Args:
        take_name: e.g. 'cmu_bike01_2'.
        fs: Optional TosFileSystem instance.

    Returns:
        DataFrame with columns including:
        tracking_timestamp_us, tx_world_device, ty_world_device, tz_world_device,
        qx_world_device, qy_world_device, qz_world_device, qw_world_device, ...
    """
    fs = _ensure_fs(fs)
    csv_path = f"{BASE_PATH}/takes/{take_name}/trajectory/closed_loop_trajectory.csv"
    with open_with_fs(csv_path, "rb", fs) as f:
        df = pd.read_csv(f)
    return df


def load_online_calibration(take_name: str, fs=None) -> list[dict]:
    """Load online_calibration.jsonl for a take.

    Returns a list of dicts, one per line.
    """
    fs = _ensure_fs(fs)
    calib_path = f"{BASE_PATH}/takes/{take_name}/trajectory/online_calibration.jsonl"
    entries = []
    with open_with_fs(calib_path, "r", fs) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _scan_ego_pose_cache(cache_file: Path) -> dict[str, str] | None:
    """Read cached take_name → uuid mapping if it exists."""
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return None


def _build_ego_pose_cache(fs) -> dict[str, str]:
    """Build take_name → uuid mapping by scanning all ego_pose JSONs."""
    mapping = {}
    for split in ["train", "val", "test"]:
        pose_dir = f"{BASE_PATH}/annotations/ego_pose/{split}/camera_pose/"
        try:
            items = list_tos_directory(pose_dir, fs)
        except Exception:
            continue
        for item in items:
            if isinstance(item, dict):
                name = item.get("name", "")
            else:
                name = str(item)
            if not name.endswith(".json"):
                continue
            # name from fs.ls is the full path without tos://
            full_path = "tos://" + name
            try:
                with open_with_fs(full_path, "r", fs) as f:
                    data = json.load(f)
                take_name = data.get("metadata", {}).get("take_name", "")
                if take_name:
                    mapping[take_name] = full_path
            except Exception:
                continue
    return mapping


def find_ego_pose(take_name: str, fs=None) -> dict[str, Any]:
    """Find the ego_pose JSON for a given take_name.

    Scans annotations/ego_pose/{train,val,test}/camera_pose/*.json,
    with a local cache to avoid re-scanning.

    Returns:
        dict with keys: metadata, aria01, cam01, cam02, ...
        aria01 has 'camera_intrinsics' (3×3) and 'camera_extrinsics' (flat list).
    """
    fs = _ensure_fs(fs)

    # Check cache first
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "ego_pose_mapping.json"
    mapping = _scan_ego_pose_cache(cache_file)
    if mapping is None:
        print("Building ego_pose take_name → uuid cache (one-time scan)...")
        mapping = _build_ego_pose_cache(fs)
        with open(cache_file, "w") as f:
            json.dump(mapping, f)
        print(f"  Cached {len(mapping)} entries.")

    if take_name not in mapping:
        raise KeyError(
            f"take_name '{take_name}' not found in ego_pose annotations. "
            f"Available: {list(mapping.keys())[:5]}..."
        )

    pose_path = mapping[take_name]
    with open_with_fs(pose_path, "r", fs) as f:
        return json.load(f)


def load_gopro_calibs(take_name: str, fs=None) -> dict[str, dict]:
    """Load gopro_calibs.csv and return a dict keyed by cam_id.

    Each row contains:
      - KANNALABRANDTK3 intrinsics: intrinsics_0..7 (fx, fy, cx, cy, k1, k2, k3, k4)
      - cam-in-world extrinsics: tx_world_cam, ty_world_cam, tz_world_cam,
                                 qx_world_cam, qy_world_cam, qz_world_cam, qw_world_cam
      - image_width, image_height

    Returns:
        dict[cam_id] = {all CSV columns as key-value pairs}
    """
    import csv

    fs = _ensure_fs(fs)
    calib_path = f"{BASE_PATH}/takes/{take_name}/trajectory/gopro_calibs.csv"
    calibs = {}
    with open_with_fs(calib_path, "r", fs) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cam_id = row.get("cam_uid") or row.get("cam_id", "")
            if cam_id:
                calibs[cam_id] = row
    return calibs


def list_ego_videos(take_name: str, fs=None) -> list[dict]:
    """List the Aria (ego) video files for a take.

    Returns list of dicts with keys: stream_id, path, size_bytes.
    """
    fs = _ensure_fs(fs)
    video_dir = f"{BASE_PATH}/takes/{take_name}/frame_aligned_videos/"
    items = list_tos_directory(video_dir, fs)

    videos = []
    for item in items:
        name = item.get("name", "") if isinstance(item, dict) else str(item)
        basename = name.rstrip("/").split("/")[-1]
        if basename.startswith("aria01_") and basename.endswith(".mp4"):
            stream_id = basename.replace("aria01_", "").replace(".mp4", "")
            videos.append(
                {
                    "stream_id": stream_id,
                    "path": "tos://" + name,
                    "size_bytes": item.get("size", 0) if isinstance(item, dict) else 0,
                }
            )
    return videos


def list_exo_videos(take_name: str, fs=None) -> list[dict]:
    """List the exo (GoPro) video files for a take.

    Returns list of dicts with keys: cam_id, path, size_bytes.
    """
    fs = _ensure_fs(fs)
    video_dir = f"{BASE_PATH}/takes/{take_name}/frame_aligned_videos/"
    items = list_tos_directory(video_dir, fs)

    videos = []
    for item in items:
        name = item.get("name", "") if isinstance(item, dict) else str(item)
        basename = name.rstrip("/").split("/")[-1]
        if basename.startswith("cam") and basename.endswith(".mp4"):
            cam_id = basename.replace(".mp4", "")
            videos.append(
                {
                    "cam_id": cam_id,
                    "path": "tos://" + name,
                    "size_bytes": item.get("size", 0) if isinstance(item, dict) else 0,
                }
            )
    return videos


def iter_video_frames(
    video_path: str,
    max_frames: int | None = None,
    fs=None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Read video frames from a TOS mp4 file.

    Uses PyAV (av) for efficient frame decoding over TOS streaming.

    Args:
        video_path: TOS path to the mp4 file.
        max_frames: Maximum number of frames to yield (None = all).
        fs: TosFileSystem instance.

    Yields:
        (frame_index, frame_as_numpy_array_bgr) tuples.
    """
    import av

    fs = _ensure_fs(fs)

    # Read the full video into a buffer (streaming from TOS)
    with open_with_fs(video_path, "rb", fs) as f:
        video_bytes = f.read()

    container = av.open(io.BytesIO(video_bytes))
    video_stream = container.streams.video[0]

    frame_idx = 0
    for frame in container.decode(video_stream):
        img = frame.to_ndarray(format="bgr24")
        yield frame_idx, img
        frame_idx += 1
        if max_frames is not None and frame_idx >= max_frames:
            break

    container.close()
