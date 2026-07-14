"""Offline ByteTrack: run tracking on pre-computed YOLO detections.

Takes a detections.json (from distributed_egoexo_detection.py) and assigns
ByteTrack track IDs frame-by-frame, without re-running YOLO.

Usage:
    # ── Local files ──────────────────────────────────────────────────
    uv run python -m src.embodiedgait.offline_tracker \
        --detections outputs/sequences/cmu_bike01_2/cam01/detections.json \
        --output outputs/sequences/cmu_bike01_2/cam01/tracks.json

    uv run python -m src.embodiedgait.offline_tracker \
        --take cmu_bike01_2 --detections-dir outputs/sequences

    # ── TOS paths ────────────────────────────────────────────────────
    uv run python -m src.embodiedgait.offline_tracker \
        --detections tos://bucket/egoexo_detections/sequences/cmu_bike01_2/cam01/detections.json \
        --output tos://bucket/egoexo_detections/sequences/cmu_bike01_2/cam01/tracks.json

    uv run python -m src.embodiedgait.offline_tracker \
        --take cmu_bike01_2 \
        --detections-tos tos://bucket/egoexo_detections/sequences \
        --output-tos tos://bucket/egoexo_detections/sequences
"""

import argparse
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from typing import Any

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


def build_tracker_args(fps: float = 30.0, **overrides) -> IterableSimpleNamespace:
    """Build tracker parameters from bytetrack.yaml, with overrides.

    Args:
        fps: Video frame rate.
        **overrides: Override any tracker arg (track_high_thresh, etc.).

    Returns:
        IterableSimpleNamespace with tracker parameters.
    """
    config = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


# ═══════════════════════════════════════════════════════════════════════
# I/O helpers — transparently handle local and tos:// paths
# ═══════════════════════════════════════════════════════════════════════


def _is_tos(path: str) -> bool:
    return path.startswith("tos://")


def _get_tosfs():
    from utils.fsspec_util import get_tosfs
    fs = get_tosfs()
    if fs is None:
        raise RuntimeError("TOS filesystem unavailable; set VOLC_ACCESSKEY and VOLC_SECRETKEY")
    return fs


def load_detections(path: str) -> dict:
    """Load a detections.json file (local or tos://)."""
    if _is_tos(path):
        from utils.fsspec_util import open_with_fs
        fs = _get_tosfs()
        with open_with_fs(path, "r", fs) as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def detections_to_boxes(
    frame_dets: list[dict],
    image_shape: tuple[int, int],
    target_class: int | None = 0,
) -> Boxes:
    """Convert a single frame's detection list to ultralytics Boxes for ByteTrack.

    Args:
        frame_dets: List of {"xyxy": [x1,y1,x2,y2], "confidence": c, "class_id": cls}.
        image_shape: (height, width) of the frame.
        target_class: If not None, keep only this class_id.

    Returns:
        ``ultralytics.engine.results.Boxes`` — the native type ByteTrack expects.
    """
    rows = []
    for d in frame_dets:
        if target_class is not None and d.get("class_id") != target_class:
            continue
        rows.append([
            *[float(v) for v in d["xyxy"]],
            float(d.get("confidence", 1.0)),
            int(d.get("class_id", 0)),
        ])

    array = np.asarray(rows, dtype=np.float32) if rows else np.empty((0, 6), dtype=np.float32)
    return Boxes(array, orig_shape=image_shape)


def run_bytetrack(
    detections_data: dict,
    target_class: int | None = 0,
    fps: float | None = None,
    img_size: tuple[int, int] | None = None,
    **tracker_kwargs,
) -> list[dict]:
    """Run ByteTrack over pre-computed per-frame detections.

    Handles non-contiguous frame indices by advancing the tracker with empty
    detections through any gaps, so track_buffer stays temporally correct.

    Args:
        detections_data: Loaded detections.json dict.
        target_class: Filter detections to this class (None = all).
        fps: Frame rate (auto-detected from detections_data["video"]["fps"]).
        img_size: (width, height) — auto-detected from metadata.
        **tracker_kwargs: Passed to build_tracker_args.

    Returns:
        List of per-frame dicts:
        {"frame_index": int, "tracks": [{"track_id": int, "xyxy": [...], "conf": float}, ...]}
    """
    from ultralytics.trackers.byte_tracker import BYTETracker

    # ── Resolve metadata ──────────────────────────────────────────────
    video_meta = detections_data.get("video", {})
    if fps is None:
        fps = float(video_meta.get("fps", 30.0))
    if img_size is None:
        w = int(video_meta.get("width", 1920))
        h = int(video_meta.get("height", 1080))
        img_size = (w, h)

    tracker_args = build_tracker_args(fps=fps, **tracker_kwargs)
    tracker = BYTETracker(tracker_args)

    image_shape = (img_size[1], img_size[0])  # (H, W) for Boxes
    empty_boxes = Boxes(np.empty((0, 6), dtype=np.float32), orig_shape=image_shape)

    frames = detections_data.get("frames", [])
    if not frames:
        return []

    # Sort by frame_index just in case
    frames = sorted(frames, key=lambda f: f["frame_index"])

    log.info(
        "ByteTrack: %d frames, fps=%.1f, img_size=%s, class_filter=%s",
        len(frames), fps, img_size, target_class,
    )

    # ── Process frame by frame with gap handling ──────────────────────
    results: list[dict] = []
    next_frame_index = frames[0]["frame_index"]

    for fr in frames:
        frame_idx = fr["frame_index"]

        # Advance tracker through any missing frames so track_buffer
        # stays temporally correct (e.g. frames 0,1,5 → feed empty frames 2,3,4).
        while next_frame_index < frame_idx:
            tracker.update(empty_boxes)
            next_frame_index += 1

        boxes = detections_to_boxes(
            fr.get("detections", []), image_shape, target_class=target_class,
        )

        # update() returns np.ndarray (M, 8):
        #   [x1, y1, x2, y2, track_id, score, cls, idx]
        active_tracks: np.ndarray = tracker.update(boxes)
        next_frame_index = frame_idx + 1

        track_list = []
        if len(active_tracks) > 0:
            for row in active_tracks:
                track_list.append({
                    "track_id": int(row[4]),
                    "xyxy": [round(float(v), 2) for v in row[:4].tolist()],
                    "conf": round(float(row[5]), 6),
                    "cls": int(row[6]),
                })

        results.append({
            "frame_index": frame_idx,
            "tracks": track_list,
        })

    return results


def save_tracks_json(
    tracks: list[dict],
    output_path: str,
    metadata: dict | None = None,
) -> None:
    """Save tracked results as JSON (local or tos://)."""
    output = {
        **(metadata or {}),
        "tracker": "bytetrack",
        "frames": tracks,
        "stats": {
            "n_frames": len(tracks),
            "n_frames_with_tracks": sum(1 for t in tracks if t["tracks"]),
            "n_tracks_total": sum(len(t["tracks"]) for t in tracks),
            "unique_track_ids": sorted(set(
                t2["track_id"] for t in tracks for t2 in t["tracks"]
            )),
        },
    }
    output["stats"]["n_unique_tracks"] = len(output["stats"]["unique_track_ids"])

    if _is_tos(output_path):
        from utils.fsspec_util import open_with_fs
        fs = _get_tosfs()
        with open_with_fs(output_path, "w", fs) as f:
            json.dump(output, f, indent=2)
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

    log.info("Saved tracked results: %s (%d frames, %d unique tracks)",
             output_path, output["stats"]["n_frames"],
             output["stats"]["n_unique_tracks"])


def process_detections_file(
    detections_path: str,
    output_path: str,
    target_class: int | None = 0,
    crop_dir: str | None = None,
    bbox_expand: float = 1.2,
    **tracker_kwargs,
) -> str:
    """Load + track + save + optionally crop.

    Args:
        detections_path: Path to detections.json (local or tos://).
        output_path: Path for tracks.json output.
        target_class: Filter detections to this class.
        crop_dir: If set, crop tracked person bboxes from video into this
                  directory (e.g. ``outputs/crops`` or ``tos://bucket/crops``).
        bbox_expand: Scale factor for bbox expansion when cropping (1.0 = no expand).
        **tracker_kwargs: Passed to ``run_bytetrack``.

    Returns the output path.
    """
    log.info("Loading detections: %s", detections_path)
    data = load_detections(detections_path)

    tracks = run_bytetrack(data, target_class=target_class, **tracker_kwargs)

    # Carry forward input metadata
    meta = {k: v for k, v in data.items() if k != "frames"}
    meta["source_detections"] = detections_path

    save_tracks_json(tracks, output_path, metadata=meta)

    # ── Optional: crop tracked persons from video ────────────────────
    if crop_dir:
        crop_persons_from_tracks(tracks, data, crop_dir, bbox_expand=bbox_expand)

    return output_path


def _discover_tos_cams(tos_base: str, take_name: str) -> list[tuple[str, str]]:
    """List camera directories under a take — one ``fs.ls`` call, no per-file checks.

    Returns list of (detections_tos_path, tracks_tos_path).
    """
    from utils.fsspec_util import list_tos_directory

    seq_dir = f"{tos_base.rstrip('/')}/{take_name}"
    items = list_tos_directory(seq_dir, _get_tosfs())

    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "directory":
            continue

        cam = item["name"].rstrip("/").split("/")[-1]
        det_path = f"{seq_dir}/{cam}/detections.json"
        if not det_path.startswith("tos://"):
            det_path = f"tos://{det_path}"

        track_path = f"{seq_dir}/{cam}/tracks.json"
        if not track_path.startswith("tos://"):
            track_path = f"tos://{track_path}"
        result.append((det_path, track_path))

    result.sort(key=lambda x: x[0])
    return result


# ── CLI ──────────────────────────────────────────────────────────────


def _resolve_node_info() -> tuple[int, int]:
    """Resolve node rank and world size from environment.

    Checks common distributed-launch env vars in order:
        torchrun / torch.distributed  (``RANK``, ``WORLD_SIZE``)
        MLP platform                   (``MLP_ROLE_INDEX``, ``MLP_WORLD_SIZE``)
        SLURM                          (``SLURM_PROCID``, ``SLURM_NTASKS``)
        OpenMPI                        (``OMPI_COMM_WORLD_RANK``, ``OMPI_COMM_WORLD_SIZE``)
        PMI                            (``PMI_RANK``, ``PMI_SIZE``)

    Falls back to (0, 1).
    """
    # Rank candidates: (rank_key, [world_key_candidates...])
    rank_candidates = [
        ("RANK",                  ["WORLD_SIZE"]),
        ("MLP_ROLE_INDEX",        ["MLP_WORLD_SIZE", "MLP_WORKER_NUM"]),
        ("SLURM_PROCID",          ["SLURM_NTASKS"]),
        ("OMPI_COMM_WORLD_RANK",  ["OMPI_COMM_WORLD_SIZE"]),
        ("PMI_RANK",              ["PMI_SIZE"]),
    ]
    for rank_key, world_keys in rank_candidates:
        rank_val = os.environ.get(rank_key)
        if rank_val is None:
            continue
        rank = int(rank_val)
        world = 1
        world_key = None
        for wk in world_keys:
            wv = os.environ.get(wk)
            if wv is not None:
                world = max(world, int(wv))  # take largest if multiple set
                world_key = wk
        if world > 1 and world_key:
            log.info("Multi-node (%s=%d, %s=%d): rank=%d/%d",
                     rank_key, rank, world_key, world, rank, world)
        return rank, world
    return 0, 1


def main():
    parser = argparse.ArgumentParser(
        description="Offline ByteTrack on pre-computed detections.json"
    )
    # ── Input sources ──────────────────────────────────────────────────
    parser.add_argument("--detections", type=str,
                        help="Path to a single detections.json (local or tos://)")
    parser.add_argument("--output", type=str,
                        help="Output path for tracks.json (local or tos://)")
    # Local batch mode
    parser.add_argument("--take", type=str,
                        help="Take name for batch discovery")
    parser.add_argument("--all-takes", action="store_true",
                        help="Process all takes under --detections-tos or --detections-dir")
    parser.add_argument("--detections-dir", type=str, default="outputs/sequences",
                        help="Local base dir: sequences/{take}/{cam}/detections.json")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Local base dir for tracks.json (defaults to --detections-dir)")
    # TOS batch mode
    parser.add_argument("--detections-tos", type=str,
                        help="TOS base path for batch discovery")
    parser.add_argument("--output-tos", type=str, default=None,
                        help="TOS base path for tracks.json (defaults to --detections-tos)")

    # Tracker config
    parser.add_argument("--target-class", type=int, default=0,
                        help="Class ID to track (default: 0 = person)")
    parser.add_argument("--fps", type=float, default=None,
                        help="FPS override (auto-detected if omitted)")
    parser.add_argument("--track-high-thresh", type=float, default=0.5)
    parser.add_argument("--track-low-thresh", type=float, default=0.1)
    parser.add_argument("--new-track-thresh", type=float, default=0.6)
    parser.add_argument("--match-thresh", type=float, default=0.8)
    parser.add_argument("--track-buffer", type=int, default=30)
    # Skip existing
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if tracks.json already exists at output path")
    # Crop
    parser.add_argument("--crop", action="store_true",
                        help="Crop tracked person bboxes from video into local --crop-dir")
    parser.add_argument("--crop-dir", type=str, default="outputs/crops",
                        help="Directory for cropped person images, local or tos:// (default: outputs/crops)")
    parser.add_argument("--bbox-expand", type=float, default=1.2,
                        help="Bbox expansion scale for cropping (default: 1.2)")

    # Multi-node / threading
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--node-rank", type=int, default=None,
                        help="Override MLP_ROLE_INDEX")
    parser.add_argument("--world-size", type=int, default=None,
                        help="Override MLP_WORLD_SIZE")

    args = parser.parse_args()

    tracker_kwargs = dict(
        track_high_thresh=args.track_high_thresh,
        track_low_thresh=args.track_low_thresh,
        new_track_thresh=args.new_track_thresh,
        match_thresh=args.match_thresh,
        track_buffer=args.track_buffer,
    )
    if args.fps is not None:
        tracker_kwargs["fps"] = args.fps

    crop_dir = args.crop_dir if args.crop else None
    bbox_expand = args.bbox_expand

    # ── Single-file mode ──────────────────────────────────────────────
    if args.detections:
        output = args.output or _default_output(args.detections)
        if args.skip_existing and _output_exists(output):
            log.info("SKIP (exists): %s", output)
            return
        process_detections_file(
            detections_path=args.detections,
            output_path=output,
            target_class=args.target_class,
            crop_dir=crop_dir,
            bbox_expand=bbox_expand,
            **tracker_kwargs,
        )
        return

    # ── Batch mode ────────────────────────────────────────────────────
    if not args.take and not args.all_takes:
        parser.error("Provide --detections (single file), --take, or --all-takes")

    node_rank = args.node_rank
    world_size = args.world_size
    if node_rank is None or world_size is None:
        env_rank, env_world = _resolve_node_info()
        if node_rank is None:
            node_rank = env_rank
        if world_size is None:
            world_size = env_world

    num_workers = args.num_workers

    if args.detections_tos:
        if args.all_takes:
            _run_batch_all_takes_tos(args, tracker_kwargs, node_rank, world_size, num_workers)
        else:
            _run_batch_tos(args, tracker_kwargs, node_rank, world_size, num_workers)
    else:
        _run_batch_local(args, tracker_kwargs, node_rank, world_size, num_workers)


def _default_output(detections_path: str) -> str:
    """Infer tracks.json path from detections.json path."""
    return detections_path.replace("detections.json", "tracks.json")


def _output_exists(path: str) -> bool:
    if _is_tos(path):
        try:
            fs = _get_tosfs()
            return bool(fs.exists(path.removeprefix("tos://")))
        except Exception:
            return False
    return os.path.exists(path)


def _run_batch_local(
    args: argparse.Namespace,
    tracker_kwargs: dict,
    node_rank: int,
    world_size: int,
    num_workers: int,
) -> None:
    detections_base = args.detections_dir.rstrip("/")
    output_base = (args.output_dir or args.detections_dir).rstrip("/")
    crop_dir = args.crop_dir if args.crop else None
    bbox_expand = args.bbox_expand

    take_dir = os.path.join(detections_base, args.take)
    if not os.path.isdir(take_dir):
        raise FileNotFoundError(f"Take directory not found: {take_dir}")

    # Discover all (det_path, out_path) pairs
    all_tasks = []
    for cam in sorted(os.listdir(take_dir)):
        cam_dir = os.path.join(take_dir, cam)
        det_path = os.path.join(cam_dir, "detections.json")
        if not os.path.isfile(det_path):
            log.warning("Skip %s: no detections.json", cam_dir)
            continue
        out_dir = os.path.join(output_base, args.take, cam)
        out_path = os.path.join(out_dir, "tracks.json")
        all_tasks.append((det_path, out_path, cam))

    _run_batch_parallel(
        all_tasks, args.target_class, tracker_kwargs,
        crop_dir, bbox_expand, node_rank, world_size, num_workers,
        args.skip_existing,
    )


def _discover_tos_takes(tos_base: str) -> list[str]:
    """List all take directories under the TOS detections base.

    Returns sorted list of take names.
    """
    from utils.fsspec_util import list_tos_directory
    fs = _get_tosfs()

    items = list_tos_directory(tos_base, fs)
    takes = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "directory":
            continue
        take_name = item["name"].rstrip("/").split("/")[-1]
        takes.append(take_name)
    return sorted(takes)


def _run_batch_all_takes_tos(
    args: argparse.Namespace,
    tracker_kwargs: dict,
    node_rank: int,
    world_size: int,
    num_workers: int,
) -> None:
    """Discover all takes under detections-tos, flatten all (take, cam) tasks,
    then shard across nodes and run with a local thread pool."""
    tos_base = args.detections_tos.rstrip("/")
    out_base = (args.output_tos or args.detections_tos).rstrip("/")
    crop_dir = args.crop_dir if args.crop else None
    bbox_expand = args.bbox_expand

    takes = _discover_tos_takes(tos_base)
    if not takes:
        log.warning("No take directories found under %s", tos_base)
        return

    log.info("Discovered %d takes, scanning for cameras...", len(takes))

    # Flatten all (take, cam) tasks
    all_tasks: list[tuple[str, str, str, str]] = []  # (det_path, out_path, take, cam)
    for i, take_name in enumerate(takes):
        raw_tasks = _discover_tos_cams(tos_base, take_name)
        if not raw_tasks:
            continue
        for det_path, default_track_path in raw_tasks:
            if args.output_tos:
                cam = default_track_path.rstrip("/").split("/")[-2]
                track_path = f"{out_base}/{take_name}/{cam}/tracks.json"
            else:
                track_path = default_track_path
            cam = track_path.rstrip("/").split("/")[-2]
            all_tasks.append((det_path, track_path, take_name, cam))
        if (i + 1) % 100 == 0:
            log.info("  Scanned %d/%d takes, %d tasks so far", i + 1, len(takes), len(all_tasks))

    if not all_tasks:
        log.warning("No detections.json found across %d takes", len(takes))
        return

    log.info("Total: %d tasks across %d takes", len(all_tasks), len(takes))

    _run_batch_parallel_all(
        all_tasks, args.target_class, tracker_kwargs,
        crop_dir, bbox_expand, node_rank, world_size, num_workers,
        args.skip_existing,
    )


def _run_batch_tos(
    args: argparse.Namespace,
    tracker_kwargs: dict,
    node_rank: int,
    world_size: int,
    num_workers: int,
) -> None:
    tos_base = args.detections_tos.rstrip("/")
    out_base = (args.output_tos or args.detections_tos).rstrip("/")
    crop_dir = args.crop_dir if args.crop else None
    bbox_expand = args.bbox_expand

    raw_tasks = _discover_tos_cams(tos_base, args.take)
    if not raw_tasks:
        log.warning("No detections.json found under %s/%s/", tos_base, args.take)
        return

    log.info("Found %d camera(s) for %s on TOS", len(raw_tasks), args.take)

    all_tasks = []
    for det_path, default_track_path in raw_tasks:
        if args.output_tos:
            cam = default_track_path.rstrip("/").split("/")[-2]
            track_path = f"{out_base}/{args.take}/{cam}/tracks.json"
        else:
            track_path = default_track_path
        cam = track_path.rstrip("/").split("/")[-2]
        all_tasks.append((det_path, track_path, cam))

    _run_batch_parallel(
        all_tasks, args.target_class, tracker_kwargs,
        crop_dir, bbox_expand, node_rank, world_size, num_workers,
        args.skip_existing,
    )


def _run_batch_parallel(
    all_tasks: list[tuple[str, str, str]],
    target_class: int,
    tracker_kwargs: dict,
    crop_dir: str | None,
    bbox_expand: float,
    node_rank: int,
    world_size: int,
    num_workers: int,
    skip_existing: bool,
) -> None:
    """Shared batch executor with node sharding + local thread pool."""
    my_tasks = all_tasks[node_rank::world_size]
    if not my_tasks:
        log.info("Node %d/%d: no tasks (total=%d)", node_rank, world_size, len(all_tasks))
        return

    log.info("Node %d/%d: %d/%d tasks, %d workers",
             node_rank, world_size, len(my_tasks), len(all_tasks), num_workers)

    def _process_one(det_path: str, out_path: str, cam: str) -> tuple[str, bool]:
        if skip_existing and _output_exists(out_path):
            return cam, False
        try:
            process_detections_file(
                detections_path=det_path,
                output_path=out_path,
                target_class=target_class,
                crop_dir=crop_dir,
                bbox_expand=bbox_expand,
                **tracker_kwargs,
            )
            return cam, True
        except Exception:
            log.exception("[%s] FAILED", cam)
            return cam, False

    _run_with_progress(my_tasks, _process_one, node_rank, world_size, num_workers)


def _run_batch_parallel_all(
    all_tasks: list[tuple[str, str, str, str]],  # (det, out, take, cam)
    target_class: int,
    tracker_kwargs: dict,
    crop_dir: str | None,
    bbox_expand: float,
    node_rank: int,
    world_size: int,
    num_workers: int,
    skip_existing: bool,
) -> None:
    """Same as ``_run_batch_parallel`` but with ``take/cam`` labels."""
    my_tasks = all_tasks[node_rank::world_size]
    if not my_tasks:
        log.info("Node %d/%d: no tasks (total=%d)", node_rank, world_size, len(all_tasks))
        return

    log.info("Node %d/%d: %d/%d tasks, %d workers",
             node_rank, world_size, len(my_tasks), len(all_tasks), num_workers)

    def _process_one(det_path, out_path, take, cam):
        label = f"{take}/{cam}"
        if skip_existing and _output_exists(out_path):
            return label, False
        try:
            process_detections_file(
                detections_path=det_path,
                output_path=out_path,
                target_class=target_class,
                crop_dir=crop_dir,
                bbox_expand=bbox_expand,
                **tracker_kwargs,
            )
            return label, True
        except Exception:
            log.exception("[%s] FAILED", label)
            return label, False

    _run_with_progress(my_tasks, _process_one, node_rank, world_size, num_workers)


def _run_with_progress(
    tasks: list,
    process_fn,
    node_rank: int,
    world_size: int,
    num_workers: int,
) -> None:
    """Thread-pool executor with 30s progress logging."""
    total = len(tasks)
    state = {"done": 0, "failed": 0, "total": total}
    start_time = time.time()
    stop_event = threading.Event()

    def _progress_loop():
        while not stop_event.is_set():
            stop_event.wait(30.0)
            if stop_event.is_set():
                break
            completed = state["done"] + state["failed"]
            remaining = total - completed
            elapsed = time.time() - start_time
            if completed > 0:
                rate = completed / elapsed
                eta = (remaining / rate) if rate > 0 else 0
                eta_str = str(timedelta(seconds=int(eta)))
            else:
                eta_str = "…"
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            log.info("[Node %d/%d] Progress: %d/%d done (%d fail) | %d left | elapsed %s | ETA %s",
                     node_rank, world_size,
                     state["done"], total, state["failed"],
                     remaining, elapsed_str, eta_str)

    progress_thread = threading.Thread(target=_progress_loop, daemon=True)
    progress_thread.start()

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(process_fn, *task): task for task in tasks}
            for future in as_completed(futures):
                _, ok = future.result()
                if ok:
                    state["done"] += 1
                else:
                    state["failed"] += 1
    finally:
        stop_event.set()
        progress_thread.join(timeout=1)

    elapsed = str(timedelta(seconds=int(time.time() - start_time)))
    log.info("Node %d/%d done: %d ok, %d failed, total %d, elapsed %s",
             node_rank, world_size, state["done"], state["failed"], total, elapsed)


# ═══════════════════════════════════════════════════════════════════════
# Crop tracked persons from video
# ═══════════════════════════════════════════════════════════════════════


def _expand_bbox(x1: float, y1: float, x2: float, y2: float, scale: float = 1.2) -> tuple[float, float, float, float]:
    """Expand bbox around center by ``scale`` factor."""
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    hw = (x2 - x1) * scale / 2.0
    hh = (y2 - y1) * scale / 2.0
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def crop_persons_from_tracks(
    tracks: list[dict],
    detections_data: dict,
    crop_dir: str,
    track_ids: set[int] | None = None,
    max_frames: int | None = None,
    bbox_expand: float = 1.2,
) -> dict[str, int]:
    """Crop tracked person bboxes from the original video frames.

    Args:
        tracks: Per-frame track list from ``run_bytetrack``.
        detections_data: The original detections.json dict (contains ``video_path``).
        crop_dir: Directory root for cropped images (local or tos://).
        track_ids: If set, only crop these track_ids (None = all).
        max_frames: Limit video frames to read (None = all).
        bbox_expand: Scale factor for bbox expansion around center (1.0 = no expand).

    Returns:
        Dict mapping track_id → number of crops saved.
    """
    import cv2

    video_path = detections_data.get("video_path")
    if not video_path:
        log.warning("No video_path in detections metadata, skipping crop")
        return {}

    take_name = detections_data.get("take_name", "unknown")
    exo_cam = detections_data.get("exo_cam", "unknown")

    # ── Build frame → tracks lookup ──────────────────────────────────
    frame_tracks: dict[int, list[dict]] = {}
    for fr in tracks:
        frame_tracks[fr["frame_index"]] = fr["tracks"]

    needed_frames = set(frame_tracks.keys())
    if not needed_frames:
        log.warning("No tracked frames to crop")
        return {}

    max_needed = max(needed_frames)
    log.info("Cropping: %d frames with tracks, max frame=%d, expand=%.1f",
             len(needed_frames), max_needed, bbox_expand)

    # ── Read video and crop ───────────────────────────────────────────
    from src.embodiedgait.loader import iter_video_frames

    to_tos = _is_tos(crop_dir)
    fs = _get_tosfs() if to_tos else None
    base_dir = crop_dir.rstrip("/") + "/" + take_name + "/" + exo_cam
    if not to_tos:
        os.makedirs(base_dir, exist_ok=True)
    track_counts: dict[str, int] = {}

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]

    for frame_idx, frame in iter_video_frames(video_path, max_frames=max_frames or max_needed + 1):
        if frame_idx not in frame_tracks:
            continue

        h, w = frame.shape[:2]
        for t in frame_tracks[frame_idx]:
            tid = t["track_id"]
            if track_ids is not None and tid not in track_ids:
                continue

            # Expand bbox around center
            ex1, ey1, ex2, ey2 = _expand_bbox(*t["xyxy"], scale=bbox_expand)
            x1 = max(0, int(ex1)); y1 = max(0, int(ey1))
            x2 = min(w, int(ex2)); y2 = min(h, int(ey2))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            jpg_rel = f"{tid}/{frame_idx:06d}.jpg"

            if to_tos:
                ok, buf = cv2.imencode(".jpg", crop, encode_params)
                if ok:
                    jpg_path = f"{base_dir}/{jpg_rel}"
                    with fs.open(jpg_path.removeprefix("tos://"), "wb") as f:
                        f.write(buf.tobytes())
            else:
                tid_dir = os.path.join(base_dir, str(tid))
                os.makedirs(tid_dir, exist_ok=True)
                jpg_path = os.path.join(tid_dir, f"{frame_idx:06d}.jpg")
                cv2.imwrite(jpg_path, crop, encode_params)

            track_counts.setdefault(str(tid), 0)
            track_counts[str(tid)] += 1

        if frame_idx >= max_needed:
            break

    log.info("Cropped %d images across %d tracks → %s",
             sum(track_counts.values()), len(track_counts), base_dir)
    return track_counts


if __name__ == "__main__":
    main()
