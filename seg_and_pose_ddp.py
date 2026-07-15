"""Distributed segmentation & pose estimation with torch.distributed (DDP).

One GPU process = one worker.  Sequences are sharded across all ranks —
each rank processes its subset independently, no IPC needed beyond the
initial discovery broadcast.

Pipeline (fully in-memory, no local temp files):
  Prefetch thread  ──[numpy arrays]──▶  GPU (main thread)  ──[results]──▶  Upload thread
  (TOS read + decode)                  (seg + pose model)                (encode + TOS write)

Images are read directly from TOS/local into memory, fed to YOLO as numpy
arrays, and results (JSON + mask PNG) are written directly to TOS/local
without ever touching local disk.

Usage:
    python -m torch.distributed.launch \\
        --nproc_per_node $MLP_WORKER_GPU \\
        --master_addr $MLP_WORKER_0_HOST \\
        --node_rank $MLP_ROLE_INDEX \\
        --master_port $MLP_WORKER_0_PORT \\
        --nnodes $MLP_WORKER_NUM \\
        /path/to/seg_and_pose_ddp.py \\
        --input-tos tos://bucket/crops/sequences \\
        --output-tos tos://bucket/seg_pose/sequences \\
        --all-takes \\
        --seg-model /path/to/yolo26x-seg.pt \\
        --pose-model /path/to/yolo26x-pose.pt

Input structure (matches tracking.py --crop output):
    {input_dir}/{take}/{cam}/{track_id}/{frame_idx:06d}.jpg

Output structure:
    {output_dir}/{take}/{cam}/{track_id}/{frame_idx:06d}.json
    {output_dir}/{take}/{cam}/{track_id}/{frame_idx:06d}_mask.png
"""

import argparse
from collections import deque
import concurrent.futures
import json
import logging
import os
import posixpath
import queue
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.distributed as dist

from utils.fsspec_util import exists_with_fs, get_tosfs, open_with_fs

# ═══════════════════════════════════════════════════════════════════════════════
# Per-rank logger — only rank 0 logs at INFO, others at WARNING
# ═══════════════════════════════════════════════════════════════════════════════


def _get_rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))


RANK = _get_rank()
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


def setup_logging():
    level = logging.INFO if RANK == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format=f"%(asctime)s | rank{RANK} | %(message)s",
    )


setup_logging()
log = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_SEG_MODEL = "yolo26x-seg.pt"
DEFAULT_POSE_MODEL = "yolo26x-pose.pt"
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class BatchPrefetcher:
    """Read and decode a bounded number of image batches in the background.

    The queue is process-local: every torchrun rank creates one prefetcher for
    the images assigned to that rank.  This avoids cross-rank IPC and lets all
    GPU ranks consume concurrently from their own in-memory queues.
    """

    _END = object()

    def __init__(
        self,
        batches: list[list[tuple[str, str, str]]],
        fs: Any,
        *,
        max_prefetch_batches: int = 4,
        num_workers: int = 4,
        label: str = "",
    ) -> None:
        if max_prefetch_batches < 1:
            raise ValueError("max_prefetch_batches must be >= 1")
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        self._batches = batches
        self._fs = fs
        self._label = label
        self._num_workers = num_workers
        self._max_prefetch_batches = max_prefetch_batches
        self._queue: queue.Queue = queue.Queue(maxsize=max_prefetch_batches)
        # A slot covers both an in-flight read and a decoded queued batch.
        # The consumer releases it as soon as it takes that batch.
        self._slots = threading.Semaphore(max_prefetch_batches)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.wait_seconds = 0.0

    def _read_one(self, item: tuple[str, str, str]):
        img_uri, rel, out_uri = item
        try:
            with open_with_fs(img_uri, "rb", self._fs) as f:
                raw_bytes = f.read()
            img = cv2.imdecode(
                np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR
            )
            if img is None:
                raise ValueError(f"Failed to decode image: {img_uri}")
            return img, img.shape[:2], (img_uri, rel, out_uri)
        except Exception:
            log.exception("[%s] Prefetch failed: %s", self._label, img_uri)
            return None, None, (img_uri, rel, out_uri)

    def _put(self, item: Any) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self._num_workers,
                thread_name_prefix=f"prefetch-rank{RANK}",
            ) as pool:
                batch_iter = iter(self._batches)
                in_flight: deque[list[concurrent.futures.Future]] = deque()

                # Submit a full sliding window up front. Reads from different
                # batches can therefore run concurrently in the same pool.
                for _ in range(self._max_prefetch_batches):
                    try:
                        batch = next(batch_iter)
                    except StopIteration:
                        break
                    self._slots.acquire()
                    in_flight.append([
                        pool.submit(self._read_one, item) for item in batch
                    ])

                while in_flight and not self._stop.is_set():
                    futures = in_flight.popleft()
                    decoded = [future.result() for future in futures]
                    arrays = [item[0] for item in decoded]
                    shapes = [item[1] for item in decoded]
                    meta = [item[2] for item in decoded]
                    if not self._put((arrays, shapes, meta)):
                        return

                    try:
                        batch = next(batch_iter)
                    except StopIteration:
                        continue

                    # Wait until the GPU consumer removes one batch before
                    # admitting another read, bounding total prefetched RAM.
                    while not self._stop.is_set():
                        if self._slots.acquire(timeout=0.5):
                            in_flight.append([
                                pool.submit(self._read_one, item) for item in batch
                            ])
                            break
        except BaseException as exc:
            self._put(exc)
        finally:
            self._put(self._END)

    def __enter__(self):
        self._thread = threading.Thread(
            target=self._run,
            name=f"batch-prefetch-rank{RANK}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __iter__(self):
        if self._thread is None:
            raise RuntimeError("BatchPrefetcher must be used as a context manager")
        while True:
            wait_started = time.perf_counter()
            item = self._queue.get()
            self.wait_seconds += time.perf_counter() - wait_started
            if item is self._END:
                return
            if isinstance(item, BaseException):
                raise RuntimeError("Batch prefetch worker failed") from item
            self._slots.release()
            yield item

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


class SequenceProgressReporter:
    """Periodically report approximate rank-0 sequence progress."""

    def __init__(self, total: int, interval_seconds: float = 30.0) -> None:
        self._total = total
        self._interval = interval_seconds
        self._completed = 0
        self._current = "waiting"
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _duration(seconds: float | None) -> str:
        if seconds is None:
            return "calculating"
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _eta_duration(seconds: float | None) -> str:
        if seconds is None:
            return "计算中"
        # Round up so a positive sub-minute ETA is not shown as 0 minutes.
        total_minutes = max(0, int((seconds + 59) // 60))
        hours, minutes = divmod(total_minutes, 60)
        return f"{hours}小时{minutes}分钟"

    def start(self) -> None:
        if RANK != 0:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="sequence-progress",
            daemon=True,
        )
        self._thread.start()

    def set_current(self, label: str) -> None:
        with self._lock:
            self._current = label

    def sequence_done(self) -> None:
        with self._lock:
            self._completed += 1
            self._current = "waiting"

    def _snapshot(self):
        with self._lock:
            completed = self._completed
            current = self._current
        elapsed = time.monotonic() - self._started
        remaining = self._total - completed
        eta_seconds = (
            elapsed / completed * remaining if completed > 0 else None
        )
        finish_at = (
            datetime.now(timezone.utc) + timedelta(seconds=eta_seconds)
            if eta_seconds is not None else None
        )
        return completed, remaining, current, elapsed, eta_seconds, finish_at

    def _log(self, final: bool = False) -> None:
        completed, remaining, current, elapsed, eta_seconds, finish_at = self._snapshot()
        finish_text = finish_at.strftime("%Y-%m-%d %H:%M:%S UTC") if finish_at else "calculating"
        log.info(
            "Sequence progress%s (approx., rank 0): completed %d/%d, "
            "remaining %d, current %s | "
            "elapsed %s, 预计还需要 %s, estimated finish %s",
            " (final)" if final else "",
            completed, self._total, remaining, current,
            self._duration(elapsed), self._eta_duration(eta_seconds), finish_text,
        )

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._log()

    def stop(self) -> None:
        if RANK != 0:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._log(final=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TOS helpers
# ═══════════════════════════════════════════════════════════════════════════════


def is_tos_path(path: str) -> bool:
    return path.startswith("tos://") or path.startswith("s3://")


def normalize_tosfs_path(path: str) -> str:
    if path.startswith("tos://"):
        return path
    if path.startswith("s3://"):
        return "tos://" + path[len("s3://"):]
    return f"tos://{path.lstrip('/')}"


def strip_storage_prefix(path: str) -> str:
    if is_tos_path(path):
        body = path.split("://", 1)[1]
        return body.split("/", 1)[1] if "/" in body else ""
    return path.lstrip(os.sep)


def _ensure_local_output_dir(path: str) -> None:
    """Create parent dirs for a local output file."""
    if not is_tos_path(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Discovery
# ═══════════════════════════════════════════════════════════════════════════════


def _is_image_file(name: str) -> bool:
    return Path(name).suffix.lower() in SUPPORTED_IMAGE_EXTS


def _list_subdirs(fs, prefix: str) -> list[str]:
    """List subdirectory names under *prefix* using the given TOS filesystem."""
    normalized = prefix.rstrip("/") + "/"
    try:
        entries = fs.ls(normalized, detail=True)
    except FileNotFoundError:
        return []
    dirs = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") == "directory":
            name = entry["name"].rstrip("/").split("/")[-1]
            dirs.append(name)
    return sorted(dirs)


def discover_sequences_tos(input_base: str, take: str | None = None) -> list[dict]:
    fs = get_tosfs()
    if fs is None:
        raise RuntimeError("TOS filesystem unavailable — check VOLC_ACCESSKEY/VOLC_SECRETKEY")
    input_base = input_base.rstrip("/")
    sequences: list[dict] = []

    if take:
        takes_to_scan = [take]
    else:
        log.info("Listing takes under %s ...", input_base)
        takes_to_scan = _list_subdirs(fs, input_base)
        log.info("Found %d take(s)", len(takes_to_scan))

    for i_take, take_name in enumerate(takes_to_scan):
        take_prefix = f"{input_base}/{take_name}"
        if not take_prefix.startswith("tos://"):
            take_prefix = f"tos://{take_prefix}"
        sequences.append({
            "label": take_name,
            "take": take_name,
            "take_prefix": take_prefix,
        })

        if (i_take + 1) % 100 == 0:
            log.info("Discovery: scanned %d/%d takes, %d sequences so far",
                     i_take + 1, len(takes_to_scan), len(sequences))

    log.info("Discovery done: %d take-level sequences across %d takes",
             len(sequences), len(takes_to_scan))
    return sequences


def discover_sequences_local(input_base: str, take: str | None = None) -> list[dict]:
    input_base = os.path.abspath(input_base.rstrip("/"))
    sequences: list[dict] = []

    takes_to_scan = (
        [take] if take
        else sorted(
            d for d in os.listdir(input_base)
            if os.path.isdir(os.path.join(input_base, d))
        )
    )

    for take_name in takes_to_scan:
        take_dir = os.path.join(input_base, take_name)
        if not os.path.isdir(take_dir):
            log.warning("Take directory not found: %s", take_dir)
            continue
        sequences.append({
            "label": take_name,
            "take": take_name,
            "take_dir": take_dir,
        })

    return sequences


# ═══════════════════════════════════════════════════════════════════════════════
# Image enumeration (runs on each worker)
# ═══════════════════════════════════════════════════════════════════════════════


def _enumerate_images_local(cam_dir: str) -> list[str]:
    images: list[str] = []
    for root, _dirs, files in os.walk(cam_dir):
        for fname in sorted(files):
            if _is_image_file(fname):
                images.append(os.path.join(root, fname))
    return sorted(images)


def _enumerate_images_tos(fs, cam_prefix: str) -> list[str]:
    """Walk a TOS prefix recursively and collect all image URIs."""
    results: list[str] = []
    _walk_collect_images(fs, cam_prefix, results)
    return sorted(results)


def _walk_collect_images(fs, prefix: str, results: list[str]) -> None:
    try:
        entries = fs.ls(prefix.rstrip("/") + "/", detail=True)
    except FileNotFoundError:
        return
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name", "")
        etype = entry.get("type", "")
        if etype == "directory":
            _walk_collect_images(fs, name, results)
        elif etype == "file":
            if _is_image_file(name):
                if not name.startswith("tos://"):
                    name = f"tos://{name}"
                results.append(name)


# ═══════════════════════════════════════════════════════════════════════════════
# Data extraction
# ═══════════════════════════════════════════════════════════════════════════════


def _numpy_to_list(obj) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _numpy_to_list(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_numpy_to_list(v) for v in obj]
    return obj


def extract_person_segmentation(seg_results: list, min_confidence: float = 0.0) -> dict:
    persons = []
    for result in seg_results:
        if result.masks is None:
            continue
        boxes = result.boxes
        masks = result.masks
        for i in range(len(boxes)):
            if int(boxes.cls[i]) != 0:
                continue
            conf = float(boxes.conf[i])
            if conf < min_confidence:
                continue
            mask_xy = masks.xy[i] if masks.xy is not None and len(masks.xy) > i else []
            mask_xyn = masks.xyn[i] if masks.xyn is not None and len(masks.xyn) > i else []
            persons.append({
                "mask_xy": _numpy_to_list(mask_xy),
                "mask_xyn": _numpy_to_list(mask_xyn),
                "bbox": _numpy_to_list(boxes.xyxy[i]),
                "confidence": round(conf, 6),
                "class_id": 0,
            })
    return {"person_count": len(persons), "persons": persons}


def extract_pose_keypoints(pose_results: list, min_confidence: float = 0.0) -> dict:
    persons = []
    for result in pose_results:
        if result.keypoints is None:
            continue
        kpts = result.keypoints
        boxes = result.boxes
        for i in range(len(kpts)):
            if boxes is not None and i < len(boxes):
                conf = float(boxes.conf[i])
                if conf < min_confidence:
                    continue
                bbox = _numpy_to_list(boxes.xyxy[i])
            else:
                conf = 1.0
                bbox = []
            persons.append({
                "keypoints_xy": _numpy_to_list(kpts.xy[i]),
                "keypoints_xyn": _numpy_to_list(kpts.xyn[i]),
                "keypoints_data": _numpy_to_list(kpts.data[i]),
                "bbox": bbox,
                "confidence": round(conf, 6),
            })
    return {"person_count": len(persons), "persons": persons}


def _build_combined_mask(seg_results, fallback_shape: tuple | None = None) -> np.ndarray:
    """Build a combined person mask at the **original image resolution**.

    Uses mask contour polygons (``result.masks.xy``) drawn directly at
    ``orig_shape`` so that letterbox padding is handled correctly by the
    YOLO coordinate pipeline — no aspect-ratio-distorting ``cv2.resize``.
    """
    if seg_results is not None:
        for result in seg_results:
            if result.masks is None or result.masks.xy is None:
                continue
            boxes = result.boxes
            masks = result.masks
            orig_h, orig_w = result.orig_shape
            combined = np.zeros((orig_h, orig_w), dtype=np.uint8)
            for i in range(len(boxes)):
                if int(boxes.cls[i]) != 0:
                    continue
                if i < len(masks.xy):
                    pts = masks.xy[i]
                    if len(pts) >= 3:  # need at least a triangle
                        cv2.fillPoly(combined, [pts.astype(np.int32)], 1)
            return combined * 255
    h, w = fallback_shape or (640, 640)
    return np.zeros((h, w), dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# Path helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _relative_to_cam(img_uri: str, cam_uri: str) -> str:
    img_body = strip_storage_prefix(img_uri)
    cam_body = strip_storage_prefix(cam_uri).rstrip("/")
    if img_body.startswith(cam_body + "/"):
        return img_body[len(cam_body) + 1:]
    if img_body.startswith(cam_body):
        return img_body[len(cam_body):].lstrip("/")
    parts = img_body.split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else Path(img_body).name


def _output_uri(output_base: str, rel: str) -> str:
    rel_json = str(Path(rel).with_suffix(".json"))
    base = output_base.rstrip("/")
    if is_tos_path(base):
        return f"{base}/{rel_json}"
    return os.path.join(base, rel_json)


def _mask_output_uri(output_base: str, rel: str) -> str:
    p = Path(rel)
    rel_mask = str(p.parent / f"{p.stem}_mask.png") if p.parent != Path(".") else f"{p.stem}_mask.png"
    base = output_base.rstrip("/")
    if is_tos_path(base):
        return f"{base}/{rel_mask}"
    return os.path.join(base, rel_mask)


# ═══════════════════════════════════════════════════════════════════════════════
# Core per-rank processing — fully in-memory pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def process_sequences(
    sequences: list[dict],
    output_base: str,
    seg_model_path: str,
    pose_model_path: str,
    seg_conf: float,
    pose_conf: float,
    overwrite: bool,
    device: str,
    batch_size: int = 16,
    prefetch_batches: int = 4,
    prefetch_workers: int = 4,
    metadata_workers: int = 16,
    upload_workers: int = 4,
) -> dict:
    """Process all sequences assigned to this rank.

    Pipeline:  prefetch (I/O)  →  GPU (main thread)  →  upload (I/O)
    Each stage runs concurrently via queues — GPU is never idle waiting for I/O.

    Returns summary stats dict.
    """
    from ultralytics import YOLO

    need_tos = any(
        is_tos_path(
            s.get("take_prefix") or s.get("take_dir")
            or s.get("cam_prefix") or s.get("cam_dir", "")
        )
        for s in sequences
    ) or is_tos_path(output_base)
    fs = get_tosfs() if need_tos else None

    # ── Load models ────────────────────────────────────────────────────
    log.info("Loading seg model: %s", seg_model_path)
    seg_model = YOLO(seg_model_path)
    log.info("Loading pose model: %s", pose_model_path)
    pose_model = YOLO(pose_model_path)

    # ── Warmup (triggers fuse once, avoids stalling first batch) ──────
    log.info("Warming up models (fuse + first inference)...")
    dummy = np.zeros((128, 128, 3), dtype=np.uint8)
    seg_model(dummy, device=device, verbose=False)
    pose_model(dummy, device=device, verbose=False)
    log.info("Warmup done")

    total_processed = 0
    total_skipped = 0
    total_errors = 0
    total_seg_persons = 0
    total_pose_persons = 0
    start_time = time.time()
    progress = SequenceProgressReporter(len(sequences), interval_seconds=30.0)
    progress.start()

    def mark_sequence_done(
        label: str,
        processed: int,
        skipped: int,
        errors: int,
    ) -> tuple[int, int, int]:
        """Record rank-local progress without synchronizing GPU ranks."""
        progress.sequence_done()
        log.info(
            "[%s] sequence complete (approx., rank 0 shard) | "
            "%d processed, %d skipped, %d errors",
            label, processed, skipped, errors,
        )
        return processed, skipped, errors

    for seq_idx, seq in enumerate(sequences):
        input_uri = (
            seq.get("take_prefix") or seq.get("take_dir")
            or seq.get("cam_prefix") or seq.get("cam_dir")
        )
        seq_label = seq["label"]
        progress.set_current(seq_label)
        skipped_at_sequence_start = total_skipped
        scan_started = time.perf_counter()

        # ── Enumerate images ────────────────────────────────────────────
        if is_tos_path(input_uri):
            image_uris = _enumerate_images_tos(fs, input_uri)
        else:
            image_uris = _enumerate_images_local(input_uri)

        n_images_global = len(image_uris)
        # Shard at image granularity rather than sequence granularity. This
        # keeps every GPU busy even when there are fewer cameras than ranks.
        world_size = dist.get_world_size()
        image_uris = image_uris[RANK::world_size]
        n_images = len(image_uris)
        scan_seconds = time.perf_counter() - scan_started
        if n_images == 0:
            mark_sequence_done(seq_label, 0, 0, 0)
            continue

        # ── Output base for this sequence ───────────────────────────────
        if is_tos_path(output_base):
            seq_out_base = f"{output_base.rstrip('/')}/{seq['take']}"
        else:
            seq_out_base = os.path.join(output_base, seq["take"])

        # ── Pre-filter: skip already-processed images ──────────────────
        candidates = []
        for img_uri in image_uris:
            # At take level rel includes cam/track/image, preserving the
            # original directory hierarchy under output_base/take.
            rel = _relative_to_cam(img_uri, input_uri)
            out_uri = _output_uri(seq_out_base, rel)
            candidates.append((img_uri, rel, out_uri))

        filter_started = time.perf_counter()
        if overwrite:
            pending = candidates
        else:
            # Remote exists() is a network round trip. Run checks in parallel
            # so the GPU does not sit idle for one RTT per image.
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=metadata_workers,
                thread_name_prefix=f"exists-rank{RANK}",
            ) as pool:
                already_done = list(pool.map(
                    lambda item: exists_with_fs(item[2], fs), candidates
                ))
            pending = [
                item for item, exists in zip(candidates, already_done)
                if not exists
            ]
            total_skipped += sum(already_done)
        filter_seconds = time.perf_counter() - filter_started

        log.info(
            "[%s] rank shard %d/%d images (%d pending) | scan %.2fs, exists %.2fs",
            seq_label, n_images, n_images_global, len(pending),
            scan_seconds, filter_seconds,
        )

        if not pending:
            mark_sequence_done(
                seq_label, 0, total_skipped - skipped_at_sequence_start, 0
            )
            continue

        seq_processed = 0
        seq_errors = 0

        # ── Split into batches ──────────────────────────────────────────
        batches = [
            pending[i:i + batch_size]
            for i in range(0, len(pending), batch_size)
        ]

        # ── Pipeline: prefetch_q → GPU → upload_q ──────────────────────
        # Queue items:
        #   prefetch_q: list of (np_array | None, orig_shape | None, meta)
        #   upload_q:   (seg_list, pose_list, shape_list, meta_list)

        upload_q: queue.Queue = queue.Queue(
            maxsize=max(prefetch_batches, upload_workers * 2)
        )

        # Upload workers update these counters under one short critical section.
        upload_stats: dict = {"errors": 0, "seg_persons": 0, "pose_persons": 0, "processed": 0}
        upload_stats_lock = threading.Lock()

        # ── Upload thread: encode + write ──────────────────────────────
        def _upload_worker():
            while True:
                item = upload_q.get()
                if item is None:
                    upload_q.task_done()
                    break

                seg_batch, pose_batch, shapes, meta_list = item
                batch_errors = 0
                batch_seg_persons = 0
                batch_pose_persons = 0
                batch_processed = 0
                for i in range(len(meta_list)):
                    img_uri, rel, out_uri = meta_list[i]
                    seg_raw = seg_batch[i] if seg_batch and i < len(seg_batch) else None
                    pose_raw = pose_batch[i] if pose_batch and i < len(pose_batch) else None
                    fallback_shape = shapes[i] if shapes and i < len(shapes) else None

                    # ── Build & upload mask PNG ─────────────────────────
                    mask_img = _build_combined_mask(
                        [seg_raw] if seg_raw is not None else None,
                        fallback_shape=fallback_shape,
                    )
                    ok, mask_png_bytes = cv2.imencode(".png", mask_img)
                    if not ok:
                        batch_errors += 1
                        continue

                    mask_uri = _mask_output_uri(seq_out_base, rel)
                    try:
                        _ensure_local_output_dir(mask_uri)
                        with open_with_fs(mask_uri, "wb", fs) as mf:
                            mf.write(mask_png_bytes.tobytes())
                    except Exception:
                        log.exception("[%s] Upload mask failed: %s", seq_label, mask_uri)
                        batch_errors += 1

                    # ── Build & upload JSON ─────────────────────────────
                    seg_count = 0
                    if seg_raw is not None:
                        try:
                            seg = extract_person_segmentation([seg_raw], min_confidence=seg_conf)
                            seg_count = seg["person_count"]
                        except Exception:
                            seg = {"person_count": 0, "persons": [], "error": "seg_failed"}
                    else:
                        seg = {"person_count": 0, "persons": [], "error": "seg_failed"}

                    pose_count = 0
                    if pose_raw is not None:
                        try:
                            ps = extract_pose_keypoints([pose_raw], min_confidence=pose_conf)
                            pose_count = ps["person_count"]
                        except Exception:
                            ps = {"person_count": 0, "persons": [], "error": "pose_failed"}
                    else:
                        ps = {"person_count": 0, "persons": [], "error": "pose_failed"}

                    output = {"image_uri": img_uri, "segmentation": seg, "pose": ps}
                    json_bytes = json.dumps(output, ensure_ascii=False).encode("utf-8")
                    try:
                        _ensure_local_output_dir(out_uri)
                        with open_with_fs(out_uri, "wb", fs) as jf:
                            jf.write(json_bytes)
                    except Exception:
                        log.exception("[%s] Upload json failed: %s", seq_label, out_uri)
                        batch_errors += 1

                    batch_seg_persons += seg_count
                    batch_pose_persons += pose_count
                    batch_processed += 1

                with upload_stats_lock:
                    upload_stats["errors"] += batch_errors
                    upload_stats["seg_persons"] += batch_seg_persons
                    upload_stats["pose_persons"] += batch_pose_persons
                    upload_stats["processed"] += batch_processed

                upload_q.task_done()

        # ── Launch I/O threads ──────────────────────────────────────────
        upload_threads = [
            threading.Thread(
                target=_upload_worker,
                name=f"upload-rank{RANK}-{worker_id}",
                daemon=True,
            )
            for worker_id in range(upload_workers)
        ]
        for upload_thread in upload_threads:
            upload_thread.start()

        gpu_seconds = 0.0
        upload_queue_wait_seconds = 0.0
        prefetch_wait_seconds = 0.0

        # ── Main thread: GPU inference ─────────────────────────────────
        with BatchPrefetcher(
            batches,
            fs,
            max_prefetch_batches=prefetch_batches,
            num_workers=prefetch_workers,
            label=seq_label,
        ) as prefetcher:
            for arrays, shapes, meta_list in prefetcher:

                # Find valid indices (images that decoded successfully)
                valid = [(j, arrays[j]) for j in range(len(arrays)) if arrays[j] is not None]

                if not valid:
                    # All images in batch failed to decode
                    seg_full = [None] * len(arrays)
                    pose_full = [None] * len(arrays)
                    upload_q.put((seg_full, pose_full, shapes, meta_list))
                    continue

                valid_indices, valid_arrays = zip(*valid)  # type: ignore[arg-type]
                valid_arrays_list = list(valid_arrays)

                # ── Segmentation ────────────────────────────────────────
                gpu_started = time.perf_counter()
                try:
                    seg_batch_raw = seg_model(valid_arrays_list, device=device, verbose=False)
                except Exception:
                    log.exception("[%s] Batch seg failed (batch)", seq_label)
                    seg_batch_raw = [None] * len(valid_arrays_list)

                # ── Pose ────────────────────────────────────────────────
                try:
                    pose_batch_raw = pose_model(valid_arrays_list, device=device, verbose=False)
                except Exception:
                    log.exception("[%s] Batch pose failed (batch)", seq_label)
                    pose_batch_raw = [None] * len(valid_arrays_list)
                gpu_seconds += time.perf_counter() - gpu_started

                # Map back: valid indices → full-batch-length lists
                seg_full = [None] * len(arrays)
                pose_full = [None] * len(arrays)
                for k, orig_idx in enumerate(valid_indices):
                    seg_full[orig_idx] = seg_batch_raw[k] if seg_batch_raw and k < len(seg_batch_raw) else None
                    pose_full[orig_idx] = pose_batch_raw[k] if pose_batch_raw and k < len(pose_batch_raw) else None

                put_started = time.perf_counter()
                upload_q.put((seg_full, pose_full, shapes, meta_list))
                upload_queue_wait_seconds += time.perf_counter() - put_started
            prefetch_wait_seconds = prefetcher.wait_seconds

        # ── Signal upload to finish & wait ─────────────────────────────
        for _ in upload_threads:
            upload_q.put(None)
        for upload_thread in upload_threads:
            upload_thread.join()

        # ── Accumulate stats ────────────────────────────────────────────
        seq_processed = upload_stats["processed"]
        seq_errors = upload_stats["errors"]
        seq_seg_persons = upload_stats["seg_persons"]
        seq_pose_persons = upload_stats["pose_persons"]

        total_processed += seq_processed
        total_skipped += 0  # already counted during pre-filter
        total_errors += seq_errors
        total_seg_persons += seq_seg_persons
        total_pose_persons += seq_pose_persons

        rank_processed, rank_skipped, rank_errors = mark_sequence_done(
            seq_label,
            seq_processed,
            total_skipped - skipped_at_sequence_start,
            seq_errors,
        )

        # ── Per-sequence log ────────────────────────────────────────────
        elapsed = time.time() - start_time
        rate = total_processed / elapsed if elapsed > 0 else 0
        log.info(
            "[%s] %d/%d seq done | rank0 shard %d processed, %d skipped (%d err) | "
            "GPU %.2fs, prefetch wait %.2fs, write-queue wait %.2fs | "
            "rank0 throughput %d imgs @ %.1f imgs/s",
            seq_label, seq_idx + 1, len(sequences),
            rank_processed, rank_skipped, rank_errors,
            gpu_seconds, prefetch_wait_seconds, upload_queue_wait_seconds,
            total_processed, rate,
        )

    progress.stop()
    return {
        "rank": RANK,
        "n_sequences": len(sequences),
        "n_processed": total_processed,
        "n_skipped": total_skipped,
        "n_errors": total_errors,
        "total_seg_persons": total_seg_persons,
        "total_pose_persons": total_pose_persons,
        "elapsed_seconds": time.time() - start_time,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DDP orchestration
# ═══════════════════════════════════════════════════════════════════════════════


def init_distributed() -> tuple[int, int, str]:
    """Initialize torch.distributed, return (global_rank, world_size, device)."""
    if "RANK" not in os.environ:
        # Single-process fallback (for testing without torchrun)
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")

    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = f"cuda:{local_rank}"
    return global_rank, world_size, device


def broadcast_sequences(sequences: list[dict], world_size: int) -> list[dict]:
    """Rank 0 broadcasts the full sequence list; all ranks return the full list."""
    obj_list = [sequences] if RANK == 0 else [None]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]


def shard_sequences(sequences: list[dict], rank: int, world_size: int) -> list[dict]:
    """Each rank gets its share: sequences[rank::world_size]."""
    return sequences[rank::world_size]


def gather_and_print_summary(rank_summary: dict, world_size: int):
    """Gather per-rank summaries to rank 0 and print."""
    summaries = [None] * world_size
    dist.all_gather_object(summaries, rank_summary)

    if RANK != 0:
        return

    total_processed = sum(s["n_processed"] for s in summaries if s)
    total_skipped = sum(s["n_skipped"] for s in summaries if s)
    total_errors = sum(s["n_errors"] for s in summaries if s)
    total_seg = sum(s["total_seg_persons"] for s in summaries if s)
    total_pose = sum(s["total_pose_persons"] for s in summaries if s)
    total_seqs = sum(s["n_sequences"] for s in summaries if s)

    log.info(
        "===== DDP summary: %d ranks, %d sequences =====\n"
        "  Images: %d processed, %d skipped, %d errors\n"
        "  Seg persons: %d, Pose persons: %d",
        world_size, total_seqs,
        total_processed, total_skipped, total_errors,
        total_seg, total_pose,
    )
    print(json.dumps({
        "status": "summary",
        "n_ranks": world_size,
        "n_sequences": total_seqs,
        "n_processed": total_processed,
        "n_skipped": total_skipped,
        "n_errors": total_errors,
        "total_seg_persons": total_seg,
        "total_pose_persons": total_pose,
        "per_rank": summaries,
    }, ensure_ascii=False), flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="DDP YOLO segmentation + pose estimation on GPU (in-memory pipeline)"
    )

    # ── Models ──────────────────────────────────────────────────────────
    parser.add_argument("--seg-model", type=str, default=DEFAULT_SEG_MODEL)
    parser.add_argument("--pose-model", type=str, default=DEFAULT_POSE_MODEL)
    parser.add_argument("--seg-conf", type=float, default=0.25)
    parser.add_argument("--pose-conf", type=float, default=0.25)

    # ── Input / output ──────────────────────────────────────────────────
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/seg_pose")
    parser.add_argument("--input-tos", type=str, default=None)
    parser.add_argument("--output-tos", type=str, default=None)

    # ── Take selection ──────────────────────────────────────────────────
    parser.add_argument("--take", type=str, default=None)
    parser.add_argument("--all-takes", action="store_true")

    # ── Misc ────────────────────────────────────────────────────────────
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for GPU inference (default: 16)")
    parser.add_argument("--prefetch-batches", type=int, default=4,
                        help="Decoded batches buffered in RAM per rank (default: 4)")
    parser.add_argument("--prefetch-workers", type=int, default=4,
                        help="Parallel image read/decode workers per rank (default: 4)")
    parser.add_argument("--metadata-workers", type=int, default=16,
                        help="Parallel output existence checks per rank (default: 16)")
    parser.add_argument("--upload-workers", type=int, default=4,
                        help="Parallel result encode/write workers per rank (default: 4)")

    args = parser.parse_args()

    # ── Validate & unify input path ─────────────────────────────────────
    if not args.input_dir and not args.input_tos:
        parser.error("Provide --input-dir or --input-tos")
    if not args.take and not args.all_takes:
        parser.error("Provide --take or --all-takes")
    for name in (
        "batch_size", "prefetch_batches", "prefetch_workers",
        "metadata_workers", "upload_workers",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")

    # Auto-detect: if --input-dir is a tos:// path, treat it as TOS
    input_src = args.input_tos or args.input_dir
    use_tos = is_tos_path(input_src)

    # ── Resolve output ──────────────────────────────────────────────────
    if use_tos:
        output_base = args.output_tos or (
            args.output_dir if args.output_dir != "outputs/seg_pose"
            else input_src.replace("/crops/", "/seg_pose/")
        )
    else:
        output_base = args.output_dir

    # ── Init DDP ────────────────────────────────────────────────────────
    global_rank, world_size, device = init_distributed()

    # ── Discovery (rank 0 only) ─────────────────────────────────────────
    if global_rank == 0:
        if use_tos:
            sequences = discover_sequences_tos(input_src, take=args.take)
        else:
            sequences = discover_sequences_local(input_src, take=args.take)
        log.info("Discovered %d sequences total", len(sequences))
    else:
        sequences = []

    # Broadcast to all ranks
    sequences = broadcast_sequences(sequences, world_size)

    # Every rank visits every sequence; process_sequences shards its image list
    # as image_uris[rank::world_size] for finer-grained load balancing.
    my_sequences = sequences
    log.info(
        "Assigned image shards from %d sequences (rank %d/%d)",
        len(my_sequences), global_rank, world_size,
    )

    # ── Process ─────────────────────────────────────────────────────────
    log.info("Processing %d sequences on %s (in-memory pipeline)", len(my_sequences), device)
    rank_summary = process_sequences(
        sequences=my_sequences,
        output_base=output_base,
        seg_model_path=args.seg_model,
        pose_model_path=args.pose_model,
        seg_conf=args.seg_conf,
        pose_conf=args.pose_conf,
        overwrite=args.overwrite,
        device=device,
        batch_size=args.batch_size,
        prefetch_batches=args.prefetch_batches,
        prefetch_workers=args.prefetch_workers,
        metadata_workers=args.metadata_workers,
        upload_workers=args.upload_workers,
    )

    log.info("Rank %d done: %d imgs processed, %d skip, %d err",
             global_rank, rank_summary["n_processed"],
             rank_summary["n_skipped"], rank_summary["n_errors"])

    dist.barrier()
    gather_and_print_summary(rank_summary, world_size)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
