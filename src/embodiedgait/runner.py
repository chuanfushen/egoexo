"""Multi-GPU / Multi-node batch runner.

Supports two modes:
  1. torchrun (multi-node): launched via torchrun, each process handles 1 GPU
  2. Local (single-node): uses mp.spawn for multi-GPU, or runs inline for 1 GPU

Usage:
    # Single node, multi-GPU:
    uv run python -m src.embodiedgait.runner --all-takes --cams cam01 cam02

    # Multi-node via torchrun:
    torchrun --nnodes=2 --nproc_per_node=4 \
        --master_addr=10.0.0.1 --master_port=29500 \
        -m src.embodiedgait.runner --all-takes --cams cam01 cam02 cam03 cam04
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from typing import Any

import torch

from utils.fsspec_util import get_tosfs, list_tos_directory

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


# ── Spawn worker (module-level, picklable) ────────────────────────


def _spawn_worker(
    local_rank: int,
    global_rank: int,
    my_tasks: list[tuple[str, str]],
    result_queue,
    model_path: str,
    output_dir: str,
    max_frames: int | None,
    prompt: str,
    crop_tos_base: str | None = None,
    crop_padding: int = 20,
    use_yolo: bool = False,
    yolo_model_path: str = "yolo26n.pt",
    yolo_conf: float = 0.25,
    track_overlap: float = 0.2,
) -> None:
    """Top-level entry for mp.spawn processes. Must be picklable."""
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    log.info("[GPU %d] global_rank=%d device=%s tasks=%d",
             local_rank, global_rank, device, len(my_tasks))
    results = _process_tasks(my_tasks, device, model_path, output_dir,
                             max_frames, prompt, crop_tos_base, crop_padding,
                             use_yolo, yolo_model_path, yolo_conf, track_overlap)
    result_queue.put((global_rank, results))


# ── Single video processing (per-process) ──────────────────────────


def _process_tasks(
    tasks: list[tuple[str, str]],
    device: str,
    model_path: str,
    output_dir: str,
    max_frames: int | None,
    prompt: str,
    crop_tos_base: str | None = None,
    crop_padding: int = 20,
    use_yolo: bool = False,
    yolo_model_path: str = "yolo26n.pt",
    yolo_conf: float = 0.25,
    track_overlap: float = 0.2,
) -> list[dict]:
    """Process a list of (take_name, cam_id) tasks on a single GPU."""
    from src.embodiedgait.tracking_pipeline import process_video, process_video_yolo

    if use_yolo:
        results = []
        for i, (take_name, exo_cam) in enumerate(tasks):
            log.info("[%s] [%d/%d] %s/%s", device, i + 1, len(tasks), take_name, exo_cam)
            try:
                result = process_video_yolo(
                    take_name=take_name, exo_cam=exo_cam,
                    yolo_model_path=yolo_model_path, output_dir=output_dir,
                    max_frames=max_frames, device=device,
                    crop_tos_base=crop_tos_base, crop_padding=crop_padding,
                    yolo_conf=yolo_conf, track_overlap_threshold=track_overlap,
                )
                results.append(result)
                stats = result.get("stats") or {}
                log.info("[%s] -> ego-frames=%d cropped=%d tracks=%d", device,
                         stats.get("n_frames_with_detections", 0),
                         stats.get("n_cropped", 0),
                         stats.get("n_tracks_total", 0))
            except Exception as e:
                log.error("[%s] FAILED %s/%s: %s", device, take_name, exo_cam, e)
                traceback.print_exc()
                results.append({
                    "take_name": take_name, "exo_cam": exo_cam,
                    "error": str(e), "stats": {"n_frames_processed": 0},
                })
        return results

    # Original SAM3 pipeline
    from src.embodiedgait.detection import PersonDetector

    detector = PersonDetector(ckpt_path=model_path)
    results = []
    for i, (take_name, exo_cam) in enumerate(tasks):
        log.info("[%s] [%d/%d] %s/%s", device, i + 1, len(tasks), take_name, exo_cam)
        try:
            result = process_video(
                take_name=take_name, exo_cam=exo_cam,
                model_path=model_path, output_dir=output_dir,
                max_frames=max_frames, prompt=prompt, detector=detector,
                crop_tos_base=crop_tos_base,
                crop_padding=crop_padding,
            )
            results.append(result)
            stats = result.get("stats") or {}
            log.info("[%s] -> detections=%d cropped=%d", device,
                     stats.get("n_frames_with_detections", 0),
                     stats.get("n_cropped", 0))
        except Exception as e:
            log.error("[%s] FAILED %s/%s: %s", device, take_name, exo_cam, e)
            traceback.print_exc()
            results.append({
                "take_name": take_name, "exo_cam": exo_cam,
                "error": str(e), "stats": {"n_frames_processed": 0},
            })
    return results


# ── torchrun entry point ───────────────────────────────────────────


def _run_torchrun(
    tasks: list[tuple[str, str]],
    args,
) -> None:
    """Entry point when launched via torchrun.

    torchrun sets: RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_WORLD_SIZE.
    Each process handles tasks[rank::world_size] on GPU local_rank.
    """
    import torch.distributed as dist

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # Each global rank gets tasks via round-robin
    my_tasks = tasks[rank::world_size]
    log.info("[torchrun] rank=%d/%d local_rank=%d device=%s tasks=%d",
             rank, world_size, local_rank, device, len(my_tasks))

    if not my_tasks:
        log.info("[torchrun] rank=%d: no tasks assigned", rank)
        dist.barrier()
        dist.destroy_process_group()
        return

    results = _process_tasks(my_tasks, device, args.model, args.output_dir,
                             args.max_frames, args.prompt,
                                 args.crop, args.crop_padding,
                                 args.yolo, args.yolo_model,
                                 args.yolo_conf, args.track_overlap)

    # Barrier: wait for all ranks to finish before cleanup
    dist.barrier()
    dist.destroy_process_group()

    # Only rank 0 writes summary
    if rank == 0:
        _write_summary(results, tasks, args.output_dir, world_size)


# ── Local (single-node) entry points ───────────────────────────────


def _run_local(
    tasks: list[tuple[str, str]],
    args,
) -> None:
    """Single-node: use mp.spawn for multi-GPU, or inline for 1 GPU."""
    num_gpus = args.num_gpus
    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    vpg = args.videos_per_gpu
    total_workers = max(1, num_gpus) * vpg

    if total_workers <= 1:
        # Single worker: run inline
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        results = _process_tasks(tasks, device, args.model, args.output_dir,
                                 args.max_frames, args.prompt,
                                 args.crop, args.crop_padding,
                                 args.yolo, args.yolo_model,
                                 args.yolo_conf, args.track_overlap)
        _write_summary(results, tasks, args.output_dir, num_gpus)
        return

    # Multi-worker: spawn subprocesses
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    processes = []
    for w in range(total_workers):
        local_gpu = w % max(1, num_gpus)  # which physical GPU
        my_tasks = tasks[w::total_workers]
        p = ctx.Process(
            target=_spawn_worker,
            args=(local_gpu, w, my_tasks, result_queue,
                  args.model, args.output_dir, args.max_frames,
                  args.prompt, args.crop, args.crop_padding,
                  args.yolo, args.yolo_model,
                  args.yolo_conf, args.track_overlap),
        )
        p.start()
        processes.append(p)

    # Collect results
    all_results = []
    for _ in range(total_workers):
        r, results = result_queue.get()
        all_results.extend(results)

    for p in processes:
        p.join()

    _write_summary(all_results, tasks, args.output_dir, num_gpus)


# ── Summary ────────────────────────────────────────────────────────


def _write_summary(
    results: list[dict],
    all_tasks: list,
    output_dir: str,
    num_gpus: int,
) -> None:
    ok = sum(1 for r in results if "error" not in r)
    fail = sum(1 for r in results if "error" in r)
    log.info("=" * 60)
    log.info("Complete: %d OK, %d failed, %d total (%d GPUs used)",
             ok, fail, len(all_tasks), num_gpus)

    summary_path = os.path.join(output_dir, "batch_summary.json")
    summary = {
        "n_total": len(all_tasks),
        "n_ok": ok,
        "n_failed": fail,
        "results": [
            {
                "take_name": r.get("take_name", "?"),
                "exo_cam": r.get("exo_cam", "?"),
                "n_frames": (r.get("stats") or {}).get("n_frames_processed", 0),
                "n_detections": (r.get("stats") or {}).get("n_frames_with_detections", 0),
                "n_cropped": (r.get("stats") or {}).get("n_cropped", 0),
                "error": r.get("error"),
            }
            for r in results
        ],
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary: %s", summary_path)


# ── Task discovery ─────────────────────────────────────────────────


def _build_capture_camera_map(fs=None) -> dict[str, list[str]]:
    """Build mapping from capture_name to list of exo camera IDs from captures.json."""
    import re
    if fs is None:
        fs = get_tosfs()
    BASE = "tos://drobotics-ailab/users/chuanfu.shen/data/egoexo4d-defaults"
    captures_path = f"{BASE}/captures.json"
    with fs.open(captures_path) as f:
        captures = json.load(f)
    cam_map = {}
    for cap in captures:
        capture_name = cap["capture_name"]
        exo_cams = [c["cam_id"] for c in cap.get("cameras", [])
                    if not c.get("is_ego", False)]
        cam_map[capture_name] = exo_cams
    return cam_map


def _take_to_capture_name(take_name: str) -> str:
    r"""Derive capture_name from take_name by removing trailing _\d+ suffix.

    Examples:
        cmu_bike01_2 → cmu_bike01
        cmu_bike01 → cmu_bike01
    """
    import re
    # Remove trailing _\d+ (e.g., _2, _3)
    # Some takes may have multiple suffixes like _2_1
    parts = take_name.rsplit("_", 1)
    return parts[0] if len(parts) > 1 and parts[1].isdigit() else take_name


def discover_all_takes(fs=None) -> list[str]:
    """Scan TOS to find all available take names."""
    if fs is None:
        fs = get_tosfs()
    BASE = "tos://drobotics-ailab/users/chuanfu.shen/data/egoexo4d-defaults"
    takes_dir = f"{BASE}/takes/"
    log.info("Scanning takes directory...")
    try:
        items = list_tos_directory(takes_dir, fs)
    except Exception as e:
        log.error("Failed to list takes: %s", e)
        return []
    take_names = []
    for item in items:
        name = item.get("name", "") if isinstance(item, dict) else str(item)
        basename = name.rstrip("/").split("/")[-1]
        if basename and not basename.startswith("."):
            take_names.append(basename)
    log.info("Found %d takes", len(take_names))
    return sorted(take_names)


def discover_tasks(
    take_names: list[str] | None = None,
    exo_cams: list[str] | None = None,
    take_list_file: str | None = None,
    all_takes: bool = False,
    auto_cams: bool = False,
) -> list[tuple[str, str]]:
    """Generate (take_name, cam_id) tasks.

    Args:
        take_names: Explicit list of take names.
        exo_cams: Explicit list of camera IDs (ignored if auto_cams=True).
        take_list_file: File with one take name per line.
        all_takes: Scan TOS for all available takes.
        auto_cams: If True, discover cameras per take from captures.json
                   instead of using the fixed --cams list.
    """
    if all_takes:
        take_names = discover_all_takes()
    if take_list_file:
        with open(take_list_file) as f:
            take_names = [line.strip() for line in f if line.strip()]
    if take_names is None:
        raise ValueError("Provide --takes, --take-list, or --all-takes")

    if auto_cams:
        cam_map = _build_capture_camera_map()
        tasks = []
        for take in take_names:
            cap_name = _take_to_capture_name(take)
            cams = cam_map.get(cap_name, [])
            if not cams:
                log.warning("No exo cameras found for take=%s (capture=%s), skipping",
                            take, cap_name)
                continue
            for cam in cams:
                tasks.append((take, cam))
        log.info("Auto-discovered %d tasks across %d takes (%d unique cameras)",
                 len(tasks), len(take_names),
                 len(set(cam for _, cam in tasks)))
        return tasks

    if exo_cams is None:
        exo_cams = ["cam01"]
    tasks = []
    for take in take_names:
        for cam in exo_cams:
            tasks.append((take, cam))
    return tasks


# ── Main ───────────────────────────────────────────────────────────


def _get_platform_info() -> dict | None:
    """Detect Volcano Cloud ML platform and return distributed config.

    Platform env vars:
        MLP_WORKER_0_HOST / MLP_WORKER_0_PORT  — master node
        MLP_ROLE_INDEX   — this node's index (0, 1, 2, ...)
        MLP_WORKER_NUM   — total number of nodes
        MLP_WORKER_GPU   — GPUs per node
    """
    if "MLP_ROLE_INDEX" not in os.environ:
        return None

    return {
        "node_rank": int(os.environ["MLP_ROLE_INDEX"]),
        "num_nodes": int(os.environ["MLP_WORKER_NUM"]),
        "gpus_per_node": int(os.environ["MLP_WORKER_GPU"]),
        "master_addr": os.environ.get("MLP_WORKER_0_HOST", "localhost"),
        "master_port": int(os.environ.get("MLP_WORKER_0_PORT", "29500")),
        "world_size": int(os.environ["MLP_WORKER_NUM"]) * int(os.environ["MLP_WORKER_GPU"]),
    }


def _run_platform(
    tasks: list[tuple[str, str]],
    args,
    pinfo: dict,
) -> None:
    """Entry point for Volcano Cloud ML platform.

    Each node independently spawns its own GPU processes.
    No cross-node communication needed — tasks are statically partitioned
    by global rank.
    """
    import torch.multiprocessing as mp

    node_rank = pinfo["node_rank"]
    gpus_per_node = pinfo["gpus_per_node"]
    world_size = pinfo["world_size"]

    vpg = args.videos_per_gpu
    total_workers = gpus_per_node * vpg  # workers on this node
    effective_world_size = world_size * vpg  # global task partition granularity

    log.info("[Platform] node %d/%d, %d GPUs × %d videos = %d workers/node, "
             "effective_world=%d",
             node_rank, pinfo["num_nodes"], gpus_per_node, vpg,
             total_workers, effective_world_size)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    processes = []
    for w in range(total_workers):
        local_gpu = w % gpus_per_node  # which physical GPU
        global_r = node_rank * total_workers + w  # global worker index
        my_tasks = tasks[global_r::effective_world_size]
        p = ctx.Process(
            target=_spawn_worker,
            args=(local_gpu, global_r, my_tasks, result_queue,
                  args.model, args.output_dir, args.max_frames,
                  args.prompt, args.crop, args.crop_padding,
                  args.yolo, args.yolo_model,
                  args.yolo_conf, args.track_overlap),
        )
        p.start()
        processes.append(p)

    # Collect results from this node
    node_results = []
    for _ in range(total_workers):
        rank, results = result_queue.get()
        node_results.extend(results)

    for p in processes:
        p.join()

    # Each node writes its own results (shared FS assumed)
    os.makedirs(args.output_dir, exist_ok=True)
    node_path = os.path.join(args.output_dir, f"node_{node_rank}_results.json")
    with open(node_path, "w") as f:
        json.dump({"node_rank": node_rank, "results": node_results}, f, indent=2)
    ok = sum(1 for r in node_results if "error" not in r)
    log.info("[Node %d] done: %d OK, %d failed → %s", node_rank, ok, len(node_results) - ok, node_path)


def main():
    parser = argparse.ArgumentParser(description="Multi-node GPU batch video processing")
    parser.add_argument("--takes", nargs="*", default=None)
    parser.add_argument("--all-takes", action="store_true")
    parser.add_argument("--cams", nargs="*", default=["cam01"],
                        help="Exo camera IDs (ignored if --auto-cams)")
    parser.add_argument("--auto-cams", action="store_true",
                        help="Auto-discover all exo cameras per take from captures.json")
    parser.add_argument("--take-list", type=str, default=None)
    parser.add_argument("--model", type=str,
                        default="/LargeModelDev/users/chuanfu.shen/ckpts/sam3/sam3.pt")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--prompt", type=str, default="person",
                        help="SAM3 text prompt (default: person)")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="GPUs to use (local mode, default: all)")
    parser.add_argument("--videos-per-gpu", type=int, default=1,
                        help="Concurrent videos per GPU (default: 1)")
    parser.add_argument("--crop", type=str, default=None,
                        help="TOS base path for cropped JPGs, e.g. tos://bucket/egoexogait")
    parser.add_argument("--crop-padding", type=int, default=20,
                        help="Extra pixels around bbox (default: 20)")

    # YOLO tracking options
    parser.add_argument("--yolo", action="store_true",
                        help="Use YOLO + ByteTrack tracking instead of SAM3")
    parser.add_argument("--yolo-model", type=str, default="yolo26n.pt",
                        help="YOLO model path (default: yolo26n.pt)")
    parser.add_argument("--yolo-conf", type=float, default=0.25,
                        help="YOLO confidence threshold (default: 0.25)")
    parser.add_argument("--track-overlap", type=float, default=0.2,
                        help="Min overlap ratio for ego-track selection (default: 0.2)")
    args = parser.parse_args()

    tasks = discover_tasks(
        take_names=args.takes, exo_cams=args.cams,
        take_list_file=args.take_list, all_takes=args.all_takes,
        auto_cams=args.auto_cams,
    )
    log.info("Total tasks: %d videos", len(tasks))

    # 1. Volcano Cloud platform mode
    pinfo = _get_platform_info()
    if pinfo:
        _run_platform(tasks, args, pinfo)
        return

    # 2. torchrun multi-node mode
    if "LOCAL_RANK" in os.environ:
        _run_torchrun(tasks, args)
        return

    # 3. Local single-node mode
    _run_local(tasks, args)


if __name__ == "__main__":
    main()
