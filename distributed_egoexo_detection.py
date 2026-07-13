"""Distributed batched YOLO detection over the EgoExo4D dataset.

The unit of distributed work is one (take, Exo camera) video. Each torchrun
rank loads YOLO once, processes its assigned videos with batched predict, and
uploads JSON directly to:

    <output-tos>/sequences/<take>/<camera>/detections.json

No tracking, ego projection, frame undistortion, or rendered MP4 is performed.

Example:
    torchrun --standalone --nproc-per-node=8 distributed_egoexo_detection.py \
        --takes cmu_soccer12_2 --auto-cams \
        --output-tos tos://bucket/path/egoexo_detections \
        --model yolo26x.pt --batch-size 64
"""

import argparse
import json
import os
import queue
import shutil
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import cv2
import torch
import torch.distributed as dist
from ultralytics import YOLO

from src.embodiedgait.loader import BASE_PATH
from src.embodiedgait.runner import discover_all_takes
from utils.fsspec_util import get_tosfs, list_tos_directory, open_with_fs


@dataclass
class PreparedTask:
    take_name: str
    camera: str
    result_tos_path: str
    video_path: str | None = None
    local_video: str | None = None
    skipped: bool = False


def init_distributed(timeout_minutes: int) -> tuple[int, int, int, str]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        backend = "nccl"
    else:
        device = "cpu"
        backend = "gloo"
    if world_size > 1:
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=timeout_minutes),
        )
    return rank, world_size, local_rank, device


def tos_exists(path: str, fs) -> bool:
    try:
        return bool(fs.exists(path))
    except Exception:
        # Some fsspec implementations expect paths without the protocol.
        return bool(fs.exists(path.removeprefix("tos://")))


def download_video(video_path: str, rank: int, fs) -> str:
    suffix = Path(video_path).suffix or ".mp4"
    temporary = tempfile.NamedTemporaryFile(
        prefix=f"egoexo-det-rank{rank:05d}-", suffix=suffix, delete=False
    )
    local_path = temporary.name
    temporary.close()
    try:
        with open_with_fs(video_path, "rb", fs) as source, open(
            local_path, "wb"
        ) as destination:
            shutil.copyfileobj(source, destination, length=64 * 1024 * 1024)
    except BaseException:
        if os.path.exists(local_path):
            os.unlink(local_path)
        raise
    return local_path


def upload_file(local_path: str, tos_path: str, fs) -> None:
    with open(local_path, "rb") as source, open_with_fs(
        tos_path, "wb", fs
    ) as destination:
        shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)


def find_video(take_name: str, camera: str, fs) -> str:
    """Find any Exo camera ID without assuming a cam01-cam04 naming scheme."""
    directory = f"{BASE_PATH}/takes/{take_name}/frame_aligned_videos/"
    items = list_tos_directory(directory, fs)
    target = f"{camera}.mp4"
    available = []
    for item in items:
        name = item.get("name", "") if isinstance(item, dict) else str(item)
        basename = name.rstrip("/").split("/")[-1]
        if basename.endswith(".mp4") and not basename.startswith("aria"):
            available.append(basename.removesuffix(".mp4"))
        if basename == target:
            return name if name.startswith("tos://") else f"tos://{name}"
    raise FileNotFoundError(
        f"No {camera} video for {take_name}; available Exo videos: {available}"
    )


def discover_exo_videos(take_name: str, fs) -> dict[str, str]:
    """Discover Exo views from videos that actually exist for a take."""
    directory = f"{BASE_PATH}/takes/{take_name}/frame_aligned_videos/"
    videos = {}
    for item in list_tos_directory(directory, fs):
        name = item.get("name", "") if isinstance(item, dict) else str(item)
        basename = name.rstrip("/").split("/")[-1]
        if not basename.endswith(".mp4") or basename.startswith("aria"):
            continue
        camera = basename.removesuffix(".mp4")
        videos[camera] = name if name.startswith("tos://") else f"tos://{name}"
    return videos


def discover_dataset_tasks(args: argparse.Namespace, fs) -> list[tuple[str, str]]:
    """Build tasks from actual frame-aligned Exo videos, not captures metadata."""
    if args.all_takes:
        take_names = discover_all_takes(fs)
    elif args.take_list:
        with open(args.take_list, encoding="utf-8") as file:
            take_names = [
                line.strip()
                for line in file
                if line.strip() and not line.lstrip().startswith("#")
            ]
    elif args.takes:
        take_names = args.takes
    else:
        raise ValueError("Provide --takes, --take-list, or --all-takes")

    tasks = []
    for take_name in take_names:
        actual_videos = discover_exo_videos(take_name, fs)
        if args.cams:
            missing = [camera for camera in args.cams if camera not in actual_videos]
            if missing:
                print(
                    f"[discovery] {take_name}: skip missing requested cameras "
                    f"{missing}; available={sorted(actual_videos)}",
                    flush=True,
                )
            cameras = [camera for camera in args.cams if camera in actual_videos]
        else:
            cameras = sorted(actual_videos)
        tasks.extend((take_name, camera) for camera in cameras)
    return tasks


def prepare_video_task(
    take_name: str,
    camera: str,
    args: argparse.Namespace,
    rank: int,
) -> PreparedTask:
    """Check TOS output and download a task, safe to run in a background thread."""
    fs = get_tosfs()
    if fs is None:
        raise RuntimeError("TOS filesystem unavailable in download thread")
    tos_dir = f"{args.output_tos.rstrip('/')}/sequences/{take_name}/{camera}"
    result_tos_path = f"{tos_dir}/{args.result_name}"
    if not args.overwrite and tos_exists(result_tos_path, fs):
        return PreparedTask(
            take_name=take_name,
            camera=camera,
            result_tos_path=result_tos_path,
            skipped=True,
        )
    video_path = find_video(take_name, camera, fs)
    print(f"[rank {rank}] prefetch download {take_name}/{camera}", flush=True)
    local_video = download_video(video_path, rank, fs)
    return PreparedTask(
        take_name=take_name,
        camera=camera,
        result_tos_path=result_tos_path,
        video_path=video_path,
        local_video=local_video,
    )


def process_video_task(
    model: YOLO,
    prepared: PreparedTask,
    args: argparse.Namespace,
    rank: int,
    device: str,
    fs,
) -> dict:
    take_name = prepared.take_name
    camera = prepared.camera
    video_path = prepared.video_path
    local_video = prepared.local_video
    result_tos_path = prepared.result_tos_path
    if not local_video or not video_path:
        raise RuntimeError(f"Task was not downloaded: {take_name}/{camera}")
    temporary_json = tempfile.NamedTemporaryFile(
        prefix=f"{take_name}-{camera}-", suffix=".json", delete=False
    )
    local_json = temporary_json.name
    temporary_json.close()

    capture = cv2.VideoCapture(local_video)
    if not capture.isOpened():
        capture.release()
        os.unlink(local_video)
        os.unlink(local_json)
        raise RuntimeError(f"Cannot open downloaded video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    batch_queue: queue.Queue = queue.Queue(maxsize=args.prefetch_batches)
    stop_event = threading.Event()

    def queue_put(item) -> bool:
        while not stop_event.is_set():
            try:
                batch_queue.put(item, timeout=0.2)
                return True
            except queue.Full:
                pass
        return False

    def producer() -> None:
        frame_index = 0
        try:
            while not stop_event.is_set():
                indices = []
                frames = []
                for _ in range(args.batch_size):
                    success, frame = capture.read()
                    if not success:
                        break
                    indices.append(frame_index)
                    frames.append(frame)
                    frame_index += 1
                    if args.max_frames and frame_index >= args.max_frames:
                        break
                if frames and not queue_put(("batch", (indices, frames))):
                    return
                if len(frames) < args.batch_size or (
                    args.max_frames and frame_index >= args.max_frames
                ):
                    queue_put(("done", None))
                    return
        except BaseException as error:
            queue_put(("error", error))

    thread = threading.Thread(target=producer, name="video-producer", daemon=True)
    thread.start()
    writer_queue: queue.Queue = queue.Queue(maxsize=args.writer_queue_batches)
    writer_errors: list[BaseException] = []
    writer_state: dict = {}
    metadata = {
        "take_name": take_name,
        "exo_cam": camera,
        "video_path": video_path,
        "coordinate_space": "raw_distorted",
        "model": args.model,
        "config": {
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "classes": args.classes,
            "batch_size": args.batch_size,
        },
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "reported_frame_count": total_frames,
        },
    }

    def writer_put(item) -> None:
        while not stop_event.is_set():
            if writer_errors:
                raise RuntimeError("JSON writer failed") from writer_errors[0]
            try:
                writer_queue.put(item, timeout=0.2)
                return
            except queue.Full:
                pass
        if writer_errors:
            raise RuntimeError("JSON writer failed") from writer_errors[0]
        raise RuntimeError("JSON writer stopped unexpectedly")

    def writer_worker() -> None:
        processed = detection_count = frames_with_detections = 0
        first_record = True
        try:
            with open(local_json, "w", encoding="utf-8") as output_file:
                output_file.write(json.dumps(metadata, ensure_ascii=False)[:-1])
                output_file.write(', "frames": [\n')
                while True:
                    try:
                        item_type, payload = writer_queue.get(timeout=0.2)
                    except queue.Empty:
                        if stop_event.is_set():
                            return
                        continue
                    if item_type == "done":
                        break
                    indices, detection_arrays, names = payload
                    for frame_index, array in zip(indices, detection_arrays):
                        detections = []
                        if array is not None:
                            for row in array:
                                class_id = int(row[5])
                                detections.append(
                                    {
                                        "class_id": class_id,
                                        "class_name": names[class_id],
                                        "confidence": round(float(row[4]), 6),
                                        "xyxy": [round(float(value), 2) for value in row[:4]],
                                    }
                                )
                        record = {
                            "frame_index": frame_index,
                            "detections": detections,
                        }
                        if not first_record:
                            output_file.write(",\n")
                        json.dump(record, output_file, ensure_ascii=False)
                        first_record = False
                        processed += 1
                        detection_count += len(detections)
                        frames_with_detections += bool(detections)
                stats = {
                    "n_frames_processed": processed,
                    "n_frames_with_detections": frames_with_detections,
                    "n_detections_total": detection_count,
                }
                output_file.write('\n], "stats": ')
                json.dump(stats, output_file)
                output_file.write("}\n")
                writer_state.update(stats)
        except BaseException as error:
            writer_errors.append(error)
            stop_event.set()

    writer_thread = threading.Thread(
        target=writer_worker, name="json-writer", daemon=True
    )
    writer_thread.start()
    processed = 0

    try:
        while True:
            if writer_errors:
                raise RuntimeError("JSON writer failed") from writer_errors[0]
            try:
                item_type, payload = batch_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item_type == "done":
                break
            if item_type == "error":
                raise RuntimeError("Video decoding failed") from payload
            indices, frames = payload
            results = model.predict(
                source=frames,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                classes=args.classes,
                device=device,
                quantize=16 if args.half and device != "cpu" else None,
                verbose=False,
            )
            if len(results) != len(indices):
                raise RuntimeError(
                    f"YOLO returned {len(results)} results for {len(indices)} frames"
                )
            detection_arrays = []
            for result in results:
                boxes = result.boxes
                detection_arrays.append(
                    boxes.data[:, :6].detach().cpu().numpy().copy()
                    if boxes is not None and len(boxes)
                    else None
                )
            names = results[0].names if results else model.names
            writer_put(("batch", (indices, detection_arrays, names)))
            processed += len(results)
            print(
                f"[rank {rank}] {take_name}/{camera}: "
                f"{processed}/{total_frames}",
                flush=True,
            )

        writer_put(("done", None))
        writer_thread.join()
        if writer_errors:
            raise RuntimeError("JSON writer failed") from writer_errors[0]
        stats = dict(writer_state)
    except BaseException:
        if os.path.exists(local_json):
            os.unlink(local_json)
        raise
    finally:
        stop_event.set()
        capture.release()
        thread.join(timeout=10)
        writer_thread.join(timeout=10)
        if os.path.exists(local_video):
            os.unlink(local_video)

    try:
        upload_file(local_json, result_tos_path, fs)
    finally:
        if os.path.exists(local_json):
            os.unlink(local_json)

    print(f"[rank {rank}] uploaded {result_tos_path}", flush=True)
    return {
        "take_name": take_name,
        "exo_cam": camera,
        "status": "complete",
        "result_path": result_tos_path,
        **stats,
    }


def upload_summary(summaries: list[dict], args: argparse.Namespace, fs) -> str:
    path = f"{args.output_tos.rstrip('/')}/batch_summary.json"
    temporary = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    local_path = temporary.name
    temporary.close()
    try:
        with open(local_path, "w", encoding="utf-8") as output:
            json.dump(
                {
                    "n_tasks": len(summaries),
                    "n_complete": sum(x.get("status") == "complete" for x in summaries),
                    "n_skipped": sum(x.get("status") == "skipped" for x in summaries),
                    "n_failed": sum(x.get("status") == "failed" for x in summaries),
                    "tasks": summaries,
                },
                output,
                indent=2,
                ensure_ascii=False,
            )
        upload_file(local_path, path, fs)
    finally:
        os.unlink(local_path)
    return path


def parse_classes(value: str) -> list[int] | None:
    if value.lower() == "all":
        return None
    return [int(item) for item in value.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--takes", nargs="*")
    parser.add_argument("--take-list", help="local file with one take per line")
    parser.add_argument("--all-takes", action="store_true")
    camera_group = parser.add_mutually_exclusive_group()
    camera_group.add_argument(
        "--cams",
        nargs="+",
        help="explicit Exo camera IDs; default discovers every non-ego camera",
    )
    camera_group.add_argument("--auto-cams", action="store_true")
    parser.add_argument("--output-tos", required=True)
    parser.add_argument("--result-name", default="detections.json")
    parser.add_argument("--model", default="yolo26x.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--writer-queue-batches", type=int, default=4)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument(
        "--classes", type=parse_classes, default=[0], help="default: 0 (person)"
    )
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout-minutes", type=int, default=120)
    args = parser.parse_args()
    if not args.output_tos.startswith("tos://"):
        parser.error("--output-tos must start with tos://")
    if (
        args.batch_size < 1
        or args.prefetch_batches < 1
        or args.writer_queue_batches < 1
    ):
        parser.error("batch and queue sizes must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    return args


def main() -> None:
    args = parse_args()
    rank, world_size, _, device = init_distributed(args.timeout_minutes)
    fs = get_tosfs()
    if fs is None:
        raise RuntimeError(
            "TOS filesystem unavailable; set VOLC_ACCESSKEY and VOLC_SECRETKEY"
        )

    try:
        tasks = discover_dataset_tasks(args, fs)
        my_tasks = tasks[rank::world_size]
        print(
            f"[rank {rank}/{world_size}] device={device}, "
            f"tasks={len(my_tasks)}/{len(tasks)}",
            flush=True,
        )
        model = YOLO(args.model)
        summaries = []
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="tos-download") as pool:
            download_future = None
            if my_tasks:
                download_future = pool.submit(
                    prepare_video_task, *my_tasks[0], args, rank
                )

            for task_index, (take_name, camera) in enumerate(my_tasks):
                print(
                    f"[rank {rank}] task {task_index + 1}/{len(my_tasks)}: "
                    f"{take_name}/{camera}",
                    flush=True,
                )
                try:
                    prepared = download_future.result()
                except Exception as error:
                    traceback.print_exc()
                    summaries.append(
                        {
                            "take_name": take_name,
                            "exo_cam": camera,
                            "status": "failed",
                            "error": f"prepare failed: {error}",
                        }
                    )
                    prepared = None

                next_index = task_index + 1
                download_future = (
                    pool.submit(
                        prepare_video_task, *my_tasks[next_index], args, rank
                    )
                    if next_index < len(my_tasks)
                    else None
                )

                if prepared is None:
                    continue
                if prepared.skipped:
                    print(
                        f"[rank {rank}] SKIP existing "
                        f"{prepared.result_tos_path}",
                        flush=True,
                    )
                    summaries.append(
                        {
                            "take_name": take_name,
                            "exo_cam": camera,
                            "status": "skipped",
                            "result_path": prepared.result_tos_path,
                        }
                    )
                    continue
                try:
                    summaries.append(
                        process_video_task(model, prepared, args, rank, device, fs)
                    )
                except Exception as error:
                    traceback.print_exc()
                    if prepared.local_video and os.path.exists(prepared.local_video):
                        os.unlink(prepared.local_video)
                    summaries.append(
                        {
                            "take_name": take_name,
                            "exo_cam": camera,
                            "status": "failed",
                            "error": str(error),
                        }
                    )

        if world_size > 1:
            gathered = [None] * world_size if rank == 0 else None
            dist.gather_object(summaries, gathered, dst=0)
        else:
            gathered = [summaries]
        if rank == 0:
            all_summaries = [item for group in gathered for item in group]
            summary_path = upload_summary(all_summaries, args, fs)
            print(f"Uploaded summary: {summary_path}", flush=True)
        if world_size > 1:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
