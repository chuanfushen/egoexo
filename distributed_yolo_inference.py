"""Multi-node, multi-GPU YOLO detection inference for MP4 videos.

Launch with torchrun. Each process owns one GPU and an independent YOLO model;
inputs are sharded across global ranks, so inference needs no gradient/DDP
synchronization. Rank-local JSONL files are merged by rank 0 on a shared FS.

Single node example (8 GPUs):
    torchrun --standalone --nproc-per-node=8 distributed_yolo_inference.py \
        video.mp4 --model yolo26x.pt --batch-size 64 \
        --output outputs/detections.jsonl

Two node example (run on both nodes, changing --node-rank):
    torchrun --nnodes=2 --nproc-per-node=8 --node-rank=0 \
        --master-addr=10.0.0.1 --master-port=29500 \
        distributed_yolo_inference.py --manifest videos.txt \
        --output /shared/detections.jsonl
"""

import argparse
import json
import os
import queue
import shutil
import tempfile
import threading
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import cv2
import torch
import torch.distributed as dist
from ultralytics import YOLO

from utils.fsspec_util import get_tosfs, open_with_fs


@dataclass
class FrameItem:
    source: str
    source_index: int
    frame_index: int
    width: int
    height: int
    image: object


def distributed_context(timeout_minutes: int) -> tuple[int, int, int, str]:
    """Initialize torch.distributed from torchrun environment variables."""
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


def load_sources(cli_sources: list[str], manifest: str | None) -> list[str]:
    sources = list(cli_sources)
    if manifest:
        with open(manifest, encoding="utf-8") as file:
            sources.extend(
                line.strip()
                for line in file
                if line.strip() and not line.lstrip().startswith("#")
            )
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(sources))


def download_tos_video(source: str, rank: int) -> str:
    fs = get_tosfs()
    if fs is None:
        raise RuntimeError(
            "TOS filesystem unavailable; set VOLC_ACCESSKEY and VOLC_SECRETKEY"
        )
    suffix = Path(source).suffix or ".mp4"
    temporary = tempfile.NamedTemporaryFile(
        prefix=f"yolo-rank{rank:05d}-", suffix=suffix, delete=False
    )
    path = temporary.name
    temporary.close()
    print(f"[rank {rank}] download {source} -> {path}", flush=True)
    try:
        with open_with_fs(source, "rb", fs) as src, open(path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=64 * 1024 * 1024)
    except BaseException:
        if os.path.exists(path):
            os.unlink(path)
        raise
    return path


def resolve_source(source: str, rank: int) -> tuple[str, str | None]:
    if source.startswith("tos://"):
        temporary = download_tos_video(source, rank)
        return temporary, temporary
    return source, None


def rank_part_path(output: Path, rank: int) -> Path:
    return output.with_name(f"{output.name}.rank{rank:05d}.part")


def write_predictions(
    model: YOLO,
    batch: list[FrameItem],
    output_file,
    args: argparse.Namespace,
    device: str,
    rank: int,
) -> int:
    results = model.predict(
        source=[item.image for item in batch],
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=args.classes,
        device=device,
        quantize=16 if args.half and device != "cpu" else None,
        verbose=False,
    )
    for item, result in zip(batch, results):
        detections = []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.detach().cpu().tolist()
            confidence = boxes.conf.detach().cpu().tolist()
            class_ids = boxes.cls.detach().cpu().int().tolist()
            for bbox, score, class_id in zip(xyxy, confidence, class_ids):
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": result.names[class_id],
                        "confidence": round(score, 6),
                        "xyxy": [round(value, 2) for value in bbox],
                    }
                )
        record = {
            "source": item.source,
            "source_index": item.source_index,
            "frame_index": item.frame_index,
            "width": item.width,
            "height": item.height,
            "rank": rank,
            "detections": detections,
        }
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(results)


def inference_rank(
    args: argparse.Namespace,
    sources: list[str],
    rank: int,
    world_size: int,
    device: str,
) -> Path:
    shard_mode = args.shard_mode
    if shard_mode == "auto":
        shard_mode = "videos" if len(sources) >= world_size else "frames"
    print(
        f"[rank {rank}/{world_size}] device={device} shard_mode={shard_mode}",
        flush=True,
    )

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
        batch: list[FrameItem] = []
        try:
            for source_index, source in enumerate(sources):
                if shard_mode == "videos" and source_index % world_size != rank:
                    continue
                resolved, temporary = resolve_source(source, rank)
                capture = cv2.VideoCapture(resolved)
                try:
                    if not capture.isOpened():
                        raise RuntimeError(f"Cannot open video: {source}")
                    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    frame_index = 0
                    while not stop_event.is_set():
                        success, frame = capture.read()
                        if not success:
                            break
                        assigned = (
                            shard_mode == "videos"
                            or frame_index % world_size == rank
                        )
                        if assigned:
                            batch.append(
                                FrameItem(
                                    source=source,
                                    source_index=source_index,
                                    frame_index=frame_index,
                                    width=width,
                                    height=height,
                                    image=frame,
                                )
                            )
                            if len(batch) == args.batch_size:
                                if not queue_put(("batch", batch)):
                                    return
                                batch = []
                        frame_index += 1
                        if args.max_frames and frame_index >= args.max_frames:
                            break
                finally:
                    capture.release()
                    if temporary and os.path.exists(temporary):
                        os.unlink(temporary)
            if batch and not queue_put(("batch", batch)):
                return
            queue_put(("done", None))
        except BaseException as error:
            queue_put(("error", error))

    model = YOLO(args.model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    part = rank_part_path(output, rank)
    thread = threading.Thread(target=producer, name="video-producer", daemon=True)
    thread.start()

    processed = 0
    try:
        with part.open("w", encoding="utf-8") as output_file:
            while True:
                item_type, payload = batch_queue.get()
                if item_type == "done":
                    break
                if item_type == "error":
                    raise RuntimeError("Video producer failed") from payload
                processed += write_predictions(
                    model, payload, output_file, args, device, rank
                )
                if processed % args.log_interval < len(payload):
                    print(f"[rank {rank}] processed={processed}", flush=True)
    finally:
        stop_event.set()
        thread.join(timeout=10)
    print(f"[rank {rank}] done, processed={processed}, part={part}", flush=True)
    return part


def merge_parts(output: Path, world_size: int, keep_parts: bool) -> None:
    temporary = output.with_name(f".{output.name}.merging")
    with temporary.open("wb") as destination:
        for rank in range(world_size):
            part = rank_part_path(output, rank)
            if not part.exists():
                raise FileNotFoundError(
                    f"Missing {part}. Multi-node merging requires a shared filesystem."
                )
            with part.open("rb") as source:
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
    os.replace(temporary, output)
    if not keep_parts:
        for rank in range(world_size):
            rank_part_path(output, rank).unlink()


def parse_classes(value: str | None) -> list[int] | None:
    if value is None or value.lower() == "all":
        return None
    return [int(part) for part in value.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*", help="local MP4 paths or tos:// URIs")
    parser.add_argument("--manifest", help="text file containing one video URI per line")
    parser.add_argument("--model", default="yolo26x.pt")
    parser.add_argument("--output", default="outputs/distributed_detections.jsonl")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--max-frames", type=int, help="limit per input video")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument(
        "--classes",
        type=parse_classes,
        default=None,
        help="comma-separated class IDs, e.g. 0 or 0,1; default: all",
    )
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--shard-mode",
        choices=["auto", "videos", "frames"],
        default="auto",
        help="auto uses video sharding when there are enough videos",
    )
    parser.add_argument("--timeout-minutes", type=int, default=60)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--keep-parts", action="store_true")
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="leave one JSONL part per rank instead of rank-0 merge",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.prefetch_batches < 1:
        parser.error("batch and prefetch sizes must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.log_interval < 1:
        parser.error("--log-interval must be positive")
    return args


def main() -> None:
    args = parse_args()
    sources = load_sources(args.sources, args.manifest)
    if not sources:
        raise SystemExit("Provide at least one video path or --manifest")

    rank, world_size, _, device = distributed_context(args.timeout_minutes)
    try:
        inference_rank(args, sources, rank, world_size, device)
        if world_size > 1:
            dist.barrier()
        if rank == 0 and not args.no_merge:
            output = Path(args.output)
            merge_parts(output, world_size, args.keep_parts)
            print(f"Merged output: {output}", flush=True)
        if world_size > 1:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
