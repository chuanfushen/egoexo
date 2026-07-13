"""YOLO batched prediction with TOS support (headless).

Supports both local paths and tos:// URIs.  For TOS paths the video is
downloaded to a temp file, processed in batches with YOLO detection, and the
detections are streamed to a local JSON file. No tracker is used.

Usage:
    uv run python test_yolo.py tos://drobotics-ailab/.../cam01.mp4
    uv run python test_yolo.py /local/path/to/video.mp4
    uv run python test_yolo.py                           # uses default TOS path
    uv run python test_yolo.py video.mp4 -o detections.json -b 64
"""

import argparse
import json
import os
import queue
import tempfile
import threading
from pathlib import Path

import cv2
from ultralytics import YOLO

from utils.fsspec_util import get_tosfs, open_with_fs

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_VIDEO = (
    "tos://drobotics-ailab/users/chuanfu.shen/data/egoexo4d-defaults/"
    "takes/cmu_soccer12_2/frame_aligned_videos/cam01.mp4"
)
OUTPUT_DIR = Path("outputs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_tos_to_temp(tos_path: str, fs) -> str:
    """Stream a TOS file to a local temporary file in chunks.

    Returns the path to the temporary file (caller is responsible for cleanup).
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    print(f"Downloading {tos_path} -> {tmp_path} ...")
    with open_with_fs(tos_path, "rb", fs) as src:
        with open(tmp_path, "wb") as dst:
            while True:
                chunk = src.read(64 * 1024 * 1024)  # 64 MiB chunks
                if not chunk:
                    break
                dst.write(chunk)
    print("Download complete.")
    return tmp_path


def _resolve_video_path(video_path: str):
    """Resolve *video_path* (local or tos://) to something OpenCV can open.

    Returns ``(resolved_path, fs, tmp_path)`` where:
    - *resolved_path* is a local path that ``cv2.VideoCapture`` can use.
    - *fs* is the TosFileSystem instance (or *None*).
    - *tmp_path* is the temp-file path if a download happened (else *None*).
      The caller **must** clean it up after use.
    """
    fs = None
    tmp_path = None

    if video_path.startswith("tos://"):
        fs = get_tosfs()
        if fs is None:
            raise RuntimeError(
                "TOS filesystem unavailable. "
                "Set VOLC_ACCESSKEY and VOLC_SECRETKEY environment variables."
            )
        tmp_path = _download_tos_to_temp(video_path, fs)
        resolved = tmp_path
    else:
        resolved = video_path

    return resolved, fs, tmp_path



# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------

def _make_output_path(input_path: str, output_path: str | None) -> str:
    """Derive output path from input filename if not explicitly given."""
    if output_path:
        return output_path

    # Strip tos:// prefix for filename extraction
    name = Path(input_path).stem
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return str(OUTPUT_DIR / f"{name}_detections.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    video_path: str,
    output_path: str | None = None,
    batch_size: int = 64,
    model_path: str = "yolo26l.pt",
    conf: float = 0.25,
    imgsz: int = 640,
    device: str | None = None,
    prefetch_batches: int = 2,
):
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if prefetch_batches < 1:
        raise ValueError("prefetch_batches must be at least 1")
    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    resolved_path, fs, tmp_path = _resolve_video_path(video_path)
    out_path = _make_output_path(video_path, output_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(resolved_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {resolved_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not fps or fps <= 0:
        fps = 30.0

    print(f"Input:  {total_frames} frames @ {fps:.1f} fps, {width}x{height}")
    print(f"Output: {out_path}")

    frame_idx = 0
    batch_queue = queue.Queue(maxsize=prefetch_batches)
    stop_event = threading.Event()

    def put_batch(item):
        """Put an item unless the consumer has stopped."""
        while not stop_event.is_set():
            try:
                batch_queue.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def decode_worker():
        """Decode video frames ahead of GPU inference."""
        next_frame_index = 0
        try:
            while not stop_event.is_set():
                batch_start = next_frame_index
                batch = []
                for _ in range(batch_size):
                    success, frame = cap.read()
                    if not success:
                        break
                    batch.append(frame)
                    next_frame_index += 1

                if batch and not put_batch(("batch", batch_start, batch)):
                    return
                if len(batch) < batch_size:
                    put_batch(("done", None, None))
                    return
        except BaseException as exc:
            put_batch(("error", None, exc))

    def predict_and_save(batch, batch_start, output_file, first_record):
        """Run one inference batch and stream its detections to JSON."""
        results = model.predict(
            source=batch,
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        for batch_offset, result in enumerate(results):
            boxes = result.boxes
            detections = []
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.detach().cpu().tolist()
                confidences = boxes.conf.detach().cpu().tolist()
                class_ids = boxes.cls.detach().cpu().int().tolist()
                for coords, confidence, class_id in zip(
                    xyxy, confidences, class_ids
                ):
                    detections.append(
                        {
                            "class_id": class_id,
                            "class_name": result.names[class_id],
                            "confidence": round(confidence, 6),
                            "xyxy": [round(value, 2) for value in coords],
                        }
                    )

            record = {
                "frame_index": batch_start + batch_offset,
                "detections": detections,
            }
            if not first_record:
                output_file.write(",\n")
            json.dump(record, output_file, ensure_ascii=False)
            first_record = False

        return len(results), first_record

    decoder_thread = threading.Thread(
        target=decode_worker, name="video-decoder", daemon=True
    )
    decoder_thread.start()

    try:
        with open(out_path, "w", encoding="utf-8") as output_file:
            output_file.write("[\n")
            first_record = True
            while True:
                item_type, batch_start, payload = batch_queue.get()
                if item_type == "done":
                    break
                if item_type == "error":
                    raise RuntimeError("Video decoding failed") from payload

                count, first_record = predict_and_save(
                    payload, batch_start, output_file, first_record
                )
                frame_idx += count
                print(f"  ... {frame_idx}/{total_frames} frames")
            output_file.write("\n]\n")

    finally:
        stop_event.set()
        cap.release()
        decoder_thread.join(timeout=5)
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            print(f"Cleaned up temp file: {tmp_path}")

    print(f"Done — saved detections for {frame_idx} frames to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", default=DEFAULT_VIDEO)
    parser.add_argument("-o", "--output", help="output detections JSON path")
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("--model", default="yolo26x.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", help="e.g. 0, cuda:0, or cpu")
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=2,
        help="number of decoded batches buffered ahead of inference (default: 2)",
    )
    args = parser.parse_args()
    main(
        args.video,
        args.output,
        args.batch_size,
        args.model,
        args.conf,
        args.imgsz,
        args.device,
        args.prefetch_batches,
    )
