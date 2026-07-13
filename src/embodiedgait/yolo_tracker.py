"""YOLO + ByteTrack person tracker for frame-by-frame video processing.

Usage:
    tracker = YOLOTracker(model_path="yolo26n.pt", device="cuda:0")
    for frame in video_frames:
        persons = tracker.track_frame(frame)
        for p in persons:
            print(f"track_id={p.track_id}, bbox={p.bbox}, conf={p.conf}")
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class TrackedPerson:
    """Single tracked person detection from YOLO."""

    track_id: int
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    conf: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, (x2 - x1) * (y2 - y1))


class YOLOTracker:
    """YOLO detection + ByteTrack tracking for persons (class 0).

    Wraps ultralytics YOLO with persist=True for frame-by-frame tracking,
    filtering to person class only.
    """

    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        device: str = "cuda",
        conf: float = 0.25,
        iou: float = 0.7,
    ):
        from ultralytics import YOLO

        self.device = device
        self.conf = conf
        self.iou = iou
        if device and "cuda" in str(device):
            gpu_id = int(str(device).split(":")[-1]) if ":" in str(device) else 0
            self.model = YOLO(model_path)
            self.model.to(f"cuda:{gpu_id}")
        else:
            self.model = YOLO(model_path)

    def track_frame(self, frame: np.ndarray) -> list[TrackedPerson]:
        """Process one BGR frame with YOLO tracking.

        Args:
            frame: BGR image (H, W, 3) as numpy array.

        Returns:
            List of TrackedPerson objects for this frame.
        """
        # Ultralytics YOLO track with persist=True for frame-by-frame tracking
        results = self.model.track(
            frame,
            persist=True,
            classes=[0],           # person only
            conf=self.conf,
            iou=self.iou,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        # results is a list; frame-by-frame mode returns [Results]
        if not results or results[0] is None:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []

        track_ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(track_ids))

        persons = []
        for i in range(len(track_ids)):
            persons.append(
                TrackedPerson(
                    track_id=int(track_ids[i]),
                    bbox=tuple(float(v) for v in xyxy[i]),
                    conf=float(confs[i]),
                )
            )
        return persons

    def reset(self):
        """Reset tracker state for a new video."""
        # Ultralytics tracker state is internal; re-create the tracker
        # by clearing the model's predictor tracker state
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            if hasattr(self.model.predictor, "trackers"):
                self.model.predictor.trackers = []
