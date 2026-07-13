"""SAM3-based person detection via text prompt."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """Single detection from SAM3."""

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    score: float
    mask: np.ndarray | None = None  # binary mask (H, W)

    @property
    def center(self) -> tuple[float, float]:
        """Bbox center (cx, cy)."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        """Bbox area in pixels."""
        x1, y1, x2, y2 = self.bbox
        return max(0.0, (x2 - x1) * (y2 - y1))


class PersonDetector:
    """SAM3-based person detector using text prompt.

    Usage:
        detector = PersonDetector(device="cuda")

        for frame in video_frames:
            detections = detector.detect(frame, prompt="person")
            for d in detections:
                print(f"bbox={d.bbox}, score={d.score:.2f}")
    """

    def __init__(
        self,
        ckpt_path: str = "/LargeModelDev/users/chuanfu.shen/ckpts/sam3/sam3.pt",
        device: str = "cuda",
    ):
        """Initialize SAM3 detector.

        Args:
            ckpt_path: Path to SAM3 checkpoint.
            device: 'cuda' or 'cpu'.
        """
        from utils.mask_extractor import Sam3Extractor

        self.extractor = Sam3Extractor(ckpt_path=ckpt_path, device=device)

    def detect(
        self,
        frame: np.ndarray,
        prompt: str = "person",
    ) -> list[Detection]:
        """Run SAM3 text-prompted detection on a BGR frame.

        Args:
            frame: BGR image (H, W, 3) as numpy array.
            prompt: Text prompt for SAM3 (default: "person").

        Returns:
            List of Detection objects sorted by score descending.
        """
        # SAM3 expects RGB
        img_rgb = frame[..., ::-1]  # BGR → RGB
        results = self.extractor.detect(img_rgb, prompt=prompt)

        detections = []
        for r in results:
            detections.append(
                Detection(
                    bbox=tuple(r["box"]),
                    score=r["score"],
                    mask=r.get("mask"),
                )
            )

        return detections
