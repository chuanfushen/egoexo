"""SAM3-based mask extraction for person segmentation."""

import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


class Sam3Extractor:
    def __init__(
        self,
        ckpt_path="/LargeModelDev/users/chuanfu.shen/ckpts/sam3/sam3.pt",
        device: str = "cuda",
    ):
        self.device = device
        self.model = build_sam3_image_model(checkpoint_path=ckpt_path)
        self.processor = Sam3Processor(self.model)

    def detect(
        self,
        img: np.ndarray | str | Image.Image,
        prompt: str = "person",
    ) -> list[dict]:
        """Run SAM3 text-prompted detection and return all results.

        Args:
            img: RGB numpy array (H,W,3), image path, or PIL Image.
            prompt: Text prompt for SAM3.

        Returns:
            List of dicts with keys: mask (np.ndarray), box (xyxy list),
            score (float), sorted by score descending.
        """
        if isinstance(img, str):
            img = Image.open(img)
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        else:
            assert isinstance(img, Image.Image)

        with torch.autocast(
            device_type=self.device,
            dtype=torch.bfloat16,
        ):
            inference_state = self.processor.set_image(img)
            output = self.processor.set_text_prompt(
                state=inference_state, prompt=prompt
            )

        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]

        results = []
        for i in range(len(masks)):
            results.append({
                "mask": masks[i],
                "box": boxes[i].tolist() if isinstance(boxes[i], torch.Tensor) else list(boxes[i]),
                "score": float(scores[i]) if isinstance(scores[i], torch.Tensor) else float(scores[i]),
            })

        return results
