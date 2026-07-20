"""Lane-line segmentation via pretrained YOLOP (BDD100K).

Chosen over CLRNet/LaneATT because it runs on plain PyTorch (no custom CUDA
ops, which don't build cleanly on Windows). Output is a per-pixel lane-line
probability mask at the original image resolution; all geometry happens
downstream in BEV space.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class LaneSegmenter:
    IN_W, IN_H = 640, 384  # YOLOP input; 1600x900 -> 640x360 + 12px pad top/bottom

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = torch.hub.load("hustvl/yolop", "yolop", pretrained=True,
                                    trust_repo=True)
        self.model.eval().to(device)

    @torch.no_grad()
    def lane_mask(self, img_bgr: np.ndarray, thresh: float = 0.5) -> np.ndarray:
        """BGR image -> boolean lane-line mask at original resolution."""
        h, w = img_bgr.shape[:2]
        scale = self.IN_W / w
        new_h = round(h * scale)
        pad = (self.IN_H - new_h) // 2

        resized = cv2.resize(img_bgr, (self.IN_W, new_h))
        canvas = np.zeros((self.IN_H, self.IN_W, 3), dtype=np.uint8)
        canvas[pad:pad + new_h] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(rgb.transpose(2, 0, 1)[None]).to(self.device)

        _, _, ll_seg = self.model(x)
        prob = torch.softmax(ll_seg[0], dim=0)[1]  # lane-line channel
        prob = prob[pad:pad + new_h].cpu().numpy()
        prob_full = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        return prob_full >= thresh
