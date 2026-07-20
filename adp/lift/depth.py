"""Metric lift v2: monocular metric depth (Depth Anything V2, outdoor-metric
ViT-S via transformers, fp16).

Range estimate per object: median metric depth over the lower-central box
region (plan spec) — robust to a few sky/background pixels, and crucially it
samples the object's BODY, so an occluded box bottom (IPM's blind spot) does
not corrupt it. The depth value is z along the optical axis; back-projecting
the sampling pixel at that depth gives the camera-frame point, transformed to
ego like any other.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from adp.calib.camera import CameraModel, SE3

MODEL_NAME = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


class DepthLift:
    def __init__(self, device: str = "cuda"):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            MODEL_NAME, dtype=torch.float16).to(device).eval()
        self.device = device
        self._map: np.ndarray | None = None
        self._scale_uv: tuple[float, float] = (1.0, 1.0)

    @torch.no_grad()
    def compute(self, img_bgr: np.ndarray) -> None:
        """Run depth once per frame; box queries then read from the cached map."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].half()
        depth = self.model(**inputs).predicted_depth[0].float().cpu().numpy()
        self._map = depth
        h, w = img_bgr.shape[:2]
        self._scale_uv = (depth.shape[1] / w, depth.shape[0] / h)

    def range_from_box(
        self,
        xyxy: np.ndarray,
        camera: CameraModel,
        T_ego_from_cam: SE3,
    ) -> tuple[np.ndarray | None, float]:
        """Box -> (ego-frame ground xy, range) from the cached depth map."""
        assert self._map is not None, "call compute(img) first"
        su, sv = self._scale_uv
        x0, y0, x1, y1 = xyxy
        h = y1 - y0
        # Lower-central region: bottom half, central 60% width.
        u0, u1 = int((x0 + 0.2 * (x1 - x0)) * su), int((x1 - 0.2 * (x1 - x0)) * su)
        v0, v1 = int((y0 + 0.5 * h) * sv), int(y1 * sv)
        patch = self._map[max(v0, 0):max(v1, v0 + 1),
                          max(u0, 0):max(u1, u0 + 1)]
        if patch.size == 0:
            return None, float("nan")
        z = float(np.median(patch))
        if not np.isfinite(z) or z <= 0.5:
            return None, float("nan")

        # Representative pixel: box bottom-center area at that depth.
        uv = np.array([[(x0 + x1) / 2, y0 + 0.75 * h]])
        p_cam = camera.backproject(uv, np.array([z]))
        p_ego = T_ego_from_cam.apply(p_cam)[0]
        return p_ego[:2], float(np.linalg.norm(p_ego[:2]))
