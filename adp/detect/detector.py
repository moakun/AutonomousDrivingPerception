"""2D detection wrapper over ultralytics (YOLOv8/v11 and RT-DETR).

Maps COCO classes onto the 5 ADP target classes and returns plain dataclasses,
so downstream code never depends on ultralytics types. Note: COCO 'bicycle' /
'motorcycle' detect the vehicle while nuScenes cyclist boxes include the rider;
IoU matching absorbs most of that discrepancy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COCO_TO_ADP = {
    0: "pedestrian",
    1: "cyclist",   # bicycle
    2: "car",
    3: "cyclist",   # motorcycle
    5: "bus",
    7: "truck",
}


@dataclass
class Detection2D:
    xyxy: np.ndarray  # (4,) float, pixels
    score: float
    category: str


class Detector:
    def __init__(
        self,
        weights: str = "yolov8s.pt",
        imgsz: int = 640,
        conf: float = 0.25,
        device: str | int = 0,
    ):
        if "rtdetr" in weights:
            from ultralytics import RTDETR
            self.model = RTDETR(weights)
        else:
            from ultralytics import YOLO
            self.model = YOLO(weights)
        self.weights = weights
        self.imgsz = imgsz
        self.conf = conf
        self.device = device
        self.last_speed: dict = {}  # ms: preprocess / inference / postprocess

    def __call__(self, img_bgr: np.ndarray) -> list[Detection2D]:
        result = self.model.predict(
            img_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            max_det=300,
            verbose=False,
        )[0]
        self.last_speed = result.speed

        detections = []
        boxes = result.boxes
        if boxes is None:
            return detections
        for xyxy, cls_id, score in zip(
            boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()
        ):
            category = COCO_TO_ADP.get(int(cls_id))
            if category is None:
                continue
            detections.append(Detection2D(xyxy=xyxy.astype(float), score=float(score), category=category))
        return detections
