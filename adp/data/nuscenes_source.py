"""Thin access layer over nuscenes-devkit.

Yields per-keyframe `Frame` objects carrying the image path, a tested
`CameraModel`, ego/camera transforms, and 3D GT boxes in the camera frame.
Downstream modules (detect/track/lift/risk) consume Frames and never touch the
devkit directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import BoxVisibility

from adp.calib.camera import CameraModel, SE3

DEFAULT_DATAROOT = os.path.join(os.path.dirname(__file__), "..", "..", "v1.0-mini")

# nuScenes fine-grained categories -> ADP target classes.
CATEGORY_MAP = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.construction": "truck",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.trailer": "truck",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "vehicle.bicycle": "cyclist",
    "vehicle.motorcycle": "cyclist",
}


@dataclass
class ObjectGT:
    """One annotated object, expressed in the camera frame of its Frame."""

    token: str
    instance_token: str
    category: str  # ADP class name, or "other" for unmapped categories
    raw_category: str
    box_cam: Box  # devkit Box in camera frame (center, wlh, orientation)
    visibility: int  # nuScenes visibility level 1..4

    @property
    def center_cam(self) -> np.ndarray:
        return self.box_cam.center

    @property
    def range_m(self) -> float:
        """Straight-line distance from camera to box center — GT for the lift."""
        return float(np.linalg.norm(self.box_cam.center))

    def corners_cam(self) -> np.ndarray:
        """(8, 3) box corners in camera frame."""
        return self.box_cam.corners().T


@dataclass
class Frame:
    scene_name: str
    sample_token: str
    timestamp_us: int
    image_path: str
    camera: CameraModel
    T_ego_from_cam: SE3
    T_global_from_ego: SE3
    objects: list[ObjectGT] = field(default_factory=list)


class NuScenesSource:
    def __init__(
        self,
        dataroot: str = DEFAULT_DATAROOT,
        version: str = "v1.0-mini",
        camera_channel: str = "CAM_FRONT",
        verbose: bool = False,
    ):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
        self.camera_channel = camera_channel

    def scene_names(self) -> list[str]:
        return [s["name"] for s in self.nusc.scene]

    def frames(self, scene_name: str, min_visibility: int = 1) -> Iterator[Frame]:
        """Iterate keyframes of a scene in time order."""
        scene = next(s for s in self.nusc.scene if s["name"] == scene_name)
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = self.nusc.get("sample", sample_token)
            yield self._build_frame(scene["name"], sample, min_visibility)
            sample_token = sample["next"]

    def _build_frame(self, scene_name: str, sample: dict, min_visibility: int) -> Frame:
        sd_token = sample["data"][self.camera_channel]
        sd = self.nusc.get("sample_data", sd_token)
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ego_pose = self.nusc.get("ego_pose", sd["ego_pose_token"])

        # get_sample_data returns boxes already transformed into the camera frame.
        image_path, boxes_cam, K = self.nusc.get_sample_data(
            sd_token, box_vis_level=BoxVisibility.ANY
        )

        camera = CameraModel(K=np.asarray(K, dtype=float), width=sd["width"], height=sd["height"])
        T_ego_from_cam = SE3.from_quat_trans(cs["rotation"], cs["translation"])
        T_global_from_ego = SE3.from_quat_trans(ego_pose["rotation"], ego_pose["translation"])

        objects = []
        for box in boxes_cam:
            ann = self.nusc.get("sample_annotation", box.token)
            # visibility_token is "1".."4" (v0-40 .. v80-100); "" means unlabeled.
            vis = int(ann["visibility_token"]) if ann["visibility_token"] else 0
            if vis < min_visibility:
                continue
            objects.append(
                ObjectGT(
                    token=box.token,
                    instance_token=ann["instance_token"],
                    category=CATEGORY_MAP.get(box.name, "other"),
                    raw_category=box.name,
                    box_cam=box,
                    visibility=vis,
                )
            )

        return Frame(
            scene_name=scene_name,
            sample_token=sample["token"],
            timestamp_us=sample["timestamp"],
            image_path=image_path,
            camera=camera,
            T_ego_from_cam=T_ego_from_cam,
            T_global_from_ego=T_global_from_ego,
            objects=objects,
        )
