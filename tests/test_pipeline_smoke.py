"""End-to-end pipeline smoke test (improvement #7).

Module-level tests structurally cannot catch wiring bugs — e.g. handing a
FusedLift to the lane code, which needs the raw homography's lift_pixels.
That shipped and only failed at eval time; this test exists so it fails in
seconds instead.

Skipped automatically when the dataset or model weights are unavailable, so
the suite still runs on a bare checkout.
"""

import os

import cv2
import numpy as np
import pytest

DATAROOT = os.path.join(os.path.dirname(__file__), "..", "v1.0-mini")
WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "yolo11m.pt")

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(DATAROOT) and os.path.isfile(WEIGHTS)),
    reason="nuScenes mini and/or yolo11m weights not present",
)


@pytest.fixture(scope="module")
def frames():
    from adp.data.nuscenes_source import NuScenesSource
    source = NuScenesSource(verbose=False)
    scene = source.scene_names()[0]
    out = []
    for cam in source.camera_frames(scene):
        out.append(cam)
        if len(out) >= 6:
            break
    return out


def run(frames, **kwargs):
    from adp.detect.detector import Detector
    from adp.lanes.detector import LaneSegmenter
    from adp.pipeline import PerceptionPipeline
    pipeline = PerceptionPipeline(
        detector=Detector(weights="yolo11m.pt", imgsz=960, conf=0.1),
        segmenter=LaneSegmenter(), **kwargs)
    outputs = []
    for cam in frames:
        img = cv2.imread(cam.image_path)
        outputs.append(pipeline.step(cam, img))
    return outputs


@pytest.mark.parametrize("kwargs", [
    {"use_width": True},    # current default: fused lift
    {"use_width": False},   # IPM-only path must still work
])
def test_pipeline_runs_and_outputs_are_wellformed(frames, kwargs):
    outputs = run(frames, **kwargs)
    assert len(outputs) == len(frames)
    for out in outputs:
        assert 0.0 <= out.ego_risk <= 1.0
        assert np.isfinite(out.ego_speed_ms)
        assert set(out.timings_ms) <= {"detect", "track_lift", "lanes"}
        for r in out.risk:
            assert np.isfinite(r.range_m) and r.range_m > 0
            assert r.reasons, "every record must explain itself"
            assert r.bucket in ("low", "caution", "high")
            if r.ttc is not None:
                assert np.isfinite(r.ttc.ttc_s) and np.isfinite(r.ttc.sigma_s)
        for t in out.tracks:
            bev = t.extra.get("bev")
            if bev is not None:
                assert np.all(np.isfinite(bev.kf.position))
                assert np.all(np.isfinite(bev.kf.velocity))


def test_lane_branch_uses_ground_plane(frames):
    """Regression: the lane branch needs lift_pixels (raw homography), which
    the fused object-lift does not implement."""
    outputs = run(frames, use_width=True)
    assert any(o.lane_lines is not None for o in outputs)
    for out in outputs:
        for line in out.lane_lines:
            assert np.all(np.isfinite(line.coeffs))


def test_composite_renderer_runs(frames):
    """The demo path must survive whatever the pipeline emits."""
    from adp.viz.compose import CAM_H, CAM_W, compose_frame
    outputs = run(frames, use_width=True)
    img = cv2.imread(frames[-1].image_path)
    canvas = compose_frame(img, outputs[-1], frames[-1])
    assert canvas.shape[0] == CAM_H and canvas.shape[1] > CAM_W
