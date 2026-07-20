"""M0 verification: render nuScenes 3D GT boxes into camera images using OUR
projection code (not the devkit's render functions).

If the wireframes sit on the objects, the whole chain is verified end-to-end:
data access -> calibration -> our CameraModel.project. Outputs one annotated
frame per scene to out/checks/, plus a console summary.

Usage: python scripts/check_data.py
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adp.data.nuscenes_source import NuScenesSource

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out", "checks")

CLASS_COLORS = {  # BGR
    "car": (255, 160, 0),
    "truck": (255, 80, 80),
    "bus": (0, 200, 255),
    "pedestrian": (0, 80, 255),
    "cyclist": (0, 220, 120),
    "other": (160, 160, 160),
}

# Edges of the devkit corner ordering: 4 front-face, 4 rear-face, 4 connecting.
BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]


def draw_box(img, obj, camera):
    uv, depth = camera.project(obj.corners_cam())
    if np.any(depth <= 0.1):  # box partly behind camera: skip drawing, not worth clipping
        return False
    color = CLASS_COLORS[obj.category]
    pts = uv.astype(int)
    for a, b in BOX_EDGES:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
    label = f"{obj.category} {obj.range_m:.0f}m"
    anchor = pts.min(axis=0)
    cv2.putText(img, label, (int(anchor[0]), int(anchor[1]) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return True


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    source = NuScenesSource(verbose=False)

    total_objects, total_drawn, in_image_hits = 0, 0, 0
    for scene_name in source.scene_names():
        frame = next(source.frames(scene_name, min_visibility=2))
        img = cv2.imread(frame.image_path)
        assert img is not None, f"failed to read {frame.image_path}"

        for obj in frame.objects:
            total_objects += 1
            if draw_box(img, obj, frame.camera):
                total_drawn += 1
                uv, depth = frame.camera.project(obj.corners_cam())
                # Sanity: visible (>=40%) GT boxes should project mostly inside the image.
                if frame.camera.in_image(uv, depth, margin=50).mean() > 0.5:
                    in_image_hits += 1

        out_path = os.path.join(OUT_DIR, f"{scene_name}.jpg")
        cv2.imwrite(out_path, img)
        print(f"{scene_name}: {len(frame.objects)} objects (vis>=2) -> {out_path}")

    frac = in_image_hits / max(total_drawn, 1)
    print(f"\nTotal: {total_objects} objects, {total_drawn} drawn, "
          f"{frac:.0%} project mostly inside the image (expect near 100%).")
    if frac < 0.9:
        print("WARNING: low in-image fraction — check calibration chain.")
        sys.exit(1)


if __name__ == "__main__":
    main()
