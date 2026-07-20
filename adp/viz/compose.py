"""Composite frame renderer: camera overlay (risk-colored boxes, IDs, ranges,
why-text) + BEV panel (corridor, lanes, tracks, velocity vectors, RSS arc).

This is the M7 deliverable view — every annotation traces to a computed
quantity; nothing is decoration.
"""

from __future__ import annotations

import cv2
import numpy as np

from adp.pipeline import PipelineOutput
from adp.viz.bev import BevCanvas, render_bev

BUCKET_COLORS = {"low": (90, 200, 90), "caution": (0, 200, 255), "high": (0, 60, 255)}
CAM_W, CAM_H = 960, 540


def sample_line(line, n=24):
    xs = np.linspace(line.x_range[0], min(line.x_range[1], 45.0), n)
    return np.stack([xs, line.y_at(xs)], axis=1)


def compose_frame(img_bgr: np.ndarray, out: PipelineOutput, cam) -> np.ndarray:
    sx = CAM_W / img_bgr.shape[1]
    cam_img = cv2.resize(img_bgr, (CAM_W, CAM_H))
    by_id = {r.track_id: r for r in out.risk}

    # Ego corridor reprojected onto the camera image.
    corr = out.corridor if out.corridor.valid else None
    if corr is not None:
        for line in (corr.left, corr.right):
            poly = sample_line(line)
            pts_ego = np.hstack([poly, np.zeros((len(poly), 1))])
            pts_cam = cam.T_ego_from_cam.inverse().apply(pts_ego)
            uv, depth = cam.camera.project(pts_cam)
            uv = (uv[depth > 0.5] * sx).astype(int)
            for a, b in zip(uv[:-1], uv[1:]):
                cv2.line(cam_img, tuple(a), tuple(b), (0, 220, 220), 2, cv2.LINE_AA)

    for t in out.tracks:
        r = by_id.get(t.track_id)
        color = BUCKET_COLORS[r.bucket] if r else (150, 150, 150)
        x0, y0, x1, y1 = (t.xyxy * sx).astype(int)
        cv2.rectangle(cam_img, (x0, y0), (x1, y1), color, 2)
        label = f"#{t.track_id} {t.category}"
        if r:
            label += f" {r.range_m:.0f}m"
            if r.ttc is not None and r.ttc.trustworthy:
                label += f" ttc{r.ttc.ttc_s:.1f}s"
        cv2.putText(cam_img, label, (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)

    if out.risk and out.risk[0].score >= 0.3:
        top = out.risk[0]
        why = (f"#{top.track_id} {top.category}: " + "; ".join(top.reasons)
               ).replace("±", "+-")
        cv2.putText(cam_img, why[:110], (8, CAM_H - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, BUCKET_COLORS[top.bucket], 1, cv2.LINE_AA)
    cv2.putText(cam_img,
                f"ego risk {out.ego_risk:.2f}   v={out.ego_speed_ms:.1f} m/s",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)

    bev_tracks = []
    for t in out.tracks:
        s = t.extra.get("bev")
        if s is None:
            continue
        pos_e, vel_e = s.in_ego(cam.T_global_from_ego)
        bev_tracks.append({"id": t.track_id, "category": t.category, "pos": pos_e,
                           "vel": vel_e, "range_std": s.range_std()})
    canvas = BevCanvas()
    bev = render_bev(
        bev_tracks, canvas=canvas,
        lanes=[sample_line(l) for l in out.lane_lines],
        ego_corridor=((sample_line(corr.left), sample_line(corr.right))
                      if corr else None),
    )
    # RSS envelope arc: min safe gap at current ego speed vs a stopped object.
    from adp.risk.rss import rss_min_gap
    d_rss = rss_min_gap(out.ego_speed_ms, 0.0)
    if d_rss > 1.0:
        u, v = canvas.to_px(d_rss + 3.5, 0)  # bumper offset
        cv2.ellipse(bev, (canvas.to_px(0, 0)[0], canvas.to_px(0, 0)[1]),
                    (int((d_rss + 3.5) * canvas.scale),) * 2,
                    0, 245, 295, (0, 80, 200), 1, cv2.LINE_AA)
        cv2.putText(bev, f"RSS {d_rss:.0f}m", (u + 8, v),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 80, 200), 1, cv2.LINE_AA)

    return np.hstack([cam_img, bev])
