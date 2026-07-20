"""Composite frame renderer: camera view + top-down view, written for a
general audience.

Design rules:
- Plain language only — no "TTC", "RSS", "ego", "corridor", track ids, or
  m/s. Speeds in km/h, times in seconds, distances in meters.
- Color always means danger level (green = fine, amber = caution,
  red = warning) — never object class.
- Text sits on solid plates for readability at video compression.
- The most important object gets one plain sentence explaining WHY it is
  highlighted (the risk record's reasons, humanized).

The underlying quantities are unchanged — this is presentation only; the
technical values remain in the risk records / eval JSONs.
"""

from __future__ import annotations

import cv2
import numpy as np

from adp.pipeline import PipelineOutput
from adp.risk.scoring import RiskRecord
from adp.viz.bev import BevCanvas, render_bev

CAM_W, CAM_H = 960, 540

BUCKET_COLORS = {"low": (110, 200, 110), "caution": (0, 190, 255),
                 "high": (60, 60, 255)}
STATUS = {"low": ("ALL CLEAR", (60, 160, 60)),
          "caution": ("CAUTION", (0, 160, 220)),
          "high": ("WARNING", (40, 40, 220))}
FRIENDLY_NAME = {"car": "Car", "truck": "Truck", "bus": "Bus",
                 "pedestrian": "Person", "cyclist": "Cyclist"}
DARK = (25, 25, 25)


def kmh(ms: float) -> int:
    return int(round(ms * 3.6))


def bucket_of(ego_risk: float) -> str:
    return "high" if ego_risk >= 0.6 else "caution" if ego_risk >= 0.3 else "low"


def humanize(r: RiskRecord) -> str:
    """One plain sentence: what the object is, where, and why it matters."""
    name = FRIENDLY_NAME.get(r.category, r.category.title())
    where = {"ego": "in your lane", "adjacent": "in the next lane",
             "off_path": "off to the side", "behind": "behind you"}[r.zone.value]
    parts = [f"{name} {where}, {r.range_m:.0f} m away"]
    if not r.velocity_plausible:
        parts.append("speed reading unreliable")
    else:
        if r.closing_ms is not None and r.closing_ms > 1.0:
            parts.append(f"getting closer at {kmh(r.closing_ms)} km/h")
        if r.ttc is not None and r.ttc.trustworthy and r.ttc.ttc_s < 6.0:
            parts.append(f"could reach you in about {r.ttc.ttc_s:.0f} second"
                         + ("s" if r.ttc.ttc_s >= 1.5 else ""))
        if r.rss_violated:
            parts.append("too close to stop safely if it brakes")
        if r.intent_cross_s is not None:
            parts.append("may step into your path")
    return " - ".join(parts)  # ASCII only: cv2 putText has no unicode glyphs


def draw_plate(img, text, x, y, bg, fg=(255, 255, 255), scale=0.5, pad=4):
    """Text on a filled rectangle; returns plate height."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (x, y - th - base - pad), (x + tw + 2 * pad, y + pad), bg, -1)
    cv2.putText(img, text, (x + pad, y - base // 2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, fg, 1, cv2.LINE_AA)
    return th + base + 2 * pad


def sample_line(line, n=24):
    xs = np.linspace(line.x_range[0], min(line.x_range[1], 45.0), n)
    return np.stack([xs, line.y_at(xs)], axis=1)


def compose_frame(img_bgr: np.ndarray, out: PipelineOutput, cam) -> np.ndarray:
    sx = CAM_W / img_bgr.shape[1]
    cam_img = cv2.resize(img_bgr, (CAM_W, CAM_H))
    by_id = {r.track_id: r for r in out.risk}

    # Detected lane edges on the road, drawn subtly.
    corr = out.corridor if out.corridor.valid else None
    if corr is not None:
        for line in (corr.left, corr.right):
            poly = sample_line(line)
            pts_ego = np.hstack([poly, np.zeros((len(poly), 1))])
            pts_cam = cam.T_ego_from_cam.inverse().apply(pts_ego)
            uv, depth = cam.camera.project(pts_cam)
            uv = (uv[depth > 0.5] * sx).astype(int)
            for a, b in zip(uv[:-1], uv[1:]):
                cv2.line(cam_img, tuple(a), tuple(b), (200, 220, 120), 2, cv2.LINE_AA)

    # Object boxes: color = danger level; label = what it is + distance.
    for t in out.tracks:
        r = by_id.get(t.track_id)
        color = BUCKET_COLORS[r.bucket] if r else (150, 150, 150)
        x0, y0, x1, y1 = (t.xyxy * sx).astype(int)
        thick = 3 if r and r.bucket == "high" else 2
        cv2.rectangle(cam_img, (x0, y0), (x1, y1), color, thick)
        name = FRIENDLY_NAME.get(t.category, t.category.title())
        # Beyond ~100m the lift is unvalidated noise — say "far" not a number.
        label = (name if r is None
                 else f"{name}  far" if r.range_m > 100
                 else f"{name}  {r.range_m:.0f} m")
        draw_plate(cam_img, label, x0, max(y0 - 2, 16), color, DARK, scale=0.45)

    # Top status banner.
    banner = cam_img.copy()
    cv2.rectangle(banner, (0, 0), (CAM_W, 34), (25, 25, 25), -1)
    cam_img = cv2.addWeighted(banner, 0.75, cam_img, 0.25, 0)
    status_text, status_color = STATUS[bucket_of(out.ego_risk)]
    cv2.putText(cam_img, f"Your speed: {kmh(out.ego_speed_ms)} km/h",
                (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(cam_img, (CAM_W - tw - 26, 4), (CAM_W - 6, 30), status_color, -1)
    cv2.putText(cam_img, status_text, (CAM_W - tw - 16, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Bottom explanation for the most important object.
    if out.risk and out.risk[0].score >= 0.3:
        top = out.risk[0]
        draw_plate(cam_img, humanize(top)[:110], 8, CAM_H - 10,
                   BUCKET_COLORS[top.bucket], DARK, scale=0.5)

    # ---- top-down panel ----------------------------------------------------
    bev_tracks = []
    for t in out.tracks:
        s = t.extra.get("bev")
        if s is None:
            continue
        pos_e, vel_e = s.in_ego(cam.T_global_from_ego)
        r = by_id.get(t.track_id)
        bev_tracks.append({
            "id": t.track_id, "category": t.category, "pos": pos_e, "vel": vel_e,
            "range_std": s.range_std(),
            "color": BUCKET_COLORS[r.bucket] if r else (150, 150, 150),
            "label": FRIENDLY_NAME.get(t.category, ""),
        })
    canvas = BevCanvas()
    bev = render_bev(
        bev_tracks, canvas=canvas,
        lanes=[sample_line(l) for l in out.lane_lines],
        ego_corridor=((sample_line(corr.left), sample_line(corr.right))
                      if corr else None),
    )
    # "You" marker label under the ego triangle.
    eu, ev = canvas.to_px(0, 0)
    cv2.putText(bev, "YOU", (eu - 14, min(ev + 22, 534)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    # Minimum safe following distance at current speed (RSS), plain words.
    from adp.risk.rss import rss_min_gap
    d_rss = rss_min_gap(out.ego_speed_ms, 0.0)
    if d_rss > 2.0:
        radius = int((d_rss + 3.5) * canvas.scale)
        cv2.ellipse(bev, (eu, ev), (radius, radius), 0, 245, 295,
                    (0, 120, 230), 1, cv2.LINE_AA)
        u, v = canvas.to_px(d_rss + 3.5, 0)
        cv2.putText(bev, f"safe gap {d_rss:.0f} m", (u + 6, v),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 140, 240), 1, cv2.LINE_AA)
    # Title + legend.
    cv2.putText(bev, "Top-down view", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    legend_y = 536
    for text, color, x in (("fine", BUCKET_COLORS["low"], 10),
                           ("caution", BUCKET_COLORS["caution"], 70),
                           ("warning", BUCKET_COLORS["high"], 160),
                           ("arrow = movement", (180, 180, 180), 260)):
        cv2.circle(bev, (x, legend_y - 4), 4, color, -1) if "=" not in text else None
        cv2.putText(bev, text, (x + 10 if "=" not in text else x, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return np.hstack([cam_img, bev])
