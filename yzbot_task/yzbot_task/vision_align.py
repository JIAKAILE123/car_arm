"""Lightweight OpenCV color blob align (no DNN).

Wrist-camera: grab ONE frame, HSV blob → joint1 delta (arm only, never Nav2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class BlobResult:
    color: str
    cx: float
    cy: float
    area: float
    image_w: int
    image_h: int
    # Bounding box in detect-resolution pixels (for FPV overlay)
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0

    @property
    def err_u(self) -> float:
        """-1..1, + means blob is on the RIGHT side of the image."""
        return (self.cx - self.image_w * 0.5) / max(self.image_w * 0.5, 1.0)

    @property
    def err_v(self) -> float:
        """-1..1, + means blob is LOWER than crosshair row."""
        target = self.image_h * 0.50
        return (self.cy - target) / max(self.image_h * 0.5, 1.0)


def _hsv_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    """Strict masks: cubes are saturated; brick walls are dull red/orange."""
    c = color.lower()
    if c == "red":
        # Higher S → keep pure red cube; reject brick / opposite wall
        m1 = cv2.inRange(hsv, (0, 110, 70), (12, 255, 255))
        m2 = cv2.inRange(hsv, (168, 110, 70), (180, 255, 255))
        return cv2.bitwise_or(m1, m2)
    if c == "blue":
        return cv2.inRange(hsv, (95, 90, 50), (130, 255, 255))
    raise ValueError(f"unsupported color: {color}")


def _cube_like(
    area: float,
    bw: int,
    bh: int,
    cx: float,
    cy: float,
    w: int,
    h: int,
    mask: np.ndarray,
    x: int,
    y: int,
) -> bool:
    """Reject wall patches: too big, too elongated, low fill, top-of-frame."""
    img_a = float(max(w * h, 1))
    frac = area / img_a
    # 3cm cube at ~0.4–1.5m is small; wall patches are huge
    if frac > 0.08:
        return False
    if area < 12:
        return False
    aspect = float(bw) / float(max(bh, 1))
    if aspect < 0.40 or aspect > 2.5:
        return False
    # extent: contour area vs bbox
    extent = area / float(max(bw * bh, 1))
    if extent < 0.25:
        return False
    # walls often fill upper FOV; cubes sit mid/lower when parked
    if cy < 0.12 * h:
        return False
    # mean saturation inside bbox — cubes high S, bricks lower
    x1 = min(w, x + bw)
    y1 = min(h, y + bh)
    roi = mask[y:y1, x:x1]
    if roi.size < 4:
        return False
    return True


def detect_blob(
    bgr: np.ndarray,
    color: str,
    max_width: int = 320,
    *,
    target_u: float = 0.0,
    target_v: float = 0.35,
    prefer_uv: Optional[Tuple[float, float]] = None,
    prefer_max_dist_frac: float = 0.25,
) -> Optional[BlobResult]:
    """Detect saturated cube blob; reject brick-wall false positives.

    prefer_uv: optional (u, v) in *original* image pixels (known-cube projection).
    When set, only keep blobs within prefer_max_dist_frac * width of that point
    so opposite red walls are ignored.
    """
    h0, w0 = bgr.shape[:2]
    scale = 1.0
    work = bgr
    if w0 > max_width:
        scale = max_width / float(w0)
        work = cv2.resize(
            bgr,
            (int(w0 * scale), int(h0 * scale)),
            interpolation=cv2.INTER_AREA,
        )
    h, w = work.shape[:2]
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    mask = _hsv_mask(hsv, color)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    tx = (0.5 + 0.5 * target_u) * w
    ty = (0.5 + 0.5 * target_v) * h
    pref = None
    if prefer_uv is not None:
        pref = (float(prefer_uv[0]) * scale, float(prefer_uv[1]) * scale)
        # aim score toward projection when available
        tx, ty = pref[0], pref[1]
    max_pref = float(prefer_max_dist_frac) * float(w)

    best: Optional[BlobResult] = None
    best_score = -1e18
    sat_floor = 100.0 if color.lower() == "red" else 85.0

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        x, y, bw, bh = cv2.boundingRect(cnt)
        m = cv2.moments(cnt)
        if m["m00"] <= 1e-6:
            continue
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        if not _cube_like(area, bw, bh, cx, cy, w, h, mask, x, y):
            continue

        if pref is not None:
            if float(np.hypot(cx - pref[0], cy - pref[1])) > max_pref:
                continue

        x1 = min(w, x + bw)
        y1 = min(h, y + bh)
        sat = float(np.mean(hsv[y:y1, x:x1, 1])) if y1 > y and x1 > x else 0.0
        if sat < sat_floor:
            continue

        dist = float(np.hypot(cx - tx, cy - ty)) / max(float(w), 1.0)
        score = (area * (0.5 + sat / 255.0)) / (1.0 + 10.0 * dist)
        score *= 0.7 + 0.3 * (cy / max(float(h), 1.0))
        if score > best_score:
            best_score = score
            best = BlobResult(
                color=color,
                cx=cx,
                cy=cy,
                area=area,
                image_w=w,
                image_h=h,
                x0=float(x),
                y0=float(y),
                x1=float(x + bw),
                y1=float(y + bh),
            )
    return best


def annotate_bgr(
    bgr: np.ndarray,
    blob: Optional[BlobResult] = None,
    *,
    target_u: float = 0.0,
    target_v: float = 0.0,
    label: str = "",
    prefer_uv: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Draw center crosshair + bounding rect (cube must sit on the cross)."""
    out = bgr.copy()
    h, w = out.shape[:2]
    tx = int((0.5 + 0.5 * target_u) * w)
    ty = int((0.5 + 0.5 * target_v) * h)
    cv2.line(out, (tx, 0), (tx, h), (0, 255, 0), 2)
    cv2.line(out, (0, ty), (w, ty), (0, 255, 0), 1)
    cv2.drawMarker(out, (tx, ty), (0, 255, 0), cv2.MARKER_CROSS, 28, 2)
    if prefer_uv is not None:
        pu, pv = int(prefer_uv[0]), int(prefer_uv[1])
        cv2.drawMarker(out, (pu, pv), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
        cv2.putText(
            out,
            "MAP_PROJ",
            (pu + 8, max(16, pv - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
        # gate radius visualization
        rad = int(0.25 * w)
        cv2.circle(out, (pu, pv), rad, (255, 0, 255), 1)
    if blob is not None:
        sx = w / max(blob.image_w, 1)
        sy = h / max(blob.image_h, 1)
        cx = int(blob.cx * sx)
        cy = int(blob.cy * sy)
        x0, y0 = int(blob.x0 * sx), int(blob.y0 * sy)
        x1, y1 = int(blob.x1 * sx), int(blob.y1 * sy)
        locked = abs(blob.err_u) < 0.08
        box = (0, 255, 0) if locked else (0, 165, 255)
        cv2.rectangle(out, (x0, y0), (x1, y1), box, 2)
        cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)
        cv2.line(out, (cx, cy), (tx, ty), (0, 255, 255), 2)
        tag = "CENTER_OK" if locked else f"err_u={blob.err_u:+.2f} err_v={blob.err_v:+.2f}"
        cv2.putText(
            out,
            f"{tag} area={blob.area:.0f}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0) if locked else (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            out,
            "NO_CUBE (wall/noise filtered)",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    if label:
        cv2.putText(
            out,
            label,
            (8, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def blob_to_joint1_delta(
    blob: BlobResult,
    *,
    gain: float = 0.35,
    max_delta: float = 0.25,
    target_u: float = 0.0,
) -> float:
    """Map image lateral error → joint1 (base yaw) delta in radians.

    delta_j1 = -gain * (err_u - target_u), clipped. No chassis motion.
    If grasp still biased after centering, shift target_u (not Nav2).
    """
    err_u = blob.err_u - target_u
    d = -gain * err_u
    return float(np.clip(d, -max_delta, max_delta))


def blob_to_yaw_cmd(
    blob: BlobResult,
    *,
    gain: float = 0.85,
    max_wz: float = 0.45,
    target_u: float = 0.0,
) -> float:
    """Front-cam lateral error → chassis yaw rate (MODE_LOCAL only).

    Blob on the RIGHT (+err_u) → turn right (−wz in ROS).
    Larger |err| uses higher gain so it actually reaches center.
    """
    err_u = blob.err_u - target_u
    g = gain * (1.0 + 1.5 * min(abs(err_u), 1.0))
    return float(np.clip(-g * err_u, -max_wz, max_wz))


def blob_range_vx(
    blob: BlobResult,
    *,
    frac_near: float = 0.06,
    frac_far: float = 0.008,
    vx_back: float = -0.06,
    vx_fwd: float = 0.04,
) -> float:
    """Heuristic stand-off from blob area fraction of the (possibly downscaled) image."""
    frac = float(blob.area) / max(float(blob.image_w * blob.image_h), 1.0)
    if frac > frac_near:
        return float(vx_back)
    if frac < frac_far:
        return float(vx_fwd)
    return 0.0


def blob_area_frac(blob: BlobResult) -> float:
    return float(blob.area) / max(float(blob.image_w * blob.image_h), 1.0)


def is_visual_lock(
    blob: Optional[BlobResult],
    *,
    min_area: float = 60.0,
    max_err_u: float = 0.55,
) -> bool:
    """True if color blob is locked (centroid usable as grasp aim). No chassis motion."""
    if blob is None:
        return False
    if float(blob.area) < min_area:
        return False
    if abs(float(blob.err_u)) > max_err_u:
        return False
    return True
