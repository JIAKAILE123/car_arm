"""Lightweight OpenCV color blob align (no DNN).

Wrist-camera: grab ONE frame, HSV blob → joint1 delta (arm only, never Nav2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
    c = color.lower()
    if c == "red":
        # Wider for Gazebo materials / lighting
        m1 = cv2.inRange(hsv, (0, 40, 40), (15, 255, 255))
        m2 = cv2.inRange(hsv, (165, 40, 40), (180, 255, 255))
        return cv2.bitwise_or(m1, m2)
    if c == "blue":
        return cv2.inRange(hsv, (90, 40, 40), (135, 255, 255))
    raise ValueError(f"unsupported color: {color}")


def detect_blob(
    bgr: np.ndarray,
    color: str,
    max_width: int = 320,
    *,
    target_u: float = 0.0,
    target_v: float = 0.0,
) -> Optional[BlobResult]:
    """Pick the color blob nearest the crosshair (not just largest).

    With 5 cubes in one FOV, largest can be the wrong neighbor.
    """
    h0, w0 = bgr.shape[:2]
    if w0 > max_width:
        scale = max_width / float(w0)
        bgr = cv2.resize(bgr, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = _hsv_mask(hsv, color)
    mask = cv2.medianBlur(mask, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    tx = (0.5 + 0.5 * target_u) * w
    ty = (0.5 + 0.5 * target_v) * h
    best: Optional[BlobResult] = None
    best_score = -1e18
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 20:
            continue
        m = cv2.moments(cnt)
        if m["m00"] <= 1e-6:
            continue
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        x, y, bw, bh = cv2.boundingRect(cnt)
        # Prefer near crosshair; slight bonus for larger area
        dist = float(np.hypot(cx - tx, cy - ty)) / max(float(w), 1.0)
        score = float(area) / (1.0 + 8.0 * dist)
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
) -> np.ndarray:
    """Draw center crosshair + bounding rect (cube must sit on the cross)."""
    out = bgr.copy()
    h, w = out.shape[:2]
    tx = int((0.5 + 0.5 * target_u) * w)
    ty = int((0.5 + 0.5 * target_v) * h)
    cv2.line(out, (tx, 0), (tx, h), (0, 255, 0), 2)
    cv2.line(out, (0, ty), (w, ty), (0, 255, 0), 1)
    cv2.drawMarker(out, (tx, ty), (0, 255, 0), cv2.MARKER_CROSS, 28, 2)
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
