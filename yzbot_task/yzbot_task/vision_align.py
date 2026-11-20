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

    def err_v_at(self, lock_v_frac: float = 0.62) -> float:
        """-1..1 vs horizontal lock line (lock_v_frac of image height)."""
        target = float(lock_v_frac) * float(self.image_h)
        return (self.cy - target) / max(self.image_h * 0.5, 1.0)

    @property
    def err_v(self) -> float:
        """-1..1 vs lower crosshair row (default lock line)."""
        return self.err_v_at(0.62)


def _hsv_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    """Strict masks: cubes are saturated pure color; walls/paint are duller."""
    c = color.lower()
    if c == "red":
        # High S/V → pure red plastic cube; reject brick / faded panels
        m1 = cv2.inRange(hsv, (0, 145, 85), (10, 255, 255))
        m2 = cv2.inRange(hsv, (170, 145, 85), (180, 255, 255))
        return cv2.bitwise_or(m1, m2)
    if c == "blue":
        # 远距蓝块 S/V 更低，略放宽
        return cv2.inRange(hsv, (92, 85, 45), (132, 255, 255))
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
    *,
    min_cy_frac: float = 0.28,
) -> bool:
    """Reject wall patches: too big, elongated, low fill, or too high in image.

    Front cam: floor cubes sit mid/lower; high cy-reject = ignore tall/wall blobs.
    """
    img_a = float(max(w * h, 1))
    frac = area / img_a
    # 3cm cube at ~0.4–1.5m is small; wall patches are huge
    if frac > 0.06:
        return False
    # 远距小块面积更小，仍须通过（原 14 易漏远处物块）
    if area < 8:
        return False
    aspect = float(bw) / float(max(bh, 1))
    if aspect < 0.45 or aspect > 2.2:
        return False
    # extent: contour area vs bbox
    extent = area / float(max(bw * bh, 1))
    if extent < 0.30:
        return False
    # Height gate (image row ≈ camera "height"): too high → not a floor cube
    if cy < float(min_cy_frac) * h:
        return False
    if float(y) < 0.18 * h:
        return False
    roi = mask[y : min(h, y + bh), x : min(w, x + bw)]
    if roi.size < 4:
        return False
    return True


def detect_blob(
    bgr: np.ndarray,
    color: str,
    max_width: int = 480,
    *,
    target_u: float = 0.0,
    target_v: float = 0.35,
    prefer_uv: Optional[Tuple[float, float]] = None,
    prefer_max_dist_frac: float = 0.25,
    min_cy_frac: float = 0.28,
) -> Optional[BlobResult]:
    """Detect saturated cube blob; reject brick-wall / high-in-frame FPs.

    prefer_uv: optional (u, v) in *original* image pixels (known-cube projection).
    When set, only keep blobs within prefer_max_dist_frac * width of that point
    so opposite red walls are ignored.

    min_cy_frac: reject blobs whose centroid is above this image-row fraction
    (floor cubes are lower; walls/ceilings are high).
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
    # 远距饱和度略降，近距仍靠 area/位置筛墙
    sat_floor = 125.0 if color.lower() == "red" else 110.0

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        x, y, bw, bh = cv2.boundingRect(cnt)
        m = cv2.moments(cnt)
        if m["m00"] <= 1e-6:
            continue
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        if not _cube_like(
            area, bw, bh, cx, cy, w, h, mask, x, y, min_cy_frac=min_cy_frac
        ):
            continue

        if pref is not None:
            if float(np.hypot(cx - pref[0], cy - pref[1])) > max_pref:
                continue

        x1 = min(w, x + bw)
        y1 = min(h, y + bh)
        sat = float(np.mean(hsv[y:y1, x:x1, 1])) if y1 > y and x1 > x else 0.0
        if sat < sat_floor:
            continue
        val = float(np.mean(hsv[y:y1, x:x1, 2])) if y1 > y and x1 > x else 0.0
        if val < 70.0:
            continue

        dist = float(np.hypot(cx - tx, cy - ty)) / max(float(w), 1.0)
        score = (area * (0.5 + sat / 255.0)) / (1.0 + 10.0 * dist)
        # Prefer lower-in-frame (floor) over high wall patches
        score *= 0.55 + 0.45 * (cy / max(float(h), 1.0))
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


def grasp_box_xyxy(
    image_w: int,
    image_h: int,
    *,
    lock_v_frac: float = 0.82,
    grasp_box_frac: float = 0.15,
    target_u: float = 0.0,
) -> Tuple[float, float, float, float, float, float]:
    """Return (x0,y0,x1,y1, cx, cy) of the square grasp window in image pixels."""
    w = float(image_w)
    h = float(image_h)
    cx = (0.5 + 0.5 * float(target_u)) * w
    cy = float(lock_v_frac) * h
    half = 0.5 * float(grasp_box_frac) * min(w, h)
    return cx - half, cy - half, cx + half, cy + half, cx, cy


def blob_bbox_in_image(
    blob: BlobResult, image_w: int, image_h: int
) -> Tuple[float, float, float, float, float, float]:
    """Scale detect-resolution bbox → full image; return x0,y0,x1,y1,bw,bh."""
    sx = float(image_w) / max(float(blob.image_w), 1.0)
    sy = float(image_h) / max(float(blob.image_h), 1.0)
    x0 = float(blob.x0) * sx
    y0 = float(blob.y0) * sy
    x1 = float(blob.x1) * sx
    y1 = float(blob.y1) * sy
    return x0, y0, x1, y1, max(0.0, x1 - x0), max(0.0, y1 - y0)


def box_fill_ratio(
    blob: BlobResult,
    image_w: int,
    image_h: int,
    *,
    lock_v_frac: float = 0.82,
    grasp_box_frac: float = 0.15,
    target_u: float = 0.0,
) -> float:
    """How much the cube bbox fills the gate (max side / gate side). ~1 = snug."""
    bx0, by0, bx1, by1, _cx, _cy = grasp_box_xyxy(
        image_w,
        image_h,
        lock_v_frac=lock_v_frac,
        grasp_box_frac=grasp_box_frac,
        target_u=target_u,
    )
    box_side = max(1e-3, min(bx1 - bx0, by1 - by0))
    _x0, _y0, _x1, _y1, bw, bh = blob_bbox_in_image(blob, image_w, image_h)
    return float(max(bw, bh) / box_side)


def estimate_range_m_pinhole(
    bbox_width_px: float,
    fx: float,
    *,
    object_width_m: float = 0.03,
) -> Optional[float]:
    """GitHub-style pinhole range: d = (W_real * fx) / W_pixels.

    Same idea as dimensionalos VisualServoing2D / classic OpenCV size→distance.
    """
    if bbox_width_px <= 1.0 or fx <= 1.0 or object_width_m <= 0.0:
        return None
    return float(object_width_m * fx) / float(bbox_width_px)


def cube_fully_in_box(
    blob: BlobResult,
    image_w: int,
    image_h: int,
    *,
    lock_v_frac: float = 0.82,
    grasp_box_frac: float = 0.15,
    target_u: float = 0.0,
) -> bool:
    """True if the cube's full bounding box lies inside the square grasp window."""
    bx0, by0, bx1, by1, _cx, _cy = grasp_box_xyxy(
        image_w,
        image_h,
        lock_v_frac=lock_v_frac,
        grasp_box_frac=grasp_box_frac,
        target_u=target_u,
    )
    x0, y0, x1, y1, _bw, _bh = blob_bbox_in_image(blob, image_w, image_h)
    return x0 >= bx0 and y0 >= by0 and x1 <= bx1 and y1 <= by1


def annotate_bgr(
    bgr: np.ndarray,
    blob: Optional[BlobResult] = None,
    *,
    target_u: float = 0.0,
    target_v: float = 0.0,
    label: str = "",
    prefer_uv: Optional[Tuple[float, float]] = None,
    lock_v_frac: float = 0.82,
    grasp_box_frac: float = 0.15,
    draw_grasp_box: bool = True,
) -> np.ndarray:
    """Draw lowered crosshair + snug square gate; cube should nearly fill the box."""
    out = bgr.copy()
    h, w = out.shape[:2]
    bx0, by0, bx1, by1, tx_f, ty_f = grasp_box_xyxy(
        w,
        h,
        lock_v_frac=lock_v_frac,
        grasp_box_frac=grasp_box_frac,
        target_u=target_u,
    )
    tx, ty = int(tx_f), int(ty_f)
    cv2.line(out, (tx, 0), (tx, h), (0, 255, 0), 2)
    cv2.line(out, (0, ty), (w, ty), (0, 255, 0), 1)
    cv2.drawMarker(out, (tx, ty), (0, 255, 0), cv2.MARKER_CROSS, 28, 2)

    in_box = False
    fill = 0.0
    if blob is not None:
        in_box = cube_fully_in_box(
            blob,
            w,
            h,
            lock_v_frac=lock_v_frac,
            grasp_box_frac=grasp_box_frac,
            target_u=target_u,
        )
        fill = box_fill_ratio(
            blob,
            w,
            h,
            lock_v_frac=lock_v_frac,
            grasp_box_frac=grasp_box_frac,
            target_u=target_u,
        )
    snug = in_box and fill >= 0.88
    if draw_grasp_box:
        gate_color = (0, 255, 0) if snug else (0, 165, 255)
        cv2.rectangle(
            out,
            (int(bx0), int(by0)),
            (int(bx1), int(by1)),
            gate_color,
            2,
        )
        tag = f"SNUG fill={fill:.2f}" if snug else f"FIT fill={fill:.2f}"
        cv2.putText(
            out,
            tag,
            (int(bx0), max(16, int(by0) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            gate_color,
            1,
            cv2.LINE_AA,
        )

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
    if blob is not None:
        sx = w / max(blob.image_w, 1)
        sy = h / max(blob.image_h, 1)
        cx = int(blob.cx * sx)
        cy = int(blob.cy * sy)
        x0, y0 = int(blob.x0 * sx), int(blob.y0 * sy)
        x1, y1 = int(blob.x1 * sx), int(blob.y1 * sy)
        locked = snug and abs(blob.err_u) < 0.08
        box = (0, 255, 0) if locked else (0, 165, 255)
        cv2.rectangle(out, (x0, y0), (x1, y1), box, 2)
        cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)
        cv2.line(out, (cx, cy), (tx, ty), (0, 255, 255), 2)
        frac = float(blob.area) / max(float(blob.image_w * blob.image_h), 1.0)
        tag = (
            "LOCK_OK"
            if locked
            else f"err_u={blob.err_u:+.2f} fill={fill:.2f}"
        )
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


def blob_to_joint1_delta(    blob: BlobResult,
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
    gain: float = 0.70,
    max_wz: float = 0.22,
    target_u: float = 0.0,
) -> float:
    """Front-cam lateral error → chassis yaw rate (MODE_LOCAL only).

    Blob on the RIGHT (+err_u) → turn right (−wz in ROS).
    Mild gain boost only — keep |wz| low so AMCL does not drift.
    """
    err_u = blob.err_u - target_u
    g = gain * (1.0 + 0.6 * min(abs(err_u), 1.0))
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


def detect_green_pad(
    bgr: np.ndarray,
    max_width: int = 480,
    *,
    min_area_frac: float = 0.012,
    max_area_frac: float = 0.55,
) -> Optional[BlobResult]:
    """Detect letter-zone green pad (large floor mat), not a cube.

    Used at place time after Nav dock + face: centroid ≈ pad center in FOV.
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
    # Gazebo / 赛场绿垫：偏亮青绿，放宽 H 以覆盖不同光照
    mask = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_a = float(max(w * h, 1))
    best: Optional[BlobResult] = None
    best_score = -1e18
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        frac = area / img_a
        if frac < float(min_area_frac) or frac > float(max_area_frac):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        # 垫子偏扁横条（长边对相机），拒细竖条/噪声
        aspect = float(bw) / float(max(bh, 1))
        if aspect < 0.6 or aspect > 6.0:
            continue
        m = cv2.moments(cnt)
        if m["m00"] <= 1e-6:
            continue
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        # 绿垫应在画面中下部（地面）；太靠上多半是墙绿/噪声
        if cy < 0.22 * h:
            continue
        # 大面积 + 靠画面中心优先
        dist = float(np.hypot(cx - 0.5 * w, cy - 0.55 * h)) / max(float(w), 1.0)
        score = area / (1.0 + 6.0 * dist)
        score *= 0.5 + 0.5 * (cy / max(float(h), 1.0))
        if score > best_score:
            best_score = score
            best = BlobResult(
                color="green",
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


def green_pad_centered(
    blob: Optional[BlobResult],
    *,
    max_err_u: float = 0.14,
    target_cy_frac: float = 0.58,
    max_err_v: float = 0.22,
    min_area_frac: float = 0.02,
) -> bool:
    """True when green-pad centroid is near image aim (place sweet spot)."""
    if blob is None or str(blob.color).lower() != "green":
        return False
    frac = blob_area_frac(blob)
    if frac < float(min_area_frac):
        return False
    if abs(float(blob.err_u)) > float(max_err_u):
        return False
    cy_frac = float(blob.cy) / float(max(blob.image_h, 1))
    if abs(cy_frac - float(target_cy_frac)) > float(max_err_v):
        return False
    return True
