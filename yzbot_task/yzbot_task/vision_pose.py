"""Wrist camera → cube 3D pose in arm_base_link (no Gazebo ground truth)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer
from cv_bridge import CvBridge
from rclpy.time import Time
from rclpy.duration import Duration

from yzbot_task.vision_align import BlobResult, detect_blob, annotate_bgr


@dataclass
class CubePoseEstimate:
    pose_arm: PoseStamped
    blob: BlobResult
    debug_bgr: Optional[np.ndarray] = None
    method: str = "shelf"


@dataclass
class EstimateResult:
    ok: bool
    estimate: Optional[CubePoseEstimate] = None
    reason: str = ""


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> Quaternion:
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    q = Quaternion()
    q.w = float(cr * cp * cy + sr * sp * sy)
    q.x = float(sr * cp * cy - cr * sp * sy)
    q.y = float(cr * sp * cy + sr * cp * sy)
    q.z = float(cr * cp * sy - sr * sp * cy)
    return q


def grasp_orientation_down() -> Quaternion:
    return _quat_from_rpy(0.0, np.pi / 2.0, 0.0)


def synthesize_camera_info(width: int, height: int, hfov_deg: float = 60.0) -> CameraInfo:
    """When Gazebo camera_info missing, approximate pinhole from image size."""
    info = CameraInfo()
    info.width = int(width)
    info.height = int(height)
    fx = width / (2.0 * np.tan(np.deg2rad(hfov_deg) * 0.5))
    fy = fx
    cx, cy = width * 0.5, height * 0.5
    info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    return info


def sample_depth_m(
    depth_msg: Image,
    bridge: CvBridge,
    u: float,
    v: float,
    scale_u: float,
    scale_v: float,
) -> Optional[float]:
    try:
        if depth_msg.encoding in ("16UC1", "mono16"):
            depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            depth = depth.astype(np.float32) * 0.001
        else:
            depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            depth = depth.astype(np.float32)
    except Exception:  # noqa: BLE001
        return None
    h, w = depth.shape[:2]
    cu = int(np.clip(u * scale_u, 0, w - 1))
    cv_ = int(np.clip(v * scale_v, 0, h - 1))
    r = 3
    patch = depth[max(0, cv_ - r) : cv_ + r + 1, max(0, cu - r) : cu + r + 1]
    vals = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 3.0)]
    if vals.size < 3:
        return None
    return float(np.median(vals))


def pixel_to_ray(u: float, v: float, info: CameraInfo) -> Tuple[float, float, float]:
    K = info.k
    fx, fy = float(K[0]), float(K[4])
    cx, cy = float(K[2]), float(K[5])
    if abs(fx) < 1e-6 or abs(fy) < 1e-6:
        fx = fy = 300.0
        cx, cy = info.width * 0.5, info.height * 0.5
    return (u - cx) / fx, (v - cy) / fy, 1.0


def _lookup(tf_buffer: Buffer, target: str, source: str):
    return tf_buffer.lookup_transform(
        target, source, Time(), timeout=Duration(seconds=0.5)
    )


def _to_arm_base(
    tf_buffer: Buffer, frame: str, x: float, y: float, z: float, stamp
) -> Optional[PoseStamped]:
    from tf2_geometry_msgs import do_transform_point

    ps = PointStamped()
    ps.header.frame_id = frame
    ps.header.stamp = stamp
    ps.point.x, ps.point.y, ps.point.z = float(x), float(y), float(z)
    try:
        tf = _lookup(tf_buffer, "arm_base_link", frame)
        pa = do_transform_point(ps, tf)
    except Exception:  # noqa: BLE001
        return None
    pose = PoseStamped()
    pose.header.frame_id = "arm_base_link"
    pose.header.stamp = stamp
    pose.pose.position.x = pa.point.x
    pose.pose.position.y = pa.point.y
    pose.pose.position.z = pa.point.z
    pose.pose.orientation = grasp_orientation_down()
    return pose


def estimate_cube_pose(
    *,
    bgr: np.ndarray,
    color: str,
    bridge: CvBridge,
    tf_buffer: Buffer,
    camera_frame: str,
    camera_info: Optional[CameraInfo],
    depth_msg: Optional[Image],
    shelf_z_base: float = 0.75,
    stamp,
) -> EstimateResult:
    """HSV blob → 3D point in arm_base_link. Always returns a reason on failure."""
    if bgr is None or bgr.size == 0:
        return EstimateResult(False, reason="no_bgr")

    blob = detect_blob(bgr, color, max_width=320)
    dbg = annotate_bgr(bgr, blob, label=f"cube {color}" if blob else f"NO_BLOB {color}")
    if blob is None:
        return EstimateResult(
            False, reason="no_hsv_blob", estimate=CubePoseEstimate(
                pose_arm=PoseStamped(), blob=BlobResult(color, 0, 0, 0, 1, 1), debug_bgr=dbg
            )
        )

    info = camera_info
    used_synth = False
    if info is None:
        info = synthesize_camera_info(bgr.shape[1], bgr.shape[0])
        used_synth = True

    full_w, full_h = info.width, info.height
    su = full_w / max(blob.image_w, 1)
    sv = full_h / max(blob.image_h, 1)
    u, v = blob.cx * su, blob.cy * sv

    if depth_msg is not None:
        dsu = depth_msg.width / max(full_w, 1)
        dsv = depth_msg.height / max(full_h, 1)
        depth = sample_depth_m(depth_msg, bridge, u, v, dsu, dsv)
        if depth is not None:
            rx, ry, rz = pixel_to_ray(u, v, info)
            pose = _to_arm_base(
                tf_buffer, camera_frame, rx * depth, ry * depth, rz * depth, stamp
            )
            if pose is not None:
                est = EstimateResult(
                    True,
                    CubePoseEstimate(
                        pose_arm=pose, blob=blob, debug_bgr=dbg, method="depth"
                    ),
                )
                return apply_pose_sanity(est)
            # fall through to shelf

    from tf2_geometry_msgs import do_transform_point

    rx, ry, rz = pixel_to_ray(u, v, info)
    try:
        tf_cb = _lookup(tf_buffer, "base_link", camera_frame)
    except Exception as exc:  # noqa: BLE001
        return EstimateResult(
            False,
            reason=f"tf_base_from_cam_fail:{exc}",
            estimate=CubePoseEstimate(
                pose_arm=PoseStamped(), blob=blob, debug_bgr=dbg
            ),
        )

    p0 = PointStamped()
    p0.header.frame_id = camera_frame
    p0.header.stamp = stamp
    p1 = PointStamped()
    p1.header.frame_id = camera_frame
    p1.header.stamp = stamp
    p1.point.x, p1.point.y, p1.point.z = rx, ry, rz
    a = do_transform_point(p0, tf_cb)
    b = do_transform_point(p1, tf_cb)
    dz = b.point.z - a.point.z
    if abs(dz) < 1e-6:
        return EstimateResult(False, reason="ray_parallel_to_shelf", estimate=CubePoseEstimate(
            pose_arm=PoseStamped(), blob=blob, debug_bgr=dbg
        ))
    t = (shelf_z_base - a.point.z) / dz
    if t < 0.05 or t > 5.0:
        return EstimateResult(
            False,
            reason=f"ray_t_out_of_range:{t:.2f}",
            estimate=CubePoseEstimate(
                pose_arm=PoseStamped(), blob=blob, debug_bgr=dbg
            ),
        )
    px = a.point.x + t * (b.point.x - a.point.x)
    py = a.point.y + t * (b.point.y - a.point.y)
    pose = _to_arm_base(tf_buffer, "base_link", px, py, shelf_z_base, stamp)
    if pose is None:
        return EstimateResult(
            False,
            reason="tf_to_arm_base_fail",
            estimate=CubePoseEstimate(
                pose_arm=PoseStamped(), blob=blob, debug_bgr=dbg
            ),
        )
    method = "shelf_synth_info" if used_synth else "shelf"
    return apply_pose_sanity(
        EstimateResult(
            True,
            CubePoseEstimate(pose_arm=pose, blob=blob, debug_bgr=dbg, method=method),
        )
    )


def clamp_grasp_pose_arm(
    pose: PoseStamped,
    *,
    r_min: float = 0.25,
    r_max: float = 0.55,
    z_min: float = 0.35,
    z_max: float = 0.72,
    max_reach: float = 0.78,
) -> PoseStamped:
    """Light workspace nudge only — preserve shelf height; do not invent XY.

    Prefer closer Nav parking over shrinking the estimated range.
    """
    out = PoseStamped()
    out.header = pose.header
    out.pose.orientation = grasp_orientation_down()
    x = float(pose.pose.position.x)
    y = float(pose.pose.position.y)
    z = float(pose.pose.position.z)
    r = float(np.hypot(x, y))
    if r < 1e-4:
        x, y, r = float(r_min), 0.0, float(r_min)
    if r < r_min:
        s = r_min / r
        x, y, r = x * s, y * s, r_min
    elif r > r_max:
        s = r_max / r
        x, y, r = x * s, y * s, r_max
    z = float(np.clip(z, z_min, z_max))
    d = float(np.sqrt(x * x + y * y + z * z))
    if d > max_reach and d > 1e-6:
        # Prefer reduce XY slightly, keep z (shelf) — last resort only
        s = max_reach / d
        x, y, z = x * s, y * s, z * s
    out.pose.position.x = x
    out.pose.position.y = y
    out.pose.position.z = z
    return out


def apply_pose_sanity(result: EstimateResult) -> EstimateResult:
    """Validate estimate; if cube is truly far, ask for closer parking.

    Standing at map x=−0.55 with cube at −1.5 leaves ~0.95 m — outside this arm.
    Do not fake-clamp that into mid-reach empty air.
    """
    if not result.ok or result.estimate is None:
        return result
    pose = result.estimate.pose_arm
    x0 = float(pose.pose.position.x)
    y0 = float(pose.pose.position.y)
    z0 = float(pose.pose.position.z)
    r0 = float(np.hypot(x0, y0))

    if r0 > 0.60:
        return EstimateResult(
            False,
            reason=(
                f"POSE_REJECT need_closer "
                f"est=({x0:.2f},{y0:.2f},{z0:.2f}) r={r0:.2f} "
                f"(arm short; park ~0.35m ahead of cube, not ~1m)"
            ),
            estimate=result.estimate,
        )

    ok, why = pose_sanity_arm_base(pose)
    if ok:
        result.reason = f"REACH_OK {why}"
        return result

    # Only mild z clip / tiny r fix — refuse large XY teleport
    clamped = clamp_grasp_pose_arm(pose)
    rc = float(np.hypot(clamped.pose.position.x, clamped.pose.position.y))
    if abs(rc - r0) > 0.12:
        return EstimateResult(
            False,
            reason=(
                f"POSE_REJECT est_mismatch r={r0:.2f}→{rc:.2f} "
                f"(fix parking distance, not fake clamp)"
            ),
            estimate=result.estimate,
        )
    ok2, why2 = pose_sanity_arm_base(clamped)
    if not ok2:
        return EstimateResult(False, reason=why2, estimate=result.estimate)

    result.estimate.pose_arm = clamped
    result.estimate.method = f"{result.estimate.method}+clamp"
    result.reason = f"REACH_OK POSE_CLAMPED ({x0:.2f},{y0:.2f},{z0:.2f}) r={r0:.2f} → {why2}"
    result.ok = True
    return result


def pose_sanity_arm_base(pose: PoseStamped) -> Tuple[bool, str]:
    """Plausible shelf grasp in arm_base (IK may still need joint fallback)."""
    x = float(pose.pose.position.x)
    y = float(pose.pose.position.y)
    z = float(pose.pose.position.z)
    r = float(np.hypot(x, y))
    d = float(np.sqrt(x * x + y * y + z * z))
    if r < 0.18:
        return False, f"POSE_REJECT too_close r={r:.3f}"
    if r > 0.60:
        return False, f"POSE_REJECT need_closer r={r:.3f}"
    if z < 0.30 or z > 0.85:
        return False, f"POSE_REJECT bad_z z={z:.3f}"
    if d > 0.90:
        return False, f"POSE_REJECT out_of_reach d={d:.3f}"
    if abs(y) > 0.40:
        return False, f"POSE_REJECT bad_y y={y:.3f}"
    return True, f"POSE_OK r={r:.3f} z={z:.3f} d={d:.3f}"


def approach_xy_for_cube(
    color: str,
    cube_id: int,
    *,
    standoff_m: float = 0.35,
    robot_xy: Optional[Tuple[float, float]] = None,
    robot_yaw: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Park axis-aligned with base +X facing the cube (camera must see it).

    Blue#2: EAST of cube (map x > cube x) — stand on +X side.
    Other blue: WEST of cube (map x < cube x).
    Red: prefer same-Y (±x); N/S only if clearly nearer.
    Yaw is always atan2(cube − park) so the front camera looks at the cube.
    """
    cx, cy, _z = known_cube_xy(color, cube_id)
    so = float(standoff_m)

    def _face(ax: float, ay: float) -> float:
        return float(np.arctan2(cy - ay, cx - ax))

    if str(color).lower() == "blue":
        # blue#2: park on +X (east), farther standoff so path does not skim the cube
        if int(cube_id) == 2:
            so2 = max(so, 0.40)
            ax, ay = float(cx + so2), float(cy)
        else:
            ax, ay = float(cx - so), float(cy)
        return ax, ay, _face(ax, ay)

    same_y = [
        (cx + so, cy),  # east
        (cx - so, cy),  # west
    ]
    same_x = [
        (cx, cy + so),  # north
        (cx, cy - so),  # south
    ]
    if robot_xy is None:
        ax, ay = same_y[0]
        return float(ax), float(ay), _face(ax, ay)

    rx, ry = float(robot_xy[0]), float(robot_xy[1])

    def _d2(p):
        return (p[0] - rx) ** 2 + (p[1] - ry) ** 2

    def _score(p):
        # Prefer nearby park; also prefer heading already close to face-cube yaw
        # so Nav2 does not stop with travel-direction yaw and miss the cube.
        ax, ay = p
        face = _face(ax, ay)
        dist = _d2(p)
        if robot_yaw is None:
            return dist
        dyaw = abs(float(np.arctan2(np.sin(face - robot_yaw), np.cos(face - robot_yaw))))
        return dist + (0.55 * dyaw) ** 2  # ~0.55m equiv per radian

    best_y = min(same_y, key=_score)
    best_x = min(same_x, key=_score)
    if _d2(best_x) < _d2(best_y) * 0.45:
        ax, ay = best_x
    else:
        ax, ay = best_y
    return float(ax), float(ay), _face(ax, ay)


def diagnose_est_vs_known(
    tf_buffer: Buffer, color: str, cube_id: int, est: PoseStamped
) -> str:
    """Compare vision estimate to map cube (debug parking/vision error)."""
    stamp = est.header.stamp
    x, y, z = known_cube_xy(color, cube_id)
    pose_gt = _to_arm_base(tf_buffer, "map", x, y, z, stamp)
    if pose_gt is None:
        pose_gt = _to_arm_base(tf_buffer, "odom", x, y, z, stamp)
    if pose_gt is None:
        return "EST_vs_KNOWN tf_fail"
    ex, ey, ez = (
        float(est.pose.position.x),
        float(est.pose.position.y),
        float(est.pose.position.z),
    )
    gx, gy, gz = (
        float(pose_gt.pose.position.x),
        float(pose_gt.pose.position.y),
        float(pose_gt.pose.position.z),
    )
    er = float(np.hypot(ex, ey))
    gr = float(np.hypot(gx, gy))
    err = float(np.sqrt((ex - gx) ** 2 + (ey - gy) ** 2 + (ez - gz) ** 2))
    return (
        f"EST_vs_KNOWN est_r={er:.2f} known_r={gr:.2f} "
        f"err={err:.2f}m known=({gx:.2f},{gy:.2f},{gz:.2f})"
    )

def joint1_from_arm_pose(pose: PoseStamped) -> float:
    x = float(pose.pose.position.x)
    y = float(pose.pose.position.y)
    j1 = float(np.arctan2(y, max(x, 0.05)))
    return float(np.clip(j1, -0.7, 0.7))


def joint1_from_blob(blob: BlobResult, span_rad: float = 0.55) -> float:
    """Rough joint1 from front/wrist blob: image-left → +j1 (shelf to robot left)."""
    nx = float(blob.cx) / max(float(blob.image_w), 1.0) - 0.5
    return float(np.clip(-nx * 2.0 * span_rad, -0.7, 0.7))


def offset_pose(pose: PoseStamped, dz: float) -> PoseStamped:
    out = PoseStamped()
    out.header = pose.header
    out.pose.orientation = pose.pose.orientation
    out.pose.position.x = pose.pose.position.x
    out.pose.position.y = pose.pose.position.y
    out.pose.position.z = pose.pose.position.z + dz
    return out


def ee_to_pose_distance_m(tf_buffer: Buffer, pose_arm: PoseStamped) -> Optional[float]:
    """Distance from grasping_frame to target pose (both in arm_base_link)."""
    try:
        tf = _lookup(tf_buffer, "arm_base_link", "grasping_frame")
    except Exception:  # noqa: BLE001
        try:
            tf = _lookup(tf_buffer, "arm_base_link", "link6")
        except Exception:  # noqa: BLE001
            return None
    t = tf.transform.translation
    dx = float(t.x) - float(pose_arm.pose.position.x)
    dy = float(t.y) - float(pose_arm.pose.position.y)
    dz = float(t.z) - float(pose_arm.pose.position.z)
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


# Layout from Gazebo export (export_cube_poses apply:=true).
# Pairwise spacing should stay ≥2m for competition.
_KNOWN_CUBES = {
    ("red", 1): (-6.89, -2.80, 0.015),
    ("red", 2): (-2.00, -5.85, 0.015),
    ("red", 3): (4.24, -10.28, 0.015),
    ("red", 4): (-2.77, 5.80, 0.015),
    ("red", 5): (1.61, 3.35, 0.015),
    ("blue", 1): (1.97, -0.24, 0.015),
    ("blue", 2): (-5.87, -6.68, 0.015),
    ("blue", 3): (-4.15, 2.35, 0.015),
    ("blue", 4): (1.82, -5.28, 0.015),
    ("blue", 5): (-1.78, -11.72, 0.015),
}

def known_cube_xy(color: str, cube_id: int):
    """Map pose of numbered cube (must match Gazebo world)."""
    key = (color.lower(), int(cube_id))
    if key not in _KNOWN_CUBES:
        raise KeyError(f"unknown cube {color}_{cube_id}")
    return _KNOWN_CUBES[key]


def cube_bearing_in_base(
    tf_buffer: Buffer, color: str, cube_id: int
) -> Optional[Tuple[float, float, Tuple[float, float, float]]]:
    """Map known cube → (bearing_rad, range_xy, map_xyz) in base_link.

    bearing = atan2(y, x) in base_link: 0 = ahead (+X), + = left (+Y), − = right.
    Do NOT clamp x>0 — that flipped the turn when the cube was slightly behind.
    """
    from tf2_geometry_msgs import do_transform_point

    x, y, z = known_cube_xy(color, cube_id)
    ps = PointStamped()
    ps.header.frame_id = "map"
    ps.point.x, ps.point.y, ps.point.z = x, y, z
    try:
        tf = _lookup(tf_buffer, "base_link", "map")
        p = do_transform_point(ps, tf)
    except Exception:  # noqa: BLE001
        try:
            tf = _lookup(tf_buffer, "base_link", "odom")
            ps.header.frame_id = "odom"
            p = do_transform_point(ps, tf)
        except Exception:  # noqa: BLE001
            return None
    bx, by = float(p.point.x), float(p.point.y)
    # Full atan2 — preserves quadrant (behind-left / behind-right)
    bearing = float(np.arctan2(by, bx))
    rng = float(np.hypot(bx, by))
    return bearing, rng, (x, y, z)


def known_cube_pose_arm_raw(
    tf_buffer: Buffer, color: str, cube_id: int, stamp
) -> EstimateResult:
    """Map cube XYZ → arm_base_link via TF only (no clamp / no reject).

    Used for 'nav near then grasp fixed coord' pipeline.
    """
    x, y, z = known_cube_xy(color, cube_id)
    pose = _to_arm_base(tf_buffer, "map", x, y, z, stamp)
    if pose is None:
        pose = _to_arm_base(tf_buffer, "odom", x, y, z, stamp)
    if pose is None:
        return EstimateResult(False, reason="known_map_tf_fail")
    r = float(np.hypot(pose.pose.position.x, pose.pose.position.y))
    return EstimateResult(
        True,
        CubePoseEstimate(
            pose_arm=pose,
            blob=BlobResult(color, 0, 0, 0, 1, 1),
            method="known_map",
        ),
        reason=(
            f"CUBE_MAP ({x:.2f},{y:.2f},{z:.2f}) → arm "
            f"({pose.pose.position.x:.3f},{pose.pose.position.y:.3f},"
            f"{pose.pose.position.z:.3f}) r={r:.3f}"
        ),
    )


def known_cube_pose_arm(
    tf_buffer: Buffer, color: str, cube_id: int, stamp
) -> EstimateResult:
    """Known map cube with reach sanity (optional / legacy)."""
    raw = known_cube_pose_arm_raw(tf_buffer, color, cube_id, stamp)
    if not raw.ok:
        return raw
    return apply_pose_sanity(raw)


def aim_joint1_toward_cube(tf_buffer: Buffer, color: str, cube_id: int) -> float:
    """Rough joint1 (rad) so arm base yaw points at known cube in base_link."""
    from tf2_geometry_msgs import do_transform_point

    x, y, z = known_cube_xy(color, cube_id)
    ps = PointStamped()
    ps.header.frame_id = "map"
    ps.point.x, ps.point.y, ps.point.z = x, y, z
    try:
        tf = _lookup(tf_buffer, "base_link", "map")
        p = do_transform_point(ps, tf)
    except Exception:  # noqa: BLE001
        try:
            tf = _lookup(tf_buffer, "base_link", "odom")
            ps.header.frame_id = "odom"
            p = do_transform_point(ps, tf)
        except Exception:  # noqa: BLE001
            return 0.0
    j1 = float(np.arctan2(p.point.y, max(p.point.x, 0.05)))
    return float(np.clip(j1, -0.7, 0.7))



def bgr_hue_hint(bgr: np.ndarray) -> str:
    """Compact debug: how much red/blue-ish pixels in frame."""
    if bgr is None or bgr.size == 0:
        return "empty"
    small = cv2.resize(bgr, (80, 45), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 40, 40), (15, 255, 255)),
        cv2.inRange(hsv, (165, 40, 40), (180, 255, 255)),
    )
    blue = cv2.inRange(hsv, (90, 40, 40), (135, 255, 255))
    n = float(small.shape[0] * small.shape[1])
    return f"red_px={np.count_nonzero(red)/n:.1%} blue_px={np.count_nonzero(blue)/n:.1%}"

