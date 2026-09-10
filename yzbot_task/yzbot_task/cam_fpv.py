"""OpenCV front-cam align assist.

Shows:
  - Green vertical = robot forward / base X (optical midline)
  - Cyan horizontal = base Y in image
  - Magenta = projected cube (from /yzbot/grasp_target map Point)
  - HUD: dy/dx to cube, yaw vs axis, ALIGN ok?

Run: ros2 run yzbot_task cam_fpv
"""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener

from yzbot_task.image_topics import resolve_image_topic, list_image_topics


def _yaw_from_quat(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _nearest_axis_yaw(yaw: float) -> Tuple[float, float]:
    """Return (snapped_yaw, abs_error) for {0, ±π/2, π}."""
    cands = (0.0, math.pi / 2.0, -math.pi / 2.0, math.pi, -math.pi)
    best = min(cands, key=lambda y: abs(_wrap_pi(yaw - y)))
    return best, abs(_wrap_pi(yaw - best))


class CamFpv(Node):
    def __init__(self):
        super().__init__("cam_fpv")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("info_topic", "/camera/camera_info")
        self.declare_parameter("debug_topic", "/yzbot/vision_debug")
        self.declare_parameter("target_topic", "/yzbot/grasp_target")
        self.declare_parameter("camera_frame", "camera_link")

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._raw = None
        self._dbg = None
        self._dbg_until = 0.0
        self._cinfo: Optional[CameraInfo] = None
        self._target: Optional[PointStamped] = None
        self._raw_sub = None
        self._bound_topic = None
        self._no_image_warned = False

        preferred = str(self.get_parameter("image_topic").value)
        self.create_subscription(
            Image, str(self.get_parameter("debug_topic").value), self._on_dbg, 10
        )
        self.create_subscription(
            CameraInfo, str(self.get_parameter("info_topic").value), self._on_info, 10
        )
        self.create_subscription(
            PointStamped, str(self.get_parameter("target_topic").value), self._on_tgt, 10
        )
        self._bind_raw(preferred)
        self.create_timer(0.05, self._tick)
        self.create_timer(1.0, self._rediscover_if_needed)
        self.get_logger().info(
            "FPV align assist: 绿竖线=车前进/base_x | 青横线=base_y | "
            "品红=物块投影 | 看是否落在绿线上=对准"
        )

    def _bind_raw(self, topic: str):
        if self._raw_sub is not None:
            self.destroy_subscription(self._raw_sub)
            self._raw_sub = None
        self._bound_topic = topic
        self._raw_sub = self.create_subscription(Image, topic, self._on_raw, 10)
        self.get_logger().info(f"FPV 订阅: {topic}")

    def _rediscover_if_needed(self):
        if self._raw is not None:
            return
        preferred = str(self.get_parameter("image_topic").value)
        topic = resolve_image_topic(self, preferred)
        if topic != self._bound_topic:
            self.get_logger().warn(f"无画面，改订: {self._bound_topic} → {topic}")
            self._bind_raw(topic)
        elif not self._no_image_warned:
            imgs = list_image_topics(self)
            self.get_logger().warn(
                f"仍无图像。Image={imgs or '(无)'}；先开 Gazebo"
            )
            self._no_image_warned = True

    def _on_raw(self, msg: Image):
        try:
            self._raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:  # noqa: BLE001
            return

    def _on_dbg(self, msg: Image):
        try:
            self._dbg = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self._dbg_until = time.time() + 2.5
        except Exception:  # noqa: BLE001
            return

    def _on_info(self, msg: CameraInfo):
        self._cinfo = msg

    def _on_tgt(self, msg: PointStamped):
        self._target = msg

    def _robot_xy_yaw(self) -> Optional[Tuple[float, float, float]]:
        for child in ("base_link", "base_footprint"):
            try:
                tf = self.tf_buffer.lookup_transform(
                    "map", child, rclpy.time.Time(), timeout=Duration(seconds=0.05)
                )
            except Exception:  # noqa: BLE001
                continue
            t = tf.transform.translation
            return float(t.x), float(t.y), _yaw_from_quat(tf.transform.rotation)
        return None

    def _project_target(self, w: int, h: int) -> Optional[Tuple[int, int, float]]:
        """Return (u, v, z_cam) of grasp target in image, or None."""
        if self._target is None or self._cinfo is None:
            return None
        cam_frame = str(self.get_parameter("camera_frame").value)
        try:
            tf = self.tf_buffer.lookup_transform(
                cam_frame,
                self._target.header.frame_id or "map",
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except Exception:  # noqa: BLE001
            return None
        # manual transform point
        tx, ty, tz = (
            float(self._target.point.x),
            float(self._target.point.y),
            float(self._target.point.z),
        )
        q = tf.transform.rotation
        t = tf.transform.translation
        # quaternion rotate
        qw, qx, qy, qz = float(q.w), float(q.x), float(q.y), float(q.z)
        # R * p
        ix = 1 - 2 * (qy * qy + qz * qz)
        iy = 2 * (qx * qy - qz * qw)
        iz = 2 * (qx * qz + qy * qw)
        jx = 2 * (qx * qy + qz * qw)
        jy = 1 - 2 * (qx * qx + qz * qz)
        jz = 2 * (qy * qz - qx * qw)
        kx = 2 * (qx * qz - qy * qw)
        ky = 2 * (qy * qz + qx * qw)
        kz = 1 - 2 * (qx * qx + qy * qy)
        px = ix * tx + iy * ty + iz * tz + float(t.x)
        py = jx * tx + jy * ty + jz * tz + float(t.y)
        pz = kx * tx + ky * ty + kz * tz + float(t.z)
        if pz <= 0.05:
            return None
        fx = float(self._cinfo.k[0])
        fy = float(self._cinfo.k[4])
        cx = float(self._cinfo.k[2])
        cy = float(self._cinfo.k[5])
        u = int(fx * (px / pz) + cx)
        v = int(fy * (py / pz) + cy)
        if u < -50 or v < -50 or u > w + 50 or v > h + 50:
            return None
        return u, v, pz

    def _draw_assist(self, bgr: np.ndarray) -> np.ndarray:
        out = bgr.copy()
        h, w = out.shape[:2]
        cx_i, cy_i = w // 2, h // 2

        # base_x (forward) = vertical midline; base_y = horizontal
        cv2.line(out, (cx_i, 0), (cx_i, h), (0, 255, 0), 2)
        cv2.line(out, (0, cy_i), (w, cy_i), (255, 255, 0), 1)
        cv2.putText(
            out,
            "base_X forward",
            (cx_i + 8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            out,
            "base_Y",
            (8, cy_i - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )

        pose = self._robot_xy_yaw()
        lines = []
        if pose is not None:
            rx, ry, yaw = pose
            snap, yerr = _nearest_axis_yaw(yaw)
            axis_ok = yerr < 0.20  # ~11°
            lines.append(f"yaw={yaw:.2f} snap={snap:.2f} err={yerr:.2f}")
            lines.append("YAW_AXIS_OK" if axis_ok else "YAW_NOT_AXIS (rotate)")

            if self._target is not None:
                tx = float(self._target.point.x)
                ty = float(self._target.point.y)
                dx = rx - tx
                dy = ry - ty
                same_y = abs(dy) < 0.12
                same_x = abs(dx) < 0.12
                lines.append(f"cube dx={dx:.2f} dy={dy:.2f}")
                if same_y:
                    lines.append("SAME_Y (base_X // map_X) OK")
                elif same_x:
                    lines.append("SAME_X (base_X // map_Y)")
                else:
                    lines.append("NOT_COLIN (need same x OR same y)")

        proj = self._project_target(w, h)
        if proj is not None:
            u, v, zc = proj
            gate_r = int(0.25 * w)
            cv2.circle(out, (u, v), gate_r, (255, 0, 255), 1)
            cv2.circle(out, (u, v), 14, (255, 0, 255), 2)
            cv2.line(out, (cx_i, cy_i), (u, v), (255, 0, 255), 2)
            lat = (u - cx_i) / max(w * 0.5, 1.0)
            lines.append(f"MAP_PROJ u_err={lat:.2f} z={zc:.2f}m gate=0.25w")
            if abs(lat) < 0.08:
                lines.append("CUBE_ON_CENTER (facing)")
                color = (0, 255, 0)
            else:
                lines.append("CUBE_OFF_CENTER (turn)")
                color = (0, 165, 255)
            cv2.putText(
                out,
                "MAP_PROJ",
                (u + 10, v - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        else:
            lines.append("no MAP_PROJ (wait PARK / camera_info)")

        y0 = h - 18 * (len(lines) + 1)
        for i, t in enumerate(lines):
            cv2.putText(
                out,
                t,
                (8, y0 + 18 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                out,
                t,
                (8, y0 + 18 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        return out

    def _tick(self):
        frame = None
        if self._dbg is not None and time.time() < self._dbg_until:
            frame = self._dbg
        elif self._raw is not None:
            frame = self._raw
        if frame is None:
            return
        vis = self._draw_assist(frame)
        cv2.imshow("yzbot FPV align (OpenCV)", vis)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = CamFpv()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
