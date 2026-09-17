"""Lightweight first-person camera window (OpenCV).

During Nav: raw front cam + green midline only (no grasp box).
After park / OpenCV align: switch to /yzbot/vision_debug (has box + blob).
"""

from __future__ import annotations

import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from yzbot_task.vision_align import annotate_bgr
from yzbot_task.image_topics import resolve_image_topic, list_image_topics


class CamFpv(Node):
    def __init__(self):
        super().__init__("cam_fpv")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("debug_topic", "/yzbot/vision_debug")
        self.bridge = CvBridge()
        self._raw = None
        self._dbg = None
        self._dbg_until = 0.0
        self._raw_sub = None
        self._bound_topic = None
        self._no_image_warned = False
        preferred = str(self.get_parameter("image_topic").value)
        dbg = str(self.get_parameter("debug_topic").value)
        self.create_subscription(Image, dbg, self._on_dbg, 10)
        self._bind_raw(preferred)
        self.create_timer(0.05, self._tick)
        self.create_timer(1.0, self._rediscover_if_needed)
        self.get_logger().info(
            "FPV=车头相机。导航中仅绿中线；到位对准后才显示 vision_debug 抓取框。"
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
                f"仍无图像帧。当前 Image 话题: {imgs or '(无)'}。请先开 Gazebo"
            )
            self._no_image_warned = True

    def _on_raw(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:  # noqa: BLE001
            return
        # Nav phase: midline only — no grasp box until vision_debug arrives
        label = f"FPV front {self._bound_topic or ''}"
        self._raw = annotate_bgr(
            bgr, None, label=label, draw_grasp_box=False
        )

    def _on_dbg(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:  # noqa: BLE001
            return
        self._dbg = bgr
        self._dbg_until = time.time() + 2.5

    def _tick(self):
        frame = None
        if self._dbg is not None and time.time() < self._dbg_until:
            # Already annotated by task_sm (box + blob) — show as-is
            frame = self._dbg
        elif self._raw is not None:
            frame = self._raw
        if frame is None:
            return
        cv2.imshow("yzbot FPV (front bumper cam)", frame)
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
