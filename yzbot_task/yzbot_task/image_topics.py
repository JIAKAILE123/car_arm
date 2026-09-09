"""Resolve which Image topic to use — prefer chassis front cam for observe-then-grasp."""

from __future__ import annotations

from typing import List

# Chassis front (eye-to-hand). Wrist RGB-D is optional FPV only.
IMAGE_TOPIC_CANDIDATES: List[str] = [
    "/camera/image_raw",
    "/camera_rgbd/image_raw",
    "/camera_rgbd/camera_rgbd/image_raw",
    "/camera_rgbd/color/image_raw",
]


def list_image_topics(node) -> List[str]:
    out: List[str] = []
    for name, types in node.get_topic_names_and_types():
        if any(t.endswith("/Image") or t == "sensor_msgs/msg/Image" for t in types):
            out.append(name)
    return sorted(out)


def resolve_image_topic(node, preferred: str = "/camera/image_raw") -> str:
    """Pick first available preferred/candidate RGB topic; log choices."""
    available = set(list_image_topics(node))
    ordered: List[str] = []
    for t in [preferred] + IMAGE_TOPIC_CANDIDATES:
        if t not in ordered:
            ordered.append(t)

    def _is_rgb(t: str) -> bool:
        return "depth" not in t and "points" not in t

    for t in ordered:
        if t in available and _is_rgb(t):
            node.get_logger().info(f"图像话题选用: {t}")
            return t

    for t in sorted(available):
        if t.endswith("/camera/image_raw") or t == "/camera/image_raw":
            if _is_rgb(t):
                node.get_logger().info(f"图像话题选用(自动前相机): {t}")
                return t

    for t in sorted(available):
        if "camera_rgbd" in t and "image_raw" in t and _is_rgb(t):
            node.get_logger().info(f"图像话题选用(腕部兜底): {t}")
            return t

    if available:
        for t in sorted(available):
            if _is_rgb(t) and "image" in t:
                node.get_logger().warn(f"图像话题兜底: {t}（可用: {sorted(available)}）")
                return t
        node.get_logger().warn(f"仅有深度图? 可用 Image: {sorted(available)}")

    node.get_logger().warn(
        f"尚未发现图像话题，先订 {preferred}；"
        f"请确认 Gazebo 已开。查话题: ros2 topic list | grep image"
    )
    return preferred
