"""Gazebo ground-truth localization.

Gazebo diff_drive uses odometry_source=WORLD, and EKF fuses absolute /odom pose,
so odom→base_footprint already equals the Gazebo world pose.

Therefore map→odom is the identity (map frame == Gazebo world). This node only:
  1) publishes identity map→odom TF
  2) republishes /amcl_pose from model_states for UI

Do NOT compose map→odom = GT * inv(odom) here — that fought EKF and skewed
costmap/inflation whenever odom briefly disagreed with model_states.
"""

from __future__ import annotations

import math
import subprocess
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from gazebo_msgs.msg import ModelStates
from tf2_ros import TransformBroadcaster


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


def _quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def _gz_model_pose(name: str) -> Optional[Tuple[float, float, float]]:
    import os

    env = os.environ.copy()
    env.setdefault("GAZEBO_MASTER_URI", "http://127.0.0.1:11345")
    try:
        proc = subprocess.run(
            ["gz", "model", "-m", name, "-p"],
            capture_output=True,
            text=True,
            timeout=0.8,
            check=False,
            env=env,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    raw = (proc.stdout or "").strip()
    if not raw:
        return None
    nums = []
    for tok in raw.replace(",", " ").split():
        try:
            nums.append(float(tok))
        except ValueError:
            continue
    if len(nums) < 3:
        return None
    x, y = float(nums[0]), float(nums[1])
    yaw = 0.0
    if len(nums) >= 7:
        yaw = _yaw_from_quat(nums[3], nums[4], nums[5], nums[6])
    return x, y, yaw


class GtLocalization(Node):
    def __init__(self) -> None:
        super().__init__("gt_localization")
        self.declare_parameter("robot_model", "six_arm")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("publish_amcl_pose", True)
        self.declare_parameter("gz_fallback", True)

        self._robot = str(self.get_parameter("robot_model").value)
        self._odom = str(self.get_parameter("odom_frame").value)
        self._map = str(self.get_parameter("map_frame").value)
        self._pub_pose = bool(self.get_parameter("publish_amcl_pose").value)
        self._gz_fallback = bool(self.get_parameter("gz_fallback").value)

        self._tf_broadcaster = TransformBroadcaster(self)
        self._gt: Optional[Tuple[float, float, float]] = None
        self._last_warn = 0.0
        self._ok_logged = False

        qos_be = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        qos_rel = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(
            ModelStates, "/gazebo/model_states", self._on_ms, qos_be
        )
        self.create_subscription(
            ModelStates, "/gazebo/model_states", self._on_ms, qos_rel
        )

        self._pose_pub = None
        if self._pub_pose:
            self._pose_pub = self.create_publisher(
                PoseWithCovarianceStamped, "/amcl_pose", 10
            )

        hz = max(5.0, float(self.get_parameter("rate_hz").value))
        self.create_timer(1.0 / hz, self._tick)
        self.create_timer(0.5, self._maybe_gz)
        self.get_logger().info(
            f"GT localization ON: identity {self._map}→{self._odom}; "
            "EKF uses Gazebo WORLD /odom (costmaps still /scan)"
        )

    def _on_ms(self, msg: ModelStates) -> None:
        try:
            i = msg.name.index(self._robot)
        except ValueError:
            return
        p = msg.pose[i].position
        q = msg.pose[i].orientation
        yaw = _yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))
        self._gt = (float(p.x), float(p.y), float(yaw))

    def _maybe_gz(self) -> None:
        if not self._gz_fallback or self._gt is not None:
            return
        pose = _gz_model_pose(self._robot)
        if pose is not None:
            self._gt = pose
            self.get_logger().info(
                f"GT from gz model -p: ({pose[0]:.2f},{pose[1]:.2f},yaw={pose[2]:.2f})"
            )

    def _tick(self) -> None:
        stamp = self.get_clock().now().to_msg()

        # map == Gazebo world; EKF owns odom→base from WORLD /odom
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self._map
        t.child_frame_id = self._odom
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self._tf_broadcaster.sendTransform(t)

        if not self._ok_logged:
            self._ok_logged = True
            self.get_logger().info("GT map→odom identity OK (pose from EKF/WORLD odom)")

        gt = self._gt
        if gt is None:
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self._last_warn > 2.0:
                self._last_warn = now
                self.get_logger().warn(
                    f"GT pose wait: no model_states for '{self._robot}' "
                    "(TF identity still published)"
                )
            return

        if self._pose_pub is not None:
            msg = PoseWithCovarianceStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = self._map
            msg.pose.pose.position.x = gt[0]
            msg.pose.pose.position.y = gt[1]
            msg.pose.pose.position.z = 0.0
            bqx, bqy, bqz, bqw = _quat_from_yaw(gt[2])
            msg.pose.pose.orientation.x = bqx
            msg.pose.pose.orientation.y = bqy
            msg.pose.pose.orientation.z = bqz
            msg.pose.pose.orientation.w = bqw
            cov = [0.0] * 36
            cov[0] = cov[7] = 1e-6
            cov[35] = 1e-6
            msg.pose.covariance = cov
            self._pose_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GtLocalization()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
