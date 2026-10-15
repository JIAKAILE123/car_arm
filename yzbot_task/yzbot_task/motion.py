"""Low-level motion helpers used by the task state machine."""

from __future__ import annotations

import math
import time
import rclpy
from typing import Optional, Tuple

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener

from nav2_msgs.action import NavigateToPose, NavigateThroughPoses
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as MsgDuration
from std_srvs.srv import Empty
from linkattacher_msgs.srv import AttachLink, DetachLink
from sensor_msgs.msg import Image, CameraInfo, JointState, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge

from yzbot_task.image_topics import resolve_image_topic
from yzbot_task.vision_align import (
    detect_blob,
    blob_to_yaw_cmd,
    blob_area_frac,
    cube_fully_in_box,
    grasp_box_xyxy,
    box_fill_ratio,
    blob_bbox_in_image,
    estimate_range_m_pinhole,
)

# Look-down survey: gripper roughly horizontal forward → wrist RGB-D (mounted
# pitch −π/2 on grasping_frame) looks DOWN at the shelf. Do NOT use a strong
# wrist-down pose (old j5≈−1.05) — that aims the camera at the floor by the base.
def survey_joints(j1: float = 0.0):
    return [float(j1), 0.50, 0.75, 0.0, -0.10, 0.0]


# Only vary base yaw; keep the same overhead camera body. Preset is for SEEING
# one cube, then MoveIt replans the grasp (preset is abandoned).
LOOK_SWEEP = [
    survey_joints(0.30),
    survey_joints(-0.30),
    survey_joints(0.50),
    survey_joints(-0.50),
    survey_joints(0.0),
]
PREGRASP = survey_joints(0.0)
# Floor cubes — keep original calibrated reach (tip not raised for “floor”).
# Stuck-below was traj time int(0.7)=0, not tip height.
PICK = [0.0, 1.2, 1.1, 0.0, -0.3, 0.0]
PRESET_REACH_R = 0.40
CARRY = [0.0, 0.25, 0.7, 0.0, -0.6, 0.0]
PLACE = [0.0, 0.7, 1.0, 0.0, -0.4, 0.0]
# arm_place_pose: physical EE was past pad center when parked at 0.38 —
# use 0.45 so base sits further back and drop lands nearer zone center.
PLACE_FRONT_M = 0.42
PLACE_FINAL = [0.0, 0.95, 1.15, 0.0, -0.55, 0.0]


def yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class MotionHelpers:
    def __init__(self, node: Node):
        self.node = node
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)
        self.bridge = CvBridge()
        self._last_image = None
        self._image_event = None
        self._image_sub = None
        self._image_topic = None
        self._last_depth = None
        self._last_cinfo = None
        self._camera_frame = "camera_rgbd_frame_optical"
        self._front_image = None
        self._front_cinfo = None
        self._front_frame = "camera_link"
        self._finger_pos: Optional[float] = None
        self._finger2_pos: Optional[float] = None
        # drive_owner: "nav" = only NavigateToPose; "local" = task cmd_vel (yaw only)
        self.drive_owner = "nav"
        self._nav_goal_handle = None
        self.cmd_vel_pub = node.create_publisher(Twist, "cmd_vel", 10)
        self.initialpose_pub = node.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )
        self._clear_local_cli = node.create_client(
            Empty, "/local_costmap/clear_entirely_local_costmap"
        )
        self._clear_global_cli = node.create_client(
            Empty, "/global_costmap/clear_entirely_global_costmap"
        )
        # Vision LOCAL: force AMCL laser updates while nearly stationary
        # (update_min_d/a otherwise freeze the filter → map pose looks "飘").
        self._nomotion_cli = node.create_client(Empty, "/request_nomotion_update")
        self._nomotion_timer = None
        # Virtual keepout over place pads so Nav2 routes around (avoid fly/crush)
        self._zone_keepout: dict = {}
        self._placed_keepout: list = []  # always-on marks for dropped cubes
        self._keepout_except: Optional[str] = None
        self._keepout_pub = node.create_publisher(
            PointCloud2, "/yzbot/zone_keepout", 1
        )
        self._keepout_timer = node.create_timer(0.4, self._publish_zone_keepout)
        self.nav = ActionClient(node, NavigateToPose, "navigate_to_pose")
        self.nav_through = ActionClient(
            node, NavigateThroughPoses, "navigate_through_poses"
        )
        self.arm = ActionClient(
            node, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory"
        )
        self.grip = ActionClient(
            node, FollowJointTrajectory, "/gripper_controller/follow_joint_trajectory"
        )
        self.attach_cli = node.create_client(AttachLink, "/ATTACHLINK")
        self.detach_cli = node.create_client(DetachLink, "/DETACHLINK")
        self._arm_goal_handle = None
        node.create_subscription(
            CameraInfo, "/camera_rgbd/camera_info", self._on_cinfo, 10
        )
        node.create_subscription(
            Image, "/camera_rgbd/depth/image_raw", self._on_depth, 10
        )
        node.create_subscription(Image, "/camera/image_raw", self._on_front, 10)
        node.create_subscription(
            CameraInfo, "/camera/camera_info", self._on_front_cinfo, 10
        )
        node.create_subscription(JointState, "/joint_states", self._on_js, 10)

    def _on_cinfo(self, msg: CameraInfo):
        self._last_cinfo = msg
        if msg.header.frame_id:
            self._camera_frame = msg.header.frame_id

    def _on_depth(self, msg: Image):
        self._last_depth = msg

    def _on_front(self, msg: Image):
        self._front_image = msg

    def _on_front_cinfo(self, msg: CameraInfo):
        self._front_cinfo = msg
        if msg.header.frame_id:
            self._front_frame = msg.header.frame_id

    def _on_js(self, msg: JointState):
        if "finger_joint1" in msg.name:
            i = msg.name.index("finger_joint1")
            if i < len(msg.position):
                self._finger_pos = float(msg.position[i])
        if "finger_joint2" in msg.name:
            j = msg.name.index("finger_joint2")
            if j < len(msg.position):
                self._finger2_pos = float(msg.position[j])

    def bind_image_topic(self, topic: Optional[str] = None, *, auto: bool = True):
        """Subscribe to wrist/front Image. If auto, resolve against live topics."""
        import threading

        preferred = topic or "/camera/image_raw"
        if auto:
            preferred = resolve_image_topic(self.node, preferred)

        if self._image_sub is not None:
            self.node.destroy_subscription(self._image_sub)
            self._image_sub = None

        self._image_event = threading.Event()
        self._last_image = None
        self._image_topic = preferred

        def _cb(msg: Image):
            self._last_image = msg
            if self._image_event is not None:
                self._image_event.set()

        self._image_sub = self.node.create_subscription(Image, preferred, _cb, 10)
        self.node.get_logger().info(f"vision subscribed: {preferred}")
        return preferred

    def grab_bgr(self, timeout: float = 3.0):
        if self._image_event is None:
            return None
        self._image_event.clear()
        # wait for a fresh frame
        if not self._image_event.wait(timeout):
            if self._last_image is None:
                return None
        try:
            return self.bridge.imgmsg_to_cv2(self._last_image, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().warn(f"cv_bridge failed: {exc}")
            return None

    def grab_front_bgr(self):
        if self._front_image is None:
            return None
        try:
            return self.bridge.imgmsg_to_cv2(self._front_image, desired_encoding="bgr8")
        except Exception:  # noqa: BLE001
            return None

    def arm_joints(self, joints, duration_sec: float = 2.5) -> bool:
        q = [float(x) for x in joints]
        self.node.get_logger().info(f"ARM look joints={['%.2f' % x for x in q]}")
        return self._send_traj(
            self.arm,
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            [
                (q, 0, 200000000),
                (q, int(duration_sec), 0),
            ],
            timeout=30.0,
        )

    def wait_servers(
        self,
        timeout: float = 8.0,
        *,
        need_arm: bool = True,
        need_attach: bool = True,
    ) -> bool:
        """Wait for Nav / arm / attach. Keep short — long waits look like 'received but no plan'."""
        ok = True
        t_nav = min(float(timeout), 8.0)
        t_arm = min(float(timeout), 8.0)
        t_att = min(float(timeout), 5.0)

        self.node.get_logger().info("WAITING_SERVERS: navigate_to_pose …")
        if not self.nav.wait_for_server(timeout_sec=t_nav):
            self.node.get_logger().error("navigate_to_pose action unavailable — 先开 Nav2")
            ok = False
        else:
            self.node.get_logger().info("WAITING_SERVERS: navigate_to_pose OK")

        # Optional: continuous multi-pose nav (door via). Soft-fail if absent.
        if not self.nav_through.wait_for_server(timeout_sec=min(3.0, t_nav)):
            self.node.get_logger().warn(
                "navigate_through_poses unavailable — door via will fall back"
            )
        if need_arm:
            self.node.get_logger().info("WAITING_SERVERS: arm_controller …")
            if not self.arm.wait_for_server(timeout_sec=t_arm):
                self.node.get_logger().error("arm_controller action unavailable")
                ok = False
            else:
                self.node.get_logger().info("WAITING_SERVERS: arm_controller OK")
            if not self.grip.wait_for_server(timeout_sec=3.0):
                self.node.get_logger().warn("gripper action unavailable (optional)")
        if need_attach:
            self.node.get_logger().info("WAITING_SERVERS: ATTACH/DETACH …")
            if not self.attach_cli.wait_for_service(timeout_sec=t_att):
                self.node.get_logger().error("/ATTACHLINK unavailable")
                ok = False
            if not self.detach_cli.wait_for_service(timeout_sec=t_att):
                self.node.get_logger().error("/DETACHLINK unavailable")
                ok = False
            if ok:
                self.node.get_logger().info("WAITING_SERVERS: attach OK")
        return ok

    def _spin_until(self, future, timeout_sec: float) -> bool:
        import time

        start = time.time()
        while not future.done():
            if time.time() - start > timeout_sec:
                return False
            # executor spins outside; here assume MultiThreadedExecutor
            time.sleep(0.05)
        return True

    def navigate(
        self,
        x: float,
        y: float,
        yaw: float,
        timeout: float = 180.0,
        *,
        early_xy_tol: Optional[float] = None,
        soft_start: bool = False,
    ) -> bool:
        if self.drive_owner != "nav":
            self.node.get_logger().warn("navigate blocked until MODE_NAV; forcing exit_local")
            self.exit_local_mode()

        # 取消上一次目标，避免「第二次发目标车不动 / 旧 BT 卡死」
        self.cancel_nav()
        if soft_start:
            # Chain via→final without hard brake (keeps AMCL smoother)
            time.sleep(0.02)
        else:
            self.stop_chassis()
            time.sleep(0.06)

        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        qx, qy, qz, qw = yaw_to_quat(yaw)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        goal = NavigateToPose.Goal()
        goal.pose = pose
        early_s = (
            f" early_xy≤{early_xy_tol:.2f}"
            if early_xy_tol is not None
            else ""
        )
        self.node.get_logger().info(
            f"NAV → ({x:.2f}, {y:.2f}, yaw={yaw:.2f}){early_s}"
        )
        # If localization is dead, Nav2 accepts nothing useful / no path in RViz
        if self.get_base_xy_yaw_map() is None:
            self.node.get_logger().error(
                "NAV abort: no map→base TF (先 2D Pose Estimate)"
            )
            return False
        send = self.nav.send_goal_async(goal)
        if not self._spin_until(send, 30.0):
            self.node.get_logger().error("nav send_goal timeout")
            return False
        goal_handle = send.result()
        if not goal_handle.accepted:
            self.node.get_logger().error("nav goal rejected")
            return False
        self._nav_goal_handle = goal_handle
        result_fut = goal_handle.get_result_async()
        t0 = time.time()
        while not result_fut.done():
            if time.time() - t0 > timeout:
                self.node.get_logger().error("nav result timeout")
                self.cancel_nav()
                return False
            # Place / via soft-disk: XY close enough → cancel (no yaw spin / no return)
            if early_xy_tol is not None:
                cur = self.get_base_xy_yaw_map()
                if cur is not None:
                    dxy = math.hypot(cur[0] - float(x), cur[1] - float(y))
                    if dxy <= float(early_xy_tol):
                        self.node.get_logger().info(
                            f"NAV_EARLY_XY dxy={dxy:.3f}≤{early_xy_tol:.2f} → continue"
                        )
                        self.cancel_nav()
                        if not soft_start:
                            self.stop_chassis()
                            time.sleep(0.06)
                        else:
                            time.sleep(0.02)
                        self._nav_goal_handle = None
                        return True
            time.sleep(0.05)
        self._nav_goal_handle = None
        from action_msgs.msg import GoalStatus

        wrapped = result_fut.result()
        status = int(getattr(wrapped, "status", GoalStatus.STATUS_UNKNOWN))
        # Early cancel returns CANCELED — still OK if we already accepted above
        if status != GoalStatus.STATUS_SUCCEEDED:
            # If XY already good after cancel race, treat as success
            if early_xy_tol is not None:
                cur = self.get_base_xy_yaw_map()
                if cur is not None:
                    dxy = math.hypot(cur[0] - float(x), cur[1] - float(y))
                    if dxy <= float(early_xy_tol) * 1.25:
                        self.node.get_logger().info(
                            f"NAV status={status} but dxy={dxy:.3f} OK for place"
                        )
                        return True
            self.node.get_logger().error(f"nav finished status={status} (not SUCCEEDED)")
            return False
        return True

    def _make_pose(self, x: float, y: float, yaw: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        qx, qy, qz, qw = yaw_to_quat(yaw)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    def navigate_through_poses(
        self,
        waypoints: list,
        timeout: float = 180.0,
        *,
        early_xy_tol: Optional[float] = None,
    ) -> bool:
        """One continuous Nav2 path through waypoints (no stop at intermediates).

        waypoints: list of (x, y, yaw). Final pose is the goal; earlier ones are vias.
        """
        if not waypoints:
            return False
        if len(waypoints) == 1:
            x, y, yaw = waypoints[0]
            return self.navigate(
                x, y, yaw, timeout=timeout, early_xy_tol=early_xy_tol
            )

        if self.drive_owner != "nav":
            self.exit_local_mode()

        if not self.nav_through.server_is_ready():
            self.node.get_logger().warn(
                "navigate_through_poses not ready — using last pose only"
            )
            x, y, yaw = waypoints[-1]
            return self.navigate(
                x, y, yaw, timeout=timeout, early_xy_tol=early_xy_tol
            )

        self.cancel_nav()
        # Do NOT long-stop here — keep continuity into the next path
        time.sleep(0.05)

        goal = NavigateThroughPoses.Goal()
        goal.poses = [
            self._make_pose(float(x), float(y), float(yaw)) for x, y, yaw in waypoints
        ]
        goal.behavior_tree = ""
        fx, fy, _ = waypoints[-1]
        self.node.get_logger().info(
            "NAV_THROUGH continuous "
            + " → ".join(f"({x:.2f},{y:.2f})" for x, y, _ in waypoints)
        )
        send = self.nav_through.send_goal_async(goal)
        if not self._spin_until(send, 30.0):
            self.node.get_logger().error("nav_through send_goal timeout")
            return False
        goal_handle = send.result()
        if not goal_handle.accepted:
            self.node.get_logger().error("nav_through goal rejected")
            # Fallback: go to final only
            x, y, yaw = waypoints[-1]
            return self.navigate(
                x, y, yaw, timeout=timeout, early_xy_tol=early_xy_tol
            )
        self._nav_goal_handle = goal_handle
        result_fut = goal_handle.get_result_async()
        t0 = time.time()
        while not result_fut.done():
            if time.time() - t0 > timeout:
                self.node.get_logger().error("nav_through result timeout")
                self.cancel_nav()
                return False
            if early_xy_tol is not None:
                cur = self.get_base_xy_yaw_map()
                if cur is not None:
                    dxy = math.hypot(cur[0] - float(fx), cur[1] - float(fy))
                    if dxy <= float(early_xy_tol):
                        self.node.get_logger().info(
                            f"NAV_THROUGH_EARLY_XY dxy={dxy:.3f}≤{early_xy_tol:.2f}"
                        )
                        self.cancel_nav()
                        self.stop_chassis()
                        time.sleep(0.1)
                        self._nav_goal_handle = None
                        return True
            time.sleep(0.08)
        self._nav_goal_handle = None
        from action_msgs.msg import GoalStatus

        wrapped = result_fut.result()
        status = int(getattr(wrapped, "status", GoalStatus.STATUS_UNKNOWN))
        if status != GoalStatus.STATUS_SUCCEEDED:
            if early_xy_tol is not None:
                cur = self.get_base_xy_yaw_map()
                if cur is not None:
                    dxy = math.hypot(cur[0] - float(fx), cur[1] - float(fy))
                    if dxy <= float(early_xy_tol) * 1.25:
                        return True
            self.node.get_logger().error(
                f"nav_through finished status={status} (not SUCCEEDED)"
            )
            return False
        return True

    def navigate_place_zone(
        self,
        x: float,
        y: float,
        *,
        xy_tol: float = 0.45,
        timeout: float = 120.0,
        approach_standoff: float = 0.38,
    ) -> bool:
        """Nav to a point ~PLACE_FRONT_M behind drop (arm EE ≈ 0.38m ahead).

        Drop XY stays ~0.38m ahead of base on +X; Nav2 goal is offset on ±X.
        """
        cur = self.get_base_xy_yaw_map()
        rx = float(cur[0]) if cur is not None else float(x) - 1.0
        # Prefer ±X dock (same as PLACE_DOCK): base behind center, face toward it
        so = max(0.15, float(approach_standoff))
        cands = [
            (float(x) - so, float(y), 0.0),     # west of zone, face +X
            (float(x) + so, float(y), math.pi),  # east of zone, face -X
        ]
        gx, gy, yaw = min(
            cands, key=lambda p: (p[0] - rx) ** 2 + (p[1] - float(y)) ** 2
        )
        self.node.get_logger().info(
            f"NAV_PLACE zone=({x:.2f},{y:.2f}) park=({gx:.2f},{gy:.2f}) "
            f"behind={so:.2f}m yaw={yaw:.2f} (区心在车前，不压区心)"
        )
        # 软 early：区前膨胀/急停易卡死；到附近后由 PLACE_DOCK 正对+补进
        soft_early = max(0.18, float(xy_tol))
        old_yaw_tol = self._set_follow_yaw_tol(3.0)
        try:
            ok = self.navigate(
                float(gx),
                float(gy),
                yaw,
                timeout=timeout,
                early_xy_tol=soft_early,
            )
        finally:
            if old_yaw_tol is not None:
                self._set_follow_yaw_tol(old_yaw_tol)
        time.sleep(0.04)
        self.stop_chassis()
        cur_n = self.get_base_xy_yaw_map()
        dxy = (
            math.hypot(float(cur_n[0]) - gx, float(cur_n[1]) - gy)
            if cur_n is not None
            else 99.0
        )
        if (not ok) or dxy > soft_early * 1.4:
            ok2 = self.navigate(
                float(gx),
                float(gy),
                yaw,
                timeout=min(40.0, timeout),
                early_xy_tol=soft_early * 1.15,
            )
            time.sleep(0.04)
            self.stop_chassis()
            cur_n = self.get_base_xy_yaw_map()
            dxy = (
                math.hypot(float(cur_n[0]) - gx, float(cur_n[1]) - gy)
                if cur_n is not None
                else 99.0
            )
            if not ok2 and dxy > 0.55:
                self.node.get_logger().error(
                    f"NAV_PLACE too far dxy={dxy:.2f}m"
                )
                return False
        # 稍远也交给 DOCK，避免区前死等
        cur2 = self.get_base_xy_yaw_map()
        use_yaw = float(cur2[2]) if cur2 is not None else yaw
        pin_x = float(cur2[0]) if cur2 is not None else float(gx)
        pin_y = float(cur2[1]) if cur2 is not None else float(gy)
        self.reanchor_amcl(pin_x, pin_y, use_yaw, settle_sec=0.03)
        return True

    def _set_follow_yaw_tol(self, value: float) -> Optional[float]:
        """Set FollowPath.yaw_goal_tolerance; return previous value if known."""
        from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
        from rcl_interfaces.srv import SetParameters, GetParameters

        try:
            get_cli = self.node.create_client(
                GetParameters, "/controller_server/get_parameters"
            )
            set_cli = self.node.create_client(
                SetParameters, "/controller_server/set_parameters"
            )
            prev = None
            if get_cli.wait_for_service(timeout_sec=0.4):
                greq = GetParameters.Request()
                greq.names = ["FollowPath.yaw_goal_tolerance"]
                gfut = get_cli.call_async(greq)
                if self._spin_until(gfut, 1.0) and gfut.result() is not None:
                    vals = gfut.result().values
                    if vals and vals[0].type != ParameterType.PARAMETER_NOT_SET:
                        prev = float(vals[0].double_value)
            if not set_cli.wait_for_service(timeout_sec=0.4):
                return prev
            preq = SetParameters.Request()
            p = Parameter()
            p.name = "FollowPath.yaw_goal_tolerance"
            p.value = ParameterValue()
            p.value.type = ParameterType.PARAMETER_DOUBLE
            p.value.double_value = float(value)
            preq.parameters = [p]
            sfut = set_cli.call_async(preq)
            self._spin_until(sfut, 1.0)
            self.node.get_logger().info(
                f"FollowPath.yaw_goal_tolerance → {value:.2f}"
                + (f" (was {prev:.2f})" if prev is not None else "")
            )
            return prev if prev is not None else 0.5
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().warn(f"yaw_tol set skipped: {exc}")
            return None

    def get_base_xy_yaw_map(self) -> Optional[Tuple[float, float, float]]:
        """Return (x, y, yaw) of base in map (prefer base_link, else base_footprint)."""
        for child in ("base_link", "base_footprint"):
            try:
                tf = self.tf_buffer.lookup_transform(
                    "map",
                    child,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.5),
                )
            except Exception:  # noqa: BLE001
                continue
            t = tf.transform.translation
            q = tf.transform.rotation
            # yaw from quaternion
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny, cosy)
            return float(t.x), float(t.y), float(yaw)
        return None

    @staticmethod
    def _wrap_pi(a: float) -> float:
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def pose_gate_ok(
        self,
        x: float,
        y: float,
        yaw: float,
        *,
        xy_tol: float = 0.20,
        yaw_tol: float = 0.20,
    ) -> Tuple[bool, str]:
        cur = self.get_base_xy_yaw_map()
        if cur is None:
            return False, "POSE_GATE no_tf map→base"
        cx, cy, cyaw = cur
        dxy = math.hypot(cx - x, cy - y)
        dyaw = abs(self._wrap_pi(cyaw - yaw))
        msg = f"dxy={dxy:.3f}m dyaw={dyaw:.3f}rad (tol xy={xy_tol} yaw={yaw_tol})"
        if dxy > xy_tol or dyaw > yaw_tol:
            return False, f"POSE_GATE_FAIL {msg}"
        return True, f"POSE_GATE_OK {msg}"

    def navigate_and_verify(
        self,
        x: float,
        y: float,
        yaw: float,
        *,
        retries: int = 2,
        xy_tol: float = 0.20,
        yaw_tol: float = 0.20,
        timeout: float = 180.0,
        settle_sec: float = 1.5,
        reanchor: bool = True,
    ) -> bool:
        """Nav2 approach + TF pose gate (standard mobile-manipulator pattern)."""
        attempts = max(int(retries), 0) + 1
        for i in range(attempts):
            if i > 0:
                # Near dyn_obs: stop, clear, then replan (don't shove through)
                self.stop_chassis(n=6)
                time.sleep(0.15)
                self.clear_costmaps()
                time.sleep(0.10)
                self.node.get_logger().warn(
                    f"NAV retry {i + 1}/{attempts}: 停车清图再规划 "
                    f"→ ({x:.2f},{y:.2f})"
                )
            if not self.navigate(x, y, yaw, timeout=timeout):
                self.node.get_logger().error(f"NAV attempt {i + 1}/{attempts} failed")
                continue
            time.sleep(max(float(settle_sec) * 0.45, 0.04))
            self.stop_chassis()
            ok, why = self.pose_gate_ok(x, y, yaw, xy_tol=xy_tol, yaw_tol=yaw_tol)
            self.node.get_logger().info(why)
            if ok:
                if reanchor:
                    # Pin AMCL to known goal — short settle (contest time)
                    self.reanchor_amcl(
                        x, y, yaw, settle_sec=min(0.10, max(0.04, settle_sec * 0.3))
                    )
                return True
            self.node.get_logger().warn(
                f"PoseGate miss → re-Nav ({i + 1}/{attempts})"
            )
        return False

    def reanchor_amcl(
        self,
        x: float,
        y: float,
        yaw: float,
        *,
        settle_sec: float = 0.8,
        xy_var: float = 0.04,
        yaw_var: float = 0.03,
    ) -> None:
        """Re-seed AMCL at a known map pose (like RViz 2D Pose Estimate).

        Call after Nav2 arrives at a verified park/zone, and after local yaw
        align, so the next NavigateToPose does not start from a skewed cloud.
        Tight covariance + chassis stop first: office walls are ambiguous and
        a spinning base right after /initialpose pulls the cloud away again.
        """
        self.stop_chassis(n=4)
        time.sleep(0.05)
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = 0.0
        qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        msg.pose.pose.orientation.z = float(qz)
        msg.pose.pose.orientation.w = float(qw)
        cov = [0.0] * 36
        cov[0] = float(xy_var)
        cov[7] = float(xy_var)
        cov[35] = float(yaw_var)
        msg.pose.covariance = cov
        for _ in range(3):
            msg.header.stamp = self.node.get_clock().now().to_msg()
            self.initialpose_pub.publish(msg)
            time.sleep(0.03)
        self.stop_chassis(n=2)
        self.node.get_logger().info(
            f"AMCL_REANCHOR ({x:.2f},{y:.2f},yaw={yaw:.2f}) → /initialpose"
        )
        time.sleep(max(float(settle_sec), 0.0))

    def configure_zone_keepouts(
        self,
        zones: dict,
        *,
        half_x: float = 0.48,
        half_y: float = 0.28,
        step: float = 0.06,
    ) -> None:
        """Register place-zone rectangles (map frame) as virtual obstacles.

        Sized to cover letter pad + margin so chassis cannot climb cubes.
        """
        self._zone_keepout = {}
        for letter, xy in zones.items():
            key = str(letter).strip().upper()
            cx, cy = float(xy[0]), float(xy[1])
            pts = []
            x0, x1 = cx - half_x, cx + half_x
            y0, y1 = cy - half_y, cy + half_y
            x = x0
            while x <= x1 + 1e-9:
                y = y0
                while y <= y1 + 1e-9:
                    pts.append((x, y, 0.05))
                    y += step
                x += step
            self._zone_keepout[key] = pts
        self.node.get_logger().info(
            f"ZONE_KEEPOUT A/B/C half=({half_x:.2f},{half_y:.2f}) "
            f"keys={list(self._zone_keepout.keys())}"
        )
        self._publish_zone_keepout()

    def clear_placed_keepouts(self) -> None:
        self._placed_keepout = []
        self._publish_zone_keepout()

    def add_placed_keepout(
        self, x: float, y: float, *, radius: float = 0.22, step: float = 0.06
    ) -> None:
        """Mark a dropped cube so Nav2 never plans over it (even if zone is open)."""
        r = max(0.12, float(radius))
        s = max(0.05, float(step))
        pts = []
        x0, x1 = float(x) - r, float(x) + r
        y0, y1 = float(y) - r, float(y) + r
        xi = x0
        while xi <= x1 + 1e-9:
            yi = y0
            while yi <= y1 + 1e-9:
                if (xi - float(x)) ** 2 + (yi - float(y)) ** 2 <= r * r + 1e-9:
                    pts.append((xi, yi, 0.06))
                yi += s
            xi += s
        self._placed_keepout.extend(pts)
        self.node.get_logger().info(
            f"PLACED_KEEPOUT +({x:.2f},{y:.2f}) r={r:.2f} "
            f"n={len(self._placed_keepout)}"
        )
        self._publish_zone_keepout()

    def set_keepout_except(self, zone: Optional[str], *, clear_maps: bool = True) -> None:
        """Exclude one letter pad while docking; placed-cube marks stay always on."""
        self._keepout_except = (
            None if zone is None else str(zone).strip().upper()
        )
        self._publish_zone_keepout()
        if clear_maps:
            self.clear_costmaps(timeout=1.0)

    def _publish_zone_keepout(self) -> None:
        pts = []
        for key, cloud in self._zone_keepout.items():
            if self._keepout_except is not None and key == self._keepout_except:
                continue
            pts.extend(cloud)
        # Dropped cubes: never cleared by zone except (avoid re-crush on same pad)
        pts.extend(self._placed_keepout)
        header = Header()
        header.stamp = self.node.get_clock().now().to_msg()
        header.frame_id = "map"
        if not pts:
            msg = point_cloud2.create_cloud(
                header,
                [
                    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                ],
                [],
            )
        else:
            msg = point_cloud2.create_cloud_xyz32(header, pts)
        self._keepout_pub.publish(msg)

    def clear_costmaps(self, timeout: float = 1.5) -> None:
        """Clear Nav2 costmaps so a skewed inflate cloud does not freeze the robot."""
        for name, cli in (
            ("local", self._clear_local_cli),
            ("global", self._clear_global_cli),
        ):
            if not cli.wait_for_service(timeout_sec=0.3):
                continue
            try:
                fut = cli.call_async(Empty.Request())
                self._spin_until(fut, timeout)
            except Exception as exc:  # noqa: BLE001
                self.node.get_logger().warn(f"clear_{name}_costmap: {exc}")

    def prepare_next_nav(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        yaw: Optional[float] = None,
        *,
        clear_maps: bool = True,
    ) -> None:
        """Before the next NavigateToPose: stop, optional reanchor, clear costmaps."""
        self.stop_chassis()
        self.exit_local_mode()
        if x is not None and y is not None and yaw is not None:
            self.reanchor_amcl(float(x), float(y), float(yaw), settle_sec=0.08)
        else:
            time.sleep(0.01)
        # After placing a cube, do NOT clear — keep laser marks so we don't plan over it
        if clear_maps:
            self.clear_costmaps(timeout=0.25)

    def stop_chassis(self, n: int = 2) -> None:
        tw = Twist()
        for _ in range(max(n, 1)):
            self.cmd_vel_pub.publish(tw)
            time.sleep(0.015)

    def publish_cmd_vel(self, vx: float, wz: float) -> bool:
        if self.drive_owner != "local":
            self.node.get_logger().warn("cmd_vel blocked: not MODE_LOCAL")
            return False
        tw = Twist()
        tw.linear.x = float(vx)
        tw.angular.z = float(wz)
        self.cmd_vel_pub.publish(tw)
        return True

    def cancel_nav(self) -> None:
        gh = self._nav_goal_handle
        self._nav_goal_handle = None
        if gh is None:
            return
        try:
            fut = gh.cancel_goal_async()
            self._spin_until(fut, 5.0)
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().warn(f"cancel_nav: {exc}")

    def _request_nomotion_update(self) -> None:
        """Ask AMCL to integrate one laser scan without requiring motion."""
        if not self._nomotion_cli.service_is_ready():
            return
        try:
            # Fire-and-forget (timer must not block / nest spin)
            self._nomotion_cli.call_async(Empty.Request())
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().warn(f"nomotion_update: {exc}")

    def _start_nomotion_heartbeat(self, period_sec: float = 1.0) -> None:
        self._stop_nomotion_heartbeat()
        if not self._nomotion_cli.service_is_ready():
            self._nomotion_cli.wait_for_service(timeout_sec=0.4)
        self._request_nomotion_update()
        self._nomotion_timer = self.node.create_timer(
            float(period_sec), self._request_nomotion_update
        )

    def _stop_nomotion_heartbeat(self) -> None:
        t = self._nomotion_timer
        self._nomotion_timer = None
        if t is not None:
            try:
                t.cancel()
                self.node.destroy_timer(t)
            except Exception:  # noqa: BLE001
                pass

    def enter_local_mode(self, settle_sec: float = 1.5) -> None:
        """Cancel Nav ownership of wheels; task_sm may publish cmd_vel."""
        self.cancel_nav()
        self.stop_chassis(n=2)
        # Cap settle — callers often pass legacy 0.08~1.5; contest wants snappy handoff
        time.sleep(min(0.06, max(float(settle_sec), 0.0)))
        self.drive_owner = "local"
        self.stop_chassis(n=1)
        self._start_nomotion_heartbeat(1.0)
        self.node.get_logger().info(
            "MODE_LOCAL (task owns cmd_vel; AMCL nomotion heartbeat on)"
        )

    def exit_local_mode(self) -> None:
        """Hand wheels back to Nav2 for NavigateToPose."""
        self._stop_nomotion_heartbeat()
        self.stop_chassis(n=1)
        self.drive_owner = "nav"
        self.node.get_logger().info("MODE_NAV (Nav2 owns cmd_vel)")

    def align_yaw_optional(
        self,
        color: str,
        *,
        timeout: float = 4.0,
        yaw_gain: float = 0.65,
        max_wz: float = 0.20,
        err_ok: float = 0.12,
    ) -> bool:
        """Short yaw-only fine face if front cam sees color. No vx (Nav owns stand-off)."""
        if self.drive_owner != "local":
            self.node.get_logger().error("align_yaw_optional requires MODE_LOCAL")
            return False

        start = time.time()
        ok_streak = 0
        saw = False
        while time.time() - start < timeout:
            bgr = self.grab_front_bgr()
            if bgr is None:
                time.sleep(0.1)
                continue
            blob = detect_blob(bgr, color)
            if blob is None:
                time.sleep(0.1)
                continue
            saw = True
            wz = blob_to_yaw_cmd(blob, gain=yaw_gain, max_wz=max_wz)
            if abs(blob.err_u) < err_ok:
                ok_streak += 1
                self.publish_cmd_vel(0.0, 0.0)
                if ok_streak >= 3:
                    self.stop_chassis()
                    self.node.get_logger().info(
                        f"YAW_ALIGN_OK err_u={blob.err_u:+.2f}"
                    )
                    return True
            else:
                ok_streak = 0
                self.publish_cmd_vel(0.0, wz)
            time.sleep(0.1)
        self.stop_chassis()
        self.node.get_logger().warn(
            f"YAW_ALIGN_SKIP ({'no_blob' if not saw else 'timeout'})"
        )
        return False

    def aim_yaw_to_bearing(
        self,
        bearing_rad: float,
        *,
        yaw_gain: float = 0.85,
        max_wz: float = 0.20,
        tol: float = 0.06,
        timeout: float = 8.0,
        bearing_fn=None,
    ) -> bool:
        """Rotate until base +X faces cube. +bearing (left) → +wz (CCW)."""
        if self.drive_owner != "local":
            self.node.get_logger().error("aim_yaw_to_bearing needs MODE_LOCAL")
            return False
        rem = float(bearing_rad)
        t0 = time.time()
        dt = 0.08
        while time.time() - t0 < timeout:
            if bearing_fn is not None:
                try:
                    live = bearing_fn()
                except Exception:  # noqa: BLE001
                    live = None
                if live is not None:
                    rem = float(live)
            if abs(rem) <= tol:
                self.stop_chassis()
                self.node.get_logger().info(
                    f"AIM_MAP_OK rem={rem:+.3f}rad (base_link +X → 物块)"
                )
                return True
            # ROS: +wz = CCW = turn left = toward +bearing
            wz = float(max(-max_wz, min(max_wz, yaw_gain * rem)))
            self.node.get_logger().info(
                f"AIM_MAP bearing={rem:+.3f} → wz={wz:+.3f} "
                f"(+bearing=左 → +wz=左转)",
                throttle_duration_sec=0.4,
            )
            self.publish_cmd_vel(0.0, wz)
            if bearing_fn is None:
                rem -= wz * dt
            time.sleep(dt)
        self.stop_chassis()
        self.node.get_logger().error(f"AIM_MAP_FAIL rem={rem:+.3f}")
        return False

    def scan_known_cube_until_seen(
        self,
        color: str,
        cube_id: int,
        *,
        max_wz: float = 0.20,
        timeout: float = 14.0,
        min_area: float = 35.0,
        on_frame=None,
    ) -> bool:
        """Face known map XY, then ±sweep about that heading until HSV sees the cube.

        Used when the cube is often outside the FOV: we know the coordinates, so
        scan around atan2(cube−robot) instead of random search.
        """
        from yzbot_task.vision_pose import known_cube_xy, cube_bearing_in_base

        if self.drive_owner != "local":
            self.node.get_logger().error("scan_known_cube_until_seen needs MODE_LOCAL")
            return False

        try:
            mx, my, _mz = known_cube_xy(color, cube_id)
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().error(f"SCAN_MAP no cube xy: {exc}")
            return False

        def _face_rem():
            cur = self.get_base_xy_yaw_map()
            if cur is None:
                return None
            yd = math.atan2(my - float(cur[1]), mx - float(cur[0]))
            return float(self._wrap_pi(yd - float(cur[2])))

        def _try_see():
            bgr = self.grab_front_bgr()
            if bgr is None:
                return None, None
            prefer = self.project_known_cube_uv(color, cube_id)
            # Loose gates while scanning into FOV; prefer map projection when available
            blob = detect_blob(
                bgr,
                color,
                prefer_uv=prefer,
                prefer_max_dist_frac=0.70,
                min_cy_frac=0.08,
            )
            if blob is None:
                blob = detect_blob(
                    bgr, color, prefer_uv=None, min_cy_frac=0.08
                )
            if on_frame is not None:
                on_frame(bgr, blob, "SCAN_MAP", prefer, None, None)
            return bgr, blob

        self.node.get_logger().info(
            f"SCAN_MAP → {color}_{cube_id} map=({mx:.2f},{my:.2f}) "
            f"先正对再±扫，直到看见色块"
        )
        rem0 = _face_rem()
        if rem0 is not None and abs(rem0) > 0.12:
            self.aim_yaw_to_bearing(
                float(rem0),
                yaw_gain=0.85,
                max_wz=float(max_wz),
                tol=0.10,
                timeout=min(6.0, abs(rem0) / max(max_wz, 0.08) + 1.2),
                bearing_fn=_face_rem,
            )
            self.stop_chassis()
            time.sleep(0.06)

        _bgr, blob = _try_see()
        if blob is not None and float(blob.area) >= min_area:
            self.node.get_logger().info(
                f"SCAN_MAP SEE area={blob.area:.0f} (已正对)"
            )
            return True

        # Sweep about face heading — wider for flaky FOV (e.g. blue near zone B)
        offsets = [0.0]
        for k in (1, 2, 3, 4, 5, 6):
            d = 0.22 * k
            offsets.extend([d, -d])
        last_off = 0.0
        t0 = time.time()
        for off in offsets:
            if time.time() - t0 > timeout:
                break
            rem_face = _face_rem()
            if rem_face is None:
                delta = float(off - last_off)
            else:
                target_rem = float(rem_face) + float(off)
                delta = float(off - last_off)
                if abs(target_rem) > abs(delta) + 0.05:
                    delta = float(self._wrap_pi(target_rem))
            if abs(delta) > 0.04:
                self.node.get_logger().info(
                    f"SCAN_MAP off={off:+.2f} spin={delta:+.3f} "
                    f"找 {color}_{cube_id}",
                    throttle_duration_sec=0.2,
                )
                self.spin_yaw_openloop(delta, max_wz=max_wz, timeout=4.0)
                self.stop_chassis()
                time.sleep(0.10)
            last_off = float(off)
            _bgr, blob = _try_see()
            if blob is not None and float(blob.area) >= min_area:
                self.node.get_logger().info(
                    f"SCAN_MAP SEE area={blob.area:.0f} off={off:+.2f}"
                )
                return True

        rem_back = _face_rem()
        if rem_back is not None and abs(rem_back) > 0.12:
            self.spin_yaw_openloop(float(rem_back), max_wz=max_wz, timeout=4.0)
        self.stop_chassis()
        self.node.get_logger().warn(
            f"SCAN_MAP MISS {color}_{cube_id} — 仍无色块，交 OpenCV 再试"
        )
        return False

    def approach_cube_servo(
        self,
        color: str,
        cube_id: int,
        *,
        target_range: float = 0.50,
        min_range: float = 0.38,
        max_vx: float = 0.12,
        max_wz: float = 0.28,
        timeout: float = 4.0,
    ) -> bool:
        """Close range while continuously correcting yaw (map + blob).

        Avoids “直走走偏”: never drive with wz=0 when bearing/err_u is off.
        """
        from yzbot_task.vision_pose import known_cube_xy, cube_bearing_in_base
        from yzbot_task.vision_align import detect_blob

        if self.drive_owner != "local":
            self.node.get_logger().error("approach_cube_servo needs MODE_LOCAL")
            return False
        try:
            mx, my, _ = known_cube_xy(color, cube_id)
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().error(f"SERVO no cube: {exc}")
            return False

        t0 = time.time()
        dt = 0.07
        self.node.get_logger().info(
            f"SERVO → {color}_{cube_id} r→[{min_range:.2f},{target_range:.2f}] "
            f"(map+视觉纠偏，禁止盲直走)"
        )
        while time.time() - t0 < timeout:
            cur = self.get_base_xy_yaw_map()
            info = cube_bearing_in_base(self.tf_buffer, color, cube_id)
            if info is not None:
                bearing, rng, _ = float(info[0]), float(info[1]), info[2]
            elif cur is not None:
                dx = float(mx) - float(cur[0])
                dy = float(my) - float(cur[1])
                rng = math.hypot(dx, dy)
                des = math.atan2(dy, dx)
                bearing = float(self._wrap_pi(des - float(cur[2])))
            else:
                time.sleep(dt)
                continue

            err_u = None
            bgr = self.grab_front_bgr()
            if bgr is not None:
                prefer = self.project_known_cube_uv(color, cube_id)
                blob = detect_blob(
                    bgr, color, prefer_uv=prefer, prefer_max_dist_frac=0.65, min_cy_frac=0.08
                )
                if blob is None:
                    blob = detect_blob(bgr, color, min_cy_frac=0.08)
                if blob is not None:
                    err_u = float(blob.err_u)

            # Fuse map bearing + image err (image wins when strong)
            yaw_cmd = 1.15 * bearing
            if err_u is not None:
                yaw_cmd = 0.45 * bearing + 0.90 * (-err_u)
            wz = float(max(-max_wz, min(max_wz, yaw_cmd)))

            # In band + aimed → done
            aimed = abs(bearing) <= 0.10 and (err_u is None or abs(err_u) <= 0.18)
            if min_range <= rng <= target_range + 0.06 and aimed:
                self.stop_chassis()
                self.node.get_logger().info(
                    f"SERVO_OK r={rng:.2f} bearing={bearing:+.2f} err_u={err_u}"
                )
                return True

            if rng < min_range:
                # Too close: reverse while still correcting yaw toward cube
                vx = -0.10
            elif abs(bearing) > 0.35 or (err_u is not None and abs(err_u) > 0.45):
                # Skewed — turn in place first (no blind forward)
                vx = 0.0
            elif rng > target_range + 0.25:
                vx = float(max_vx)
            elif rng > target_range:
                vx = float(max_vx) * 0.55
            else:
                vx = 0.0

            self.publish_cmd_vel(vx, wz)
            time.sleep(dt)

        self.stop_chassis()
        info2 = cube_bearing_in_base(self.tf_buffer, color, cube_id)
        ok = info2 is not None and float(info2[1]) <= target_range + 0.20
        self.node.get_logger().warn(
            f"SERVO_TIMEOUT r="
            f"{float(info2[1]) if info2 is not None else -1:.2f} ok={ok}"
        )
        return ok

    def creep_to_known_cube(
        self,
        color: str,
        cube_id: int,
        *,
        target_range: float = 0.42,
        max_vx: float = 0.14,
        max_wz: float = 0.22,
        timeout: float = 12.0,
    ) -> bool:
        """Face known cube map XY, then drive forward until range ≤ target_range.

        Used when Nav2 stops short (inflation / goal checker) ~0.7–1.0 m away.
        Requires MODE_LOCAL.
        """
        from yzbot_task.vision_pose import known_cube_xy

        if self.drive_owner != "local":
            self.node.get_logger().error("creep_to_known_cube needs MODE_LOCAL")
            return False
        try:
            mx, my, _mz = known_cube_xy(color, cube_id)
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().error(f"CREEP no cube xy: {exc}")
            return False

        def _rng_yaw():
            cur = self.get_base_xy_yaw_map()
            if cur is None:
                return None, None
            dx = float(mx) - float(cur[0])
            dy = float(my) - float(cur[1])
            rng = math.hypot(dx, dy)
            des = math.atan2(dy, dx)
            rem = float(self._wrap_pi(des - float(cur[2])))
            return rng, rem

        rng0, rem0 = _rng_yaw()
        if rng0 is None:
            self.node.get_logger().warn("CREEP skip (no map pose)")
            return False
        self.node.get_logger().info(
            f"CREEP → {color}_{cube_id} r={rng0:.2f}→≤{target_range:.2f}m "
            f"map=({mx:.2f},{my:.2f})"
        )
        # Already in band — do not drive closer (overshoot kills FOV / under-chassis)
        min_r = max(0.42, float(target_range) - 0.02)
        if float(rng0) <= float(target_range) + 0.04:
            if float(rng0) < min_r:
                self.node.get_logger().info(
                    f"CREEP already too close r={rng0:.2f} → reverse to ≥{min_r:.2f}"
                )
                t_b = time.time()
                while time.time() - t_b < 4.0:
                    rng, rem = _rng_yaw()
                    if rng is None or float(rng) >= min_r:
                        break
                    self.publish_cmd_vel(-0.14, 0.0)
                    time.sleep(0.06)
                self.stop_chassis()
            else:
                self.node.get_logger().info(f"CREEP_OK already r={rng0:.2f}m")
            return True
        if rem0 is not None and abs(rem0) > 0.12:
            self.aim_yaw_to_bearing(
                float(rem0),
                yaw_gain=0.75,
                max_wz=min(float(max_wz), 0.16),
                tol=0.08,
                timeout=min(8.0, abs(rem0) / max(max_wz, 0.08) + 1.5),
                bearing_fn=lambda: (_rng_yaw()[1]),
            )
            self.stop_chassis()

        t0 = time.time()
        dt = 0.08
        last_rng = float(rng0)
        while time.time() - t0 < timeout:
            rng, rem = _rng_yaw()
            if rng is None:
                time.sleep(dt)
                continue
            # Passed the cube (range grew while driving forward) → reverse out
            if float(rng) < min_r or (
                rem is not None and abs(float(rem)) > 1.2 and float(rng) < 0.55
            ):
                self.node.get_logger().warn(
                    f"CREEP under/beside cube r={rng:.2f} → reverse"
                )
                t_b = time.time()
                while time.time() - t_b < 3.5:
                    rng2, _ = _rng_yaw()
                    if rng2 is None or float(rng2) >= float(target_range):
                        break
                    self.publish_cmd_vel(-0.16, 0.0)
                    time.sleep(dt)
                self.stop_chassis()
                rng, rem = _rng_yaw()
                if rng is not None and float(rng) <= float(target_range) + 0.08:
                    return True
                continue
            if rng <= float(target_range):
                self.stop_chassis()
                self.node.get_logger().info(f"CREEP_OK r={rng:.2f}m")
                return True
            # Overshoot guard: if we somehow got closer than min, reverse
            if rng < float(target_range) - 0.06:
                self.publish_cmd_vel(-0.12, 0.0)
                time.sleep(dt)
                continue
            wz = 0.0
            if rem is not None and abs(rem) > 0.08:
                wz = float(max(-max_wz, min(max_wz, 1.0 * rem)))
            # Slow when close — less overshoot past FOV（末段再快一点）
            if rng > target_range + 0.28:
                scale = 1.0
            elif rng > target_range + 0.10:
                scale = 0.72
            else:
                scale = 0.50
            vx = float(max_vx * scale)
            if rem is not None and abs(rem) > 0.45:
                vx = 0.0  # too skewed — yaw first
            # Range increasing while we think we're approaching → stop (drove past)
            if float(rng) > last_rng + 0.08 and last_rng < float(target_range) + 0.25:
                self.node.get_logger().warn(
                    f"CREEP range rose {last_rng:.2f}→{rng:.2f} (past cube?) → reverse"
                )
                self.publish_cmd_vel(-0.14, 0.0)
                time.sleep(0.35)
                self.stop_chassis()
                break
            last_rng = float(rng)
            self.publish_cmd_vel(vx, wz)
            time.sleep(dt)
        self.stop_chassis()
        rng_f, _ = _rng_yaw()
        self.node.get_logger().warn(
            f"CREEP_TIMEOUT r={rng_f if rng_f is not None else -1:.2f}m "
            f"(target≤{target_range:.2f})"
        )
        return rng_f is not None and float(rng_f) <= float(target_range) + 0.20

    def spin_yaw_openloop(
        self,
        delta_yaw: float,
        *,
        max_wz: float = 0.25,
        timeout: float = 12.0,
    ) -> bool:
        """In-place yaw turn. Prefer map closed-loop with taper near target.

        Fast constant-wz openloop overshoots and blurs the front cam during SCAN.
        """
        if self.drive_owner != "local":
            self.node.get_logger().error("spin_yaw_openloop needs MODE_LOCAL")
            return False
        rem = float(self._wrap_pi(delta_yaw))
        if abs(rem) < 0.04:
            return True
        max_wz = float(max(0.08, abs(max_wz)))
        t_lim = min(float(timeout), abs(rem) / max(max_wz * 0.55, 0.06) + 0.8)
        cur0 = self.get_base_xy_yaw_map()
        self.node.get_logger().info(
            f"SPIN_YAW delta={rem:+.3f} max_wz={max_wz:.3f} t≈{t_lim:.1f}s"
        )
        t0 = time.time()
        dt = 0.08
        if cur0 is not None:
            target = float(self._wrap_pi(float(cur0[2]) + rem))
            while time.time() - t0 < t_lim:
                c = self.get_base_xy_yaw_map()
                if c is None:
                    break
                err = float(self._wrap_pi(target - float(c[2])))
                if abs(err) <= 0.045:
                    break
                # Taper near target — less overshoot / motion blur for HSV
                speed = min(max_wz, max(0.07, 0.55 * abs(err) + 0.05))
                wz = float(math.copysign(speed, err))
                self.publish_cmd_vel(0.0, wz)
                time.sleep(dt)
            self.stop_chassis()
            return True
        # Fallback: open-loop with soft taper by remaining angle
        wz_cap = float(math.copysign(max_wz, rem))
        turned = 0.0
        while time.time() - t0 < t_lim and abs(turned) < abs(rem) - 0.04:
            left = abs(rem) - abs(turned)
            speed = min(abs(wz_cap), max(0.07, 0.55 * left + 0.05))
            wz = float(math.copysign(speed, rem))
            self.publish_cmd_vel(0.0, wz)
            time.sleep(dt)
            turned += wz * dt
        self.stop_chassis()
        return True

    def drive_openloop_distance(
        self,
        distance_m: float,
        *,
        vx: float = 0.12,
        timeout: float = 12.0,
    ) -> bool:
        """Straight open-loop drive along current heading (wz=0). MODE_LOCAL.

        distance_m > 0 → forward; < 0 → reverse. No in-place spin — safe near cubes.
        """
        if self.drive_owner != "local":
            self.node.get_logger().error("drive_openloop_distance needs MODE_LOCAL")
            return False
        dist = float(distance_m)
        if abs(dist) < 0.04:
            return True
        speed = float(abs(vx))
        if speed < 0.06:
            speed = 0.06
        signed_vx = float(math.copysign(speed, dist))
        need_t = abs(dist) / speed
        t_lim = min(float(timeout), need_t + 0.5)
        self.node.get_logger().info(
            f"DRIVE_OPENLOOP dist={dist:+.2f}m vx={signed_vx:+.2f} t≈{t_lim:.1f}s "
            f"(直行，不原地转)"
        )
        t0 = time.time()
        dt = 0.08
        traveled = 0.0
        while time.time() - t0 < t_lim and abs(traveled) < abs(dist) - 0.03:
            self.publish_cmd_vel(signed_vx, 0.0)
            time.sleep(dt)
            traveled += signed_vx * dt
        self.stop_chassis()
        return True

    def project_known_cube_uv(
        self, color: str, cube_id: int
    ) -> Optional[Tuple[float, float]]:
        """Map known cube → front-camera pixel (u, v) for detect_blob gate."""
        from yzbot_task.vision_pose import known_cube_xy
        from geometry_msgs.msg import PointStamped

        if self._front_cinfo is None:
            return None
        try:
            x, y, z = known_cube_xy(color, cube_id)
        except Exception:  # noqa: BLE001
            return None
        cam = self._front_frame or "camera_link"
        ps = PointStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = self.node.get_clock().now().to_msg()
        ps.point.x = float(x)
        ps.point.y = float(y)
        ps.point.z = float(z)
        try:
            tf = self.tf_buffer.lookup_transform(
                cam,
                "map",
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except Exception:  # noqa: BLE001
            return None
        q = tf.transform.rotation
        t = tf.transform.translation
        qw, qx, qy, qz = float(q.w), float(q.x), float(q.y), float(q.z)
        tx, ty, tz = float(ps.point.x), float(ps.point.y), float(ps.point.z)
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
        fx = float(self._front_cinfo.k[0])
        fy = float(self._front_cinfo.k[4])
        cx = float(self._front_cinfo.k[2])
        cy = float(self._front_cinfo.k[5])
        u = fx * (px / pz) + cx
        v = fy * (py / pz) + cy
        return float(u), float(v)

    def opencv_align_to_cube(
        self,
        color: str,
        cube_id: int,
        *,
        target_r: float = 0.40,
        r_tol: float = 0.05,
        err_ok: float = 0.045,
        yaw_gain: float = 0.70,
        vx_gain: float = 0.45,
        max_wz: float = 0.14,
        max_vx: float = 0.12,
        timeout: float = 22.0,
        stable_hold_sec: float = 0.40,
        min_area: float = 40.0,
        lock_v_frac: float = 0.82,
        grasp_box_frac: float = 0.15,
        target_area_frac: float = 0.028,
        area_frac_tol: float = 0.012,
        target_fill: float = 1.26,
        fill_tol: float = 0.12,
        cube_width_m: float = 0.03,
        on_frame=None,
    ) -> bool:
        """Nav后对准：先按已知地图坐标朝向拧，再视觉贴合前进。"""
        from yzbot_task.vision_pose import cube_bearing_in_base

        if self.drive_owner != "local":
            self.node.get_logger().error("opencv_align_to_cube needs MODE_LOCAL")
            return False

        detect_tv = 2.0 * float(lock_v_frac) - 1.0

        def _bearing_live():
            info = cube_bearing_in_base(self.tf_buffer, color, cube_id)
            return None if info is None else float(info[0])

        # Phase 0: only aim if clearly off; skip micro-yaw (saves the final left twist)
        info0 = cube_bearing_in_base(self.tf_buffer, color, cube_id)
        if info0 is not None:
            b0, r0, xyz = info0
            if abs(float(b0)) > 0.35:
                self.node.get_logger().info(
                    f"AIM_MAP → known {color}_{cube_id} @ "
                    f"({xyz[0]:.2f},{xyz[1]:.2f}) bearing={b0:+.3f} r={r0:.2f}m"
                )
                aim_budget = min(
                    6.0, max(2.0, abs(float(b0)) / max(max_wz, 0.08) + 0.5)
                )
                self.aim_yaw_to_bearing(
                    float(b0),
                    yaw_gain=0.90,
                    max_wz=min(float(max_wz), 0.14),
                    tol=0.18,
                    timeout=aim_budget,
                    bearing_fn=_bearing_live,
                )
                self.stop_chassis()
                time.sleep(0.10)
            else:
                self.node.get_logger().info(
                    f"AIM_MAP skip (已朝向 bearing={b0:+.3f}，不拧头)"
                )
        else:
            self.node.get_logger().warn(
                "AIM_MAP skip (无 TF/坐标)；仅靠视觉"
            )

        self.node.get_logger().info(
            f"OPENCV_ALIGN start {color}_{cube_id} "
            f"box={grasp_box_frac:.2f} v={lock_v_frac:.2f} "
            f"fill≈{target_fill:.2f}±{fill_tol:.2f} "
            f"max_wz={max_wz:.2f} (先对中线再贴合方块)"
        )
        t0 = time.time()
        ok_streak = 0
        stable_t0: Optional[float] = None
        center_streak = 0
        phase = "CENTER"  # CENTER → BOX → LOCK
        last_rng: Optional[float] = None
        saw_blob = False

        while time.time() - t0 < timeout:
            bgr = self.grab_front_bgr()
            prefer = self.project_known_cube_uv(color, cube_id)
            blob = None
            img_h = img_w = 0
            if bgr is not None:
                img_h, img_w = bgr.shape[:2]
                if prefer is not None:
                    blob = detect_blob(
                        bgr,
                        color,
                        target_v=detect_tv,
                        prefer_uv=prefer,
                        prefer_max_dist_frac=0.55,
                        min_cy_frac=0.22,
                    )
                if blob is None:
                    blob = detect_blob(
                        bgr,
                        color,
                        target_v=detect_tv,
                        prefer_uv=None,
                        min_cy_frac=0.22,
                    )
            if on_frame is not None and bgr is not None:
                on_frame(
                    bgr,
                    blob,
                    f"OC_{phase}",
                    prefer,
                    lock_v_frac,
                    grasp_box_frac,
                )

            bearing = None
            rng_tf = None
            info = cube_bearing_in_base(self.tf_buffer, color, cube_id)
            if info is not None:
                bearing, rng_tf, _xyz = info
                bearing = float(bearing)
                rng_tf = float(rng_tf)
                last_rng = rng_tf
            if blob is not None and float(blob.area) >= min_area:
                saw_blob = True

            err_u = None
            fill = None
            in_box = False
            rng_vis = None
            bw = bh = 0.0
            if blob is not None and float(blob.area) >= min_area and img_w > 0:
                err_u = float(blob.err_u)
                in_box = cube_fully_in_box(
                    blob,
                    img_w,
                    img_h,
                    lock_v_frac=lock_v_frac,
                    grasp_box_frac=grasp_box_frac,
                )
                fill = box_fill_ratio(
                    blob,
                    img_w,
                    img_h,
                    lock_v_frac=lock_v_frac,
                    grasp_box_frac=grasp_box_frac,
                )
                _x0, _y0, _x1, _y1, bw, bh = blob_bbox_in_image(
                    blob, img_w, img_h
                )
                fx = None
                if self._front_cinfo is not None:
                    fx = float(self._front_cinfo.k[0])
                rng_vis = estimate_range_m_pinhole(
                    max(bw, bh), float(fx or 0.0), object_width_m=cube_width_m
                )
                if rng_vis is not None:
                    # Prefer TF range when available; else keep visual estimate
                    if last_rng is None:
                        last_rng = float(rng_vis)

            wz = 0.0
            vx = 0.0

            # Depth = pixel size (fill = cube bbox / grasp gate).
            # Floor cubes already sit low in FPV when still far — cy alone ≠ near.
            # Lock when cube slightly overfills gate (~1.26 = snug/close)
            fill_gate = max(1.18, float(target_fill) - float(fill_tol))  # ~1.18
            fill_creep = max(1.08, fill_gate - 0.10)  # CLOSE until near lock
            fill_now = float(fill) if fill is not None else 0.0
            fill_low = fill is None or fill_now < fill_creep

            if err_u is not None:
                yaw_ok = abs(err_u) <= max(err_ok, 0.12)
                rng_now_pre = rng_tf if rng_tf is not None else (
                    float(rng_vis) if rng_vis is not None else last_rng
                )
                cy_frac = float(blob.cy) / float(max(blob.image_h, 1))
                # Primary: small pixels → CLOSE. TF/cy only as backup.
                tf_far = rng_now_pre is not None and float(rng_now_pre) > (
                    float(target_r) + 0.08
                )
                need_close = bool(fill_low or tf_far)

                if not yaw_ok:
                    phase = "CENTER"
                    center_streak = 0
                    wz = blob_to_yaw_cmd(blob, gain=yaw_gain, max_wz=max_wz)
                    # Mild inch only if cube still well inside FOV (not near edges)
                    cy_ok = 0.22 <= cy_frac <= 0.82
                    vx = min(max_vx, 0.05) if (need_close and cy_ok and abs(err_u) < 0.22) else 0.0
                elif need_close:
                    # Seen + centered → can surge forward; never push cube out of FOV
                    phase = "CLOSE"
                    center_streak = 0
                    wz = blob_to_yaw_cmd(
                        blob, gain=yaw_gain * 0.25, max_wz=max_wz * 0.35
                    )
                    cy_frac_g = float(blob.cy) / float(max(blob.image_h, 1))
                    # FOV keep: near left/right/bottom/top → stop or reverse
                    leaving = (
                        abs(err_u) > 0.32
                        or cy_frac_g > 0.88
                        or cy_frac_g < 0.16
                    )
                    if leaving:
                        vx = -0.06 if cy_frac_g > 0.88 else 0.0
                        self.node.get_logger().info(
                            "OPENCV_ALIGN CLOSE: 块贴边 → 停/倒（保存在画面内）",
                            throttle_duration_sec=0.4,
                        )
                    elif fill_now < 0.50:
                        vx = min(max_vx, 0.16)  # far / tiny → 往前冲
                    elif fill_now < 0.80:
                        vx = min(max_vx, 0.12)
                    elif fill_now < fill_creep:
                        vx = min(max_vx, 0.08)
                    else:
                        vx = min(max_vx, 0.05)
                    if tf_far and vx > 0.0:
                        vx = max(vx, min(max_vx, 0.10))
                else:
                    center_streak += 1
                    if center_streak < 2:
                        phase = "CENTER"
                        wz = 0.0
                        vx = 0.0
                    else:
                        phase = "BOX"
                        wz = 0.0
                        if img_w > 0:
                            _bx0, _by0, _bx1, _by1, _cx, _cy = grasp_box_xyxy(
                                img_w,
                                img_h,
                                lock_v_frac=lock_v_frac,
                                grasp_box_frac=grasp_box_frac,
                            )
                            box_side = min(_bx1 - _bx0, _by1 - _by0)
                            y1_img = float(blob.y1) * (
                                float(img_h) / max(float(blob.image_h), 1.0)
                            )
                            if max(bw, bh) > 1.35 * box_side or y1_img > 0.94 * float(
                                img_h
                            ):
                                vx = -min(max_vx, 0.05)
                            elif fill_now >= fill_gate:
                                vx = 0.0
                            else:
                                vx = min(max_vx, 0.05)
            else:
                # No blob in FOV — NEVER drive forward on TF alone (walks past cube)
                phase = "SEEK"
                center_streak = 0
                vx = 0.0
                if bearing is not None and abs(bearing) > 0.08:
                    wz = float(max(-max_wz, min(max_wz, 0.75 * bearing)))
                else:
                    wz = 0.0
                if saw_blob:
                    # Had cube, then lost → reverse back into FOV
                    vx = -0.08
                    self.node.get_logger().info(
                        "OPENCV_ALIGN SEEK: 丢块 → 倒退回画面（禁止盲目前进）",
                        throttle_duration_sec=0.5,
                    )

            # Final FOV clamp: if we still see the cube on an edge, kill forward
            if blob is not None and err_u is not None and vx > 0.0:
                cy_f = float(blob.cy) / float(max(blob.image_h, 1))
                if abs(err_u) > 0.30 or cy_f > 0.86 or cy_f < 0.18:
                    vx = 0.0

            yaw_ok = err_u is not None and abs(err_u) <= max(err_ok, 0.14)
            rng_gate = float(target_r) + 0.08
            rng_now = rng_tf if rng_tf is not None else (
                float(rng_vis) if rng_vis is not None else last_rng
            )
            tf_ok = bool(
                rng_now is not None and float(rng_now) <= rng_gate
            )
            cy_frac = None
            if blob is not None and blob.image_h > 0:
                cy_frac = float(blob.cy) / float(blob.image_h)
            # Lock ONLY when pixels are big enough — never on err_u/cy alone
            vis_ok = bool(fill is not None and float(fill) >= fill_gate)
            if vis_ok and yaw_ok:
                vx = min(0.0, vx) if vx > 0 else vx
            # Prefer vision fill; TF alone cannot approve pick
            range_ok = bool(vis_ok and (tf_ok or float(fill) >= fill_gate + 0.05))
            soft_snug = bool(vis_ok)
            # Block lock while map range still far (avoids “远处就夹”)
            too_far = bool(
                rng_now is not None
                and float(rng_now) > float(target_r) + 0.12
            )
            lock_ok = bool(yaw_ok and soft_snug and not too_far)

            self.node.get_logger().info(
                f"OPENCV_ALIGN[{phase}] vx={vx:+.2f} wz={wz:+.2f} "
                f"err_u={('n/a' if err_u is None else f'{err_u:+.3f}')} "
                f"cy={('n/a' if cy_frac is None else f'{cy_frac:.2f}')} "
                f"r={('n/a' if rng_now is None else f'{rng_now:.2f}')} "
                f"fill={('n/a' if fill is None else f'{fill:.2f}')} "
                f"range_ok={range_ok} soft_snug={soft_snug}",
                throttle_duration_sec=0.35,
            )

            if lock_ok:
                ok_streak += 1
                self.publish_cmd_vel(0.0, 0.0)
                if ok_streak >= 2:
                    if stable_t0 is None:
                        stable_t0 = time.time()
                        phase = "LOCK"
                        self.node.get_logger().info(
                            "OPENCV_ALIGN_OK → hold (中线+距离OK，准备下探)"
                        )
                    elif time.time() - stable_t0 >= max(0.20, stable_hold_sec * 0.6):
                        self.stop_chassis()
                        self.node.get_logger().info(
                            f"OPENCV_ALIGN_STABLE err_u={err_u} fill={fill} "
                            f"r={rng_now} → PICK"
                        )
                        return True
            else:
                ok_streak = 0
                stable_t0 = None
                self.publish_cmd_vel(vx, wz)
            time.sleep(0.08)

        self.stop_chassis()
        # NEVER soft-OK on TF alone — screenshot case: r~0.36 fill~0.37 still far
        bgr = self.grab_front_bgr()
        fill_last = None
        cy_frac = None
        if bgr is not None and saw_blob:
            prefer = self.project_known_cube_uv(color, cube_id)
            blob = detect_blob(
                bgr,
                color,
                target_v=detect_tv,
                prefer_uv=prefer,
                prefer_max_dist_frac=0.55,
                min_cy_frac=0.12,
            )
            if blob is None:
                blob = detect_blob(
                    bgr, color, target_v=detect_tv, min_cy_frac=0.12
                )
            if blob is not None:
                h, w = bgr.shape[:2]
                fill_last = box_fill_ratio(
                    blob,
                    w,
                    h,
                    lock_v_frac=lock_v_frac,
                    grasp_box_frac=grasp_box_frac,
                )
                cy_frac = float(blob.cy) / float(max(blob.image_h, 1))
        # Soft OK only if pixels are large enough — never cy alone (floor cubes look low when far)
        vis_close = bool(
            fill_last is not None and float(fill_last) >= 1.15
        )
        if (
            last_rng is not None
            and float(last_rng) <= float(target_r) + 0.05
            and vis_close
        ):
            self.node.get_logger().warn(
                f"OPENCV_ALIGN_TIMEOUT → SOFT_OK r={last_rng:.2f} "
                f"fill={fill_last} cy={cy_frac}"
            )
            return True
        self.node.get_logger().error(
            f"OPENCV_ALIGN_TIMEOUT → FAIL r="
            f"{'n/a' if last_rng is None else f'{last_rng:.2f}'} "
            f"fill={fill_last} cy={cy_frac} (像素仍小，禁止下探)"
        )
        return False

    def drive_until_visually_near(
        self,
        color: str,
        cube_id: int,
        *,
        lock_v_frac: float = 0.82,
        grasp_box_frac: float = 0.15,
        min_fill: float = 0.55,
        min_cy_frac: float = 0.55,
        max_vx: float = 0.08,
        timeout: float = 8.0,
        max_travel_m: float = 0.22,
    ) -> bool:
        """Creep until cube looks near — but stop before it leaves the FOV.

        Stop early if cy/fill already OK. If cube is too low / too big / lost
        after being seen, brake (optionally reverse a bit) so we don't pull it
        out of the camera.
        """
        if self.drive_owner != "local":
            self.node.get_logger().error("drive_until_visually_near needs MODE_LOCAL")
            return False
        detect_tv = 2.0 * float(lock_v_frac) - 1.0
        t0 = time.time()
        traveled = 0.0
        dt = 0.08
        last_fill = None
        last_cy = None
        saw = False
        while time.time() - t0 < timeout and traveled < max_travel_m:
            bgr = self.grab_front_bgr()
            if bgr is None:
                time.sleep(dt)
                continue
            prefer = self.project_known_cube_uv(color, cube_id)
            blob = detect_blob(
                bgr,
                color,
                target_v=detect_tv,
                prefer_uv=prefer,
                prefer_max_dist_frac=0.60,
                min_cy_frac=0.08,
            )
            if blob is None:
                blob = detect_blob(
                    bgr, color, target_v=detect_tv, min_cy_frac=0.08
                )
            wz = 0.0
            vx = 0.0
            if blob is not None:
                saw = True
                h, w = bgr.shape[:2]
                last_fill = box_fill_ratio(
                    blob,
                    w,
                    h,
                    lock_v_frac=lock_v_frac,
                    grasp_box_frac=grasp_box_frac,
                )
                last_cy = float(blob.cy) / float(max(blob.image_h, 1))
                y1_frac = float(blob.y1) / float(max(blob.image_h, 1))
                wz = blob_to_yaw_cmd(blob, gain=0.65, max_wz=0.16)

                # Too close / about to leave FOV → stop (only if already fairly near)
                if (
                    float(last_fill) >= 1.32
                    or (
                        float(last_fill) >= 1.20
                        and (float(last_cy) >= 0.84 or y1_frac >= 0.93)
                    )
                ):
                    self.publish_cmd_vel(-0.04, 0.0)
                    time.sleep(0.20)
                    self.stop_chassis()
                    self.node.get_logger().info(
                        f"VIS_NEAR_HOLD (防出视野) fill={last_fill:.2f} "
                        f"cy={last_cy:.2f} y1={y1_frac:.2f}"
                    )
                    return True

                # Good enough — fill (pixel size) only; cy alone ≠ near
                if float(last_fill) >= min_fill:
                    self.stop_chassis()
                    self.node.get_logger().info(
                        f"VIS_NEAR_OK fill={last_fill:.2f} cy={last_cy:.2f}"
                    )
                    return True

                # Still small pixels → keep driving; speed by fill
                if float(last_fill) < 0.70:
                    vx = min(max_vx, 0.10)
                elif float(last_fill) < 1.00:
                    vx = min(max_vx, 0.08)
                else:
                    vx = min(max_vx, 0.06)
            else:
                if saw:
                    # Had cube, then lost → reverse, don't chase out of FOV
                    self.publish_cmd_vel(-0.08, 0.0)
                    time.sleep(0.35)
                    self.stop_chassis()
                    self.node.get_logger().warn(
                        "VIS_NEAR_LOST → 微退停 (避免拉出视野)"
                    )
                    return bool(
                        last_fill is not None and float(last_fill) >= 1.10
                    )
                from yzbot_task.vision_pose import cube_bearing_in_base

                info = cube_bearing_in_base(self.tf_buffer, color, cube_id)
                if info is not None and abs(float(info[0])) > 0.08:
                    wz = float(max(-0.16, min(0.16, 0.75 * float(info[0]))))
                vx = 0.05
            self.publish_cmd_vel(vx, wz)
            traveled += abs(vx) * dt
            time.sleep(dt)
        self.stop_chassis()
        self.node.get_logger().warn(
            f"VIS_NEAR_TIMEOUT fill={last_fill} cy={last_cy} "
            f"travel={traveled:.2f}m"
        )
        # Accept only if pixels grew enough — never cy alone
        return bool(
            last_fill is not None and float(last_fill) >= max(1.12, min_fill - 0.08)
        )

    def opencv_center_on_cube(self, color: str, **kwargs) -> bool:
        """Yaw-only legacy wrapper → align with cube_id=1 fallback."""
        cid = int(kwargs.pop("cube_id", 1))
        return self.opencv_align_to_cube(color, cid, **kwargs)

    def face_center_then_range(
        self,
        color: str,
        *,
        err_ok: float = 0.10,
        yaw_gain: float = 0.70,
        max_wz: float = 0.20,
        target_r: float = 0.38,
        r_tol: float = 0.06,
        vx_gain: float = 0.35,
        max_vx: float = 0.07,
        timeout_see: float = 4.0,
        timeout_yaw: float = 14.0,
        timeout_range: float = 12.0,
        sweep_yaw_rad: float = 0.40,
        stable_hold_sec: float = 0.25,
        map_bearing: Optional[float] = None,
        bearing_fn=None,
        estimate_fn=None,
        on_frame=None,
    ) -> Tuple[bool, Optional[float]]:
        """AIM_MAP(TF慢速朝向) → SEE / 小幅扫 → 视觉锁十字中线 → RANGE."""
        if self.drive_owner != "local":
            self.node.get_logger().error("face_center_then_range needs MODE_LOCAL")
            return False, None

        if map_bearing is not None or bearing_fn is not None:
            b0 = float(map_bearing) if map_bearing is not None else 0.0
            if bearing_fn is not None:
                try:
                    live0 = bearing_fn()
                    if live0 is not None:
                        b0 = float(live0)
                except Exception:  # noqa: BLE001
                    pass
            self.node.get_logger().info(
                f"AIM_MAP bearing={b0:+.3f}rad → 慢速朝向物块"
            )
            if not self.aim_yaw_to_bearing(
                b0,
                max_wz=max_wz,
                yaw_gain=min(0.9, yaw_gain + 0.1),
                timeout=min(10.0, timeout_yaw),
                bearing_fn=bearing_fn,
            ):
                self.stop_chassis()
                return False, None

        def _see_blob(timeout: float) -> bool:
            t0 = time.time()
            streak = 0
            while time.time() - t0 < timeout:
                bgr = self.grab_front_bgr()
                if bgr is None:
                    time.sleep(0.08)
                    continue
                blob = detect_blob(bgr, color, min_cy_frac=0.28)
                if on_frame is not None:
                    on_frame(bgr, blob, "SEE")
                if blob is not None and blob.area >= 40:
                    streak += 1
                    if streak >= 2:
                        self.node.get_logger().info(
                            f"SEE_OK area={blob.area:.0f} err_u={blob.err_u:+.2f}"
                        )
                        return True
                else:
                    streak = 0
                time.sleep(0.08)
            return False

        if not _see_blob(timeout_see):
            self.node.get_logger().info(
                f"SEE miss → 慢速小扫 ±{sweep_yaw_rad:.2f}rad"
            )
            found = False
            sweep_wz = min(float(max_wz), 0.18)
            for sign in (+1.0, -1.0, +1.0, -1.0, 0.0):
                if found:
                    break
                rem = sign * float(sweep_yaw_rad)
                t_sw = time.time()
                while time.time() - t_sw < 3.5 and abs(rem) > 0.05:
                    bgr = self.grab_front_bgr()
                    blob = (
                        detect_blob(bgr, color, min_cy_frac=0.28)
                        if bgr is not None
                        else None
                    )
                    if on_frame is not None and bgr is not None:
                        on_frame(bgr, blob, "SWEEP")
                    if blob is not None and blob.area >= 40:
                        found = True
                        self.stop_chassis()
                        self.node.get_logger().info("SEE_OK during SWEEP")
                        break
                    wz = float(max(-sweep_wz, min(sweep_wz, 0.9 * rem)))
                    self.publish_cmd_vel(0.0, wz)
                    rem -= wz * 0.08
                    time.sleep(0.08)
                self.stop_chassis()
            if not found and not _see_blob(1.5):
                self.stop_chassis()
                self.node.get_logger().error(
                    "SEARCH_FAIL NO_BLOB (AIM后仍无色块；已停，不无限找)"
                )
                return False, None

        # Vision lock: keep turning until red box center sits on green cross
        t0 = time.time()
        ok_streak = 0
        stable_t0: Optional[float] = None
        while time.time() - t0 < timeout_yaw:
            bgr = self.grab_front_bgr()
            if bgr is None:
                time.sleep(0.08)
                continue
            blob = detect_blob(bgr, color, min_cy_frac=0.28)
            if on_frame is not None:
                on_frame(bgr, blob, "CENTER")
            if blob is None:
                ok_streak = 0
                stable_t0 = None
                # lost blob: nudge slowly with TF bearing if available
                if bearing_fn is not None:
                    try:
                        b = bearing_fn()
                    except Exception:  # noqa: BLE001
                        b = None
                    if b is not None and abs(float(b)) > 0.08:
                        wz = float(max(-max_wz, min(max_wz, 0.8 * float(b))))
                        self.publish_cmd_vel(0.0, wz)
                    else:
                        self.publish_cmd_vel(0.0, 0.0)
                else:
                    self.publish_cmd_vel(0.0, 0.0)
                time.sleep(0.08)
                continue
            if abs(blob.err_u) <= err_ok:
                ok_streak += 1
                self.publish_cmd_vel(0.0, 0.0)
                if ok_streak >= 3:
                    if stable_t0 is None:
                        stable_t0 = time.time()
                        self.node.get_logger().info(
                            f"CENTER_OK err_u={blob.err_u:+.3f} → hold STABLE"
                        )
                    elif time.time() - stable_t0 >= stable_hold_sec:
                        self.stop_chassis()
                        self.node.get_logger().info("STABLE_OK (十字对准色块中心)")
                        break
            else:
                ok_streak = 0
                stable_t0 = None
                wz = blob_to_yaw_cmd(blob, gain=yaw_gain, max_wz=max_wz)
                self.publish_cmd_vel(0.0, wz)
            time.sleep(0.08)
        else:
            self.stop_chassis()
            self.node.get_logger().error("CENTER_FAIL (timeout; 未锁到中线)")
            return False, None

        t1 = time.time()
        range_streak = 0
        last_r: Optional[float] = None
        while time.time() - t1 < timeout_range:
            bgr = self.grab_front_bgr()
            if bgr is None:
                time.sleep(0.08)
                continue
            blob = detect_blob(bgr, color, min_cy_frac=0.28)
            if on_frame is not None:
                on_frame(bgr, blob, "RANGE")
            if blob is None or abs(blob.err_u) > err_ok * 1.8:
                if blob is not None:
                    wz = blob_to_yaw_cmd(blob, gain=yaw_gain, max_wz=max_wz)
                    self.publish_cmd_vel(0.0, wz)
                else:
                    self.publish_cmd_vel(0.0, 0.0)
                range_streak = 0
                time.sleep(0.08)
                continue

            r = None
            if estimate_fn is not None:
                try:
                    r = estimate_fn(bgr, blob)
                except Exception:  # noqa: BLE001
                    r = None
            if r is None:
                frac = blob_area_frac(blob)
                r = float(0.55 - (frac - 0.02) * (0.25 / 0.06))
                r = float(max(0.22, min(0.80, r)))
            last_r = r
            err = float(r - target_r)
            if abs(err) <= r_tol:
                range_streak += 1
                self.publish_cmd_vel(0.0, 0.0)
                if range_streak >= 4:
                    self.stop_chassis()
                    self.node.get_logger().info(
                        f"RANGE_OK r={r:.3f}m target={target_r:.3f}±{r_tol:.3f}"
                    )
                    return True, r
            else:
                range_streak = 0
                vx = float(max(-max_vx, min(max_vx, vx_gain * err)))
                self.publish_cmd_vel(vx, 0.0)
            time.sleep(0.1)

        self.stop_chassis()
        self.node.get_logger().error(f"RANGE_FAIL last_r={last_r} target={target_r}")
        return False, last_r

    def local_chassis_align(self, color: str, **kwargs) -> bool:
        """Deprecated alias → yaw-only (no forward IBVS)."""
        allowed = {
            k: kwargs[k]
            for k in ("timeout", "yaw_gain", "max_wz", "err_ok")
            if k in kwargs
        }
        return self.align_yaw_optional(color, **allowed)

    def _cancel_arm_goal(self) -> None:
        """Cancel stuck arm traj so the next lift/pick is not blocked."""
        gh = self._arm_goal_handle
        self._arm_goal_handle = None
        if gh is None:
            return
        try:
            fut = gh.cancel_goal_async()
            self._spin_until(fut, 2.0)
            self.node.get_logger().info("ARM goal canceled (clear stuck traj)")
            time.sleep(0.05)  # let JTC release before next goal
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().warn(f"ARM cancel: {exc}")

    def _traj_duration(self, sec, nsec=0) -> MsgDuration:
        """Float seconds → Duration (int(sec) alone turned 0.7s into 0s and jammed lifts)."""
        total_ns = int(round(float(sec) * 1_000_000_000.0)) + int(nsec)
        if total_ns < 0:
            total_ns = 0
        # Never emit a zero-length first segment (controller may ignore traj)
        if total_ns == 0:
            total_ns = 100_000_000  # 0.1 s
        return MsgDuration(
            sec=total_ns // 1_000_000_000,
            nanosec=total_ns % 1_000_000_000,
        )

    def _send_traj(self, client: ActionClient, joint_names, points, timeout=60.0) -> bool:
        if client is self.arm:
            self._cancel_arm_goal()
            if not self.arm.wait_for_server(timeout_sec=2.0):
                self.node.get_logger().error("ARM action server unavailable")
                return False
        traj = JointTrajectory()
        traj.joint_names = list(joint_names)
        # stamp=0 → execute immediately (non-zero future stamp can be ignored)
        traj.header.stamp.sec = 0
        traj.header.stamp.nanosec = 0
        last_t = -1
        for positions, sec, nsec in points:
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in positions]
            pt.time_from_start = self._traj_duration(sec, nsec)
            t_ns = int(pt.time_from_start.sec) * 1_000_000_000 + int(
                pt.time_from_start.nanosec
            )
            if t_ns <= last_t:
                t_ns = last_t + 200_000_000
                pt.time_from_start = MsgDuration(
                    sec=t_ns // 1_000_000_000,
                    nanosec=t_ns % 1_000_000_000,
                )
            last_t = t_ns
            traj.points.append(pt)

        # Log times so we can see if duration encoding broke again
        times = [
            f"{p.time_from_start.sec + p.time_from_start.nanosec * 1e-9:.2f}s"
            for p in traj.points
        ]
        self.node.get_logger().info(
            f"ARM_TRAJ n={len(traj.points)} t={times} → "
            f"{['%.2f' % x for x in traj.points[-1].positions]}"
        )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        send = client.send_goal_async(goal)
        if not self._spin_until(send, 20.0):
            self.node.get_logger().error("ARM_TRAJ send_goal timeout")
            return False
        handle = send.result()
        if handle is None or not handle.accepted:
            self.node.get_logger().error("ARM_TRAJ goal REJECTED")
            return False
        if client is self.arm:
            self._arm_goal_handle = handle
        result_fut = handle.get_result_async()
        ok = self._spin_until(result_fut, timeout)
        if not ok:
            self.node.get_logger().error(
                f"ARM_TRAJ result timeout ({timeout:.1f}s) → cancel"
            )
            if client is self.arm:
                self._cancel_arm_goal()
            return False
        wrapped = result_fut.result()
        # Humble: wrapped.result.error_code (0 = SUCCESSFUL)
        err = 0
        try:
            err = int(wrapped.result.error_code)
        except Exception:  # noqa: BLE001
            err = -1
        if client is self.arm:
            self._arm_goal_handle = None
        if err != 0:
            self.node.get_logger().error(
                f"ARM_TRAJ finished with error_code={err}"
            )
            return False
        return True

    def arm_survey_pose(self, j1: float = 0.0) -> bool:
        """Overhead look: wrist cam pitches down at shelf (not at floor by base)."""
        j1 = float(j1)
        q = survey_joints(j1)
        self.node.get_logger().info(
            f"ARM look_down (wrist overhead view) j1={j1:.3f} q={['%.2f' % x for x in q]}"
        )
        return self._send_traj(
            self.arm,
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            [
                ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0, 500),
                (q, 1.4, 0),
            ],
        )

    def arm_pregrasp_pose(self, j1: float = 0.0) -> bool:
        """Raise arm to first look pose (wrist FOV toward shelf)."""
        return self.arm_survey_pose(j1)

    def arm_pregrasp_nudge(self, j1: float) -> bool:
        """Small lateral adjust at pregrasp height (chassis stays put)."""
        j1 = float(j1)
        q = list(PREGRASP)
        q[0] = j1
        self.node.get_logger().info(f"ARM pregrasp nudge j1={j1:.3f}")
        return self._send_traj(
            self.arm,
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            [
                (q, 0, 500),
                (q, 1, 500),
            ],
            timeout=20.0,
        )

    def arm_pick_pose(self, j1: float = 0.0) -> bool:
        """Reach to calibrated PICK (faster ~20%)."""
        j1 = float(j1)
        pick = list(PICK)
        pick[0] = j1
        self.node.get_logger().info(
            f"ARM pick pose j1={j1:.3f} q={['%.2f' % x for x in pick]}"
        )
        ok = self._send_traj(
            self.arm,
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            [
                ([j1, 0.55, 0.85, 0.0, -0.35, 0.0], 0.14, 0),
                (pick, 0.36, 0),
            ],
            timeout=1.5,
        )
        if not ok:
            self.node.get_logger().warn(
                "ARM pick traj incomplete → cancel，继续吸附"
            )
            self._cancel_arm_goal()
            self.stop_chassis()
        return True

    def arm_carry_pose(self, j1: float = 0.0) -> bool:
        """Lift from PICK then tuck (faster ~20%)."""
        j1 = float(j1)
        pick = list(PICK)
        pick[0] = j1
        self.node.get_logger().info("ARM carry pose (快速抬起)")
        self._cancel_arm_goal()
        time.sleep(0.02)
        ok = self._send_traj(
            self.arm,
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            [
                (pick, 0.05, 0),
                ([j1, 0.70, 1.05, 0.0, -0.55, 0.0], 0.18, 0),
                ([0.0, 0.25, 1.20, 0.0, -1.00, 0.0], 0.38, 0),
            ],
            timeout=1.7,
        )
        if ok:
            return True
        self.node.get_logger().warn("ARM carry failed → cancel+retry")
        self._cancel_arm_goal()
        time.sleep(0.02)
        ok2 = self._send_traj(
            self.arm,
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            [
                ([0.0, 0.50, 1.00, 0.0, -0.60, 0.0], 0.16, 0),
                ([0.0, 0.20, 1.20, 0.0, -1.00, 0.0], 0.40, 0),
            ],
            timeout=1.5,
        )
        if not ok2:
            self.node.get_logger().error(
                "ARM carry still failed — continue anyway"
            )
            self._cancel_arm_goal()
            return False
        return True

    def arm_place_pose(self) -> bool:
        """Lower toward ground before detach — EE ends ~PLACE_FRONT_M ahead of base."""
        self.node.get_logger().info(
            f"ARM place pose (投放约前方 {PLACE_FRONT_M:.2f}m，快速)"
        )
        return self._send_traj(
            self.arm,
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            [
                ([0.0, 0.45, 0.85, 0.0, -0.50, 0.0], 0.11, 0),
                (list(PLACE_FINAL), 0.34, 0),
            ],
            timeout=1.4,
        )

    def arm_lift_after_place(self) -> bool:
        """Immediately tuck up after drop — no dwell."""
        self.node.get_logger().info("ARM lift after place (马上抬起)")
        self._cancel_arm_goal()
        ok = self._send_traj(
            self.arm,
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            [
                (list(PLACE_FINAL), 0.04, 0),
                ([0.0, 0.45, 1.00, 0.0, -0.70, 0.0], 0.14, 0),
                ([0.0, 0.15, 1.20, 0.0, -1.00, 0.0], 0.30, 0),
            ],
            timeout=1.4,
        )
        if not ok:
            self._cancel_arm_goal()
            self.node.get_logger().warn("place-lift incomplete → soft continue")
        return True

    def arm_home(self) -> bool:
        self.node.get_logger().info("ARM home")
        return self._send_traj(
            self.arm,
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            [
                ([0.0, 0.25, 0.7, 0.0, -0.6, 0.0], 0.10, 0),
                ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.45, 0),
            ],
            timeout=1.6,
        )

    def gripper_open(self) -> bool:
        """Open both fingers (URDF upper≈0.06). Never slam closed first."""
        self.node.get_logger().info("GRIPPER open cmd (both fingers)")
        # Stay at open — starting at [0,0] looked like a false clamp before attach.
        return self._send_traj(
            self.grip,
            ["finger_joint1", "finger_joint2"],
            [([0.05, 0.05], 0, 200000000), ([0.05, 0.05], 1, 0)],
            timeout=15.0,
        )

    def ensure_gripper_open(
        self, *, min_pos: float = 0.035, timeout: float = 6.0
    ) -> bool:
        """Open + verify both fingers via /joint_states (MTC open stage)."""
        f1 = self._finger_pos
        f2 = self._finger2_pos
        if f1 is not None and f1 >= min_pos and (f2 is None or f2 >= min_pos):
            f2s = f"{f2:.3f}" if f2 is not None else "n/a"
            self.node.get_logger().info(
                f"GRIPPER already open finger1={f1:.3f} finger2={f2s} — skip traj"
            )
            return True
        self.gripper_open()
        t0 = time.time()
        while time.time() - t0 < timeout:
            f1 = self._finger_pos
            f2 = self._finger2_pos
            # If finger2 not in joint_states yet (old URDF), require f1 only
            f1_ok = f1 is not None and f1 >= min_pos
            f2_ok = f2 is None or f2 >= min_pos
            if f1_ok and f2_ok:
                f2s = f"{f2:.3f}" if f2 is not None else "n/a"
                self.node.get_logger().info(
                    f"GRIPPER_OPEN_OK finger1={f1:.3f} finger2={f2s}"
                )
                return True
            time.sleep(0.1)
        self.node.get_logger().error(
            f"GRIPPER_OPEN_FAIL finger1={self._finger_pos} "
            f"finger2={self._finger2_pos} need>={min_pos}"
        )
        return False

    def gripper_close(self) -> bool:
        self.node.get_logger().info("GRIPPER close")
        return self._send_traj(
            self.grip,
            ["finger_joint1", "finger_joint2"],
            [([0.05, 0.05], 0, 200000000), ([0.0, 0.0], 1, 0)],
            timeout=15.0,
        )

    def attach(self, color: str, cube_id: int) -> bool:
        req = AttachLink.Request()
        req.model1_name = "six_arm"
        req.link1_name = "link6"
        req.model2_name = f"{color}_cube_{cube_id}"
        req.link2_name = "link"
        self.node.get_logger().info(f"ATTACH {req.model2_name}")
        fut = self.attach_cli.call_async(req)
        if not self._spin_until(fut, 4.0):
            return False
        res = fut.result()
        return bool(res and res.success)

    def detach(self, color: str, cube_id: int) -> bool:
        req = DetachLink.Request()
        req.model1_name = "six_arm"
        req.link1_name = "link6"
        req.model2_name = f"{color}_cube_{cube_id}"
        req.link2_name = "link"
        self.node.get_logger().info(f"DETACH {req.model2_name}")
        fut = self.detach_cli.call_async(req)
        if not self._spin_until(fut, 4.0):
            return False
        res = fut.result()
        return bool(res and res.success)
