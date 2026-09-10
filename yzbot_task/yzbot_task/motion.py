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

from nav2_msgs.action import NavigateToPose
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as MsgDuration
from std_srvs.srv import Empty
from linkattacher_msgs.srv import AttachLink, DetachLink
from sensor_msgs.msg import Image, CameraInfo, JointState
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
# Calibrated stand-off for this preset (arm_base forward reach ≈ 0.40 m at j1=0).
# visual_target_r MUST match PRESET_REACH_R or the fixed arm motion will miss the cube.
PICK = [0.0, 1.2, 1.17, 0.0, -0.3, 0.0]
PRESET_REACH_R = 0.40
CARRY = [0.0, 0.25, 0.7, 0.0, -0.6, 0.0]
PLACE = [0.0, 0.7, 1.0, 0.0, -0.4, 0.0]


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
        self.nav = ActionClient(node, NavigateToPose, "navigate_to_pose")
        self.arm = ActionClient(
            node, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory"
        )
        self.grip = ActionClient(
            node, FollowJointTrajectory, "/gripper_controller/follow_joint_trajectory"
        )
        self.attach_cli = node.create_client(AttachLink, "/ATTACHLINK")
        self.detach_cli = node.create_client(DetachLink, "/DETACHLINK")
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
        timeout: float = 30.0,
        *,
        need_arm: bool = True,
        need_attach: bool = True,
    ) -> bool:
        ok = True
        if not self.nav.wait_for_server(timeout_sec=timeout):
            self.node.get_logger().error("navigate_to_pose action unavailable — 先开 Nav2")
            ok = False
        if need_arm:
            if not self.arm.wait_for_server(timeout_sec=timeout):
                self.node.get_logger().error("arm_controller action unavailable")
                ok = False
            if not self.grip.wait_for_server(timeout_sec=5.0):
                self.node.get_logger().warn("gripper action unavailable (optional)")
        if need_attach:
            if not self.attach_cli.wait_for_service(timeout_sec=timeout):
                self.node.get_logger().error("/ATTACHLINK unavailable")
                ok = False
            if not self.detach_cli.wait_for_service(timeout_sec=timeout):
                self.node.get_logger().error("/DETACHLINK unavailable")
                ok = False
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

    def navigate(self, x: float, y: float, yaw: float, timeout: float = 180.0) -> bool:
        if self.drive_owner != "nav":
            self.node.get_logger().warn("navigate blocked until MODE_NAV; forcing exit_local")
            self.exit_local_mode()

        # 取消上一次目标，避免「第二次发目标车不动 / 旧 BT 卡死」
        self.cancel_nav()
        self.stop_chassis()
        time.sleep(0.15)

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
        self.node.get_logger().info(f"NAV → ({x:.2f}, {y:.2f}, yaw={yaw:.2f})")
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
        if not self._spin_until(result_fut, timeout):
            self.node.get_logger().error("nav result timeout")
            self.cancel_nav()
            return False
        self._nav_goal_handle = None
        from action_msgs.msg import GoalStatus

        wrapped = result_fut.result()
        status = int(getattr(wrapped, "status", GoalStatus.STATUS_UNKNOWN))
        if status != GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().error(f"nav finished status={status} (not SUCCEEDED)")
            return False
        return True

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
            if not self.navigate(x, y, yaw, timeout=timeout):
                self.node.get_logger().error(f"NAV attempt {i + 1}/{attempts} failed")
                continue
            time.sleep(max(float(settle_sec), 0.5))
            self.stop_chassis()
            ok, why = self.pose_gate_ok(x, y, yaw, xy_tol=xy_tol, yaw_tol=yaw_tol)
            self.node.get_logger().info(why)
            if ok:
                if reanchor:
                    # Pin AMCL to known goal — in-place turns before/after drift it
                    self.reanchor_amcl(x, y, yaw, settle_sec=0.6)
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
        xy_var: float = 0.12,
        yaw_var: float = 0.05,
    ) -> None:
        """Re-seed AMCL at a known map pose (like RViz 2D Pose Estimate).

        Call after Nav2 arrives at a verified park/zone, and after local yaw
        align, so the next NavigateToPose does not start from a skewed cloud.
        """
        self.stop_chassis()
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
            self.initialpose_pub.publish(msg)
            time.sleep(0.05)
        self.node.get_logger().info(
            f"AMCL_REANCHOR ({x:.2f},{y:.2f},yaw={yaw:.2f}) → /initialpose"
        )
        time.sleep(max(float(settle_sec), 0.0))

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
    ) -> None:
        """Before the next NavigateToPose: stop, optional reanchor, clear costmaps."""
        self.stop_chassis()
        self.exit_local_mode()
        if x is not None and y is not None and yaw is not None:
            self.reanchor_amcl(float(x), float(y), float(yaw), settle_sec=0.7)
        else:
            time.sleep(0.4)
        self.clear_costmaps()
        time.sleep(0.3)

    def stop_chassis(self, n: int = 5) -> None:
        tw = Twist()
        for _ in range(max(n, 1)):
            self.cmd_vel_pub.publish(tw)
            time.sleep(0.05)

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

    def enter_local_mode(self, settle_sec: float = 1.5) -> None:
        """Cancel Nav ownership of wheels; task_sm may publish cmd_vel."""
        self.cancel_nav()
        self.stop_chassis()
        time.sleep(max(float(settle_sec), 0.0))
        self.drive_owner = "local"
        self.stop_chassis()
        self.node.get_logger().info("MODE_LOCAL (task owns cmd_vel)")

    def exit_local_mode(self) -> None:
        """Hand wheels back to Nav2 for NavigateToPose."""
        self.stop_chassis()
        time.sleep(0.25)
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
        """Rotate slowly until base +x faces cube. Prefer closed-loop TF bearing_fn."""
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
                    f"AIM_MAP_OK rem={rem:+.3f}rad (慢速朝向物块)"
                )
                return True
            # Soft proportional + hard cap — protects AMCL
            wz = float(max(-max_wz, min(max_wz, yaw_gain * rem)))
            self.publish_cmd_vel(0.0, wz)
            if bearing_fn is None:
                rem -= wz * dt
            time.sleep(dt)
        self.stop_chassis()
        self.node.get_logger().error(f"AIM_MAP_FAIL rem={rem:+.3f}")
        return False

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
        lock_v_frac: float = 0.75,
        grasp_box_frac: float = 0.15,
        target_area_frac: float = 0.028,
        area_frac_tol: float = 0.012,
        target_fill: float = 0.90,
        fill_tol: float = 0.14,
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
                time.sleep(0.15)
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

            wz = 0.0
            vx = 0.0

            if err_u is not None:
                # === Have cube in camera: NEVER fight with TF bearing ===
                # Soft center: small lateral error → skip yaw, go straight to BOX
                yaw_ok = abs(err_u) <= max(err_ok, 0.14)
                if not yaw_ok:
                    phase = "CENTER"
                    center_streak = 0
                    wz = blob_to_yaw_cmd(blob, gain=yaw_gain, max_wz=max_wz)
                    vx = 0.0
                else:
                    center_streak += 1
                    phase = "BOX" if center_streak >= 1 else "CENTER"
                    # No micro-yaw once roughly on centerline (saves the left-twist)
                    wz = 0.0
                    # Phase 2: only after centered — approach for box fill
                    if phase == "BOX" and fill is not None and img_w > 0:
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
                        if max(bw, bh) > 1.05 * box_side:
                            vx = -min(max_vx, 0.08)
                        elif y1_img > 0.92 * float(img_h):
                            vx = -min(max_vx, 0.05)
                        else:
                            df = float(target_fill) - float(fill)
                            vx = float(max(-max_vx, min(max_vx, 1.1 * df)))
                            if rng_vis is not None:
                                dr = float(rng_vis) - float(target_r)
                                vx_r = float(
                                    max(-max_vx, min(max_vx, vx_gain * dr))
                                )
                                vx = 0.40 * vx + 0.60 * vx_r
                            # Keep creeping forward until nearly fill target
                            if fill < 0.85 and vx < 0.05:
                                vx = max(vx, 0.06)
            else:
                # No cube in view: slow twist toward known map coordinates only
                phase = "SEEK"
                center_streak = 0
                if bearing is not None and abs(bearing) > 0.08:
                    wz = float(max(-max_wz, min(max_wz, 0.75 * bearing)))
                if rng_tf is not None and (
                    bearing is None or abs(bearing) < 0.30
                ):
                    if float(rng_tf) > float(target_r) + 0.10:
                        vx = 0.04

            yaw_ok = err_u is not None and abs(err_u) <= max(err_ok, 0.14)
            # Soft snug: roughly in/near box is enough — don't demand perfect fill
            soft_snug = bool(
                fill is not None
                and (
                    in_box
                    or float(fill) >= 0.45
                    or (
                        abs(float(fill) - float(target_fill))
                        <= float(fill_tol) * 1.5
                    )
                )
            )
            # Centered + roughly close → OK (preset arm + attach can finish)
            lock_ok = bool(yaw_ok and soft_snug)

            self.node.get_logger().info(
                f"OPENCV_ALIGN[{phase}] vx={vx:+.2f} wz={wz:+.2f} "
                f"err_u={('n/a' if err_u is None else f'{err_u:+.3f}')} "
                f"bear={('n/a' if bearing is None else f'{bearing:+.3f}')} "
                f"fill={('n/a' if fill is None else f'{fill:.2f}')} "
                f"in_box={in_box} soft_snug={soft_snug}",
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
                            "OPENCV_ALIGN_OK → hold (中线大致OK，准备下探)"
                        )
                    elif time.time() - stable_t0 >= max(0.20, stable_hold_sec * 0.6):
                        self.stop_chassis()
                        self.node.get_logger().info(
                            f"OPENCV_ALIGN_STABLE err_u={err_u} fill={fill} → PICK"
                        )
                        return True
            else:
                ok_streak = 0
                stable_t0 = None
                self.publish_cmd_vel(vx, wz)
            time.sleep(0.08)

        self.stop_chassis()
        # Soft timeout: if we saw the cube at all, still allow pick (task_sm continues)
        self.node.get_logger().warn(
            "OPENCV_ALIGN_TIMEOUT → SOFT_OK (仍允许预设臂下探吸附)"
        )
        return True

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

    def _send_traj(self, client: ActionClient, joint_names, points, timeout=60.0) -> bool:
        traj = JointTrajectory()
        traj.joint_names = list(joint_names)
        for positions, sec, nsec in points:
            pt = JointTrajectoryPoint()
            pt.positions = list(positions)
            pt.time_from_start = MsgDuration(sec=int(sec), nanosec=int(nsec))
            traj.points.append(pt)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        send = client.send_goal_async(goal)
        if not self._spin_until(send, 30.0):
            return False
        handle = send.result()
        if not handle.accepted:
            return False
        result = handle.get_result_async()
        return self._spin_until(result, timeout)

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
                (q, 3, 0),
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
        """Reach-down to PICK. No high PREGRASP hop (looked like raise→clamp→down)."""
        j1 = float(j1)
        pick = list(PICK)
        pick[0] = j1
        mid = [j1, 1.05, 1.12, 0.0, -0.3, 0.0]
        self.node.get_logger().info(f"ARM pick pose j1={j1:.3f} (direct mid→pick)")
        return self._send_traj(
            self.arm,
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            [
                (mid, 0, 500),
                (pick, 3, 0),
            ],
        )

    def arm_carry_pose(self, j1: float = 0.0) -> bool:
        """Fold arm tight over base while driving (exit corridors / middle walls).

        Prior end still stuck forward → scraped walls on place-nav after pick.
        Final pose keeps EE near base so 2D footprint better matches reality.
        """
        j1 = float(j1)
        self.node.get_logger().info("ARM carry pose (tucked tight)")
        return self._send_traj(
            self.arm,
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            [
                ([j1, 1.15, 1.15, 0.0, -0.4, 0.0], 0, 400),
                ([0.0, 0.35, 1.10, 0.0, -0.95, 0.0], 2.0, 0),
                # 收得更靠车体，减少过中间墙时臂尖/夹爪刮墙
                ([0.0, 0.08, 1.35, 0.0, -1.25, 0.0], 4.0, 0),
            ],
        )

    def arm_place_pose(self) -> bool:
        """Lower a bit before detach at place zone."""
        self.node.get_logger().info("ARM place pose")
        return self._send_traj(
            self.arm,
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            [
                ([0.0, 0.25, 0.7, 0.0, -0.6, 0.0], 0, 500),
                ([0.0, 0.7, 1.0, 0.0, -0.4, 0.0], 3, 0),
            ],
        )

    def arm_home(self) -> bool:
        self.node.get_logger().info("ARM home")
        return self._send_traj(
            self.arm,
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            [
                ([0.0, 0.25, 0.7, 0.0, -0.6, 0.0], 0, 500),
                ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 4, 0),
            ],
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

    def snap_cube_fixed(self, color: str, cube_id: int) -> bool:
        """Snap numbered cube back to official fixed map pose (does not drift)."""
        from yzbot_task.vision_pose import known_cube_xy

        import subprocess

        x, y, z = known_cube_xy(color, cube_id)
        name = f"{color}_cube_{cube_id}"
        self.node.get_logger().info(
            f"CUBE_SNAP_FIXED {name} → ({x:.2f},{y:.2f},{z:.2f})"
        )
        try:
            subprocess.run(
                [
                    "gz",
                    "model",
                    "-m",
                    name,
                    "-x",
                    f"{x:.4f}",
                    "-y",
                    f"{y:.4f}",
                    "-z",
                    f"{z:.4f}",
                ],
                check=False,
                timeout=5,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().warn(f"snap_cube_fixed failed: {exc}")
            return False

    def raise_cube_inplace(self, color: str, cube_id: int, z: float = 0.82) -> bool:
        """Only bump cube Z at its shelf XY — keep collision, avoid flying to EE."""
        from yzbot_task.vision_pose import known_cube_xy

        import subprocess

        name = f"{color}_cube_{cube_id}"
        x, y, _z = known_cube_xy(color, cube_id)
        self.node.get_logger().info(
            f"raise {name} inplace → ({x:.2f}, {y:.2f}, {z:.2f})"
        )
        try:
            subprocess.run(
                [
                    "gz",
                    "model",
                    "-m",
                    name,
                    "-x",
                    f"{x:.4f}",
                    "-y",
                    f"{y:.4f}",
                    "-z",
                    f"{z:.4f}",
                ],
                check=False,
                timeout=5,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().warn(f"raise cube failed: {exc}")
            return False

    def snap_cube_near_ee(self, color: str, cube_id: int) -> bool:
        """Deprecated: caused cubes to 'fly' to the arm. Use raise_cube_inplace."""
        return self.raise_cube_inplace(color, cube_id)

    def attach(self, color: str, cube_id: int) -> bool:
        req = AttachLink.Request()
        req.model1_name = "six_arm"
        req.link1_name = "link6"
        req.model2_name = f"{color}_cube_{cube_id}"
        req.link2_name = "link"
        self.node.get_logger().info(f"ATTACH {req.model2_name}")
        fut = self.attach_cli.call_async(req)
        if not self._spin_until(fut, 30.0):
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
        if not self._spin_until(fut, 30.0):
            return False
        res = fut.result()
        return bool(res and res.success)
