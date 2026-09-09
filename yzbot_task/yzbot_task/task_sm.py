"""Task SM: center-lock → fixed range → preset arm grasp.

1) Nav coarse approach
2) Yaw until red/blue blob sits on camera crosshair (car faces cube)
3) Creep until estimated range ≈ fixed stand-off
4) STOP → preset PREGRASP/PICK (j1=0) → near ATTACH

Easier & more reliable than free Cartesian IK (user request).
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from yzbot_task.command_parser import parse_command, PickTask
from yzbot_task.motion import MotionHelpers, CARRY, PLACE, PICK, PREGRASP, PRESET_REACH_R
from yzbot_task.vision_align import annotate_bgr, detect_blob, is_visual_lock
from yzbot_task.vision_pose import (
    EstimateResult,
    estimate_cube_pose,
    offset_pose,
    ee_to_pose_distance_m,
    known_cube_pose_arm_raw,
    known_cube_xy,
    cube_bearing_in_base,
    bgr_hue_hint,
    approach_xy_for_cube,
    diagnose_est_vs_known,
    joint1_from_blob,
)
from yzbot_task.moveit_arm import MoveItArm
from yzbot_task.image_topics import resolve_image_topic


def _load_poses() -> Dict:
    try:
        share = Path(get_package_share_directory("yzbot_task"))
        path = share / "config" / "poses.yaml"
    except Exception:  # noqa: BLE001
        path = Path(__file__).resolve().parents[1] / "config" / "poses.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TaskStateMachine(Node):
    def __init__(self):
        super().__init__("task_sm")
        self.declare_parameter("dry_run", False)
        self.declare_parameter("skip_nav", False)
        self.declare_parameter("max_cubes", 5)
        self.declare_parameter("use_moveit", True)
        self.declare_parameter("use_vision_align", True)
        # OFF by default: continuous yaw IBVS made the car spin away
        self.declare_parameter("use_chassis_align", True)
        self.declare_parameter("align_timeout", 14.0)
        self.declare_parameter("align_yaw_gain", 0.95)
        self.declare_parameter("local_settle_sec", 0.5)
        # Nav2 常停在 0.25~0.35m / ~0.5rad 误差内，过严会反复失败
        self.declare_parameter("pose_gate_xy", 0.40)
        self.declare_parameter("pose_gate_yaw", 0.55)
        self.declare_parameter("nav_retries", 3)
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("shelf_z", 0.75)
        self.declare_parameter("grasp_above", 0.05)
        self.declare_parameter("grasp_down", 0.0)
        self.declare_parameter("attach_max_dist", 0.45)
        # known_coords = 按官方序号固定坐标：导航到物块前方 → 预设臂 → 吸附
        self.declare_parameter("grasp_mode", "known_coords")
        self.declare_parameter("prefer_joint_pick", True)
        self.declare_parameter("allow_known_map", True)
        self.declare_parameter("use_cube_standoff", True)

        self.declare_parameter("grasp_standoff_m", 0.40)

        self.declare_parameter("grasp_standoff_retry_m", 0.28)
        self.declare_parameter("lock_min_area", 60.0)
        self.declare_parameter("lock_max_err_u", 0.10)
        self.declare_parameter("lock_timeout", 14.0)
        # MUST match PRESET_REACH_R (~0.40): chassis stops where fixed PICK can touch
        self.declare_parameter("visual_target_r", float(PRESET_REACH_R))
        self.declare_parameter("visual_r_tol", 0.04)
        self.declare_parameter("range_timeout", 12.0)
        self.declare_parameter("timeout_see", 4.0)
        self.declare_parameter("sweep_yaw_rad", 0.55)
        self.declare_parameter("stable_hold_sec", 0.25)
        self.declare_parameter("preset_r_slack", 0.15)

        self.poses = _load_poses()
        self.cb_group = ReentrantCallbackGroup()
        self.motion = MotionHelpers(self)
        preferred = str(self.get_parameter("image_topic").value)
        bound = self.motion.bind_image_topic(preferred, auto=True)
        self.moveit = MoveItArm(self, self.motion.arm)
        self._cv_bridge = CvBridge()
        self.vision_dbg_pub = self.create_publisher(Image, "/yzbot/vision_debug", 10)
        self.busy = False
        self.cube_next = {"red": 1, "blue": 1}
        self.create_timer(2.0, self._maybe_rebind_image, callback_group=self.cb_group)
        self._image_rebound = False

        self.status_pub = self.create_publisher(String, "/task_status", 10)
        self.info_pub = self.create_publisher(String, "/task_info", 10)
        self.create_subscription(
            String,
            "/task_command",
            self._on_command,
            10,
            callback_group=self.cb_group,
        )
        mode = str(self.get_parameter("grasp_mode").value)
        self.get_logger().info(
            f"task_sm ready — grasp_mode={mode} "
            f"(导航到物块前→预设下探臂→吸附); FPV={bound}"
        )
        self.get_logger().info(
            "发任务: ros2 topic pub --once /task_command "
            "std_msgs/msg/String \"{data: '[red,1,A]'}\""
        )

    def _maybe_rebind_image(self):
        if self._image_rebound or self.busy:
            return
        if self.motion._last_image is not None:
            self._image_rebound = True
            return
        preferred = str(self.get_parameter("image_topic").value)
        topic = resolve_image_topic(self, preferred)
        if topic != getattr(self.motion, "_image_topic", None):
            self.get_logger().warn(f"无图像帧，重新订阅: {topic}")
            self.motion.bind_image_topic(topic, auto=False)
        self._image_rebound = True

    def _pub_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text)

    def _on_command(self, msg: String):
        if self.busy:
            self.get_logger().warn("busy, ignore new command")
            return
        self.busy = True
        text = msg.data
        self.get_logger().info(f"收到 /task_command: {text!r}")

        def _worker():
            try:
                self._run_command(text)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"task failed: {exc}")
                self._pub_status(f"ERROR: {exc}")
            finally:
                self.busy = False

        threading.Thread(target=_worker, daemon=True).start()

    def _expand(self, tasks: List[PickTask]) -> List[Tuple[str, int, str]]:
        queue: List[Tuple[str, int, str]] = []
        max_c = int(self.get_parameter("max_cubes").value)
        used = {"red": set(), "blue": set()}
        for t in tasks:
            if t.cube_id is not None:
                cid = int(t.cube_id)
                if cid < 1 or cid > max_c:
                    self.get_logger().error(f"bad cube id {t.color}_{cid}")
                    continue
                if cid in used[t.color]:
                    self.get_logger().warn(f"duplicate {t.color}_{cid}, skip")
                    continue
                used[t.color].add(cid)
                queue.append((t.color, cid, t.zone))
                # keep cube_next past this id so count-mode stays consistent
                self.cube_next[t.color] = max(self.cube_next[t.color], cid + 1)
                continue
            for _ in range(t.count):
                cid = self.cube_next[t.color]
                while cid <= max_c and cid in used[t.color]:
                    cid += 1
                if cid > max_c:
                    self.get_logger().error(f"no more {t.color} cubes")
                    break
                used[t.color].add(cid)
                queue.append((t.color, cid, t.zone))
                self.cube_next[t.color] = cid + 1
        return queue

    def _pose(self, key: str):
        p = self.poses[key]
        return float(p["x"]), float(p["y"]), float(p["yaw"])

    def _parking_xy(
        self, color: str, cid: int, *, standoff_m: Optional[float] = None
    ) -> Tuple[float, float, float]:
        if bool(self.get_parameter("use_cube_standoff").value):
            so = (
                float(self.get_parameter("grasp_standoff_m").value)
                if standoff_m is None
                else float(standoff_m)
            )
            return approach_xy_for_cube(color, cid, standoff_m=so)
        return self._pose(f"{color}_approach")

    def _nav_to_pick(
        self, color: str, cid: int, *, standoff_m: Optional[float] = None
    ) -> bool:
        ax, ay, ayaw = self._parking_xy(color, cid, standoff_m=standoff_m)
        so = (
            float(self.get_parameter("grasp_standoff_m").value)
            if standoff_m is None
            else float(standoff_m)
        )
        mx, my, mz = known_cube_xy(color, cid)
        self._pub_status(
            f"PARK_FIXED {color}_cube_{cid} 前方 "
            f"({ax:.2f},{ay:.2f},yaw={ayaw:.2f}) | "
            f"物块固定坐标 map≈({mx:.2f},{my:.2f},{mz:.3f}) standoff={so:.2f}m"
        )
        self.motion.exit_local_mode()
        xy_tol = float(self.get_parameter("pose_gate_xy").value)
        yaw_tol = float(self.get_parameter("pose_gate_yaw").value)
        retries = int(self.get_parameter("nav_retries").value)
        if not self.motion.navigate_and_verify(
            ax, ay, ayaw, retries=retries, xy_tol=xy_tol, yaw_tol=yaw_tol
        ):
            self._pub_status(f"NAV_PICK_FAIL / POSE_GATE_FAIL {color}_{cid}")
            return False
        self.motion.stop_chassis()
        self._pub_status("NAV_OK + POSE_GATE_OK (chassis STOP)")
        return True

    def _publish_dbg(self, bgr, blob, label: str):
        try:
            if bgr is None:
                return
            vis = annotate_bgr(bgr, blob, label=label)
            msg = self._cv_bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_link"
            self.vision_dbg_pub.publish(msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"vision_debug failed: {exc}")

    def _wait_front_lock(self, color: str):
        """Look-then-move gate: must see color blob. Does NOT spin the base."""
        min_a = float(self.get_parameter("lock_min_area").value)
        max_e = float(self.get_parameter("lock_max_err_u").value)
        timeout = float(self.get_parameter("lock_timeout").value)
        t0 = time.time()
        best = None
        while time.time() - t0 < timeout:
            self.motion.stop_chassis()
            bgr = self.motion.grab_front_bgr()
            if bgr is None:
                time.sleep(0.1)
                continue
            blob = detect_blob(bgr, color)
            self._publish_dbg(bgr, blob, f"lock:{color}")
            if is_visual_lock(blob, min_area=min_a, max_err_u=max_e):
                self._pub_status(
                    f"FRONT_LOCK {color} err_u={blob.err_u:+.2f} "
                    f"area={blob.area:.0f} center=({blob.cx:.0f},{blob.cy:.0f})"
                )
                try:
                    import cv2

                    cv2.imwrite("/tmp/yzbot_front_lock.jpg", bgr)
                except Exception:  # noqa: BLE001
                    pass
                return bgr, blob
            if blob is not None:
                best = blob
                self._pub_status(
                    f"FRONT_SEE {color} err_u={blob.err_u:+.2f} area={blob.area:.0f} "
                    f"(waiting lock…)"
                )
            else:
                self._pub_status(f"FRONT_NO_BLOB {bgr_hue_hint(bgr)}")
            time.sleep(0.15)
        self._pub_status(
            f"FRONT_LOCK_FAIL best="
            f"{'none' if best is None else f'area={best.area:.0f} err={best.err_u:+.2f}'}"
        )
        return None, None

    def _joint_grasp_j1(self, j1: float, cube_pose) -> Tuple[bool, object]:
        j1 = float(max(-0.7, min(0.7, j1)))
        self._pub_status(f"JOINT_GRASP j1={j1:+.2f} (aim blob center)")
        pre = list(PREGRASP)
        pre[0] = j1
        pick = list(PICK)
        pick[0] = j1
        if not self.moveit.execute_joints(pre, duration_sec=3.0):
            if not self.motion.arm_joints(pre, duration_sec=3.0):
                self._pub_status("PICK_ABORT JOINT_PRE_FAIL")
                return False, cube_pose
        if not self.moveit.execute_joints(pick, duration_sec=3.0):
            if not self.motion.arm_joints(pick, duration_sec=3.0):
                self._pub_status("PICK_ABORT JOINT_PICK_FAIL")
                return False, cube_pose
        self._pub_status("JOINT_GRASP_OK")
        return True, cube_pose

    def _finish_attach_lift(self, color: str, cid: int, grasp_pose) -> bool:
        max_d = float(self.get_parameter("attach_max_dist").value)
        dist = ee_to_pose_distance_m(self.motion.tf_buffer, grasp_pose)
        if dist is None:
            self._pub_status("PICK_ABORT DIST_UNKNOWN (无 grasping_frame TF)")
            return False
        self._pub_status(f"EE_TO_CUBE dist={dist:.3f}m (max={max_d:.3f})")
        if dist > max_d:
            self._pub_status(
                f"ATTACH_REJECTED_TOO_FAR ({dist:.3f}>{max_d:.3f}) "
                f"— 未吸附，避免把块扯飞"
            )
            return False

        self._pub_status("MTC_ATTACH")
        if not self.motion.attach(color, cid):
            self._pub_status(f"ATTACH_FAIL {color}_{cid}")
            return False
        self._pub_status(f"MTC_ATTACH_OK {color}_{cid}")
        time.sleep(0.2)

        self._pub_status("MTC_LIFT")
        if self.moveit.execute_joints(CARRY, duration_sec=4.0):
            self._pub_status("MTC_LIFT_OK")
        else:
            if not self.motion.arm_carry_pose(0.0):
                self._pub_status("PICK_ABORT MTC_LIFT_FAIL")
                return False
            self._pub_status("MTC_LIFT_OK (joint)")
        return True

    def _estimate_blob_range_m(self, bgr, color: str) -> Optional[float]:
        """Shelf-ray distance in arm_base (r=hypot(x,y)) for centered blob."""
        stamp = self.get_clock().now().to_msg()
        result = estimate_cube_pose(
            bgr=bgr,
            color=color,
            bridge=self.motion.bridge,
            tf_buffer=self.motion.tf_buffer,
            camera_frame=self.motion._front_frame,
            camera_info=self.motion._front_cinfo,
            depth_msg=None,
            shelf_z_base=float(self.get_parameter("shelf_z").value),
            stamp=stamp,
        )
        if not result.ok or result.estimate is None:
            return None
        # Skip apply_pose_sanity rejection path: use raw before sanity if needed
        # estimate_cube_pose already applies sanity; if need_closer, still return r
        p = result.estimate.pose_arm.pose.position
        return float(math.hypot(p.x, p.y))

    def _pick_center_range_preset(self, color: str, cid: int) -> bool:
        """User pipeline: center crosshair → fixed range → preset arm → attach."""
        self.motion.stop_chassis()
        self._pub_status("LOOK: fold arm (front cam clear)")
        self.motion.arm_joints([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], duration_sec=2.0)
        self.motion.stop_chassis()

        self._pub_status("MTC_OPEN")
        if not self.motion.ensure_gripper_open():
            self._pub_status("PICK_ABORT MTC_OPEN_FAIL")
            return False
        self._pub_status("MTC_OPEN_OK")

        if self.motion.drive_owner != "local":
            self.motion.enter_local_mode(
                settle_sec=float(self.get_parameter("local_settle_sec").value)
            )

        def _on_frame(bgr, blob, phase: str):
            self._publish_dbg(bgr, blob, phase)
            if blob is not None and abs(blob.err_u) < 0.08:
                try:
                    import cv2

                    cv2.imwrite("/tmp/yzbot_center_ok.jpg", bgr)
                except Exception:  # noqa: BLE001
                    pass

        def _est_r(bgr, _blob):
            return self._estimate_blob_range_m(bgr, color)

        bearing_info = cube_bearing_in_base(self.motion.tf_buffer, color, cid)
        map_bearing = None
        if bearing_info is not None:
            map_bearing, map_r, mxyz = bearing_info
            self._pub_status(
                f"CUBE_MAP ({mxyz[0]:.2f},{mxyz[1]:.2f},{mxyz[2]:.2f}) "
                f"bearing={map_bearing:+.3f}rad map_r={map_r:.2f}m → AIM_MAP"
            )
        else:
            self._pub_status("AIM_MAP_SKIP (TF无坐标；仅靠视觉)")

        def _bearing_live():
            info = cube_bearing_in_base(self.motion.tf_buffer, color, cid)
            return None if info is None else float(info[0])

        self._pub_status(
            "ALIGN: TF闭环朝向→看见色块→锁绿十字中线 "
            f"r≈{float(self.get_parameter('visual_target_r').value):.2f}m"
        )
        ok, measured_r = self.motion.face_center_then_range(
            color,
            err_ok=float(self.get_parameter("lock_max_err_u").value),
            yaw_gain=float(self.get_parameter("align_yaw_gain").value),
            target_r=float(self.get_parameter("visual_target_r").value),
            r_tol=float(self.get_parameter("visual_r_tol").value),
            timeout_see=float(self.get_parameter("timeout_see").value),
            timeout_yaw=float(self.get_parameter("lock_timeout").value),
            timeout_range=float(self.get_parameter("range_timeout").value),
            sweep_yaw_rad=float(self.get_parameter("sweep_yaw_rad").value),
            stable_hold_sec=float(self.get_parameter("stable_hold_sec").value),
            map_bearing=map_bearing,
            bearing_fn=_bearing_live,
            estimate_fn=_est_r,
            on_frame=_on_frame,
        )
        self.motion.stop_chassis()
        if not ok:
            self._pub_status(
                "PICK_ABORT SEARCH/CENTER/RANGE_FAIL (已停；看FPV绿十字/红圈)"
            )
            return False

        # Gate: chassis range must match preset arm reach or PICK will miss
        reach = float(PRESET_REACH_R)
        slack = float(self.get_parameter("preset_r_slack").value)
        if measured_r is None or abs(float(measured_r) - reach) > slack:
            self._pub_status(
                f"PICK_ABORT RANGE_MISMATCH_PRESET "
                f"r={measured_r} need {reach:.2f}±{slack:.2f} "
                f"(预设臂只够到≈{reach:.2f}m，站位不对不能夹)"
            )
            return False
        self._pub_status(
            f"READY_PRESET: r={float(measured_r):.3f}≈PRESET_REACH_R={reach:.2f} → 预设臂"
        )

        # After facing cube, preset pick with j1=0 (cube on centerline)
        stamp = self.get_clock().now().to_msg()
        bgr = self.motion.grab_front_bgr()
        cube_pose = None
        if bgr is not None:
            res = estimate_cube_pose(
                bgr=bgr,
                color=color,
                bridge=self.motion.bridge,
                tf_buffer=self.motion.tf_buffer,
                camera_frame=self.motion._front_frame,
                camera_info=self.motion._front_cinfo,
                depth_msg=None,
                shelf_z_base=float(self.get_parameter("shelf_z").value),
                stamp=stamp,
            )
            if res.ok and res.estimate is not None:
                cube_pose = res.estimate.pose_arm
                self._pub_status(
                    f"VISION_POSE[{res.estimate.method}] "
                    f"({cube_pose.pose.position.x:.3f},"
                    f"{cube_pose.pose.position.y:.3f},"
                    f"{cube_pose.pose.position.z:.3f})"
                )
        if cube_pose is None:
            km = known_cube_pose_arm_raw(self.motion.tf_buffer, color, cid, stamp)
            if not km.ok or km.estimate is None:
                self._pub_status("PICK_ABORT NO_POSE_FOR_ATTACH_GATE")
                return False
            cube_pose = km.estimate.pose_arm
            self._pub_status("ATTACH_GATE uses known_map XY")

        self._pub_status("PRESET_ARM j1=0 (正对后固定动作)")
        ok2, grasp_pose = self._joint_grasp_j1(0.0, cube_pose)
        if not ok2:
            return False
        time.sleep(0.25)
        # After preset: EE must be near cube — otherwise do NOT attach (avoid yanking)
        dist = ee_to_pose_distance_m(self.motion.tf_buffer, grasp_pose)
        max_d = float(self.get_parameter("attach_max_dist").value)
        if dist is None:
            self._pub_status("PICK_ABORT PRESET_EE_TF_UNKNOWN")
            return False
        self._pub_status(
            f"PRESET_REACH_CHECK ee_to_cube={dist:.3f}m (max={max_d:.3f}) "
            f"preset_r={PRESET_REACH_R:.2f}"
        )
        if dist > max_d:
            self._pub_status(
                f"PICK_ABORT PRESET_MISS ({dist:.3f}>{max_d:.3f}) "
                f"— 预设伸到位但离块仍远；调 visual_target_r 或 PICK 关节"
            )
            # retract to home so arm is not left poking
            self.motion.arm_joints([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], duration_sec=2.0)
            return False
        return self._finish_attach_lift(color, cid, grasp_pose)

    def _pick_vision_look_then_move(self, color: str, cid: int) -> bool:
        """Alias → center_range_preset (user preferred pipeline)."""
        return self._pick_center_range_preset(color, cid)

    def _pick_known_coords(self, color: str, cid: int) -> bool:
        """Park done → 开爪 → 预设下探到位 → 再吸附 → 抬起。

        严禁在臂未到位时 ATTACH：LinkAttacher 用单位姿对齐，会先把块扯到
        link6（看起来像「夹一下块就飞起来，然后臂才下来」）。
        """
        import subprocess
        import os

        self.motion.stop_chassis()

        # 清掉上次残留吸附，否则一动臂块就会跟着飞
        self._pub_status(f"DETACH_CLEAR {color}_{cid} (before arm)")
        try:
            self.motion.detach(color, cid)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"pre-detach ignored: {exc}")
        time.sleep(0.2)

        mx, my, mz = known_cube_xy(color, cid)
        if self.motion.snap_cube_fixed(color, cid):
            self._pub_status(
                f"CUBE_FIXED {color}_cube_{cid} @ map=({mx:.2f},{my:.2f},{mz:.2f})"
            )

        # 假吸附不需要夹爪动作：禁止开合，避免「举着先夹一下」
        self._pub_status("GRIPPER_SKIP (LinkAttacher，不发夹爪轨迹)")

        # 预设下探到位 —— 必须在吸附之前完成
        self._pub_status("PRESET_ARM 下探 (mid→PICK) — 到位后再吸")
        if not self.motion.arm_pick_pose(0.0):
            self._pub_status("PICK_ABORT PRESET_ARM_FAIL")
            return False
        time.sleep(0.5)

        # 臂到位后再吸附（cube_control / ATTACHLINK）
        script = os.path.join(
            os.path.expanduser("~/dev_ws/src/yzbot"), "cube_control.sh"
        )
        self._pub_status(f"ATTACH → cube_control on {color} {cid} (arm already at PICK)")
        ok = False
        if os.path.isfile(script):
            try:
                r = subprocess.run(
                    ["bash", script, "on", color, str(cid)],
                    check=False,
                    timeout=20,
                    capture_output=True,
                    text=True,
                )
                self.get_logger().info(r.stdout or "")
                ok = r.returncode == 0
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"cube_control.sh failed: {exc}")
        if not ok and not self.motion.attach(color, cid):
            self._pub_status(f"ATTACH_FAIL {color}_{cid}")
            return False
        self._pub_status(f"MTC_ATTACH_OK {color}_{cid}")
        time.sleep(0.2)

        # 抬起带走，再等底盘/定位稳定后再去放置区
        self._pub_status("MTC_LIFT")
        if not self.motion.arm_carry_pose(0.0):
            self.get_logger().warn("carry pose failed; cube may still be attached")
        else:
            self._pub_status("MTC_LIFT_OK")
        self.motion.stop_chassis()
        time.sleep(1.5)
        return True

    def _place_with_moveit(self, color: str, cid: int) -> bool:
        """到放置区：预设放置姿态 → Detach。"""
        import subprocess
        import os

        self._pub_status("PLACE_ARM 预设放置姿态")
        if not self.motion.arm_place_pose():
            self.get_logger().warn("place pose failed; still try detach")
        self._pub_status(f"DETACH → cube_control off {color} {cid}")
        script = os.path.join(
            os.path.expanduser("~/dev_ws/src/yzbot"), "cube_control.sh"
        )
        ok = False
        if os.path.isfile(script):
            try:
                r = subprocess.run(
                    ["bash", script, "off", color, str(cid)],
                    check=False,
                    timeout=20,
                    capture_output=True,
                    text=True,
                )
                self.get_logger().info(r.stdout or "")
                ok = r.returncode == 0
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"cube_control off failed: {exc}")
        if not ok and not self.motion.detach(color, cid):
            self._pub_status(f"DETACH_FAIL {color}_{cid}")
            return False
        self._pub_status(f"DETACH_OK {color}_{cid}")
        self.motion.arm_home()
        return True

    def _run_command(self, text: str):
        dry = bool(self.get_parameter("dry_run").value)
        skip_nav = bool(self.get_parameter("skip_nav").value) or dry

        try:
            tasks = parse_command(text)
        except ValueError as exc:
            self._pub_status(f"PARSE_FAIL: {exc}")
            return

        info = String()
        info.data = ";".join(str(t) for t in tasks)
        self.info_pub.publish(info)
        self._pub_status(f"PARSED: {info.data}")

        queue = self._expand(tasks)
        if not queue:
            self._pub_status("EMPTY_QUEUE")
            return

        if dry:
            for color, cid, zone in queue:
                self._pub_status(f"DRY: pick {color}_{cid} -> zone {zone}")
            self._pub_status("DONE_DRY")
            return

        mode_needs_arm = True  # known_coords 也要臂下探
        if not self.motion.wait_servers(
            timeout=20.0,
            need_arm=mode_needs_arm,
            need_attach=True,
        ):
            self._pub_status("SERVERS_UNAVAILABLE (先开 Gazebo + Nav2 + 臂控制器；吸附要 /ATTACHLINK)")
            return

        for idx, (color, cid, zone) in enumerate(queue, start=1):
            self._pub_status(f"STEP {idx}/{len(queue)}: {color}_{cid} -> {zone}")
            zx, zy, zyaw = self._pose(f"zone_{zone.lower()}")

            mode = str(self.get_parameter("grasp_mode").value).strip().lower()
            self.motion.exit_local_mode()

            if not skip_nav:
                self._pub_status(f"NAV_START → pick {color}_cube_{cid}")
                if not self._nav_to_pick(color, cid):
                    return

            pick_ok = False
            if mode in ("known_coords", "known", "map", "coord", "attach_only"):
                self.motion.stop_chassis()
                self._pub_status("PICK: 预设下探 → 吸附")
                pick_ok = self._pick_known_coords(color, cid)
            else:
                settle = float(self.get_parameter("local_settle_sec").value)
                self._pub_status("MODE_LOCAL (视觉对准)")
                self.motion.enter_local_mode(settle_sec=settle)
                self.motion.stop_chassis()
                try:
                    pick_ok = self._pick_center_range_preset(color, cid)
                finally:
                    self.motion.stop_chassis()
                    self._pub_status("MODE_NAV (交回导航)")
                    self.motion.exit_local_mode()

            if not pick_ok:
                self._pub_status("STOP (未夹到，不去放置区)")
                return

            if not skip_nav:
                self.motion.exit_local_mode()
                self._pub_status(f"NAV_START → place zone {zone}")
                if not self.motion.navigate_and_verify(
                    zx,
                    zy,
                    zyaw,
                    retries=3,
                    xy_tol=0.45,
                    yaw_tol=0.50,
                    settle_sec=2.0,
                ):
                    self._pub_status(f"NAV_PLACE_FAIL {zone}")
                    return

            if not self._place_with_moveit(color, cid):
                return
            self._pub_status(f"PLACED {color}_{cid} at {zone}")

        self._pub_status("DONE")


def main(args=None):
    rclpy.init(args=args)
    node = TaskStateMachine()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown()
        except Exception:  # noqa: BLE001
            pass
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
