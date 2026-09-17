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
from geometry_msgs.msg import PointStamped
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from yzbot_task.command_parser import parse_command, PickTask
from yzbot_task.route_planner import plan_route
from yzbot_task.motion import MotionHelpers, CARRY, PLACE, PICK, PRESET_REACH_R
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
        self.declare_parameter("align_yaw_gain", 0.70)
        # Local align yaw (before “slow search” cut to 0.10)
        self.declare_parameter("align_max_wz", 0.20)
        self.declare_parameter("local_settle_sec", 0.15)
        # Nav2 often stops ~0.5–0.8m short (inflation); soft gate then continuous local
        self.declare_parameter("pose_gate_xy", 0.55)
        # Place can be loose; pick must face cube or front cam sees nothing
        self.declare_parameter("pose_gate_yaw", 0.75)
        self.declare_parameter("pose_gate_yaw_pick", 0.25)
        self.declare_parameter("nav_retries", 1)
        # Door vias (code-only): first red cube pick/place A|B must pass these points
        self.declare_parameter("door_via_enable", True)
        # Fixed waypoints (map): (-4.2,0) and (-3.2,0); small XY error OK
        self.declare_parameter("door_via_xy_tol", 0.35)
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("shelf_z", 0.75)
        self.declare_parameter("grasp_above", 0.05)
        self.declare_parameter("grasp_down", 0.0)
        self.declare_parameter("attach_max_dist", 0.45)
        # Place: park so ~place_front_m ahead of base_link = zone center (cube on letter)
        self.declare_parameter("place_front_m", 0.20)
        # known_coords = 按官方序号固定坐标：导航到物块前方 → 预设臂 → 吸附
        self.declare_parameter("grasp_mode", "known_coords")
        self.declare_parameter("prefer_joint_pick", True)
        self.declare_parameter("allow_known_map", True)
        self.declare_parameter("use_cube_standoff", True)
        # Nav park slightly outside inflate; final stop farther for vision room
        self.declare_parameter("grasp_standoff_m", 0.50)
        self.declare_parameter("grasp_standoff_retry_m", 0.55)
        # Final stand-off ≈ preset arm reach — straight-on must close to this before pick
        self.declare_parameter("grasp_creep_m", 0.40)
        # Last snap before arm-down — leave a little room so FPV keeps the cube
        self.declare_parameter("pick_final_r", 0.34)
        self.declare_parameter("lock_min_area", 60.0)
        self.declare_parameter("lock_max_err_u", 0.10)
        self.declare_parameter("lock_timeout", 14.0)
        # MUST match PRESET_REACH_R (~0.40): chassis stops where fixed PICK can touch
        self.declare_parameter("visual_target_r", float(PRESET_REACH_R))
        self.declare_parameter("visual_r_tol", 0.04)
        self.declare_parameter("range_timeout", 12.0)
        self.declare_parameter("timeout_see", 4.0)
        self.declare_parameter("sweep_yaw_rad", 0.25)
        self.declare_parameter("stable_hold_sec", 0.40)
        self.declare_parameter("preset_r_slack", 0.15)
        # Fixed pick visual gate: whole cube inside large square + pixel range
        self.declare_parameter("lock_v_frac", 0.82)
        self.declare_parameter("grasp_box_frac", 0.15)
        self.declare_parameter("target_area_frac", 0.028)
        self.declare_parameter("area_frac_tol", 0.012)
        self.declare_parameter("target_fill", 1.26)
        self.declare_parameter("fill_tol", 0.08)
        self.declare_parameter("cube_width_m", 0.03)
        # Global shortest tour over pick→place legs (may reorder LLM groups)
        self.declare_parameter("use_route_planner", True)
        self.declare_parameter("plan_start_x", 0.0)
        self.declare_parameter("plan_start_y", 0.0)

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
        self.plan_pub = self.create_publisher(String, "/task_plan", 10)
        self.grasp_target_pub = self.create_publisher(
            PointStamped, "/yzbot/grasp_target", 10
        )
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

    def _is_first_red(self, color: str, cid: int) -> bool:
        return str(color).lower() == "red" and int(cid) == 1

    def _door_via_for_zone(self, zone: str) -> bool:
        """Door vias only for first-red when place zone is A or B (not C)."""
        return str(zone).strip().upper() in ("A", "B")

    def _door_via_points(self) -> List[Tuple[float, float]]:
        """Fixed mid-path points for red,1 → A/B (no visualization)."""
        return [(-4.2, 0.0), (-3.2, 0.0)]

    def _ordered_door_vias(
        self, fx: float, fy: float
    ) -> List[Tuple[float, float]]:
        """Visit vias in order along current robot → final goal."""
        vias = list(self._door_via_points())
        cur = self.motion.get_base_xy_yaw_map()
        rx, ry = (float(cur[0]), float(cur[1])) if cur is not None else (0.0, 0.0)
        dx, dy = float(fx) - rx, float(fy) - ry
        L = math.hypot(dx, dy) or 1.0

        def _along(v: Tuple[float, float]) -> float:
            return ((v[0] - rx) * dx + (v[1] - ry) * dy) / L

        return sorted(vias, key=_along)

    def _via_point_still_needed(
        self, vx: float, vy: float, fx: float, fy: float, tol: float
    ) -> bool:
        """Skip only if already at/past this fixed point toward the goal."""
        cur = self.motion.get_base_xy_yaw_map()
        if cur is None:
            return True
        rx, ry = float(cur[0]), float(cur[1])
        if math.hypot(rx - vx, ry - vy) <= tol:
            return False
        gx, gy = float(fx) - vx, float(fy) - vy
        gl = math.hypot(gx, gy)
        if gl < 1e-3:
            return False
        along = ((rx - vx) * gx + (ry - vy) * gy) / gl
        # Past the point toward goal by more than tol → don't turn back
        return along <= tol

    def _door_vias_needed_for_goal(self, fx: float, fy: float) -> bool:
        """Only use door vias when the robot is still east of the corridor.

        Pick red#1 park is near (-6.5,-2.8); stopping at (-3.2,0) looks 'far
        from cube + idle' if the final leg never runs. Skip vias when already
        on the west side (x ≲ -3.5) or goal does not need the door.
        """
        cur = self.motion.get_base_xy_yaw_map()
        if cur is None:
            return True
        rx = float(cur[0])
        # Already past / at door corridor toward western reds
        if rx <= -3.5:
            return False
        # Goal also east of door — vias would be a detour
        if float(fx) >= -3.0:
            return False
        return True

    def _nav_continuous_with_door(
        self,
        fx: float,
        fy: float,
        fyaw: float,
        *,
        place_mode: bool = False,
        xy_tol: Optional[float] = None,
        yaw_tol: Optional[float] = None,
        use_door_via: bool = True,
    ) -> bool:
        """Fixed vias (-4.2,0) then (-3.2,0) [order by path], then final.

        Point goals with modest XY tol (not a soft area). Never replan back to
        a via already passed. Always finish with navigate_and_verify to (fx,fy).
        """
        self.motion.exit_local_mode()
        tol = max(0.15, float(self.get_parameter("door_via_xy_tol").value))
        used_via = False
        want_via = (
            bool(self.get_parameter("door_via_enable").value)
            and bool(use_door_via)
            and self._door_vias_needed_for_goal(fx, fy)
        )
        if bool(use_door_via) and not want_via:
            self._pub_status("VIA skip — already west of door / not needed")

        if want_via:
            for vx, vy in self._ordered_door_vias(fx, fy):
                if not self._via_point_still_needed(vx, vy, fx, fy, tol):
                    self._pub_status(
                        f"VIA skip ({vx:.2f},{vy:.2f}) — already at/past"
                    )
                    continue
                via_yaw = math.atan2(float(fy) - vy, float(fx) - vx)
                self._pub_status(
                    f"VIA fixed → ({vx:.2f},{vy:.2f}) tol={tol:.2f}m then onward"
                )
                used_via = True
                ok = self.motion.navigate(
                    vx,
                    vy,
                    via_yaw,
                    timeout=90.0,
                    early_xy_tol=tol,
                    soft_start=True,
                )
                if not ok and self._via_point_still_needed(vx, vy, fx, fy, tol):
                    self._pub_status(f"VIA_FAIL ({vx:.2f},{vy:.2f})")
                    return False
                time.sleep(0.05)

        if place_mode:
            ok = self.motion.navigate_place_zone(
                fx, fy, xy_tol=float(xy_tol or 0.45)
            )
        else:
            ok = self.motion.navigate_and_verify(
                fx,
                fy,
                fyaw,
                retries=int(self.get_parameter("nav_retries").value),
                xy_tol=float(
                    xy_tol
                    if xy_tol is not None
                    else self.get_parameter("pose_gate_xy").value
                ),
                yaw_tol=float(
                    yaw_tol
                    if yaw_tol is not None
                    else self.get_parameter("pose_gate_yaw").value
                ),
            )
        if not ok:
            self._pub_status("VIA_NAV_FAIL" if used_via else "NAV_FAIL")
            return False
        return True

    def _parking_xy(
        self, color: str, cid: int, *, standoff_m: Optional[float] = None
    ) -> Tuple[float, float, float]:
        if bool(self.get_parameter("use_cube_standoff").value):
            so = (
                float(self.get_parameter("grasp_standoff_m").value)
                if standoff_m is None
                else float(standoff_m)
            )
            robot_xy = None
            robot_yaw = None
            cur = self.motion.get_base_xy_yaw_map()
            if cur is not None:
                robot_xy = (cur[0], cur[1])
                robot_yaw = float(cur[2])
            return approach_xy_for_cube(
                color,
                cid,
                standoff_m=so,
                robot_xy=robot_xy,
                robot_yaw=robot_yaw,
            )
        return self._pose(f"{color}_approach")

    def _blue2_east_approach_vias(
        self, cx: float, cy: float, park_x: float, park_y: float
    ) -> List[Tuple[float, float]]:
        """Vias so nav reaches blue#2 from +X without driving over the cube.

        If the robot is west/near the cube, swing north or south first, then to a
        staging point east of the cube, then to the park (still east).
        """
        cur = self.motion.get_base_xy_yaw_map()
        rx = float(cur[0]) if cur is not None else park_x + 1.0
        ry = float(cur[1]) if cur is not None else park_y
        stage_x = max(float(park_x), float(cx) + 0.85)
        stage = (stage_x, float(cy))
        vias: List[Tuple[float, float]] = []

        # Would a straight path cross the cube? (robot still on −X / over cube)
        if rx < float(cx) + 0.25:
            north = (stage_x, float(cy) + 0.95)
            south = (stage_x, float(cy) - 0.95)
            clear = north if abs(ry - north[1]) <= abs(ry - south[1]) else south
            vias.append(clear)
            vias.append(stage)
        elif rx < stage_x - 0.25:
            # Already past cube in x but not yet at staging — go east first
            vias.append(stage)
        return vias

    def _nav_with_vias_then_park(
        self,
        vias: List[Tuple[float, float]],
        fx: float,
        fy: float,
        fyaw: float,
        *,
        xy_tol: float,
        yaw_tol: float,
        via_tol: float = 0.40,
    ) -> bool:
        """Soft via points (early XY), then final navigate_and_verify."""
        self.motion.exit_local_mode()
        for vx, vy in vias:
            if not self._via_point_still_needed(vx, vy, fx, fy, via_tol):
                self._pub_status(f"VIA skip ({vx:.2f},{vy:.2f}) — already at/past")
                continue
            via_yaw = math.atan2(float(fy) - vy, float(fx) - vx)
            self._pub_status(
                f"VIA approach → ({vx:.2f},{vy:.2f}) (绕开物块，从 +X 靠近)"
            )
            ok = self.motion.navigate(
                vx,
                vy,
                via_yaw,
                timeout=90.0,
                early_xy_tol=via_tol,
                soft_start=True,
            )
            if not ok and self._via_point_still_needed(vx, vy, fx, fy, via_tol):
                self._pub_status(f"VIA_FAIL ({vx:.2f},{vy:.2f})")
                return False
            time.sleep(0.05)
        retries = int(self.get_parameter("nav_retries").value)
        return self.motion.navigate_and_verify(
            fx, fy, fyaw, retries=retries, xy_tol=xy_tol, yaw_tol=yaw_tol
        )

    def _nav_to_pick(
        self,
        color: str,
        cid: int,
        *,
        standoff_m: Optional[float] = None,
        place_zone: str = "",
    ) -> bool:
        ax, ay, ayaw = self._parking_xy(color, cid, standoff_m=standoff_m)
        so = (
            float(self.get_parameter("grasp_standoff_m").value)
            if standoff_m is None
            else float(standoff_m)
        )
        mx, my, mz = known_cube_xy(color, cid)
        axis = "同y沿x(baseX//mapX)" if abs(ay - my) < 1e-6 else "同x沿y(baseX//mapY)"
        # OpenCV FPV assist: publish cube map point for projection
        tgt = PointStamped()
        tgt.header.frame_id = "map"
        tgt.header.stamp = self.get_clock().now().to_msg()
        tgt.point.x = float(mx)
        tgt.point.y = float(my)
        tgt.point.z = float(mz)
        self.grasp_target_pub.publish(tgt)
        self._pub_status(
            f"PARK_AXIS {color}_cube_{cid} {axis} "
            f"({ax:.2f},{ay:.2f},yaw={ayaw:.2f}) | "
            f"cube=({mx:.2f},{my:.2f}) standoff={so:.2f}m"
        )
        xy_tol = float(self.get_parameter("pose_gate_xy").value)
        # Pick: must face cube — loose place yaw (0.75) leaves cam looking away
        yaw_tol = float(self.get_parameter("pose_gate_yaw_pick").value)
        # vision-v0.5: pick goes straight to park (door vias only on PLACE).
        # Using door vias on pick left the robot idle at (-3.2,0) far from cube.
        if str(color).lower() == "blue" and int(cid) == 2:
            # Never drive west→east over the cube; swing to +X then park
            vias = self._blue2_east_approach_vias(mx, my, ax, ay)
            self._pub_status(
                f"BLUE2_EAST_APPROACH vias={len(vias)} → park +X "
                f"({ax:.2f},{ay:.2f}) 不压块"
            )
            nav_ok = self._nav_with_vias_then_park(
                vias, ax, ay, ayaw, xy_tol=xy_tol, yaw_tol=yaw_tol
            )
            if not nav_ok:
                cur = self.motion.get_base_xy_yaw_map()
                dpark = (
                    math.hypot(float(cur[0]) - ax, float(cur[1]) - ay)
                    if cur is not None
                    else 99.0
                )
                if dpark > 1.20:
                    self._pub_status(
                        f"NAV_PICK_FAIL / POSE_GATE_FAIL {color}_{cid} "
                        f"dpark={dpark:.2f}m"
                    )
                    return False
                self._pub_status(
                    f"NAV_PICK_SOFT dpark={dpark:.2f}m → 本地按坐标 creep 靠近"
                )
        else:
            self.motion.exit_local_mode()
            self._pub_status(
                f"NAV_PICK_DIRECT → ({ax:.2f},{ay:.2f}) "
                f"(取块不绕门点，避免停门口)"
            )
            retries = int(self.get_parameter("nav_retries").value)
            nav_ok = self.motion.navigate_and_verify(
                ax,
                ay,
                ayaw,
                retries=retries,
                xy_tol=xy_tol,
                yaw_tol=yaw_tol,
                settle_sec=0.35,
                reanchor=True,
            )
            if not nav_ok:
                cur = self.motion.get_base_xy_yaw_map()
                dpark = (
                    math.hypot(float(cur[0]) - ax, float(cur[1]) - ay)
                    if cur is not None
                    else 99.0
                )
                # Nav2 often "succeeds"/gives up ~0.7m short of park (cube inflate)
                if dpark > 1.20:
                    self._pub_status(
                        f"NAV_PICK_FAIL / POSE_GATE_FAIL {color}_{cid} "
                        f"dpark={dpark:.2f}m"
                    )
                    return False
                self._pub_status(
                    f"NAV_PICK_SOFT dpark={dpark:.2f}m → 本地按坐标 creep 靠近"
                )
                self.motion.clear_costmaps()
        self.motion.stop_chassis()
        # Distance check only — face/creep/scan/opencv run in ONE continuous
        # MODE_LOCAL in the pick block (no stop-start between phases).
        if not self._ensure_near_cube(color, cid, standoff_m=so, max_range_m=1.10):
            self._pub_status(f"NAV_PICK_FAIL / TOO_FAR {color}_{cid}")
            return False
        self._pub_status(
            f"NAV_PICK_OK → 连续本地: 正对+停在~{float(self.get_parameter('grasp_creep_m').value):.2f}m"
        )
        return True

    def _ensure_near_cube(
        self,
        color: str,
        cid: int,
        *,
        standoff_m: float,
        max_range_m: float = 0.85,
        retries: int = 2,
    ) -> bool:
        """If TF range to known cube is large, re-park — do not idle in OpenCV far away."""
        from yzbot_task.vision_pose import cube_bearing_in_base

        for attempt in range(max(1, int(retries) + 1)):
            info = cube_bearing_in_base(self.motion.tf_buffer, color, cid)
            cur = self.motion.get_base_xy_yaw_map()
            if info is not None:
                _b, rng, xyz = info
                rng = float(rng)
            elif cur is not None:
                mx, my, _mz = known_cube_xy(color, cid)
                rng = math.hypot(float(cur[0]) - mx, float(cur[1]) - my)
                xyz = (mx, my, _mz)
            else:
                self.get_logger().warn("NEAR_CUBE skip (no TF/pose)")
                return True
            self._pub_status(
                f"NEAR_CUBE check r={rng:.2f}m (max {max_range_m:.2f}) "
                f"cube=({xyz[0]:.2f},{xyz[1]:.2f}) try={attempt + 1}"
            )
            if rng <= float(max_range_m):
                return True
            # Prefer local creep toward known cube (Nav often cannot close last 0.5–0.8m)
            settle = float(self.get_parameter("local_settle_sec").value)
            creep_r = float(self.get_parameter("grasp_creep_m").value)
            self._pub_status(
                f"TOO_FAR r={rng:.2f}m → CREEP 按坐标开到 ≤{creep_r:.2f}m "
                f"（try={attempt + 1}）"
            )
            self.motion.enter_local_mode(settle_sec=min(0.20, settle))
            try:
                if self.motion.creep_to_known_cube(
                    color,
                    cid,
                    target_range=creep_r,
                    max_vx=0.12,
                    timeout=12.0,
                ):
                    return True
            finally:
                self.motion.stop_chassis()
                self.motion.exit_local_mode()
            ax, ay, ayaw = self._parking_xy(color, cid, standoff_m=standoff_m)
            self._pub_status(
                f"TOO_FAR r={rng:.2f}m → 重停 PARK ({ax:.2f},{ay:.2f})"
            )
            self.motion.exit_local_mode()
            xy_tol = float(self.get_parameter("pose_gate_xy").value)
            yaw_tol = float(self.get_parameter("pose_gate_yaw_pick").value)
            if not self.motion.navigate_and_verify(
                ax,
                ay,
                ayaw,
                retries=1,
                xy_tol=xy_tol,
                yaw_tol=yaw_tol,
            ):
                continue
            self.motion.stop_chassis()
        info2 = cube_bearing_in_base(self.motion.tf_buffer, color, cid)
        if info2 is not None and float(info2[1]) <= float(max_range_m) + 0.15:
            return True
        return False

    def _face_cube_yaw(
        self, color: str, cid: int, *, manage_mode: bool = True, reanchor: bool = False
    ) -> bool:
        """Park 后按已知 map 坐标规矩转向：yaw → atan2(cube−robot)，直到车头对准。

        manage_mode=False: 调用方已在 MODE_LOCAL，不进出模式、不长时间 settle（流畅串联）。
        """
        settle = float(self.get_parameter("local_settle_sec").value)
        # Face-turn: a bit faster than fine OpenCV align
        max_wz = max(0.22, float(self.get_parameter("align_max_wz").value))
        yaw_gain = max(0.85, float(self.get_parameter("align_yaw_gain").value))
        mx, my, _mz = known_cube_xy(color, cid)
        tol = 0.12

        if manage_mode:
            self.motion.enter_local_mode(settle_sec=min(0.12, settle))
        try:
            ok = False
            for attempt in range(1, 3):
                cur = self.motion.get_base_xy_yaw_map()
                if cur is None:
                    self.get_logger().warn("FACE_CUBE skip (no map pose)")
                    return False
                rx, ry, ryaw = float(cur[0]), float(cur[1]), float(cur[2])
                yaw_des = math.atan2(my - ry, mx - rx)
                rem = self.motion._wrap_pi(yaw_des - ryaw)
                self._pub_status(
                    f"FACE_CUBE#{attempt} {color}_{cid} "
                    f"cube=({mx:.2f},{my:.2f}) robot=({rx:.2f},{ry:.2f}) "
                    f"yaw={ryaw:+.2f}→{yaw_des:+.2f} rem={rem:+.3f}rad"
                )
                if abs(rem) <= tol:
                    ok = True
                    break

                def _rem_live():
                    c = self.motion.get_base_xy_yaw_map()
                    if c is None:
                        return None
                    yd = math.atan2(my - float(c[1]), mx - float(c[0]))
                    return self.motion._wrap_pi(yd - float(c[2]))

                budget = min(8.0, abs(rem) / max(max_wz, 0.10) + 1.5)
                closed = self.motion.aim_yaw_to_bearing(
                    float(rem),
                    yaw_gain=yaw_gain,
                    max_wz=max_wz,
                    tol=tol,
                    timeout=budget,
                    bearing_fn=_rem_live,
                )
                if closed:
                    ok = True
                    break
                rem2 = _rem_live()
                if rem2 is None:
                    rem2 = rem
                if abs(float(rem2)) <= tol:
                    ok = True
                    break
                self.motion.spin_yaw_openloop(
                    float(rem2), max_wz=max_wz, timeout=budget
                )
                rem3 = _rem_live()
                if rem3 is not None and abs(float(rem3)) <= tol * 1.5:
                    ok = True
                    break

            if reanchor:
                c2 = self.motion.get_base_xy_yaw_map()
                if c2 is not None:
                    yd2 = math.atan2(my - float(c2[1]), mx - float(c2[0]))
                    self.motion.reanchor_amcl(
                        float(c2[0]), float(c2[1]), float(yd2), settle_sec=0.15
                    )
            self._pub_status(f"FACE_CUBE {'OK' if ok else 'SOFT'}")
            return ok
        finally:
            self.motion.stop_chassis()
            if manage_mode:
                self.motion.exit_local_mode()

    def _publish_dbg(
        self,
        bgr,
        blob,
        label: str,
        prefer_uv=None,
        lock_v_frac: Optional[float] = None,
        grasp_box_frac: Optional[float] = None,
    ):
        try:
            if bgr is None:
                return
            lv = (
                float(self.get_parameter("lock_v_frac").value)
                if lock_v_frac is None
                else float(lock_v_frac)
            )
            gb = (
                float(self.get_parameter("grasp_box_frac").value)
                if grasp_box_frac is None
                else float(grasp_box_frac)
            )
            vis = annotate_bgr(
                bgr,
                blob,
                label=label,
                prefer_uv=prefer_uv,
                lock_v_frac=lv,
                grasp_box_frac=gb,
            )
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
        """Direct to PICK only — no PREGRASP hop (that lateral swing throws off aim)."""
        j1 = float(max(-0.7, min(0.7, j1)))
        self._pub_status(f"JOINT_GRASP j1={j1:+.2f} (直达PICK，无预摆)")
        pick = list(PICK)
        pick[0] = j1
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
            max_wz=float(self.get_parameter("align_max_wz").value),
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

        # 假吸附不需要夹爪动作：禁止开合，避免「举着先夹一下」
        self._pub_status("GRIPPER_SKIP (LinkAttacher，不发夹爪轨迹)")

        # 预设下探到位 —— 抬高防顶地；超时也不空抖
        self._pub_status("PRESET_ARM 下探 → PICK（抬高防顶地）")
        if not self.motion.arm_pick_pose(0.0):
            self._pub_status("PICK_ABORT PRESET_ARM_FAIL")
            try:
                self.motion.arm_home()
            except Exception:  # noqa: BLE001
                pass
            return False
        time.sleep(0.25)

        # 臂到位后再吸附（cube_control / ATTACHLINK）
        script = os.path.join(
            os.path.expanduser("~/dev_ws/yzbot/src"), "cube_control.sh"
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
        # 等臂收稳再导航出门，避免中间墙刮臂
        time.sleep(2.0)
        return True

    def _place_with_moveit(self, color: str, cid: int) -> bool:
        """到放置区：预设放置姿态 → Detach → 收臂（不做瞬移/冻结/回货架）。"""
        import subprocess
        import os

        self._pub_status("PLACE_ARM 预设放置姿态")
        if not self.motion.arm_place_pose():
            self.get_logger().warn("place pose failed; still try detach")
        self._pub_status(f"DETACH → cube_control off {color} {cid}")
        script = os.path.join(
            os.path.expanduser("~/dev_ws/yzbot/src"), "cube_control.sh"
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

        # Unstick leftover Nav/local ownership after Ctrl-C / kill
        try:
            self.motion.cancel_nav()
            self.motion.stop_chassis()
            self.motion.exit_local_mode()
            self.motion.clear_costmaps()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"pre-clear ignored: {exc}")

        try:
            tasks = parse_command(text, max_cubes=int(self.get_parameter("max_cubes").value))
        except ValueError as exc:
            self._pub_status(f"PARSE_FAIL: {exc}")
            return

        info = String()
        info.data = ";".join(str(t) for t in tasks)
        self.info_pub.publish(info)
        self._pub_status(f"PARSED: {info.data}")

        max_c = int(self.get_parameter("max_cubes").value)
        if bool(self.get_parameter("use_route_planner").value):
            # Prefer live robot pose as tour start
            cur = self.motion.get_base_xy_yaw_map()
            if cur is not None:
                sx, sy = float(cur[0]), float(cur[1])
            else:
                sx = float(self.get_parameter("plan_start_x").value)
                sy = float(self.get_parameter("plan_start_y").value)
            try:
                plan = plan_route(
                    text,
                    start=(sx, sy),
                    max_cubes=max_c,
                    use_door_via=True,
                )
            except Exception as exc:  # noqa: BLE001
                self._pub_status(f"PLAN_FAIL: {exc} → fallback expand")
                queue = self._expand(tasks)
            else:
                for line in plan.summary().splitlines():
                    self.get_logger().info(line)
                pm = String()
                pm.data = plan.as_command()
                self.plan_pub.publish(pm)
                self._pub_status(
                    f"PLAN_OK ≈{plan.total_m:.1f}m → {pm.data}"
                )
                queue = list(plan.legs)
                # Keep cube_next consistent with planned ids
                for color, cid, _z in queue:
                    self.cube_next[color] = max(
                        self.cube_next[color], int(cid) + 1
                    )
        else:
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
        self._pub_status("WAITING_SERVERS (Nav2/臂/吸附，最多约 8s)…")
        if not self.motion.wait_servers(
            timeout=8.0,
            need_arm=mode_needs_arm,
            need_attach=True,
        ):
            self._pub_status("SERVERS_UNAVAILABLE (先开 Gazebo + Nav2 + 臂控制器；吸附要 /ATTACHLINK)")
            return
        # No map→odom ⇒ RViz 无全局路径、车也不走（常见于重启后未 2D Pose Estimate）
        loc = self.motion.get_base_xy_yaw_map()
        if loc is None:
            self._pub_status(
                "LOC_FAIL：无 map→odom。请先开 Nav2，在 RViz 对 Gazebo 真车位做 "
                "2D Pose Estimate，再重新发任务"
            )
            return
        self._pub_status(
            f"SERVERS_OK + LOC ({loc[0]:.2f},{loc[1]:.2f},yaw={loc[2]:.2f}) → 开始导航"
        )

        for idx, (color, cid, zone) in enumerate(queue, start=1):
            self._pub_status(f"STEP {idx}/{len(queue)}: {color}_{cid} -> {zone}")
            zx, zy, zyaw = self._pose(f"zone_{zone.lower()}")

            mode = str(self.get_parameter("grasp_mode").value).strip().lower()
            self.motion.exit_local_mode()

            if not skip_nav:
                self._pub_status(f"NAV_START → pick {color}_cube_{cid}")
                if not self._nav_to_pick(color, cid, place_zone=zone):
                    return

            pick_ok = False
            if mode in ("known_coords", "known", "map", "coord", "attach_only"):
                # ONE continuous MODE_LOCAL: face → creep → scan → opencv (no mid stops)
                mx, my, _mz = known_cube_xy(color, cid)
                so = float(self.get_parameter("grasp_creep_m").value)
                # Align / pick to preset reach (not a looser visual-only distance)
                reach_r = float(self.get_parameter("visual_target_r").value)
                settle = float(self.get_parameter("local_settle_sec").value)
                self._pub_status(
                    f"LOCAL_FLOW → 正对+开到~{reach_r:.2f}m(臂长)再贴框 "
                    f"{color}_{cid} map=({mx:.2f},{my:.2f})"
                )
                self.motion.enter_local_mode(settle_sec=min(0.08, settle))
                aligned = False
                try:
                    # Fast path: face → creep → opencv → pick (no post-spin!)
                    self._face_cube_yaw(
                        color, cid, manage_mode=False, reanchor=False
                    )
                    self._pub_status(f"CREEP_MAP → ≤{reach_r:.2f}m")
                    self.motion.creep_to_known_cube(
                        color,
                        cid,
                        target_range=reach_r,
                        max_vx=0.14,
                        max_wz=0.20,
                        timeout=8.0,
                    )

                    def _on_oc(
                        bgr,
                        blob,
                        phase: str,
                        prefer_uv=None,
                        lock_v_frac=None,
                        grasp_box_frac=None,
                    ):
                        self._publish_dbg(
                            bgr,
                            blob,
                            phase,
                            prefer_uv=prefer_uv,
                            lock_v_frac=lock_v_frac,
                            grasp_box_frac=grasp_box_frac,
                        )

                    # Quick see? skip long SCAN if already in FOV
                    bgr0 = self.motion.grab_front_bgr()
                    seen0 = False
                    if bgr0 is not None:
                        from yzbot_task.vision_align import detect_blob
                        seen0 = detect_blob(
                            bgr0, color, min_cy_frac=0.10
                        ) is not None
                    if not seen0:
                        self._pub_status("SCAN_MAP 短扫（未见色块）")
                        self.motion.scan_known_cube_until_seen(
                            color,
                            cid,
                            max_wz=0.18,
                            timeout=4.0,
                            on_frame=lambda b, bl, p, prefer_uv=None, lv=None, gb=None: _on_oc(
                                b, bl, p, prefer_uv, lv, gb
                            ),
                        )

                    self._pub_status("OPENCV_ALIGN")
                    aligned = self.motion.opencv_align_to_cube(
                        color,
                        cid,
                        target_r=reach_r,
                        r_tol=float(self.get_parameter("visual_r_tol").value),
                        err_ok=min(
                            0.15,
                            float(self.get_parameter("lock_max_err_u").value) + 0.05,
                        ),
                        yaw_gain=float(self.get_parameter("align_yaw_gain").value),
                        max_wz=min(
                            0.16, float(self.get_parameter("align_max_wz").value)
                        ),
                        max_vx=0.12,
                        timeout=16.0,
                        stable_hold_sec=0.25,
                        lock_v_frac=float(self.get_parameter("lock_v_frac").value),
                        grasp_box_frac=float(
                            self.get_parameter("grasp_box_frac").value
                        ),
                        target_area_frac=float(
                            self.get_parameter("target_area_frac").value
                        ),
                        area_frac_tol=float(
                            self.get_parameter("area_frac_tol").value
                        ),
                        target_fill=float(self.get_parameter("target_fill").value),
                        fill_tol=float(self.get_parameter("fill_tol").value),
                        cube_width_m=float(
                            self.get_parameter("cube_width_m").value
                        ),
                        on_frame=_on_oc,
                    )
                    if aligned:
                        self._pub_status(
                            "OPENCV_OK → 直接下探（不再 FINAL 拧头/硬拉）"
                        )
                    else:
                        # Only if failed: gentle visual creep, NO map face spin
                        self._pub_status("OPENCV_SOFT → VIS 轻拉（不拧头）")
                        aligned = self.motion.drive_until_visually_near(
                            color,
                            cid,
                            lock_v_frac=float(
                                self.get_parameter("lock_v_frac").value
                            ),
                            grasp_box_frac=float(
                                self.get_parameter("grasp_box_frac").value
                            ),
                            min_fill=1.20,
                            min_cy_frac=0.72,
                            max_vx=0.09,
                            timeout=8.0,
                            max_travel_m=0.28,
                        )
                        if not aligned:
                            self._pub_status(
                                "PICK_ABORT 画面未够近 — 不下探"
                            )
                            return
                finally:
                    self.motion.stop_chassis()
                    self.motion.exit_local_mode()

                # Quick pin — short settle only
                ax, ay, ayaw = self._parking_xy(color, cid, standoff_m=so)
                cur = self.motion.get_base_xy_yaw_map()
                if cur is not None:
                    pin_x, pin_y, pin_yaw = (
                        float(cur[0]),
                        float(cur[1]),
                        float(cur[2]),
                    )
                    self.motion.reanchor_amcl(
                        pin_x, pin_y, pin_yaw, settle_sec=0.15
                    )
                else:
                    self.motion.reanchor_amcl(ax, ay, ayaw, settle_sec=0.15)
                self.motion.clear_costmaps()

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
                front_m = float(self.get_parameter("place_front_m").value)
                self._pub_status(
                    f"NAV_PLACE → zone {zone} 中心 "
                    f"(车前≈{front_m:.2f}m 对准区心再放下)"
                )
                self.motion.prepare_next_nav()
                if self._is_first_red(color, cid) and self._door_via_for_zone(zone):
                    if not self._nav_continuous_with_door(
                        zx,
                        zy,
                        0.0,
                        place_mode=True,
                        xy_tol=0.40,
                        use_door_via=True,
                    ):
                        self._pub_status(f"NAV_PLACE_FAIL {zone}")
                        return
                elif not self.motion.navigate_place_zone(
                    zx, zy, xy_tol=0.35, approach_standoff=front_m
                ):
                    self._pub_status(f"NAV_PLACE_FAIL {zone}")
                    return
                # Dock: base at zone − front_m·heading, so 车前≈front_m 落在区心
                self._dock_place_front(zx, zy, front_m=front_m)

            if not self._place_with_moveit(color, cid):
                return
            self._pub_status(f"PLACED {color}_{cid} at {zone}")
            # Leave zone so next path does not drive over the just-placed cube
            self._leave_zone_after_place(zx, zy)
            cur = self.motion.get_base_xy_yaw_map()
            pyaw = float(cur[2]) if cur is not None else zyaw
            lx = float(cur[0]) if cur is not None else zx
            ly = float(cur[1]) if cur is not None else zy
            # Keep costmap marks of the placed cube
            self.motion.prepare_next_nav(lx, ly, pyaw, clear_maps=False)

        self._pub_status("DONE")

    def _dock_place_front(
        self, zx: float, zy: float, *, front_m: float = 0.20
    ) -> None:
        """Dock with car heading along zone long axis (front ⊥ long edge).

        Zone pad is 1.0m (X) × 0.5m (Y). Approach from ±X so yaw is 0 or π:
        car body along the long side, bumper perpendicular to the long edge;
        车前 front_m = zone center.
        """
        settle = float(self.get_parameter("local_settle_sec").value)
        cur = self.motion.get_base_xy_yaw_map()
        if cur is None:
            return
        rx, ry = float(cur[0]), float(cur[1])
        # Prefer ±X docks (perpendicular to long edge) — not ±Y parallel
        cands = [
            (float(zx) - front_m, float(zy), 0.0),       # from -X, face +X
            (float(zx) + front_m, float(zy), math.pi),   # from +X, face -X
        ]
        px, py, pyaw = min(
            cands, key=lambda p: (p[0] - rx) ** 2 + (p[1] - ry) ** 2
        )
        self._pub_status(
            f"PLACE_DOCK 垂直长边 → base=({px:.2f},{py:.2f}) "
            f"yaw={pyaw:+.2f} 车前{front_m:.2f}m=区心({zx:.2f},{zy:.2f})"
        )
        # Nav coarse to dock (keep intended yaw)
        self.motion.exit_local_mode()
        self.motion.navigate(
            px,
            py,
            pyaw,
            timeout=60.0,
            early_xy_tol=0.30,
            soft_start=True,
        )
        self.motion.enter_local_mode(settle_sec=min(0.10, settle))
        try:
            # Spin to pad-perpendicular yaw (along long axis)
            for _ in range(2):
                c = self.motion.get_base_xy_yaw_map()
                if c is None:
                    break
                rem = self.motion._wrap_pi(pyaw - float(c[2]))
                if abs(rem) <= 0.10:
                    break
                self.motion.spin_yaw_openloop(
                    float(rem), max_wz=0.22, timeout=5.0
                )
            # Creep so front point lands on zone center
            t0 = time.time()
            while time.time() - t0 < 10.0:
                c = self.motion.get_base_xy_yaw_map()
                if c is None:
                    break
                # Desired: base at (px,py) with yaw≈pyaw
                dx = px - float(c[0])
                dy = py - float(c[1])
                rng = math.hypot(dx, dy)
                fx = float(c[0]) + front_m * math.cos(float(c[2]))
                fy = float(c[1]) + front_m * math.sin(float(c[2]))
                d_drop = math.hypot(fx - zx, fy - zy)
                if rng <= 0.06 and d_drop <= 0.10:
                    break
                if d_drop <= 0.08 and abs(
                    self.motion._wrap_pi(pyaw - float(c[2]))
                ) <= 0.15:
                    break
                rem = self.motion._wrap_pi(pyaw - float(c[2]))
                wz = max(-0.20, min(0.20, 1.0 * rem))
                des = math.atan2(dy, dx)
                rem_drive = self.motion._wrap_pi(des - float(c[2]))
                if abs(rem) > 0.35:
                    vx = 0.0
                elif abs(rem_drive) < 0.60:
                    vx = 0.10 if rng > 0.06 else 0.0
                else:
                    vx = 0.0
                # Overshoot: zone closer than front_m along forward → reverse
                fwd_x, fwd_y = math.cos(float(c[2])), math.sin(float(c[2]))
                to_zx, to_zy = zx - float(c[0]), zy - float(c[1])
                ahead = to_zx * fwd_x + to_zy * fwd_y
                if ahead < front_m - 0.06:
                    vx = -0.08
                    wz = max(-0.15, min(0.15, rem))
                self.motion.publish_cmd_vel(vx, wz)
                time.sleep(0.08)
            # Final yaw lock (perpendicular to long edge)
            c2 = self.motion.get_base_xy_yaw_map()
            if c2 is not None:
                rem2 = self.motion._wrap_pi(pyaw - float(c2[2]))
                if abs(rem2) > 0.08:
                    self.motion.spin_yaw_openloop(
                        float(rem2), max_wz=0.18, timeout=4.0
                    )
            self.motion.stop_chassis()
            c3 = self.motion.get_base_xy_yaw_map()
            if c3 is not None:
                fx = float(c3[0]) + front_m * math.cos(float(c3[2]))
                fy = float(c3[1]) + front_m * math.sin(float(c3[2]))
                yaw_err = self.motion._wrap_pi(pyaw - float(c3[2]))
                self._pub_status(
                    f"PLACE_DOCK_OK 垂直长边 yaw_err={yaw_err:+.2f} "
                    f"车前=({fx:.2f},{fy:.2f}) 区心=({zx:.2f},{zy:.2f}) "
                    f"err={math.hypot(fx - zx, fy - zy):.2f}m"
                )
        finally:
            self.motion.stop_chassis()
            self.motion.exit_local_mode()

    def _leave_zone_after_place(self, zx: float, zy: float) -> None:
        """Get clear of zone pile BEFORE any yaw turn (in-place spin crushes cubes).

        1) Straight reverse open-loop (wz=0) — never pivot on the pile
        2) Drive radially outward from zone center to ≥1.35 m
        3) Only then hand wheels to Nav2 for the next pick
        """
        settle = float(self.get_parameter("local_settle_sec").value)
        leave_r = 1.35
        cur = self.motion.get_base_xy_yaw_map()
        if cur is None:
            yaw = 0.0
            rx, ry = float(zx), float(zy)
        else:
            rx, ry, yaw = float(cur[0]), float(cur[1]), float(cur[2])

        # Unit vector away from zone center (radial exit)
        dx, dy = rx - float(zx), ry - float(zy)
        dist0 = math.hypot(dx, dy)
        if dist0 < 0.12:
            ux, uy = -math.cos(yaw), -math.sin(yaw)
        else:
            ux, uy = dx / dist0, dy / dist0
        lx = float(zx) + ux * leave_r
        ly = float(zy) + uy * leave_r

        self._pub_status(
            f"LEAVE_ZONE 先直行倒出，再离开区心 → ({lx:.2f},{ly:.2f}) "
            f"r≥{leave_r:.2f}m（禁止区上原地转向压块）"
        )
        self.motion.enter_local_mode(settle_sec=min(0.20, settle))
        try:
            # 1) Always reverse straight first — cubes are under/near bumper
            self.motion.drive_openloop_distance(-0.85, vx=0.14, timeout=10.0)
            time.sleep(0.15)
            cur2 = self.motion.get_base_xy_yaw_map()
            if cur2 is not None:
                rx, ry, yaw = float(cur2[0]), float(cur2[1]), float(cur2[2])
            # Recompute radial target from new pose
            dx, dy = rx - float(zx), ry - float(zy)
            dist1 = math.hypot(dx, dy)
            if dist1 < 0.12:
                ux, uy = -math.cos(yaw), -math.sin(yaw)
            else:
                ux, uy = dx / dist1, dy / dist1
            lx = float(zx) + ux * leave_r
            ly = float(zy) + uy * leave_r

            # 2) Face leave point gently, then drive forward (still no big in-place spin on pile)
            des = math.atan2(ly - ry, lx - rx)
            rem = self.motion._wrap_pi(des - yaw)
            # Only nudge yaw if already somewhat clear (>0.45m from center)
            if dist1 >= 0.45 and abs(rem) > 0.20:
                # Cap spin: prefer small correction, not full pivot on cubes
                self.motion.spin_yaw_openloop(
                    float(max(-0.55, min(0.55, rem))),
                    max_wz=0.18,
                    timeout=4.0,
                )
            need = math.hypot(lx - rx, ly - ry)
            if need > 0.20:
                # Drive mostly straight toward leave; if still skewed, reverse more
                if abs(rem) > 1.0 and dist1 < 0.55:
                    self.motion.drive_openloop_distance(-0.50, vx=0.12, timeout=6.0)
                else:
                    self.motion.drive_openloop_distance(
                        min(need, 1.10), vx=0.14, timeout=12.0
                    )
        finally:
            self.motion.stop_chassis()
            self.motion.exit_local_mode()

        # 3) Soft Nav polish to leave XY (keep yaw — avoid RotateToGoal on pile)
        cur3 = self.motion.get_base_xy_yaw_map()
        keep_yaw = float(cur3[2]) if cur3 is not None else yaw
        dleft = (
            math.hypot(float(cur3[0]) - float(zx), float(cur3[1]) - float(zy))
            if cur3 is not None
            else 0.0
        )
        if dleft < leave_r - 0.15:
            self._pub_status(
                f"LEAVE_ZONE_NAV → ({lx:.2f},{ly:.2f}) 补齐离区 "
                f"(now r={dleft:.2f}m)"
            )
            ok = self.motion.navigate(
                lx,
                ly,
                keep_yaw,
                timeout=45.0,
                early_xy_tol=0.40,
                soft_start=True,
            )
            if not ok:
                self.get_logger().warn("LEAVE_ZONE nav soft-fail — continue anyway")
        else:
            self._pub_status(f"LEAVE_ZONE_OK r={dleft:.2f}m 已离开堆放区")
        self.motion.stop_chassis()
        time.sleep(0.2)


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
