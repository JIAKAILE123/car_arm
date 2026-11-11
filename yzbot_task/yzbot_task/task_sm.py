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
from yzbot_task.motion import (
    MotionHelpers,
    HOME,
    PLACE,
    PICK,
    PRESET_REACH_R,
    PLACE_FRONT_M,
)
from yzbot_task.vision_align import (
    annotate_bgr,
    blob_area_frac,
    detect_blob,
    detect_green_pad,
    green_pad_centered,
    is_visual_lock,
)
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
        # True: map→odom from Gazebo GT (gt_localization); no AMCL laser loc / reanchor
        self.declare_parameter("use_gt_localization", True)
        self.declare_parameter("max_cubes", 5)
        self.declare_parameter("use_moveit", True)
        self.declare_parameter("use_vision_align", True)
        # OFF by default: continuous yaw IBVS made the car spin away
        self.declare_parameter("use_chassis_align", True)
        self.declare_parameter("align_timeout", 7.9)
        # 降增益/角速度：少拧过头再拧回（差速+脚轮惯性大）
        self.declare_parameter("align_yaw_gain", 0.50)
        self.declare_parameter("align_max_wz", 0.14)
        self.declare_parameter("local_settle_sec", 0.02)
        # Nav2 often stops ~0.5–0.8m short (inflation); soft gate then continuous local
        self.declare_parameter("pose_gate_xy", 0.55)
        # Pick: tighter stop so different paths don't leave huge park variance
        self.declare_parameter("pick_pose_gate_xy", 0.22)
        # Place / pick：到点 yaw 放宽，少在家具/物块旁空拧；朝向交给 FACE/视觉
        self.declare_parameter("pose_gate_yaw", 0.90)
        self.declare_parameter("pose_gate_yaw_pick", 0.45)
        self.declare_parameter("nav_retries", 1)
        # Door vias (code-only): first red cube pick/place A|B must pass these points
        self.declare_parameter("door_via_enable", True)
        # Fixed waypoints (map): (-4.2,0) and (-3.2,0); small XY error OK
        self.declare_parameter("door_via_xy_tol", 0.35)
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("shelf_z", 0.75)
        self.declare_parameter("grasp_above", 0.05)
        self.declare_parameter("grasp_down", 0.0)
        # EE 到块超过此距禁止 ATTACH（LinkAttacher 会远处拽块）
        self.declare_parameter("attach_max_dist", 0.32)
        # Place: park by place_front_m (calibrated physical EE). Optional pull-back.
        self.declare_parameter("place_front_m", float(PLACE_FRONT_M))
        # 沿长边(X≈1.0m)往区内收；短边(Y≈0.5m)只允许很小侧偏
        self.declare_parameter("place_inward_m", 0.18)
        self.declare_parameter("place_park_extra_m", 0.0)
        # 去放置区的路上预计算绿垫，不发速度；到点不再停 4 秒对垫。
        self.declare_parameter("place_vision_enable", True)
        self.declare_parameter("place_vision_target_cy", 0.58)
        # known_coords = 按官方序号固定坐标：导航到物块前方 → 预设臂 → 吸附
        self.declare_parameter("grasp_mode", "known_coords")
        self.declare_parameter("prefer_joint_pick", True)
        self.declare_parameter("allow_known_map", True)
        self.declare_parameter("use_cube_standoff", True)
        # Nav park: far enough for FOV; creep must not overshoot past this
        self.declare_parameter("grasp_standoff_m", 0.55)
        self.declare_parameter("grasp_standoff_retry_m", 0.62)
        # Local creep target (= vision sweet spot); never drive closer than this
        self.declare_parameter("grasp_creep_m", 0.50)
        self.declare_parameter("grasp_min_range_m", 0.38)
        # Last snap before arm-down — leave a little room so FPV keeps the cube
        self.declare_parameter("pick_final_r", 0.34)
        # 远距小块：略降面积门槛（配合更高检测分辨率）
        self.declare_parameter("lock_min_area", 40.0)
        self.declare_parameter("lock_max_err_u", 0.10)
        self.declare_parameter("lock_timeout", 7.9)
        # MUST match PRESET_REACH_R (~0.40): chassis stops where fixed PICK can touch
        self.declare_parameter("visual_target_r", float(PRESET_REACH_R))
        self.declare_parameter("visual_r_tol", 0.045)
        self.declare_parameter("range_timeout", 6.8)
        self.declare_parameter("timeout_see", 2.3)
        self.declare_parameter("sweep_yaw_rad", 0.25)
        self.declare_parameter("stable_hold_sec", 0.12)
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
        # Same letter zone: side-by-side slots so later cubes don't crush earlier ones
        self._zone_place_count = {"A": 0, "B": 0, "C": 0}
        # Attached cube not yet dropped — must place before next pick
        self._holding: Optional[Tuple[str, int, str]] = None
        self._place_vis_stop: Optional[threading.Event] = None
        self._place_vis_thread: Optional[threading.Thread] = None
        self._place_vis_lock = threading.Lock()
        self._place_vis_sample: Optional[dict] = None
        self._place_vis_zone: Optional[Tuple[float, float, float, float, float, float]] = None
        self._place_vis_ahead: Optional[dict] = None
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
        # Virtual keepout over A/B/C so Nav2 detours (reconfigured from poses)
        try:
            zmap = {}
            for letter in ("A", "B", "C"):
                zx, zy, _ = self._pose(f"zone_{letter.lower()}")
                zmap[letter] = (zx, zy)
            # 只挡字母中心。垫子外圈留给膨胀渐变，车可以贴边、偶尔压上去。
            # 墙膨胀不改。放下的物块另有 placed keepout。
            self.motion.configure_zone_keepouts(
                zmap, half_x=0.20, half_y=0.10, step=0.06
            )
            # 红2西侧竖墙（x≈-4.26）。地图上这面墙是空的，全局规划会直线穿过去。
            # 只标墙身下段。北端口（y>0.7）不标，避免这圈虚拟膨胀把口封死。
            wall_pts = []
            y = -1.35
            while y <= 0.40 + 1e-9:
                for x in (-4.36, -4.26, -4.16):
                    wall_pts.append((x, y, 0.25))
                y += 0.08
            self.motion.add_named_keepout("WALL75", wall_pts)
            # 这面竖墙东侧（x≈-3.2）和活动杆之间只剩约 0.9m。
            # 从南边贴墙上来会卡在 y≈-1.2，杆再过来就把车挤进墙里弹飞。
            # 只封这条缝，不改墙，东边绕开杆的路还在。
            pinch_pts = []
            y = -2.40
            while y <= -1.00 + 1e-9:
                x = -3.05
                while x <= -2.55 + 1e-9:
                    pinch_pts.append((x, y, 0.25))
                    x += 0.10
                y += 0.10
            self.motion.add_named_keepout("WALL_PINCH", pinch_pts)
            # 斜墙，从 (1.85,-1.4) 往东南。车在西侧贴着走。
            # 跟红2那面墙一样，只沿墙身标一窄条，靠膨胀把车稍微推离墙。
            diag_pts = []
            y = -1.40
            nx, ny = -0.775, -0.630
            while y >= -2.20 - 1e-9:
                x = 1.85 - 0.8125 * (y + 1.4)
                for off in (0.0, 0.10, 0.18):
                    diag_pts.append((x + off * nx, y + off * ny, 0.25))
                y -= 0.08
            self.motion.add_named_keepout("WALL_DIAG", diag_pts)
            self.motion.clear_placed_keepouts()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"zone keepout init skipped: {exc}")

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

    def _face_travel(self, yaw: float) -> None:
        """朝向差得多时先原地调头，后面用车头往前走，不倒着开。"""
        cur = self.motion.get_base_xy_yaw_map()
        if cur is None:
            return
        rem = self.motion._wrap_pi(float(yaw) - float(cur[2]))
        if abs(rem) < 0.8:
            return
        self._pub_status(f"FACE_TRAVEL 先调头 {abs(rem)*180/math.pi:.0f}° 再前进")
        self.motion.snap_yaw_map(float(yaw), tol=0.28, max_wz=0.40, timeout=4.5)

    def _pose(self, key: str):
        p = self.poses[key]
        return float(p["x"]), float(p["y"]), float(p["yaw"])

    def _is_first_red(self, color: str, cid: int) -> bool:
        return str(color).lower() == "red" and int(cid) == 1

    def _door_via_for_zone(self, zone: str) -> bool:
        """Door vias only for first-red when place zone is A or B (not C)."""
        return str(zone).strip().upper() in ("A", "B")

    def _place_slot_xy(self, zx: float, zy: float, zone: str) -> Tuple[float, float, int]:
        """Same-zone multi-place: keep first on center; later tiny Y only."""
        zkey = str(zone).strip().upper()
        n = int(self._zone_place_count.get(zkey, 0))
        # 同区多块：只在短边(Y)上小错开，勿贴短边外
        slots_y = (0.0, 0.025, -0.025, 0.04, -0.04)
        dy = float(slots_y[n % len(slots_y)])
        return float(zx), float(zy) + dy, n

    def _place_drop_inward(
        self, tx: float, ty: float, zx: float, zy: float, *, zone: str = ""
    ) -> Tuple[float, float, float]:
        """Bias drop along approach into pad.

        绿垫：长边沿 X≈1.0m，短边沿 Y≈0.5m。
        从 ±X 接近时面对的是短边——目标沿长边往区心收，短边容差更紧。
        """
        inward = float(self.get_parameter("place_inward_m").value)
        if str(zone).strip().upper() == "B":
            inward = max(inward, 0.0) + 0.05
        # 长边半长 0.5 → 夹 ±0.32；短边半宽 0.25 → 夹 ±0.12
        max_dx, max_dy = 0.32, 0.12
        cur = self.motion.get_base_xy_yaw_map()
        rx = float(cur[0]) if cur is not None else float(tx) - 1.0
        from_west = (rx - (tx - 0.1)) ** 2 <= (rx - (tx + 0.1)) ** 2
        yaw = 0.0 if from_west else math.pi
        if abs(inward) >= 1e-4:
            if from_west:
                tx = float(tx) + inward
            else:
                tx = float(tx) - inward
        # 略向区心，仍保持矩形区（长边可偏，短边更贴中）
        tx = 0.75 * float(tx) + 0.25 * float(zx)
        ty = 0.70 * float(ty) + 0.30 * float(zy)
        tx = float(zx) + max(-max_dx, min(max_dx, float(tx) - float(zx)))
        ty = float(zy) + max(-max_dy, min(max_dy, float(ty) - float(zy)))
        return float(tx), float(ty), yaw

    def _skirt_wall_above_a(self, goal_y: float) -> None:
        """不再把墙角当成必须再经过的点，避免走到上角后又被拉回去。"""
        return

    def _door_via_points(self) -> List[Tuple[float, float]]:
        """Fixed mid-path points for red,1 → A/B (no visualization)."""
        return [(-4.2, 0.0), (-3.2, 0.0)]

    def _nav_through_c_door_then(
        self,
        fx: float,
        fy: float,
        fyaw: float,
        *,
        xy_tol: float,
        yaw_tol: float = 0.55,
        place_mode: bool = False,
        approach_standoff: Optional[float] = None,
        from_east: bool = False,
    ) -> bool:
        """Go straight to the C drop. No fixed via points."""
        self._pub_status(f"NAV_C → ({fx:.2f},{fy:.2f})")
        self.motion.exit_local_mode()
        if place_mode and approach_standoff is not None:
            return self.motion.navigate_place_zone(
                fx,
                fy,
                xy_tol=xy_tol,
                approach_standoff=float(approach_standoff),
                from_east=from_east,
            )
        return self.motion.navigate_and_verify(
            fx,
            fy,
            fyaw,
            retries=1,
            xy_tol=xy_tol,
            yaw_tol=yaw_tol,
            stay_if_xy=max(0.30, xy_tol * 1.4),
            early_yaw_tol=0.55,
            snap_yaw=False,
        )

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
        want_via = False

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
            self._skirt_wall_above_a(fy)
            front_m = float(self.get_parameter("place_front_m").value)
            park_m = front_m + float(
                self.get_parameter("place_park_extra_m").value
            )
            ok = self.motion.navigate_place_zone(
                fx,
                fy,
                xy_tol=float(xy_tol or 0.28),
                approach_standoff=park_m,
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

    def _blue_open_side_vias(
        self, color: str, cid: int, cx: float, cy: float
    ) -> List[Tuple[float, float]]:
        """No fixed vias — Nav2 paths around walls; layout keeps cubes off walls."""
        del color, cid, cx, cy
        return []

    def _cube_safe_approach_vias(
        self, cx: float, cy: float, park_x: float, park_y: float
    ) -> List[Tuple[float, float]]:
        """Vias so pick never drives across the cube to the park side.

        Handles opposite ±X or ±Y: swing clear, then stage on the park side.
        """
        cur = self.motion.get_base_xy_yaw_map()
        rx = float(cur[0]) if cur is not None else float(park_x)
        ry = float(cur[1]) if cur is not None else float(park_y)
        vias: List[Tuple[float, float]] = []

        opp_x = (float(park_x) - float(cx)) * (rx - float(cx)) < 0.0
        opp_y = (float(park_y) - float(cy)) * (ry - float(cy)) < 0.0
        park_east = float(park_x) >= float(cx)
        park_north = float(park_y) >= float(cy)

        if opp_x:
            stage_x = float(cx) + (1.00 if park_east else -1.00)
            north = (stage_x, float(cy) + 1.10)
            south = (stage_x, float(cy) - 1.10)
            clear = north if abs(ry - north[1]) <= abs(ry - south[1]) else south
            vias.append(clear)
            vias.append((stage_x, float(cy)))
            return vias

        if opp_y:
            # e.g. forced north park while robot is south — swing E/W first
            stage_y = float(cy) + (1.00 if park_north else -1.00)
            east = (float(cx) + 1.10, stage_y)
            west = (float(cx) - 1.10, stage_y)
            clear = east if abs(rx - east[0]) <= abs(rx - west[0]) else west
            vias.append(clear)
            vias.append((float(cx), stage_y))
            return vias

        # Same side but sitting on/near the cube — back out to stage first
        if math.hypot(rx - float(cx), ry - float(cy)) < 0.70:
            if abs(float(park_x) - float(cx)) >= abs(float(park_y) - float(cy)):
                stage_x = float(cx) + (1.00 if park_east else -1.00)
                vias.append((stage_x, float(cy)))
            else:
                stage_y = float(cy) + (1.00 if park_north else -1.00)
                vias.append((float(cx), stage_y))
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
        """Soft via points (early XY), then final navigate_and_verify.

        One pose at a time — not NavigateThroughPoses — so a passed corner is
        never kept as a live goal that pulls the robot back on replan.
        """
        self.motion.exit_local_mode()
        for vx, vy in vias:
            if not self._via_point_still_needed(vx, vy, fx, fy, via_tol):
                self._pub_status(
                    f"VIA skip ({vx:.2f},{vy:.2f}) — already at/past"
                )
                continue
            via_yaw = math.atan2(float(fy) - vy, float(fx) - vx)
            self._pub_status(
                f"VIA approach → ({vx:.2f},{vy:.2f}) (绕墙/绕块，再到取货点)"
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
        xy_tol = float(self.get_parameter("pick_pose_gate_xy").value)
        # 只有红3：位置到了就停，最后几度别在导航里拧，交给视觉。其它块保持原容差。
        red3_soft = str(color).lower() == "red" and int(cid) == 3
        red2 = str(color).lower() == "red" and int(cid) == 2
        yaw_tol = 0.45
        early_yaw = 0.45 if (red3_soft or red2) else 0.55
        # All blue + all red: never drive across/over the cube (protects AMCL)
        use_safe = str(color).lower() in ("blue", "red")
        if use_safe:
            if abs(ax - mx) >= abs(ay - my):
                side = "东" if ax >= mx else "西"
            else:
                side = "北" if ay >= my else "南"
            self._pub_status(
                f"SAFE_PICK {color}_{cid} park{side} ({ax:.2f},{ay:.2f}) "
                f"yaw→块={ayaw:.2f} 直达取货点（无途经点）"
            )
            self.motion.exit_local_mode()
            retries = int(self.get_parameter("nav_retries").value)
            # 红3/红4：第一次停就要朝向对，勿再二次走路
            sticky_park = (
                (str(color).lower() == "red" and int(cid) in (3, 4))
                or (str(color).lower() == "blue" and int(cid) in (1, 3))
            )
            if sticky_park:
                retries = 0
            # 蓝3：已经贴着取货点时别再发短导航（会倒一下再空等很久才识别）
            blue3 = str(color).lower() == "blue" and int(cid) == 3
            if blue3:
                cur_b3 = self.motion.get_base_xy_yaw_map()
                if cur_b3 is not None:
                    d_b3 = math.hypot(
                        float(cur_b3[0]) - ax, float(cur_b3[1]) - ay
                    )
                    if d_b3 <= max(0.50, xy_tol * 2.2):
                        self._pub_status(
                            f"BLUE3_NEAR dxy={d_b3:.2f}m → 跳过短导航，直接识别"
                        )
                        self.motion.stop_chassis()
                        return True
            nav_ok = False
            if (
                not nav_ok
                and str(color).lower() == "blue"
                and int(cid) == 5
            ):
                # 先把车头调到朝西，再从 x 正侧开过去，避免倒车绕北
                self._face_travel(math.pi)
                lead_x = float(ax) + 1.05
                self._pub_status(
                    f"BLUE5_EAST ({lead_x:.2f},{ay:.2f}) → ({ax:.2f},{ay:.2f}) 车头朝前"
                )
                self.motion.navigate(
                    lead_x,
                    ay,
                    math.pi,
                    timeout=70.0,
                    early_xy_tol=0.45,
                    early_yaw_tol=0.90,
                    soft_start=True,
                )
            if not nav_ok:
                nav_ok = self.motion.navigate_and_verify(
                    ax,
                    ay,
                    ayaw,
                    retries=retries,
                    xy_tol=xy_tol,
                    yaw_tol=yaw_tol,
                    settle_sec=0.04,
                    reanchor=True,
                    stay_if_xy=max(0.30, xy_tol * 1.4),
                    early_yaw_tol=early_yaw,
                    snap_yaw=False,
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
                if not sticky_park:
                    self.motion.clear_costmaps(timeout=0.25)
            cur2 = self.motion.get_base_xy_yaw_map()
            if cur2 is not None:
                self.motion.reanchor_amcl(
                    float(cur2[0]), float(cur2[1]), float(ayaw), settle_sec=0.03
                )
        else:
            self.motion.exit_local_mode()
            self._pub_status(
                f"NAV_PICK_DIRECT → ({ax:.2f},{ay:.2f}) yaw={ayaw:.2f} "
                f"(朝向物块，边走边转)"
            )
            retries = int(self.get_parameter("nav_retries").value)
            sticky_park = (
                (str(color).lower() == "red" and int(cid) in (3, 4))
                or (str(color).lower() == "blue" and int(cid) == 1)
            )
            if sticky_park:
                retries = 0
            nav_ok = self.motion.navigate_and_verify(
                ax,
                ay,
                ayaw,
                retries=retries,
                xy_tol=xy_tol,
                yaw_tol=yaw_tol,
                settle_sec=0.04,
                reanchor=True,
                stay_if_xy=max(0.30, xy_tol * 1.4),
                early_yaw_tol=early_yaw,
                snap_yaw=False,
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
                if not sticky_park:
                    self.motion.clear_costmaps(timeout=0.4)
        self.motion.stop_chassis()
        creep_r = float(self.get_parameter("grasp_creep_m").value)
        # 红3(近B)/红4：略放宽近距门，避免刚停又二次 Navigate
        near_max = creep_r + (0.28 if sticky_park else 0.12)
        if not self._ensure_near_cube(
            color, cid, standoff_m=so, max_range_m=near_max
        ):
            self._pub_status(f"NAV_PICK_FAIL / TOO_FAR {color}_{cid}")
            return False
        self._pub_status(
            f"NAV_PICK_OK → 本地: 补朝向(若需要)+OpenCV贴框 "
            f"(park≈{float(self.get_parameter('grasp_standoff_m').value):.2f}m)"
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
        """If TF range to known cube is large, re-park — do not idle in OpenCV far away.

        红3(近B区)/红4：禁止对这些块重发 Navigate。
        """
        from yzbot_task.vision_pose import cube_bearing_in_base

        sticky = (
            (str(color).lower() == "red" and int(cid) in (3, 4))
            or (str(color).lower() == "blue" and int(cid) in (1, 3))
        )
        n_try = 1 if sticky else max(1, int(retries) + 1)

        for attempt in range(n_try):
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
            if sticky and rng <= 1.15:
                self._pub_status(
                    f"NEAR_CUBE red#{cid} soft r={rng:.2f}m → 停住不二次导航"
                )
                return True
            settle = float(self.get_parameter("local_settle_sec").value)
            creep_r = float(self.get_parameter("grasp_creep_m").value)
            self._pub_status(
                f"TOO_FAR r={rng:.2f}m → CREEP 按坐标开到 ≤{creep_r:.2f}m "
                f"（try={attempt + 1}）"
            )
            self.motion.enter_local_mode(settle_sec=min(0.03, settle))
            try:
                if self.motion.creep_to_known_cube(
                    color,
                    cid,
                    target_range=creep_r,
                    max_vx=0.10,
                    max_wz=0.14,
                    timeout=8.0,
                ):
                    return True
            finally:
                self.motion.stop_chassis()
                self.motion.exit_local_mode()
            if sticky:
                # 红3/4：creep 不够也别再 Navigate，交给视觉
                info2 = cube_bearing_in_base(self.motion.tf_buffer, color, cid)
                if info2 is not None and float(info2[1]) <= 1.20:
                    self._pub_status(
                        f"NEAR_CUBE red#{cid} after creep → 不重停，交视觉"
                    )
                    return True
                return True
            ax, ay, ayaw = self._parking_xy(color, cid, standoff_m=standoff_m)
            self._pub_status(
                f"TOO_FAR r={rng:.2f}m → 重停 PARK ({ax:.2f},{ay:.2f})"
            )
            self.motion.exit_local_mode()
            xy_tol = float(self.get_parameter("pick_pose_gate_xy").value)
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

    def _hold_pick_range(
        self,
        color: str,
        cid: int,
        *,
        target_m: Optional[float] = None,
    ) -> None:
        """Keep pick range with continuous yaw correction (no blind straight drive)."""
        target = float(
            target_m
            if target_m is not None
            else self.get_parameter("grasp_creep_m").value
        )
        min_r = float(self.get_parameter("grasp_min_range_m").value)
        settle = float(self.get_parameter("local_settle_sec").value)

        from yzbot_task.vision_pose import cube_bearing_in_base, known_cube_xy

        def _rng():
            info = cube_bearing_in_base(self.motion.tf_buffer, color, cid)
            if info is not None:
                return float(info[1])
            cur = self.motion.get_base_xy_yaw_map()
            if cur is None:
                return None
            mx, my, _ = known_cube_xy(color, cid)
            return math.hypot(float(cur[0]) - mx, float(cur[1]) - my)

        rng0 = _rng()
        if rng0 is None:
            return
        if min_r <= rng0 <= target + 0.08:
            self._pub_status(f"RANGE_OK r={rng0:.2f}m")
            return

        owned = self.motion.drive_owner == "local"
        if not owned:
            self.motion.enter_local_mode(settle_sec=min(0.04, settle))
        try:
            self._pub_status(
                f"SERVO_APPROACH r={rng0:.2f}m → [{min_r:.2f},{target:.2f}] "
                f"（边走边纠偏）"
            )
            self.motion.approach_cube_servo(
                color,
                cid,
                target_range=target,
                min_range=min_r,
                max_vx=0.12,
                max_wz=0.24,
                timeout=3.8,
            )
            r1 = _rng()
            if r1 is not None:
                self._pub_status(f"RANGE_HOLD r={r1:.2f}m")
        finally:
            self.motion.stop_chassis()
            if not owned:
                self.motion.exit_local_mode()

    def _face_cube_yaw(
        self, color: str, cid: int, *, manage_mode: bool = True, reanchor: bool = False
    ) -> bool:
        """Park 后按已知 map 坐标规矩转向：yaw → atan2(cube−robot)，直到车头对准。

        manage_mode=False: 调用方已在 MODE_LOCAL，不进出模式、不长时间 settle（流畅串联）。
        """
        settle = float(self.get_parameter("local_settle_sec").value)
        # 夹取正对：慢转对准 map 目标块（保证相机前就是该序号，不是去扫别的）
        max_wz = min(0.12, float(self.get_parameter("align_max_wz").value))
        yaw_gain = min(0.55, float(self.get_parameter("align_yaw_gain").value))
        mx, my, _mz = known_cube_xy(color, cid)
        tol = 0.16

        if manage_mode:
            self.motion.enter_local_mode(settle_sec=min(0.03, settle))
        try:
            ok = False
            cur = self.motion.get_base_xy_yaw_map()
            if cur is None:
                self.get_logger().warn("FACE_CUBE skip (no map pose)")
                return False
            rx, ry, ryaw = float(cur[0]), float(cur[1]), float(cur[2])
            yaw_des = math.atan2(my - ry, mx - rx)
            rem = self.motion._wrap_pi(yaw_des - ryaw)
            self._pub_status(
                f"FACE_CUBE {color}_{cid} "
                f"cube=({mx:.2f},{my:.2f}) robot=({rx:.2f},{ry:.2f}) "
                f"yaw={ryaw:+.2f}→{yaw_des:+.2f} rem={rem:+.3f}rad"
            )
            if abs(rem) <= tol:
                ok = True
            else:

                def _rem_live():
                    c = self.motion.get_base_xy_yaw_map()
                    if c is None:
                        return None
                    yd = math.atan2(my - float(c[1]), mx - float(c[0]))
                    return self.motion._wrap_pi(yd - float(c[2]))

                budget = min(4.0, abs(rem) / max(max_wz, 0.10) + 1.0)
                ok = self.motion.aim_yaw_to_bearing(
                    float(rem),
                    yaw_gain=yaw_gain,
                    max_wz=max_wz,
                    tol=tol,
                    timeout=budget,
                    bearing_fn=_rem_live,
                )
                if not ok:
                    rem2 = _rem_live()
                    if rem2 is not None and abs(float(rem2)) <= tol * 1.35:
                        ok = True
                    # 不再 spin_yaw_openloop 二次硬拧（夹取时最容易转过去）

            if reanchor:
                c2 = self.motion.get_base_xy_yaw_map()
                if c2 is not None:
                    yd2 = math.atan2(my - float(c2[1]), mx - float(c2[0]))
                    self.motion.reanchor_amcl(
                        float(c2[0]), float(c2[1]), float(yd2), settle_sec=0.08
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
            time.sleep(0.10)
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
        if not self.moveit.execute_joints(pick, duration_sec=1.35):
            if not self.motion.arm_joints(pick, duration_sec=1.35):
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
        time.sleep(0.02)

        self._pub_status("MTC_LIFT → HOME")
        if self.moveit.execute_joints(HOME, duration_sec=1.5):
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
        self.motion.arm_joints([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], duration_sec=0.9)
        self.motion.stop_chassis()

        self._pub_status("MTC_OPEN")
        if not self.motion.ensure_gripper_open(timeout=3.5):
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
        time.sleep(0.05)
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
            self.motion.arm_joints([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], duration_sec=0.9)
            return False
        return self._finish_attach_lift(color, cid, grasp_pose)

    def _pick_vision_look_then_move(self, color: str, cid: int) -> bool:
        """Alias → center_range_preset (user preferred pipeline)."""
        return self._pick_center_range_preset(color, cid)

    def _pick_known_coords(self, color: str, cid: int) -> bool:
        """Park done → 开爪 → 预设下探到位 → 再吸附 → 抬起。

        严禁在臂未到位时 ATTACH：LinkAttacher 用单位姿对齐，会先把块扯到
        link6（看起来像「夹一下块就飞起来，然后臂才下来」）。
        严禁在底盘仍远时下探/吸附（远处「看见」也会误夹拽块）。
        """
        import subprocess
        import os

        from yzbot_task.vision_pose import (
            cube_bearing_in_base,
            ee_to_pose_distance_m,
            known_cube_pose_arm_raw,
            known_cube_xy,
        )

        self.motion.stop_chassis()

        # —— 下探前硬距离门：底盘到块必须在预设臂可达带内 ——
        reach = float(self.get_parameter("visual_target_r").value)
        max_base_r = reach + 0.10
        info = cube_bearing_in_base(self.motion.tf_buffer, color, cid)
        if info is not None:
            base_r = float(info[1])
        else:
            cur = self.motion.get_base_xy_yaw_map()
            if cur is None:
                self._pub_status("PICK_ABORT NO_POSE_BEFORE_ARM")
                return False
            mx, my, _ = known_cube_xy(color, cid)
            base_r = math.hypot(float(cur[0]) - mx, float(cur[1]) - my)
        if base_r > max_base_r:
            self._pub_status(
                f"PICK_ABORT FAR_BEFORE_ARM r={base_r:.2f}>{max_base_r:.2f}m "
                f"— 离得远不夹"
            )
            return False
        self._pub_status(f"PICK_RANGE_OK r={base_r:.2f}≤{max_base_r:.2f} → 下探")

        # 清掉上次残留吸附，否则一动臂块就会跟着飞
        self._pub_status(f"DETACH_CLEAR {color}_{cid} (before arm)")
        try:
            self.motion.detach(color, cid)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"pre-detach ignored: {exc}")
        time.sleep(0.02)

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
        time.sleep(0.02)

        # 臂到位后：EE 离块仍远则禁止 ATTACH（防拽飞）
        stamp = self.get_clock().now().to_msg()
        km = known_cube_pose_arm_raw(self.motion.tf_buffer, color, cid, stamp)
        max_ee = float(self.get_parameter("attach_max_dist").value)
        if km.ok and km.estimate is not None:
            dist = ee_to_pose_distance_m(
                self.motion.tf_buffer, km.estimate.pose_arm
            )
            if dist is not None and float(dist) > max_ee:
                self._pub_status(
                    f"PICK_ABORT EE_FAR {dist:.3f}>{max_ee:.3f}m — 不吸附"
                )
                try:
                    self.motion.arm_home()
                except Exception:  # noqa: BLE001
                    pass
                return False
            if dist is not None:
                self._pub_status(f"EE_OK dist={dist:.3f}≤{max_ee:.3f}")

        # 臂到位后再吸附
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
                    timeout=1.0,
                    capture_output=True,
                    text=True,
                )
                self.get_logger().info(r.stdout or "")
                ok = r.returncode == 0
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"cube_control.sh failed: {exc}")
        if not ok:
            ok = bool(self.motion.attach(color, cid))
        if ok:
            self._pub_status(f"MTC_ATTACH_OK {color}_{cid}")
        else:
            self._pub_status(f"ATTACH_FAIL {color}_{cid} → 收臂跳过")
            try:
                self.motion.arm_home()
            except Exception:  # noqa: BLE001
                pass
            return False
        time.sleep(0.02)

        # 抬起带走（先离地再收，避免卡在下探姿态）
        self._pub_status("MTC_LIFT")
        if not self.motion.arm_carry_pose(0.0):
            self.get_logger().warn("carry pose failed; continue to place anyway")
        else:
            self._pub_status("MTC_LIFT_OK")
        self.motion.stop_chassis()
        time.sleep(0.02)
        return True

    def _cube_still_on_ee(self, color: str, cid: int) -> bool:
        """True if cube TF is still glued near EE (attach not really cleared)."""
        from yzbot_task.vision_pose import (
            ee_to_pose_distance_m,
            known_cube_pose_arm_raw,
        )

        stamp = self.get_clock().now().to_msg()
        km = known_cube_pose_arm_raw(
            self.motion.tf_buffer, color, cid, stamp
        )
        if not km.ok or km.estimate is None:
            return False
        dist = ee_to_pose_distance_m(
            self.motion.tf_buffer, km.estimate.pose_arm
        )
        if dist is None:
            return False
        # After real detach the block falls away; <12cm means still stuck
        return float(dist) < 0.12

    def _place_with_moveit(self, color: str, cid: int) -> bool:
        """放置：下探 → 松开确认 → 抬臂。未松开成功则失败，禁止去下一块。"""
        import subprocess
        import os

        self._pub_status("PLACE_ARM 下放投放")
        if not self.motion.arm_place_pose():
            self.get_logger().warn("place pose failed; still try detach")
        time.sleep(0.12)

        ok = False
        for attempt in range(1, 4):
            self._pub_status(
                f"DETACH → {color}_{cid} (尝试 {attempt}/3)"
            )
            # LinkAttacher must succeed; cube_control alone must not clear gate
            ok = bool(self.motion.detach(color, cid))
            script = os.path.join(
                os.path.expanduser("~/dev_ws/yzbot/src"), "cube_control.sh"
            )
            if os.path.isfile(script):
                try:
                    r = subprocess.run(
                        ["bash", script, "off", color, str(cid)],
                        check=False,
                        timeout=1.0,
                        capture_output=True,
                        text=True,
                    )
                    if r.stdout:
                        self.get_logger().info(r.stdout)
                    if not ok and r.returncode == 0:
                        # script claimed off — accept only if EE no longer glued
                        time.sleep(0.08)
                        ok = not self._cube_still_on_ee(color, cid)
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warn(f"cube_control off skipped: {exc}")
            if ok:
                time.sleep(0.08)
                if self._cube_still_on_ee(color, cid):
                    self._pub_status(
                        f"DETACH_STUCK {color}_{cid} 块仍贴EE → 再松"
                    )
                    ok = False
                    try:
                        self.motion.detach(color, cid)
                    except Exception:  # noqa: BLE001
                        pass
                    continue
                break
            time.sleep(0.15)

        if not ok:
            self._pub_status(
                f"DETACH_FAIL {color}_{cid} — 仍持物，不进入下一块"
            )
            return False

        self._pub_status(f"DETACH_OK {color}_{cid}")
        self._pub_status("PLACE_LIFT 抬起到初始姿态")
        self.motion.arm_lift_after_place()
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
                    use_door_via=False,
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

        self._zone_place_count = {"A": 0, "B": 0, "C": 0}
        # 不在此清空 _holding：若上次异常中断仍吸着块，首轮会 HOLDING_BLOCK 先强制放
        if self._holding is not None:
            hc, hid, hz = self._holding
            self._pub_status(
                f"RESUME_HOLDING {hc}_{hid} → {hz}（上次未放下，先补放）"
            )
        try:
            self.motion.clear_placed_keepouts()
            self.motion.set_keepout_except(None, clear_maps=False)
        except Exception:  # noqa: BLE001
            pass

        if dry:
            for color, cid, zone in queue:
                self._pub_status(f"DRY: pick {color}_{cid} -> zone {zone}")
            self._pub_status("DONE_DRY")
            return

        mode_needs_arm = True  # known_coords 也要臂下探
        self._pub_status("WAITING_SERVERS (Nav2/臂/吸附，最多约 2s)…")
        if not self.motion.wait_servers(
            timeout=2.0,
            need_arm=mode_needs_arm,
            need_attach=True,
        ):
            self._pub_status(
                "SERVERS_UNAVAILABLE (先开 Gazebo + Nav2 + 臂控制器；吸附要 /ATTACHLINK)"
            )
            return
        # 等 map→odom（GT 可能稍晚于 Nav2 / EKF）
        loc = None
        use_gt = bool(self.get_parameter("use_gt_localization").value)
        self._pub_status("LOC_WAIT 等待定位 TF（map→base）…")
        t_loc0 = time.time()
        while time.time() - t_loc0 < 8.0:
            loc = self.motion.get_base_xy_yaw_map()
            if loc is not None:
                break
            time.sleep(0.05)
        if loc is None:
            if use_gt:
                self._pub_status(
                    "LOC_WAIT_TIMEOUT：仍无 map→odom。"
                    "请确认 Gazebo 已 spawn 小车且 gt_localization 在跑"
                )
            else:
                self._pub_status(
                    "LOC_FAIL：无 map→odom。请先开 Nav2，在 RViz 做 2D Pose Estimate"
                )
            return
        loc_tag = "GT" if self.motion.use_gt_localization else "AMCL"
        self._pub_status(
            f"SERVERS_OK + LOC_{loc_tag} "
            f"({loc[0]:.2f},{loc[1]:.2f},yaw={loc[2]:.2f}) → 开始导航"
        )

        for idx, (color, cid, zone) in enumerate(queue, start=1):
            # 硬门：手里还有块时绝不开始下一块的取货导航
            if self._holding is not None:
                hc, hid, hz = self._holding
                self._pub_status(
                    f"HOLDING_BLOCK 仍持 {hc}_{hid} → 禁止 STEP "
                    f"{color}_{cid}，先强制放到 {hz}"
                )
                zx_h, zy_h, zyaw_h = self._pose(f"zone_{hz.lower()}")
                tx_h, ty_h, slot_h = self._place_slot_xy(zx_h, zy_h, hz)
                tx_h, ty_h, _ = self._place_drop_inward(
                    tx_h, ty_h, zx_h, zy_h, zone=hz
                )
                arm_m = float(self.get_parameter("place_front_m").value)
                park_m = arm_m + float(
                    self.get_parameter("place_park_extra_m").value
                )
                if not skip_nav:
                    self._start_place_vision_preview(
                        zx_h, zy_h, arm_m, tx_h, ty_h, park_m
                    )
                    self.motion.set_keepout_except(hz, clear_maps=False)
                    self.motion.navigate_place_zone(
                        tx_h, ty_h, xy_tol=0.28, approach_standoff=park_m
                    )
                    self._dock_place_front(
                        tx_h, ty_h, front_m=arm_m, park_m=park_m
                    )
                if not self._place_with_moveit(hc, hid):
                    self.motion.set_keepout_except(None)
                    self._pub_status(
                        f"PLACE_ABORT {hc}_{hid} 未放下 — 中止，不去下一块"
                    )
                    return
                self._holding = None
                hz_key = str(hz).upper()
                self._zone_place_count[hz_key] = int(
                    self._zone_place_count.get(hz_key, 0)
                ) + 1
                self._pub_status(
                    f"PLACED {hc}_{hid} at {hz} slot#{slot_h}（门控补放）"
                )
                if hz_key in ("A", "B", "C"):
                    if self.motion.attach_cube_to_zone(hc, hid, hz_key):
                        self._pub_status(
                            f"ZONE_STICK_OK {hc}_{hid} → zone_{hz_key.lower()}"
                        )
                    else:
                        self._pub_status(
                            f"ZONE_STICK_SOFT {hc}_{hid} → 未粘上"
                        )
                self.motion.add_placed_keepout(tx_h, ty_h, radius=0.14)
                self.motion.set_keepout_except(None, clear_maps=False)
                self._leave_zone_after_place(tx_h, ty_h)

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
                # Nav already parked near cube → face → scan/opencv (no blind creep)
                mx, my, _mz = known_cube_xy(color, cid)
                reach_r = float(self.get_parameter("visual_target_r").value)
                settle = float(self.get_parameter("local_settle_sec").value)
                self._pub_status(
                    f"LOCAL_FLOW → 视觉识别抓取 "
                    f"{color}_{cid} map=({mx:.2f},{my:.2f})"
                )
                self.motion.enter_local_mode(settle_sec=min(0.03, settle))
                aligned = False
                try:
                    # 导航已到位：先看相机；已看见则跳过按 map 拧头，直接 OpenCV
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

                    def _cube_r() -> Optional[float]:
                        from yzbot_task.vision_pose import cube_bearing_in_base
                        info = cube_bearing_in_base(
                            self.motion.tf_buffer, color, cid
                        )
                        if info is not None:
                            return float(info[1])
                        cur_r = self.motion.get_base_xy_yaw_map()
                        if cur_r is None:
                            return None
                        return float(
                            math.hypot(
                                float(cur_r[0]) - mx, float(cur_r[1]) - my
                            )
                        )

                    def _see_now(*, min_area: float = 14.0) -> bool:
                        from yzbot_task.vision_align import detect_blob
                        bgr = self.motion.grab_front_bgr()
                        if bgr is None:
                            return False
                        blob = detect_blob(bgr, color, min_cy_frac=0.08)
                        if blob is None:
                            return False
                        return float(blob.area) >= float(min_area)

                    def _scan(timeout: float, wz: float, min_area: float) -> bool:
                        return bool(
                            self.motion.scan_known_cube_until_seen(
                                color,
                                cid,
                                max_wz=wz,
                                timeout=timeout,
                                min_area=min_area,
                                on_frame=lambda b, bl, p, prefer_uv=None, lv=None, gb=None: _on_oc(
                                    b, bl, p, prefer_uv, lv, gb
                                ),
                            )
                        )

                    def _unbury_if_under() -> None:
                        """If cube is under/beside chassis, reverse until FOV range."""
                        from yzbot_task.vision_pose import cube_bearing_in_base
                        info = cube_bearing_in_base(
                            self.motion.tf_buffer, color, cid
                        )
                        cur = self.motion.get_base_xy_yaw_map()
                        if info is not None:
                            brg, rng = float(info[0]), float(info[1])
                        elif cur is not None:
                            rng = math.hypot(
                                float(cur[0]) - mx, float(cur[1]) - my
                            )
                            brg = 0.0
                        else:
                            return
                        # Under / too close / behind → must back up before any scan
                        # 蓝3：只在压到块上才微退，别空倒再干等
                        is_b3 = str(color).lower() == "blue" and int(cid) == 3
                        need = bool(rng < (0.22 if is_b3 else 0.30))
                        if not need:
                            return
                        self._pub_status(
                            f"CUBE_UNDER {color}_{cid} r={rng:.2f} "
                            f"brg={brg:+.2f} → 倒退出 FOV"
                        )
                        t0 = time.time()
                        max_back = 1.2 if is_b3 else 4.5
                        while time.time() - t0 < max_back:
                            info2 = cube_bearing_in_base(
                                self.motion.tf_buffer, color, cid
                            )
                            if info2 is not None:
                                brg2, rng2 = float(info2[0]), float(info2[1])
                            else:
                                c2 = self.motion.get_base_xy_yaw_map()
                                if c2 is None:
                                    break
                                rng2 = math.hypot(
                                    float(c2[0]) - mx, float(c2[1]) - my
                                )
                                brg2 = 0.0
                            if rng2 >= (0.30 if is_b3 else 0.36):
                                break
                            self.motion.publish_cmd_vel(-0.16, 0.0)
                            time.sleep(0.06)
                        self.motion.stop_chassis()

                    def _recover_see() -> bool:
                        """FOV miss：轻正对 + 窄扫描（勿大幅拧头转过去）。"""
                        _unbury_if_under()
                        self._pub_status(
                            f"SEE_RECOVER {color}_{cid} → 轻正对+窄扫"
                        )
                        self._face_cube_yaw(
                            color, cid, manage_mode=False, reanchor=False
                        )
                        if _see_now():
                            return True
                        return _scan(4.0, 0.45, 10.0)

                    _unbury_if_under()
                    # 常规：大偏差才补朝向；已看见则直接视觉（勿无故二次走路——见 red#3/#4 专项）
                    cur_f = self.motion.get_base_xy_yaw_map()
                    need_face = True
                    if cur_f is not None:
                        yaw_des = math.atan2(
                            my - float(cur_f[1]), mx - float(cur_f[0])
                        )
                        rem_f = abs(
                            self.motion._wrap_pi(yaw_des - float(cur_f[2]))
                        )
                        need_face = rem_f > 0.45
                    if need_face:
                        self._pub_status(
                            f"FACE_TOPUP {color}_{cid} → Nav后补朝向再视觉"
                        )
                        self._face_cube_yaw(
                            color, cid, manage_mode=False, reanchor=False
                        )
                        self.motion.stop_chassis()
                        time.sleep(0.04)
                    else:
                        self._pub_status(
                            f"FACE_SKIP {color}_{cid} → Nav途中已对准"
                        )
                    seen0 = _see_now()
                    if seen0:
                        self._pub_status(
                            f"SEE_OK {color}_{cid} → 视觉贴近"
                        )
                    if not seen0:
                        self._pub_status("SCAN_MAP 未见 → 正对后快扫")
                        seen0 = _scan(4.0, 0.45, 10.0)
                    if not seen0:
                        seen0 = _recover_see()
                    if not seen0:
                        # 蓝3：短距到点后别再倒退空等，直接窄扫识别
                        if str(color).lower() == "blue" and int(cid) == 3:
                            self._pub_status(
                                f"SEE_RECOVER2 blue_3 → 不倒退，直接窄扫"
                            )
                            seen0 = _scan(2.5, 0.45, 10.0)
                        else:
                            self._pub_status(
                                f"SEE_RECOVER2 {color}_{cid} → 小倒退+窄扫"
                            )
                            self.motion.drive_openloop_distance(
                                -0.18, vx=0.18, timeout=1.6
                            )
                            self._face_cube_yaw(
                                color, cid, manage_mode=False, reanchor=False
                            )
                            seen0 = _recover_see()
                    if not seen0:
                        self._pub_status(
                            f"SCAN_STILL_MISS {color}_{cid} "
                            f"map=({mx:.2f},{my:.2f}) → OpenCV 最后试"
                        )

                    oc_timeout = 6.5 if not seen0 else 4.5
                    self._pub_status(
                        "OPENCV_ALIGN"
                        + ("（已看见→快贴）" if seen0 else "（搜索贴近）")
                    )
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
                            0.12, float(self.get_parameter("align_max_wz").value)
                        ),
                        max_vx=0.25 if seen0 else 0.20,
                        timeout=oc_timeout,
                        stable_hold_sec=0.10,
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
                        self._pub_status("OPENCV_OK → 距离复核")
                    else:
                        self._pub_status("OPENCV_SOFT → VIS 轻拉")
                        aligned = self.motion.drive_until_visually_near(
                            color,
                            cid,
                            lock_v_frac=float(
                                self.get_parameter("lock_v_frac").value
                            ),
                            grasp_box_frac=float(
                                self.get_parameter("grasp_box_frac").value
                            ),
                            min_fill=1.15,
                            min_cy_frac=0.70,
                            max_vx=0.11,
                            timeout=3.5,
                            max_travel_m=0.22,
                        )
                        if not aligned:
                            if _recover_see():
                                aligned = True
                                self._pub_status("VIS_RECOVER_OK → 准备下探")
                            else:
                                self._pub_status(
                                    f"VIS_FAIL {color}_{cid} → 仅按坐标贴近"
                                )

                    # ALL cubes: never arm-down while base is still far
                    # 「看见」≠够近：远处色块也能进 FOV，必须 r 到位才夹
                    pick_r = float(reach_r)
                    if str(color).lower() == "blue" and int(cid) in (4, 5):
                        pick_r = max(pick_r, 0.46)
                    # 硬门限：超过此距离禁止下探（短于旧 +0.28，不靠「看见」放行）
                    max_pick_r = pick_r + 0.10
                    info_r = _cube_r()
                    if info_r is not None and info_r > pick_r + 0.06:
                        self._pub_status(
                            f"PRE_PICK_CREEP {color}_{cid} "
                            f"r={info_r:.2f}m → ≤{pick_r:.2f}m（快贴）"
                        )
                        self.motion.creep_to_known_cube(
                            color,
                            cid,
                            target_range=pick_r,
                            max_vx=0.18,
                            max_wz=0.17,
                            timeout=3.2,
                        )
                        _unbury_if_under()
                        info_r = _cube_r()
                    if info_r is not None:
                        self._pub_status(
                            f"PRE_PICK_R {color}_{cid} r={info_r:.2f}m "
                            f"need≤{max_pick_r:.2f} see={_see_now()}"
                        )

                    # Hard gate: still far → abort（看见也不能夹，防远处吸附拽块）
                    info_r = _cube_r()
                    if info_r is None:
                        self._pub_status(
                            f"PICK_ABORT NO_RANGE {color}_{cid} — 不空夹"
                        )
                        aligned = False
                    elif float(info_r) > float(max_pick_r):
                        self._pub_status(
                            f"PICK_ABORT STILL_FAR {color}_{cid} "
                            f"r={info_r:.2f}>{max_pick_r:.2f}m — 不空夹"
                        )
                        aligned = False
                    else:
                        aligned = True
                finally:
                    self.motion.stop_chassis()
                    self.motion.exit_local_mode()

                if not aligned:
                    self._pub_status(
                        f"STOP (搜不到 {color}_{cid}，跳过本块)"
                    )
                    pick_ok = False
                else:
                    # Quick pin — short settle only
                    so = float(self.get_parameter("grasp_standoff_m").value)
                    ax, ay, ayaw = self._parking_xy(color, cid, standoff_m=so)
                    cur = self.motion.get_base_xy_yaw_map()
                    if cur is not None:
                        pin_x, pin_y, pin_yaw = (
                            float(cur[0]),
                            float(cur[1]),
                            float(cur[2]),
                        )
                        self.motion.reanchor_amcl(
                            pin_x, pin_y, pin_yaw, settle_sec=0.03
                        )
                    else:
                        self.motion.reanchor_amcl(ax, ay, ayaw, settle_sec=0.03)

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
                self._pub_status(
                    f"SKIP_CUBE {color}_{cid}（未可靠看到/够近，不空夹）→ 下一块"
                )
                continue

            # 已夹起：标记持物，必须放置成功后才能去下一块
            self._holding = (str(color), int(cid), str(zone).upper())
            self._pub_status(
                f"HOLDING {color}_{cid} → 先运放到 {zone}，禁止下一块"
            )

            # 投放：槽位 + 略内收，夹在绿垫长方形内（不收成一个点）
            tx, ty, slot_i = self._place_slot_xy(zx, zy, zone)
            tx, ty, _approach_yaw = self._place_drop_inward(
                tx, ty, zx, zy, zone=zone
            )
            # 3号蓝回 A：Y 只往负方向移 0.2m。0.5m 会落到垫子南边外面。
            if (
                str(color).lower() == "blue"
                and int(cid) == 3
                and str(zone).strip().upper() == "A"
            ):
                ty = float(ty) - 0.20
            self._pub_status(
                f"PLACE_TARGET zone={zone} #{slot_i} "
                f"drop=({tx:.2f},{ty:.2f}) 区心=({zx:.2f},{zy:.2f})"
            )

            if not skip_nav:
                arm_m = float(self.get_parameter("place_front_m").value)
                park_m = arm_m + float(
                    self.get_parameter("place_park_extra_m").value
                )
                self._pub_status(
                    f"NAV_PLACE → zone {zone} 停靠={park_m:.2f}m "
                    f"臂伸={arm_m:.2f}m 投放=({tx:.2f},{ty:.2f})"
                )
                # Allow path into this zone; keep other pads as keepout
                self.motion.set_keepout_except(zone, clear_maps=False)
                self.motion.prepare_next_nav()
                # 接近容差放宽：太严会在区前空等/卡死；精调交给 PLACE_DOCK
                ok_nav = True
                blue3_place = str(color).lower() == "blue" and int(cid) == 3
                if blue3_place and str(zone).strip().upper() == "C":
                    self._face_travel(math.pi)
                    self._start_place_vision_preview(
                        zx, zy, arm_m, tx, ty, park_m, from_east=True
                    )
                    ok_nav = self._nav_through_c_door_then(
                        tx,
                        ty,
                        math.pi,
                        xy_tol=0.22,
                        place_mode=True,
                        approach_standoff=park_m,
                        from_east=True,
                    )
                elif blue3_place:
                    # 3号蓝在A区东边：先朝东侧停靠点往前开，到了再对区。
                    # 路上就追朝西，车会倒着拧，块放不回垫子。
                    so_b3 = max(0.15, float(park_m))
                    gx_b3, gy_b3 = float(tx) + so_b3, float(ty)
                    cur_b3 = self.motion.get_base_xy_yaw_map()
                    if cur_b3 is not None:
                        travel_b3 = math.atan2(
                            gy_b3 - float(cur_b3[1]),
                            gx_b3 - float(cur_b3[0]),
                        )
                    else:
                        travel_b3 = math.pi
                    self._face_travel(travel_b3)
                    self._start_place_vision_preview(
                        zx, zy, arm_m, tx, ty, park_m, from_east=True
                    )
                    ok_nav = self.motion.navigate_place_zone(
                        tx,
                        ty,
                        xy_tol=0.22,
                        approach_standoff=park_m,
                        from_east=True,
                        goal_yaw=travel_b3,
                    )
                    cur_b3 = self.motion.get_base_xy_yaw_map()
                    if cur_b3 is not None:
                        rem_b3 = self.motion._wrap_pi(
                            math.pi - float(cur_b3[2])
                        )
                        if abs(rem_b3) > 0.25:
                            self._pub_status(
                                "PLACE_FACE_EAST 东侧停靠后再对区 "
                                f"{abs(rem_b3) * 180 / math.pi:.0f}°"
                            )
                            self.motion.snap_yaw_map(
                                math.pi, tol=0.22, max_wz=0.28, timeout=3.5
                            )
                elif str(zone).strip().upper() == "C":
                    self._start_place_vision_preview(
                        zx, zy, arm_m, tx, ty, park_m
                    )
                    ok_nav = self._nav_through_c_door_then(
                        tx,
                        ty,
                        0.0,
                        xy_tol=0.22,
                        place_mode=True,
                        approach_standoff=park_m,
                    )
                elif self._is_first_red(color, cid) and self._door_via_for_zone(zone):
                    self._start_place_vision_preview(
                        zx, zy, arm_m, tx, ty, park_m
                    )
                    ok_nav = self._nav_continuous_with_door(
                        tx,
                        ty,
                        0.0,
                        place_mode=True,
                        xy_tol=0.22,
                        use_door_via=True,
                    )
                else:
                    self._skirt_wall_above_a(ty)
                    self._start_place_vision_preview(
                        zx, zy, arm_m, tx, ty, park_m
                    )
                    ok_nav = self.motion.navigate_place_zone(
                        tx, ty, xy_tol=0.22, approach_standoff=park_m
                    )
                if not ok_nav:
                    self._pub_status(
                        f"NAV_PLACE 软失败 {zone} → 仍本地对区心补进"
                    )
                self._dock_place_front(tx, ty, front_m=arm_m, park_m=park_m)

            arm_m = float(self.get_parameter("place_front_m").value)
            park_m = arm_m + float(
                self.get_parameter("place_park_extra_m").value
            )
            placed = self._place_with_moveit(color, cid)
            if not placed:
                # One more full place attempt before aborting the queue
                self._pub_status(
                    f"PLACE_RETRY {color}_{cid} → {zone}（仍持物）"
                )
                if not skip_nav:
                    self._start_place_vision_preview(
                        zx, zy, arm_m, tx, ty, park_m
                    )
                    self.motion.set_keepout_except(zone, clear_maps=False)
                    self.motion.navigate_place_zone(
                        tx, ty, xy_tol=0.28, approach_standoff=park_m
                    )
                    self._dock_place_front(tx, ty, front_m=arm_m, park_m=park_m)
                placed = self._place_with_moveit(color, cid)
            if not placed:
                self.motion.set_keepout_except(None)
                self._pub_status(
                    f"PLACE_ABORT {color}_{cid} 未放下 — 中止，不去下一块"
                )
                return

            self._holding = None
            zkey = str(zone).strip().upper()
            self._zone_place_count[zkey] = int(self._zone_place_count.get(zkey, 0)) + 1
            self._pub_status(f"PLACED {color}_{cid} at {zone} slot#{slot_i}")
            # A/B/C：放下后粘到区垫，避免被车蹭出区（与夹爪 LinkAttacher 同机制）
            if zkey in ("A", "B", "C"):
                if self.motion.attach_cube_to_zone(color, cid, zkey):
                    self._pub_status(
                        f"ZONE_STICK_OK {color}_{cid} → zone_{zkey.lower()}"
                    )
                else:
                    self._pub_status(
                        f"ZONE_STICK_SOFT {color}_{cid} → 未粘上，靠 keepout"
                    )
            # Mark cube as permanent obstacle BEFORE leave/next nav (never drive over it)
            self.motion.add_placed_keepout(tx, ty, radius=0.14)
            self.motion.set_keepout_except(None, clear_maps=False)
            # Reverse clear of pad so footprint is off the letter before planning
            self._leave_zone_after_place(tx, ty)
            cur = self.motion.get_base_xy_yaw_map()
            pyaw = float(cur[2]) if cur is not None else zyaw
            lx = float(cur[0]) if cur is not None else zx
            ly = float(cur[1]) if cur is not None else zy
            # Keep costmap marks of the placed cube; do not clear
            self.motion.prepare_next_nav(lx, ly, pyaw, clear_maps=False)

        # 队列跑完≠件件成功；面板只认 PLACED，勿暗示全部完成
        self._pub_status("DONE")

    def _drop_inside_pad(
        self,
        fx: float,
        fy: float,
        zx: float,
        zy: float,
        *,
        from_west: Optional[bool] = None,
    ) -> bool:
        """绿垫 长边X≈1.0m × 短边Y≈0.5m；长/短轴容差不同。

        从 ±X 接近时面对短边：不能停在短边外侧/贴边，要沿长边进到区内。
        """
        # 长边(X)半长 0.5，留边 → ±0.40；短边(Y)半宽 0.25，留边 → ±0.14
        if abs(float(fy) - float(zy)) > 0.14:
            return False
        if abs(float(fx) - float(zx)) > 0.40:
            return False
        # 禁止停在靠近侧的短边外缘：至少越过短边、靠近区心一侧
        if from_west is True and float(fx) < float(zx) - 0.18:
            return False
        if from_west is False and float(fx) > float(zx) + 0.18:
            return False
        return True

    def _dock_place_front(
        self,
        zx: float,
        zy: float,
        *,
        front_m: float = 0.38,
        park_m: Optional[float] = None,
    ) -> None:
        """沿长边推进绿垫；短边只允许很小侧偏。朝向大致对着即可。"""
        settle = float(self.get_parameter("local_settle_sec").value)
        arm_m = float(front_m)
        so = max(0.28, float(park_m) if park_m is not None else arm_m)
        cur = self.motion.get_base_xy_yaw_map()
        if cur is None:
            self._stop_place_vision_preview()
            return
        rx, ry = float(cur[0]), float(cur[1])
        from_west = (rx - (float(zx) - 0.1)) ** 2 <= (
            rx - (float(zx) + 0.1)
        ) ** 2
        cands = [
            (float(zx) - so, float(zy), 0.0),
            (float(zx) + so, float(zy), math.pi),
        ]
        px, py, pyaw = min(
            cands, key=lambda p: (p[0] - rx) ** 2 + (p[1] - ry) ** 2
        )
        from_west = abs(px - (float(zx) - so)) < abs(px - (float(zx) + so))
        self._pub_status(
            f"PLACE_DOCK 长边进区/短边对齐 → park=({px:.2f},{py:.2f}) "
            f"目标=({zx:.2f},{zy:.2f}) from={'W' if from_west else 'E'}"
        )
        self.motion.exit_local_mode()
        # 区前不再重发导航。那一段会边走边转，A 区最容易把车头带歪。
        self.motion.stop_chassis()

        self.motion.enter_local_mode(settle_sec=min(0.04, settle))
        try:
            # 到点后只原地对一次区，少转一点，避免拧过再拧回来。
            c0 = self.motion.get_base_xy_yaw_map()
            if c0 is not None:
                rem = self.motion._wrap_pi(pyaw - float(c0[2]))
                if abs(rem) > 0.22:
                    turn = float(rem) * 0.75
                    self._pub_status(
                        f"PLACE_FACE 原地对区 {abs(turn)*180/math.pi:.0f}°"
                    )
                    self.motion.spin_yaw_openloop(
                        turn, max_wz=0.16, timeout=2.2
                    )
                    self.motion.stop_chassis()
                    time.sleep(0.03)

            # 朝向已经对着区，才沿车头短走一段。已经在区内就停，不再顶出区。
            c1 = self.motion.get_base_xy_yaw_map()
            if c1 is not None:
                bx, by, yaw = float(c1[0]), float(c1[1]), float(c1[2])
                yaw_err = abs(self.motion._wrap_pi(pyaw - yaw))
                fx = bx + arm_m * math.cos(yaw)
                fy = by + arm_m * math.sin(yaw)
                ex_b = math.cos(yaw) * (float(zx) - fx) + math.sin(yaw) * (
                    float(zy) - fy
                )
                inside = self._drop_inside_pad(
                    fx, fy, zx, zy, from_west=from_west
                )
                if (not inside) and yaw_err <= 0.30 and ex_b > 0.04:
                    crawl = min(0.18, float(ex_b))
                    self._pub_status(f"PLACE_CRAWL +{crawl:.2f}m 沿车头进区")
                    self.motion.drive_openloop_distance(
                        crawl, vx=0.12, timeout=1.6
                    )
                    self.motion.stop_chassis()
                c3 = self.motion.get_base_xy_yaw_map()
                if c3 is not None:
                    yaw3 = float(c3[2])
                    fx3 = float(c3[0]) + arm_m * math.cos(yaw3)
                    fy3 = float(c3[1]) + arm_m * math.sin(yaw3)
                    ok = self._drop_inside_pad(
                        fx3, fy3, zx, zy, from_west=from_west
                    )
                    self._pub_status(
                        f"PLACE_DOCK_OK in_pad={'Y' if ok else 'N'} "
                        f"dx={fx3-zx:+.2f}(长) dy={fy3-zy:+.2f}(短) "
                        f"预测=({fx3:.2f},{fy3:.2f})"
                    )

            # 4) 绿垫视觉微调：转身正对后再判像素中心（最后1m，马上放之前）
            if bool(self.get_parameter("place_vision_enable").value):
                self._place_vision_center_pad(
                    zx, zy, arm_m=arm_m, from_west=from_west
                )
        finally:
            self._stop_place_vision_preview()
            self.motion.stop_chassis()
            self.motion.exit_local_mode()

    def _start_place_vision_preview(
        self,
        zx: float,
        zy: float,
        arm_m: float,
        tx: float,
        ty: float,
        park_m: float,
        from_east: bool = False,
    ) -> None:
        """On the way to the zone, sample the pad and predict the drop. No cmd_vel."""
        self._stop_place_vision_preview()
        self._place_vis_sample = None
        self._place_vis_ahead = None
        so = max(0.15, float(park_m))
        if from_east:
            gx, gy, face = float(tx) + so, float(ty), math.pi
        else:
            cur = self.motion.get_base_xy_yaw_map()
            rx = float(cur[0]) if cur is not None else float(tx) - 1.0
            cands = [
                (float(tx) - so, float(ty), 0.0),
                (float(tx) + so, float(ty), math.pi),
            ]
            gx, gy, face = min(
                cands, key=lambda p: (p[0] - rx) ** 2 + (p[1] - float(ty)) ** 2
            )
        self._place_vis_zone = (
            float(zx),
            float(zy),
            float(arm_m),
            float(gx),
            float(gy),
            float(face),
        )
        self._place_vis_stop = threading.Event()
        self._place_vis_thread = threading.Thread(
            target=self._place_vision_preview_loop,
            name="place_vision_preview",
            daemon=True,
        )
        self._place_vis_thread.start()
        self._pub_status("PLACE_VISION_PREVIEW 路上算绿垫和落点，不控车")

    def _stop_place_vision_preview(self) -> None:
        ev = self._place_vis_stop
        th = self._place_vis_thread
        if ev is not None:
            ev.set()
        if (
            th is not None
            and th.is_alive()
            and threading.current_thread() is not th
        ):
            th.join(timeout=0.5)
        self._place_vis_thread = None

    def _place_vision_preview_loop(self) -> None:
        target_cy = float(self.get_parameter("place_vision_target_cy").value)
        vision_on = bool(self.get_parameter("place_vision_enable").value)
        stop = self._place_vis_stop
        while stop is not None and not stop.is_set():
            if vision_on:
                bgr = self.motion.grab_front_bgr()
                pad = detect_green_pad(bgr) if bgr is not None else None
                if pad is not None:
                    cy_frac = float(pad.cy) / float(max(pad.image_h, 1))
                    sample = {
                        "t": time.time(),
                        "err_u": float(pad.err_u),
                        "cy_frac": cy_frac,
                        "frac": blob_area_frac(pad),
                        "centered": green_pad_centered(
                            pad,
                            max_err_u=0.12,
                            target_cy_frac=target_cy,
                            max_err_v=0.18,
                            min_area_frac=0.018,
                        ),
                    }
                    with self._place_vis_lock:
                        self._place_vis_sample = sample
            if stop.wait(0.08):
                break
            try:
                self._note_place_landing_ahead()
            except Exception:  # noqa: BLE001
                pass

    def _note_place_landing_ahead(self) -> None:
        """About 4s before the park, predict the drop. No cmd_vel."""
        zone = self._place_vis_zone
        if zone is None or self._place_vis_ahead is not None:
            return
        cur = self.motion.get_base_xy_yaw_map()
        if cur is None:
            return
        zx, zy, arm_m, gx, gy, face = zone
        rx, ry = float(cur[0]), float(cur[1])
        dist = math.hypot(rx - gx, ry - gy)
        # 约 0.40m/s，离停靠点 1.6m 大约还要 4 秒。只记一次，不控车。
        if dist > 1.60:
            return
        land_x = float(gx) + float(arm_m) * math.cos(float(face))
        land_y = float(gy) + float(arm_m) * math.sin(float(face))
        from_west = float(gx) < float(zx)
        in_pad = self._drop_inside_pad(
            land_x, land_y, zx, zy, from_west=from_west
        )
        self._place_vis_ahead = {
            "land_x": land_x,
            "land_y": land_y,
            "in_pad": in_pad,
            "dist": dist,
        }
        self._pub_status(
            f"PLACE_AHEAD_4S 落点=({land_x:.2f},{land_y:.2f}) "
            f"区心=({zx:.2f},{zy:.2f}) 在区内={'Y' if in_pad else 'N'} "
            f"d={dist:.2f}m 不控车"
        )

    def _place_vision_center_pad(
        self,
        zx: float,
        zy: float,
        *,
        arm_m: float,
        from_west: bool,
    ) -> None:
        """Use the pad sample computed on the way in. Does not command the base."""
        del zx, zy, arm_m, from_west
        self._stop_place_vision_preview()
        with self._place_vis_lock:
            sample = self._place_vis_sample
        if sample is None:
            bgr = self.motion.grab_front_bgr()
            pad = detect_green_pad(bgr) if bgr is not None else None
            if pad is not None:
                target_cy = float(self.get_parameter("place_vision_target_cy").value)
                cy_frac = float(pad.cy) / float(max(pad.image_h, 1))
                sample = {
                    "t": time.time(),
                    "err_u": float(pad.err_u),
                    "cy_frac": cy_frac,
                    "frac": blob_area_frac(pad),
                    "centered": green_pad_centered(
                        pad,
                        max_err_u=0.12,
                        target_cy_frac=target_cy,
                        max_err_v=0.18,
                        min_area_frac=0.018,
                    ),
                }
        if sample is None:
            self._pub_status("PLACE_VISION_MISS 途中未看到绿垫 → 直接放")
            return
        age = max(0.0, time.time() - float(sample["t"]))
        self._pub_status(
            f"PLACE_VISION_READY u={float(sample['err_u']):+.2f} "
            f"cy={float(sample['cy_frac']):.2f} frac={float(sample['frac']):.3f} "
            f"centered={'Y' if sample['centered'] else 'N'} age={age:.2f}s → 直接放"
        )

    def _leave_zone_after_place(self, zx: float, zy: float) -> None:
        """Straight reverse only — clear the pad so next approach won't crush cubes."""
        settle = float(self.get_parameter("local_settle_sec").value)
        self._pub_status("LEAVE_ZONE 直行倒退离开堆放区（不转向）→ 直接规划")
        self.motion.enter_local_mode(settle_sec=min(0.05, settle))
        try:
            # Leave pad（略缩短倒退，省时；仍离开字母垫）
            self.motion.drive_openloop_distance(-0.36, vx=0.85, timeout=0.9)
        finally:
            self.motion.stop_chassis()
            self.motion.exit_local_mode()
        self._pub_status("LEAVE_ZONE_OK 倒退完成 → 下一段导航直接规划")


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
