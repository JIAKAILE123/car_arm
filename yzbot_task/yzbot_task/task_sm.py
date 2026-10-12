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
    CARRY,
    PLACE,
    PICK,
    PRESET_REACH_R,
    PLACE_FRONT_M,
)
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
        self.declare_parameter("align_timeout", 7.9)
        self.declare_parameter("align_yaw_gain", 0.75)
        # Local align yaw (before “slow search” cut to 0.10)
        self.declare_parameter("align_max_wz", 0.24)
        self.declare_parameter("local_settle_sec", 0.02)
        # Nav2 often stops ~0.5–0.8m short (inflation); soft gate then continuous local
        self.declare_parameter("pose_gate_xy", 0.55)
        # Pick: tighter stop so different paths don't leave huge park variance
        self.declare_parameter("pick_pose_gate_xy", 0.22)
        # Place can be loose; pick must face cube or front cam sees nothing
        self.declare_parameter("pose_gate_yaw", 0.75)
        self.declare_parameter("pose_gate_yaw_pick", 0.18)
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
        # Negative = toward robot / outer edge; Positive = deeper into pad (区心)
        # 0.06 偶发偏区外（停靠偏短）；略加大内收，仍远小于垫半宽 0.5m
        self.declare_parameter("place_inward_m", 0.10)
        self.declare_parameter("place_park_extra_m", 0.0)
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
        self.declare_parameter("lock_min_area", 60.0)
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
            self.motion.configure_zone_keepouts(
                zmap, half_x=0.48, half_y=0.28, step=0.06
            )
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
        # Prefer letter center; later ±4cm so stacks don't collide
        slots_y = (0.0, 0.04, -0.04, 0.06, -0.06)
        dy = float(slots_y[n % len(slots_y)])
        return float(zx), float(zy) + dy, n

    def _place_drop_inward(
        self, tx: float, ty: float, zx: float, zy: float, *, zone: str = ""
    ) -> Tuple[float, float, float]:
        """Bias drop along approach: +inward = deeper into pad, − = toward robot."""
        inward = float(self.get_parameter("place_inward_m").value)
        # Zone B letter is shallow on approach side — push further into pad
        if str(zone).strip().upper() == "B":
            inward = max(inward, 0.0) + 0.05
        # Pad is 1.0×0.5 m; keep drop inside with margin (avoid 区外)
        max_dx, max_dy = 0.32, 0.16
        cur = self.motion.get_base_xy_yaw_map()
        rx = float(cur[0]) if cur is not None else float(tx) - 1.0
        from_west = (rx - (tx - 0.1)) ** 2 <= (rx - (tx + 0.1)) ** 2
        yaw = 0.0 if from_west else math.pi
        if abs(inward) >= 1e-4:
            if from_west:
                tx = float(tx) + inward
            else:
                tx = float(tx) - inward
        # Clamp vs 区心 so slot+inward never leaves the letter pad
        tx = float(zx) + max(-max_dx, min(max_dx, float(tx) - float(zx)))
        ty = float(zy) + max(-max_dy, min(max_dy, float(ty) - float(zy)))
        return float(tx), float(ty), yaw

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
        xy_tol = float(self.get_parameter("pick_pose_gate_xy").value)
        # Pick: must face cube — loose place yaw leaves cam looking away
        yaw_tol = float(self.get_parameter("pose_gate_yaw_pick").value)
        # All blue + all red: never drive across/over the cube (protects AMCL)
        use_safe = str(color).lower() in ("blue", "red")
        if use_safe:
            vias = self._cube_safe_approach_vias(mx, my, ax, ay)
            if abs(ax - mx) >= abs(ay - my):
                side = "东" if ax >= mx else "西"
            else:
                side = "北" if ay >= my else "南"
            self._pub_status(
                f"SAFE_PICK {color}_{cid} park{side} ({ax:.2f},{ay:.2f}) "
                f"yaw→块={ayaw:.2f} vias={len(vias)} 不压块"
            )
            if vias:
                nav_ok = self._nav_with_vias_then_park(
                    vias, ax, ay, ayaw, xy_tol=xy_tol, yaw_tol=yaw_tol
                )
            else:
                self.motion.exit_local_mode()
                retries = int(self.get_parameter("nav_retries").value)
                nav_ok = self.motion.navigate_and_verify(
                    ax,
                    ay,
                    ayaw,
                    retries=retries,
                    xy_tol=xy_tol,
                    yaw_tol=yaw_tol,
                    settle_sec=0.04,
                    reanchor=True,
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
                self.motion.clear_costmaps(timeout=0.25)
            # Re-anchor at park facing cube — recovers if path brushed obstacles
            cur2 = self.motion.get_base_xy_yaw_map()
            if cur2 is not None:
                self.motion.reanchor_amcl(
                    float(cur2[0]), float(cur2[1]), float(ayaw), settle_sec=0.03
                )
        else:
            self.motion.exit_local_mode()
            self._pub_status(
                f"NAV_PICK_DIRECT → ({ax:.2f},{ay:.2f}) yaw={ayaw:.2f} "
                f"(朝向物块)"
            )
            retries = int(self.get_parameter("nav_retries").value)
            nav_ok = self.motion.navigate_and_verify(
                ax,
                ay,
                ayaw,
                retries=retries,
                xy_tol=xy_tol,
                yaw_tol=yaw_tol,
                settle_sec=0.04,
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
                self.motion.clear_costmaps(timeout=0.4)
        self.motion.stop_chassis()
        # Distance check — must be within creep band (was 1.10m → 仍很远就进 OpenCV)
        creep_r = float(self.get_parameter("grasp_creep_m").value)
        if not self._ensure_near_cube(
            color, cid, standoff_m=so, max_range_m=creep_r + 0.12
        ):
            self._pub_status(f"NAV_PICK_FAIL / TOO_FAR {color}_{cid}")
            return False
        self._pub_status(
            f"NAV_PICK_OK → 本地: 转向识别+OpenCV贴框 "
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
            self.motion.enter_local_mode(settle_sec=min(0.03, settle))
            try:
                if self.motion.creep_to_known_cube(
                    color,
                    cid,
                    target_range=creep_r,
                    max_vx=0.10,
                    max_wz=0.16,
                    timeout=8.0,
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
                max_wz=0.28,
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
        # Face-turn: a bit faster than fine OpenCV align
        max_wz = max(0.22, float(self.get_parameter("align_max_wz").value))
        yaw_gain = max(0.85, float(self.get_parameter("align_yaw_gain").value))
        mx, my, _mz = known_cube_xy(color, cid)
        tol = 0.12

        if manage_mode:
            self.motion.enter_local_mode(settle_sec=min(0.03, settle))
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

        self._pub_status("MTC_LIFT")
        if self.moveit.execute_joints(CARRY, duration_sec=1.5):
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

    def _place_with_moveit(self, color: str, cid: int) -> bool:
        """放置：快速下探 → 立刻松开 → 马上抬臂（不等待）。"""
        import subprocess
        import os

        self._pub_status("PLACE_ARM 快速下放")
        if not self.motion.arm_place_pose():
            self.get_logger().warn("place pose failed; still try detach")

        # 先走 ROS detach（快），cube_control 短超时，绝不干等
        self._pub_status(f"DETACH → {color}_{cid}（立刻抬臂）")
        ok = bool(self.motion.detach(color, cid))
        script = os.path.join(
            os.path.expanduser("~/dev_ws/yzbot/src"), "cube_control.sh"
        )
        if os.path.isfile(script):
            try:
                r = subprocess.run(
                    ["bash", script, "off", color, str(cid)],
                    check=False,
                    timeout=0.8,
                    capture_output=True,
                    text=True,
                )
                if r.returncode == 0:
                    ok = True
                if r.stdout:
                    self.get_logger().info(r.stdout)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"cube_control off skipped: {exc}")
        if ok:
            self._pub_status(f"DETACH_OK {color}_{cid}")
        else:
            self._pub_status(f"DETACH_SOFT_FAIL {color}_{cid} → 仍马上抬臂")

        # 放下后立刻抬起，不 sleep
        self._pub_status("PLACE_LIFT 马上抬起")
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

        self._zone_place_count = {"A": 0, "B": 0, "C": 0}
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
        self._pub_status("WAITING_SERVERS (Nav2/臂/吸附，最多约 6.5s)…")
        if not self.motion.wait_servers(
            timeout=6.5,
            need_arm=mode_needs_arm,
            need_attach=True,
        ):
            self._pub_status(
                "SERVERS_UNAVAILABLE (先开 Gazebo + Nav2 + 臂控制器；吸附要 /ATTACHLINK)"
            )
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
                # Nav already parked near cube → face → scan/opencv (no blind creep)
                mx, my, _mz = known_cube_xy(color, cid)
                reach_r = float(self.get_parameter("visual_target_r").value)
                settle = float(self.get_parameter("local_settle_sec").value)
                self._pub_status(
                    f"LOCAL_FLOW → 导航到位后转向识别 "
                    f"{color}_{cid} map=({mx:.2f},{my:.2f})"
                )
                self.motion.enter_local_mode(settle_sec=min(0.03, settle))
                aligned = False
                try:
                    # Restore original grasp vision: face → scan → OpenCV → soft nudge
                    self._face_cube_yaw(
                        color, cid, manage_mode=False, reanchor=False
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
                        need = bool(
                            rng < 0.48 or abs(brg) > 1.15
                        )
                        if not need:
                            return
                        self._pub_status(
                            f"CUBE_UNDER {color}_{cid} r={rng:.2f} "
                            f"brg={brg:+.2f} → 倒退出 FOV"
                        )
                        t0 = time.time()
                        while time.time() - t0 < 4.5:
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
                            if rng2 >= 0.58 and abs(brg2) < 0.9:
                                break
                            self.motion.publish_cmd_vel(-0.16, 0.0)
                            time.sleep(0.06)
                        self.motion.stop_chassis()

                    def _recover_see() -> bool:
                        """FOV miss: if under→reverse; else creep to safe FOV band."""
                        _unbury_if_under()
                        # Prefer FOV band ~0.55–0.65; never crawl onto the cube
                        fov_r = 0.58 if (
                            str(color).lower() == "blue" and int(cid) in (4, 5)
                        ) else max(float(reach_r) + 0.12, 0.52)
                        self._pub_status(
                            f"SEE_RECOVER {color}_{cid} → 保持r≈{fov_r:.2f}+正对+再扫"
                        )
                        self.motion.creep_to_known_cube(
                            color,
                            cid,
                            target_range=float(fov_r),
                            max_vx=0.10,
                            max_wz=0.20,
                            timeout=5.5,
                        )
                        _unbury_if_under()
                        self._face_cube_yaw(
                            color, cid, manage_mode=False, reanchor=False
                        )
                        if _see_now():
                            return True
                        return _scan(9.0, 0.30, 14.0)

                    _unbury_if_under()
                    seen0 = _see_now()
                    if not seen0:
                        self._pub_status("SCAN_MAP 按坐标扫（未见色块）")
                        seen0 = _scan(9.0, 0.26, 16.0)
                    if not seen0:
                        seen0 = _recover_see()
                    if not seen0:
                        self._pub_status(
                            f"SEE_RECOVER2 {color}_{cid} → 再倒退+扫描"
                        )
                        self.motion.drive_openloop_distance(
                            -0.28, vx=0.22, timeout=2.0
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

                    oc_timeout = 7.5 if not seen0 else 5.5
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
                            0.18, float(self.get_parameter("align_max_wz").value)
                        ),
                        max_vx=0.16,
                        timeout=oc_timeout,
                        stable_hold_sec=0.14,
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
                            # Last chance for any cube — do not pretend OK
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
                        # 短快一次，不拖长时间
                        self.motion.creep_to_known_cube(
                            color,
                            cid,
                            target_range=pick_r,
                            max_vx=0.20,
                            max_wz=0.20,
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

            # Drop on zone center (arm reach == park distance). Tiny Y only if re-place.
            tx, ty, slot_i = self._place_slot_xy(zx, zy, zone)
            tx, ty, _approach_yaw = self._place_drop_inward(
                tx, ty, zx, zy, zone=zone
            )
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
                self.motion.set_keepout_except(zone)
                self.motion.prepare_next_nav()
                if self._is_first_red(color, cid) and self._door_via_for_zone(zone):
                    if not self._nav_continuous_with_door(
                        tx,
                        ty,
                        0.0,
                        place_mode=True,
                        xy_tol=0.08,
                        use_door_via=True,
                    ):
                        self._pub_status(f"NAV_PLACE_FAIL {zone}")
                        self.motion.set_keepout_except(None)
                        return
                elif not self.motion.navigate_place_zone(
                    tx, ty, xy_tol=0.08, approach_standoff=park_m
                ):
                    self._pub_status(f"NAV_PLACE_FAIL {zone}")
                    self.motion.set_keepout_except(None)
                    return
                # Fine dock uses arm reach to predict EE, not park_extra
                self._dock_place_front(tx, ty, front_m=arm_m, park_m=park_m)

            if not self._place_with_moveit(color, cid):
                self.motion.set_keepout_except(None)
                return
            zkey = str(zone).strip().upper()
            self._zone_place_count[zkey] = int(self._zone_place_count.get(zkey, 0)) + 1
            self._pub_status(f"PLACED {color}_{cid} at {zone} slot#{slot_i}")
            # Mark cube as permanent obstacle BEFORE leave/next nav (never drive over it)
            self.motion.add_placed_keepout(tx, ty, radius=0.22)
            self.motion.set_keepout_except(None, clear_maps=False)
            # Reverse clear of pad so footprint is off the letter before planning
            self._leave_zone_after_place(tx, ty)
            cur = self.motion.get_base_xy_yaw_map()
            pyaw = float(cur[2]) if cur is not None else zyaw
            lx = float(cur[0]) if cur is not None else zx
            ly = float(cur[1]) if cur is not None else zy
            # Keep costmap marks of the placed cube; do not clear
            self.motion.prepare_next_nav(lx, ly, pyaw, clear_maps=False)

        self._pub_status("DONE")

    def _dock_place_front(
        self,
        zx: float,
        zy: float,
        *,
        front_m: float = 0.38,
        park_m: Optional[float] = None,
    ) -> None:
        """Park so arm EE (front_m ahead) lands on (zx,zy) — zone center.

        Nav goal uses park_m (usually == front_m). Fine-tune predicts drop with
        front_m and nulls both depth and lateral error in base frame.
        """
        settle = float(self.get_parameter("local_settle_sec").value)
        arm_m = float(front_m)
        so = float(park_m) if park_m is not None else arm_m
        cur = self.motion.get_base_xy_yaw_map()
        if cur is None:
            return
        rx, ry = float(cur[0]), float(cur[1])
        # Only ±X docks so arm (j1=0) points straight into the letter pad
        cands = [
            (float(zx) - so, float(zy), 0.0),
            (float(zx) + so, float(zy), math.pi),
        ]
        px, py, pyaw = min(
            cands, key=lambda p: (p[0] - rx) ** 2 + (p[1] - ry) ** 2
        )
        self._pub_status(
            f"PLACE_DOCK 对准区心 → base=({px:.2f},{py:.2f}) yaw={pyaw:+.2f} "
            f"臂伸{arm_m:.2f}m → ({zx:.2f},{zy:.2f})"
        )
        self.motion.exit_local_mode()
        self.motion.navigate(
            px,
            py,
            pyaw,
            timeout=32.0,
            early_xy_tol=0.10,
            soft_start=True,
        )
        self.motion.enter_local_mode(settle_sec=min(0.04, settle))
        try:
            # 1) Face pad tightly — yaw bias is the main lateral place error
            for _ in range(3):
                c = self.motion.get_base_xy_yaw_map()
                if c is None:
                    break
                rem = self.motion._wrap_pi(pyaw - float(c[2]))
                if abs(rem) <= 0.06:
                    break
                self.motion.spin_yaw_openloop(
                    float(rem), max_wz=0.22, timeout=2.0
                )
                self.motion.stop_chassis()
                time.sleep(0.03)

            # 2) Null predicted EE error vs zone target (depth + lateral)
            # Prefer closing depth shortfall (偏外) before fine lateral
            t0 = time.time()
            while time.time() - t0 < 5.5:
                c = self.motion.get_base_xy_yaw_map()
                if c is None:
                    break
                bx, by, yaw = float(c[0]), float(c[1]), float(c[2])
                fx = bx + arm_m * math.cos(yaw)
                fy = by + arm_m * math.sin(yaw)
                ex = float(zx) - fx
                ey = float(zy) - fy
                d_drop = math.hypot(ex, ey)
                rem = self.motion._wrap_pi(pyaw - yaw)
                if d_drop <= 0.025 and abs(rem) <= 0.06:
                    break
                # Map error → base frame (x forward, y left)
                ex_b = math.cos(yaw) * ex + math.sin(yaw) * ey
                ey_b = -math.sin(yaw) * ex + math.cos(yaw) * ey
                # Shortfall (ex_b>0) → drive in; bias forward so less 区外
                vx = max(-0.10, min(0.16, 1.15 * ex_b))
                # Lateral: prefer yaw nudge; keep small if yaw already ok
                if abs(rem) > 0.08:
                    wz = max(-0.20, min(0.20, 1.2 * rem))
                    vx *= 0.35
                else:
                    wz = max(-0.18, min(0.18, 1.5 * ey_b))
                self.motion.publish_cmd_vel(vx, wz)
                time.sleep(0.05)
            self.motion.stop_chassis()
            # 3) Still short on approach → open-loop crawl into pad (no reverse)
            c2 = self.motion.get_base_xy_yaw_map()
            if c2 is not None:
                bx, by, yaw = float(c2[0]), float(c2[1]), float(c2[2])
                fx = bx + arm_m * math.cos(yaw)
                fy = by + arm_m * math.sin(yaw)
                ex_b = math.cos(yaw) * (float(zx) - fx) + math.sin(yaw) * (
                    float(zy) - fy
                )
                if ex_b > 0.03:
                    crawl = min(0.14, float(ex_b))
                    self._pub_status(
                        f"PLACE_CRAWL_IN +{crawl:.2f}m（补进，防区外）"
                    )
                    self.motion.drive_openloop_distance(
                        crawl, vx=0.18, timeout=1.2
                    )
                    self.motion.stop_chassis()
            c3 = self.motion.get_base_xy_yaw_map()
            if c3 is not None:
                yaw = float(c3[2])
                fx = float(c3[0]) + arm_m * math.cos(yaw)
                fy = float(c3[1]) + arm_m * math.sin(yaw)
                yaw_err = self.motion._wrap_pi(pyaw - yaw)
                self._pub_status(
                    f"PLACE_DOCK_OK yaw_err={yaw_err:+.2f} "
                    f"预测投放=({fx:.2f},{fy:.2f}) "
                    f"目标=({zx:.2f},{zy:.2f}) "
                    f"err={math.hypot(fx - zx, fy - zy):.3f}m"
                )
        finally:
            self.motion.stop_chassis()
            self.motion.exit_local_mode()

    def _leave_zone_after_place(self, zx: float, zy: float) -> None:
        """Straight reverse only — clear the pad so next approach won't crush cubes."""
        settle = float(self.get_parameter("local_settle_sec").value)
        self._pub_status("LEAVE_ZONE 直行倒退离开堆放区（不转向）→ 直接规划")
        self.motion.enter_local_mode(settle_sec=min(0.05, settle))
        try:
            # Leave pad (local reverse only — not Nav speed). Shorter + slightly faster.
            self.motion.drive_openloop_distance(-0.48, vx=0.70, timeout=1.1)
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
