"""Dynamic-obstacle dodge guard (speed chain only).

Path:
    velocity_smoother -> /cmd_vel_smoothed
        -> dyn_obs_guard -> /cmd_vel_guard
        -> collision_monitor -> /cmd_vel

Pass-through by default, so navigation is unchanged. When the Gazebo ground
truth says a moving bar is about to reach the base, the guard takes over and
drives the base *across* the bar's direction of travel (perpendicular escape),
not along its own heading — running forward/back along the corridor keeps the
base on the bar's sweep line and just gets pushed into a wall.

Also contains a small stuck detector: if the base has a velocity command but
has not moved for ~2 s (wedged in a wall after a hit), it reverses briefly to
break free before handing control back to Nav2.
"""

from __future__ import annotations

import math
import os
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener


# Must match offic_room.world: all three bars are 0.385 x 0.28 boxes.
_OBS_RADIUS = math.hypot(0.385 / 2.0, 0.28 / 2.0)
# Base footprint is 0.76 x 0.80 -> circumscribed radius.
_ROBOT_RADIUS = math.hypot(0.76, 0.80) / 2.0


class DynObsGuard(Node):
    def __init__(self) -> None:
        super().__init__("dyn_obs_guard")
        self.declare_parameter("detect_radius_m", 3.00)
        self.declare_parameter("ttc_trigger_sec", 1.50)
        self.declare_parameter("escape_speed", 0.25)
        self.declare_parameter("escape_turn", 0.00)
        self.declare_parameter("max_dodge_sec", 1.00)
        self.declare_parameter("cooldown_sec", 0.35)
        self.declare_parameter("cmd_timeout_sec", 0.40)
        self.declare_parameter("stuck_moved_m", 0.06)
        self.declare_parameter("stuck_window_sec", 2.0)
        self.declare_parameter("stuck_back_speed", 0.30)
        self.declare_parameter("stuck_back_sec", 0.65)
        self.declare_parameter("stuck_cooldown_sec", 2.50)
        self.declare_parameter(
            "trace_path",
            os.path.expanduser("~/dev_ws/yzbot/log/dyn_obs_guard_trace.log"),
        )

        self._pub = self.create_publisher(Twist, "cmd_vel_guard", 10)
        self.create_subscription(Twist, "cmd_vel_smoothed", self._on_cmd, 10)
        for qos in (qos_profile_sensor_data, 10):
            self.create_subscription(
                ModelStates, "/gazebo/model_states", self._on_models, qos
            )
        self.create_subscription(
            LaserScan, "/scan", self._on_scan, qos_profile_sensor_data
        )
        # The lidar has range_min=0.5 m, so anything closer is invisible to it.
        # Use the global costmap (static walls / furniture / scan marks) to see
        # what the laser is blind to, otherwise the escape path drives into it.
        self.create_subscription(
            OccupancyGrid, "/global_costmap/costmap", self._on_costmap, 10
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(0.05, self._tick)

        self._scan: Optional[LaserScan] = None
        self._costmap: Optional[OccupancyGrid] = None
        self._cmd: Optional[Twist] = None
        self._cmd_t = 0.0
        self._last_sent_vx = 0.0
        self._obs: Dict[str, Tuple[float, float]] = {}
        self._obs_prev: Dict[str, Tuple[float, float]] = {}
        self._obs_t = 0.0
        self._obs_prev_t = 0.0

        self._dodging = False
        self._dodge_from = 0.0
        self._dodge_vx = 0.0
        self._dodge_wz = 0.0
        self._cooldown_until = 0.0

        self._stuck_until = 0.0
        self._stuck_vx = 0.0
        self._stuck_cooldown_until = 0.0
        self._hist: Deque[Tuple[float, float, float]] = deque(maxlen=60)
        self._trace_path = str(self.get_parameter("trace_path").value)

        self.get_logger().info(
            "dyn_obs_guard ready — passthrough, perpendicular dodge, stuck release"
        )

    def _trace(self, text: str) -> None:
        """Append one diagnostic line; ROS node output is not persisted by default."""
        path = self._trace_path
        if not path:
            return
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(f"{self._now():.3f} {text}\n")
        except Exception:  # noqa: BLE001
            self._trace_path = ""

    def _now(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    # ------------------------------------------------------------------ input
    def _on_cmd(self, msg: Twist) -> None:
        self._cmd = msg
        self._cmd_t = self._now()
        if not self._dodging and self._stuck_until <= self._now():
            self._send(msg.linear.x, msg.angular.z)

    def _on_models(self, msg: ModelStates) -> None:
        t = self._now()
        if self._obs_t > 0.0 and (t - self._obs_t) < 0.02:
            return
        cur: Dict[str, Tuple[float, float]] = {}
        for name, pose in zip(msg.name, msg.pose):
            if name.startswith("dyn_obs_"):
                cur[name] = (float(pose.position.x), float(pose.position.y))
        if not cur:
            return
        self._obs_prev = self._obs
        self._obs_prev_t = self._obs_t
        self._obs = cur
        self._obs_t = t

    def _on_scan(self, msg: LaserScan) -> None:
        self._scan = msg

    # ------------------------------------------------------------------ utils
    def _send(self, vx: float, wz: float = 0.0) -> None:
        tw = Twist()
        tw.linear.x = float(vx)
        tw.angular.z = float(wz)
        self._last_sent_vx = float(vx)
        self._pub.publish(tw)

    def _on_costmap(self, msg: OccupancyGrid) -> None:
        self._costmap = msg

    def _world_blocked(self, x: float, y: float) -> bool:
        cm = self._costmap
        if cm is None:
            return False
        info = cm.info
        res = float(info.resolution) or 0.05
        mx = int((x - info.origin.position.x) / res)
        my = int((y - info.origin.position.y) / res)
        if mx < 0 or my < 0 or mx >= info.width or my >= info.height:
            return False
        idx = my * info.width + mx
        if idx >= len(cm.data):
            return False
        val = int(cm.data[idx])
        return val >= 90

    def _dir_blocked(
        self, x: float, y: float, heading: float, dist: float, half_w: float = 0.28
    ) -> bool:
        cx = x + dist * math.cos(heading)
        cy = y + dist * math.sin(heading)
        nx, ny = -math.sin(heading), math.cos(heading)
        for off in (-half_w, 0.0, half_w):
            if self._world_blocked(cx + off * nx, cy + off * ny):
                return True
        return False

    def _side_blocked(
        self, x: float, y: float, yaw: float, direction_rad: float, dist: float = 0.55
    ) -> bool:
        """True if moving in body-frame direction_rad is unsafe.

        0 rad = ahead, pi = behind. Costmap covers the <0.5 m blind zone of the
        lidar; the laser still contributes for the 0.5-0.9 m band.
        """
        heading = yaw + float(direction_rad)
        if self._costmap is not None and self._dir_blocked(x, y, heading, dist):
            return True
        clear = self._sector_clear(float(direction_rad), math.radians(45), 0.90)
        return clear < 0.55

    def _sector_clear(self, center_rad: float, half_rad: float, limit: float) -> float:
        """Smallest laser range inside a body-frame sector (0 rad = ahead)."""
        s = self._scan
        if s is None or not s.ranges:
            return limit
        best = limit
        inc = float(s.angle_increment)
        a = float(s.angle_min)
        for r in s.ranges:
            ang = a
            a += inc
            if not math.isfinite(r) or r <= 0.02 or r >= best:
                continue
            d = abs(ang - center_rad)
            if d > math.pi:
                d = 2.0 * math.pi - d
            if d <= half_rad:
                best = float(r)
        return best

    def _robot_pose(self) -> Optional[Tuple[float, float, float]]:
        for child in ("base_footprint", "base_link"):
            try:
                tf = self.tf_buffer.lookup_transform(
                    "map", child, rclpy.time.Time(), timeout=Duration(seconds=0.05)
                )
            except Exception:  # noqa: BLE001
                continue
            tr = tf.transform.translation
            q = tf.transform.rotation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            return float(tr.x), float(tr.y), math.atan2(siny, cosy)
        return None

    def _obs_vel(self, name: str) -> Tuple[float, float]:
        dt = self._obs_t - self._obs_prev_t
        if dt <= 1e-3 or name not in self._obs or name not in self._obs_prev:
            return 0.0, 0.0
        a = self._obs[name]
        b = self._obs_prev[name]
        return (a[0] - b[0]) / dt, (a[1] - b[1]) / dt

    # ------------------------------------------------------------------ threat
    def _closest_threat(
        self, rx: float, ry: float, yaw: float
    ) -> Optional[Tuple[float, str, float]]:
        """Return (time_to_hit, obs_name, current_distance) or None.

        Uses straight-line extrapolation of both the bar and the base. A bar
        sweeping *across* the corridor has ~zero instantaneous closing speed,
        so an instantaneous TTC never fires — but its future sweep does cross
        the footprint, which is exactly the hit we keep seeing.
        """
        reach = float(self.get_parameter("detect_radius_m").value)
        horizon = float(self.get_parameter("ttc_trigger_sec").value)
        v_self = self._last_sent_vx
        rvx = v_self * math.cos(yaw)
        rvy = v_self * math.sin(yaw)
        safe = _ROBOT_RADIUS + _OBS_RADIUS + 0.05
        steps = max(1, int(horizon / 0.10))
        best: Optional[Tuple[float, str, float]] = None
        for name, (ox, oy) in self._obs.items():
            d_now = math.hypot(ox - rx, oy - ry)
            if d_now > reach:
                continue
            vx, vy = self._obs_vel(name)
            hit_t: Optional[float] = None
            for i in range(1, steps + 1):
                t = 0.10 * i
                bx = rx + rvx * t
                by = ry + rvy * t
                d = math.hypot(ox + vx * t - bx, oy + vy * t - by)
                if d <= safe:
                    hit_t = t
                    break
            if hit_t is not None and (best is None or hit_t < best[0]):
                best = (hit_t, name, d_now)
        return best

    def _escape_cmd(
        self, rx: float, ry: float, yaw: float, name: str, ttc: float
    ) -> Tuple[float, float]:
        """Bar coming: just ease away at normal driving speed, no turning.

        The previous version swung the base at 0.8 rad/s and drove it in an
        arc — it yawed hard enough to tip over. Keep the escape purely
        longitudinal and gentle: reverse if the bar is in front, creep forward
        if it is behind, and never rotate.
        """
        del ttc
        ox, oy = self._obs[name]
        px, py = ox - rx, oy - ry
        speed = float(self.get_parameter("escape_speed").value)
        back_blocked = self._side_blocked(rx, ry, yaw, math.pi, 0.55)
        front_blocked = self._side_blocked(rx, ry, yaw, 0.0, 0.55)
        along = px * math.cos(yaw) + py * math.sin(yaw)
        if along >= 0.0:
            # Bar ahead: back off, but only if there is room behind.
            if not back_blocked:
                return -speed, 0.0
            if not front_blocked:
                return speed, 0.0
            return 0.0, 0.0
        # Bar behind: ease forward.
        if not front_blocked:
            return speed, 0.0
        if not back_blocked:
            return -speed, 0.0
        return 0.0, 0.0

    @staticmethod
    def _wrap(a: float) -> float:
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    # ------------------------------------------------------------------ stuck
    def _stuck_release(self, now: float, pose: Tuple[float, float, float]):
        """Return a reverse vx when the base is commanded but not moving."""
        if now < self._stuck_cooldown_until:
            return None
        self._hist.append((now, pose[0], pose[1]))
        window = float(self.get_parameter("stuck_window_sec").value)
        while self._hist and (now - self._hist[0][0]) > window:
            self._hist.popleft()
        if len(self._hist) < 15:
            return None
        t0, x0, y0 = self._hist[0]
        if (now - t0) < window * 0.8:
            return None
        moved = math.hypot(pose[0] - x0, pose[1] - y0)
        if moved >= float(self.get_parameter("stuck_moved_m").value):
            return None
        if abs(self._last_sent_vx) < 0.05:
            return None
        # Commanded motion but no displacement -> wedged. Back out the other way.
        return -float(self.get_parameter("stuck_back_speed").value) * (
            1.0 if self._last_sent_vx > 0.0 else -1.0
        )

    # ------------------------------------------------------------------- tick
    def _tick(self) -> None:
        now = self._now()
        trigger = float(self.get_parameter("ttc_trigger_sec").value)
        max_dodge = float(self.get_parameter("max_dodge_sec").value)
        timeout = float(self.get_parameter("cmd_timeout_sec").value)

        if self._stuck_until > now:
            self._send(self._stuck_vx, 0.0)
            return

        pose = self._robot_pose()
        if pose is not None:
            rel = self._stuck_release(now, pose)
            if rel is not None and rel != 0.0:
                # Do not back into a wall while trying to escape one.
                if rel < 0.0:
                    rear_blocked = self._side_blocked(
                        pose[0], pose[1], pose[2], math.pi, 0.40
                    )
                    front_blocked = self._side_blocked(
                        pose[0], pose[1], pose[2], 0.0, 0.40
                    )
                    if rear_blocked:
                        rel = abs(rel) if not front_blocked else 0.0
                else:
                    front_blocked = self._side_blocked(
                        pose[0], pose[1], pose[2], 0.0, 0.40
                    )
                    rear_blocked = self._side_blocked(
                        pose[0], pose[1], pose[2], math.pi, 0.40
                    )
                    if front_blocked:
                        rel = -abs(rel) if not rear_blocked else 0.0
            if rel is not None and rel != 0.0:
                back_sec = float(self.get_parameter("stuck_back_sec").value)
                self._stuck_until = now + back_sec
                self._stuck_vx = rel
                self._stuck_cooldown_until = (
                    now
                    + back_sec
                    + float(self.get_parameter("stuck_cooldown_sec").value)
                )
                self._hist.clear()
                self._trace(f"STUCK_RELEASE vx={rel:+.2f} pose=({pose[0]:.2f},{pose[1]:.2f})")
                self.get_logger().warn(f"STUCK_RELEASE vx={rel:+.2f}")
                self._send(rel, 0.0)
                return

        stale = self._cmd is None or (now - self._cmd_t) > timeout
        if stale and not self._dodging:
            self._send(0.0, 0.0)
            return

        if pose is None:
            if not self._dodging:
                self._send(self._cmd.linear.x, self._cmd.angular.z)
            return

        threat = self._closest_threat(pose[0], pose[1], pose[2])

        if self._dodging:
            gone = threat is None or threat[0] > trigger * 1.8
            # "Hold still and let it pass" must not be cut short by the normal
            # dodge timeout, otherwise the base lurches out again and gets hit.
            parked = abs(self._dodge_vx) < 1e-3 and abs(self._dodge_wz) < 1e-3
            limit = max_dodge * 3.0 if parked else max_dodge
            if (now - self._dodge_from) > limit or gone:
                self._dodging = False
                self._cooldown_until = now + float(
                    self.get_parameter("cooldown_sec").value
                )
            else:
                vx = self._dodge_vx
                if vx < 0.0 and self._side_blocked(
                    pose[0], pose[1], pose[2], math.pi, 0.45
                ):
                    vx = 0.0
                elif vx > 0.0 and self._side_blocked(
                    pose[0], pose[1], pose[2], 0.0, 0.45
                ):
                    vx = 0.0
                if vx == 0.0 and self._dodge_vx != 0.0:
                    self._trace("DYN_PARK both sides blocked -> hold")
                self._send(vx, 0.0)
                return

        if now < self._cooldown_until:
            self._send(self._cmd.linear.x, self._cmd.angular.z)
            return

        if threat is not None and threat[0] < trigger:
            ttc, name, d = threat
            vx, wz = self._escape_cmd(pose[0], pose[1], pose[2], name, ttc)
            ox, oy = self._obs[name]
            along = (ox - pose[0]) * math.cos(pose[2]) + (oy - pose[1]) * math.sin(
                pose[2]
            )
            self._trace(
                f"DYN_DODGE {name} ttc={ttc:.2f} d={d:.2f} "
                f"along={along:+.2f} -> vx={vx:+.2f} wz={wz:+.2f}"
            )
            self._dodging = True
            self._dodge_from = now
            self._dodge_vx, self._dodge_wz = vx, wz
            self._send(vx, wz)
            self.get_logger().warn(
                f"DYN_DODGE {name} ttc={ttc:.2f}s d={d:.2f}m → "
                f"vx={vx:+.2f} wz={wz:+.2f}"
            )
            return

        self._send(self._cmd.linear.x, self._cmd.angular.z)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DynObsGuard()
    executor = MultiThreadedExecutor(num_threads=2)
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
