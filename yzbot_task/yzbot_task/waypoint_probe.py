"""Visible waypoint probe: colored sphere + print map XYZ.

Use this to pick a doorway / via point when Nav goes crooked.

RViz:
  - Add display → MarkerArray → topic `/waypoint_probe/markers`
  - Toolbar → Publish Point → click map → terminal prints WAYPOINT (x,y,z)
    and the magenta ball jumps there.

Gazebo: tries to spawn/move model `wp_probe` (small magenta sphere).

Params:
  x,y,z  — initial probe pose (map)
  show_park_for — e.g. "red,1" to also draw green park sphere for that cube
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from gazebo_msgs.srv import SpawnEntity, SetEntityState, DeleteEntity
from gazebo_msgs.msg import EntityState


_SPHERE_SDF = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="wp_probe">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry><sphere><radius>0.12</radius></sphere></geometry>
        <material>
          <ambient>1 0 1 1</ambient>
          <diffuse>1 0 1 1</diffuse>
          <emissive>0.8 0 0.8 1</emissive>
        </material>
      </visual>
      <collision name="collision">
        <geometry><sphere><radius>0.01</radius></sphere></geometry>
      </collision>
    </link>
  </model>
</sdf>
"""


class WaypointProbe(Node):
    def __init__(self):
        super().__init__("waypoint_probe")
        self.declare_parameter("x", 0.5)
        self.declare_parameter("y", -2.5)
        self.declare_parameter("z", 0.15)
        self.declare_parameter("show_park_for", "red,1")
        self.declare_parameter("grasp_standoff_m", 0.23)

        self._x = float(self.get_parameter("x").value)
        self._y = float(self.get_parameter("y").value)
        self._z = float(self.get_parameter("z").value)
        self._gz_ready = False

        self._pub = self.create_publisher(MarkerArray, "/waypoint_probe/markers", 10)
        self.create_subscription(PointStamped, "/clicked_point", self._on_click, 10)

        self._spawn_cli = self.create_client(SpawnEntity, "/spawn_entity")
        self._spawn_cli2 = self.create_client(SpawnEntity, "/gazebo/spawn_entity")
        self._set_cli = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        self._del_cli = self.create_client(DeleteEntity, "/delete_entity")

        self.create_timer(0.5, self._tick)
        self.create_timer(2.0, self._try_spawn_gz)

        self.get_logger().info(
            f"waypoint_probe READY | magenta ball at ({self._x:.3f},{self._y:.3f},{self._z:.3f}) | "
            "RViz: MarkerArray=/waypoint_probe/markers + Publish Point → /clicked_point"
        )
        self._log_coords("INIT")

    def _log_coords(self, tag: str):
        self.get_logger().info(
            f"===== WAYPOINT {tag}  map  x={self._x:.4f}  y={self._y:.4f}  z={self._z:.4f} ====="
        )

    def _park_xy(self) -> Optional[Tuple[float, float, float]]:
        raw = str(self.get_parameter("show_park_for").value).strip()
        if not raw:
            return None
        try:
            color, cid_s = raw.split(",")
            cid = int(cid_s)
            from yzbot_task.vision_pose import approach_xy_for_cube

            so = float(self.get_parameter("grasp_standoff_m").value)
            ax, ay, ayaw = approach_xy_for_cube(color.strip(), cid, standoff_m=so)
            return ax, ay, ayaw
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"park pose skip: {exc}")
            return None

    def _sphere_marker(
        self, mid: int, x: float, y: float, z: float, rgba, scale: float, ns: str
    ) -> Marker:
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = mid
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = float(scale)
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        m.lifetime.sec = 0
        return m

    def _text_marker(self, mid: int, x: float, y: float, z: float, text: str) -> Marker:
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "wp_text"
        m.id = mid
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z) + 0.35
        m.pose.orientation.w = 1.0
        m.scale.z = 0.18
        m.color.r, m.color.g, m.color.b, m.color.a = (1.0, 1.0, 1.0, 1.0)
        m.text = text
        m.lifetime.sec = 0
        return m

    def _tick(self):
        arr = MarkerArray()
        arr.markers.append(
            self._sphere_marker(
                1, self._x, self._y, self._z, (1.0, 0.0, 1.0, 0.95), 0.28, "wp_probe"
            )
        )
        arr.markers.append(
            self._text_marker(
                2,
                self._x,
                self._y,
                self._z,
                f"WP ({self._x:.2f},{self._y:.2f})",
            )
        )
        park = self._park_xy()
        if park is not None:
            ax, ay, ayaw = park
            arr.markers.append(
                self._sphere_marker(
                    3, ax, ay, 0.12, (0.1, 1.0, 0.2, 0.9), 0.22, "park"
                )
            )
            arr.markers.append(
                self._text_marker(
                    4,
                    ax,
                    ay,
                    0.12,
                    f"PARK yaw={ayaw:.2f} ({ax:.2f},{ay:.2f})",
                )
            )
            # yaw arrow
            arrow = Marker()
            arrow.header.frame_id = "map"
            arrow.header.stamp = self.get_clock().now().to_msg()
            arrow.ns = "park_yaw"
            arrow.id = 5
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose.position.x = ax
            arrow.pose.position.y = ay
            arrow.pose.position.z = 0.12
            arrow.pose.orientation.z = math.sin(ayaw / 2.0)
            arrow.pose.orientation.w = math.cos(ayaw / 2.0)
            arrow.scale.x = 0.45
            arrow.scale.y = 0.06
            arrow.scale.z = 0.06
            arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = (
                0.1,
                1.0,
                0.2,
                1.0,
            )
            arr.markers.append(arrow)
        self._pub.publish(arr)

    def _on_click(self, msg: PointStamped):
        self._x = float(msg.point.x)
        self._y = float(msg.point.y)
        self._z = max(float(msg.point.z), 0.12)
        self._log_coords("CLICK")
        self._move_gz()

    def _spawn_client(self):
        if self._spawn_cli.service_is_ready():
            return self._spawn_cli
        if self._spawn_cli2.service_is_ready():
            return self._spawn_cli2
        return None

    def _try_spawn_gz(self):
        if self._gz_ready:
            return
        cli = self._spawn_client()
        if cli is None:
            self._spawn_cli.wait_for_service(timeout_sec=0.0)
            self._spawn_cli2.wait_for_service(timeout_sec=0.0)
            return
        req = SpawnEntity.Request()
        req.name = "wp_probe"
        req.xml = _SPHERE_SDF
        req.robot_namespace = ""
        req.initial_pose.position.x = self._x
        req.initial_pose.position.y = self._y
        req.initial_pose.position.z = self._z
        req.initial_pose.orientation.w = 1.0
        req.reference_frame = "world"
        fut = cli.call_async(req)

        def _done(f):
            try:
                res = f.result()
                ok = bool(res and getattr(res, "success", False))
                self._gz_ready = ok or "already" in str(
                    getattr(res, "status_message", "")
                ).lower()
                self.get_logger().info(
                    f"gazebo wp_probe spawn: success={ok} "
                    f"msg={getattr(res, 'status_message', '')}"
                )
                if self._gz_ready:
                    self._move_gz()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"spawn wp_probe: {exc}")

        fut.add_done_callback(_done)

    def _move_gz(self):
        if not self._set_cli.service_is_ready():
            return
        req = SetEntityState.Request()
        st = EntityState()
        st.name = "wp_probe"
        st.pose.position.x = self._x
        st.pose.position.y = self._y
        st.pose.position.z = self._z
        st.pose.orientation.w = 1.0
        st.reference_frame = "world"
        req.state = st
        self._set_cli.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointProbe()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
