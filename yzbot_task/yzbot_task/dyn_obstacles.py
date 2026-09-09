"""Move dyn_obs_1 / dyn_obs_2 along patrol paths (Gazebo set_entity_state).

Obstacles sit on the real pick/place corridors so Nav2 must replan.
Attach / LinkAttacher is unchanged — this node only moves the orange/cyan blockers.
"""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Pose, Twist


class DynObstacles(Node):
    def __init__(self):
        super().__init__("dyn_obstacles")
        self.declare_parameter("rate_hz", 10.0)
        self.cli = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        if not self.cli.wait_for_service(timeout_sec=20.0):
            self.get_logger().error("/gazebo/set_entity_state missing — start gazebo_world2")
        self.t0 = self.get_clock().now()
        period = 1.0 / max(float(self.get_parameter("rate_hz").value), 1.0)
        self.create_timer(period, self._tick)
        self.get_logger().info(
            "dyn_obstacles: dyn_obs_1 patrols pick corridor; dyn_obs_2 patrols place corridor"
        )

    def _elapsed(self) -> float:
        return (self.get_clock().now() - self.t0).nanoseconds * 1e-9

    def _set(self, name: str, x: float, y: float, z: float = 0.45, yaw: float = 0.0):
        if not self.cli.service_is_ready():
            return
        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)
        req = SetEntityState.Request()
        st = EntityState()
        st.name = name
        st.pose = Pose()
        st.pose.position.x = float(x)
        st.pose.position.y = float(y)
        st.pose.position.z = float(z)
        st.pose.orientation.z = float(qz)
        st.pose.orientation.w = float(qw)
        st.twist = Twist()
        st.reference_frame = "world"
        req.state = st
        self.cli.call_async(req)

    def _tick(self):
        t = self._elapsed()
        # obs1: on path from origin toward red/blue cargo (must be "effective")
        y1 = 2.2 * math.sin(2.0 * math.pi * t / 14.0)
        self._set("dyn_obs_1", 1.0, y1, 0.45)
        # obs2: on path from mid-map down to place zones A/B/C
        x2 = 3.2 + 1.0 * math.sin(2.0 * math.pi * t / 16.0)
        y2 = -4.5 - 2.5 * abs(math.sin(2.0 * math.pi * t / 16.0))
        self._set("dyn_obs_2", x2, y2, 0.45)


def main(args=None):
    rclpy.init(args=args)
    node = DynObstacles()
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
