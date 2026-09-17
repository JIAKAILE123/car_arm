"""Periodically clear local costmap ghosts left by moving dyn_obs.

Laser raytrace alone often leaves trails behind kinematic sliders; a light
local clear keeps Nav2 from treating residue as real walls.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty


class CostmapGhostClear(Node):
    def __init__(self):
        super().__init__("costmap_ghost_clear")
        self.declare_parameter("period_sec", 1.0)
        self.declare_parameter("clear_local", True)
        self.declare_parameter("clear_global", False)

        self._local = self.create_client(
            Empty, "/local_costmap/clear_entirely_local_costmap"
        )
        self._global = self.create_client(
            Empty, "/global_costmap/clear_entirely_global_costmap"
        )
        period = max(0.3, float(self.get_parameter("period_sec").value))
        self.create_timer(period, self._tick)
        self.get_logger().info(
            f"costmap_ghost_clear every {period:.1f}s "
            f"(local={bool(self.get_parameter('clear_local').value)}, "
            f"global={bool(self.get_parameter('clear_global').value)})"
        )

    def _tick(self):
        if bool(self.get_parameter("clear_local").value):
            if self._local.service_is_ready():
                self._local.call_async(Empty.Request())
            else:
                self._local.wait_for_service(timeout_sec=0.0)
        if bool(self.get_parameter("clear_global").value):
            if self._global.service_is_ready():
                self._global.call_async(Empty.Request())
            else:
                self._global.wait_for_service(timeout_sec=0.0)


def main(args=None):
    rclpy.init(args=args)
    node = CostmapGhostClear()
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
