"""Deprecated: dyn obstacles are moved by Gazebo plugin libdyn_obs_mover.so.

Running this node fights the plugin (gz CLI) and can make obstacles look frozen.
"""

import sys

import rclpy
from rclpy.node import Node


def main(args=None):
    rclpy.init(args=args)
    node = Node("dyn_obstacles")
    node.get_logger().error(
        "dyn_obstacles ROS node is DISABLED. "
        "Motion comes from Gazebo plugin libdyn_obs_mover.so "
        "(gazebo_world2 sets GAZEBO_PLUGIN_PATH). "
        "Do not ros2 run this — it fights the plugin. Exiting."
    )
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
