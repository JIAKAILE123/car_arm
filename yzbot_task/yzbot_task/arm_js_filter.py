"""Republish /joint_states with only arm + gripper joints for MoveIt."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

ARM_JOINTS = {
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "finger_joint1",
    "finger_joint2",
}


class ArmJsFilter(Node):
    def __init__(self):
        super().__init__("arm_js_filter")
        self.pub = self.create_publisher(JointState, "/yzbot/arm_joint_states", 10)
        self.create_subscription(JointState, "/joint_states", self._cb, 50)
        self.get_logger().info(
            "arm_js_filter: /joint_states → /yzbot/arm_joint_states (no wheels)"
        )

    def _cb(self, msg: JointState):
        out = JointState()
        out.header = msg.header
        for i, name in enumerate(msg.name):
            if name not in ARM_JOINTS:
                continue
            out.name.append(name)
            if i < len(msg.position):
                out.position.append(msg.position[i])
            if i < len(msg.velocity):
                out.velocity.append(msg.velocity[i])
            if i < len(msg.effort):
                out.effort.append(msg.effort[i])
        if out.name:
            self.pub.publish(out)


def main():
    rclpy.init()
    node = ArmJsFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
