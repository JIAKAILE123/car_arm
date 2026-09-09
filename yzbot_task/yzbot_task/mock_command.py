"""Publish a mock structured command (no API)."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

DEFAULT_COMMAND = "[red,1,A]"


class MockCommand(Node):
    def __init__(self):
        super().__init__("mock_command")
        self.declare_parameter("command", DEFAULT_COMMAND)
        self.cmd = str(self.get_parameter("command").value)
        self.pub = self.create_publisher(String, "/task_command", 10)
        self.timer = self.create_timer(0.5, self._tick)
        self.sent = 0

    def _tick(self):
        msg = String()
        msg.data = self.cmd
        self.pub.publish(msg)
        self.sent += 1
        self.get_logger().info(f"published /task_command: {self.cmd} ({self.sent})")
        if self.sent >= 5:
            self.timer.cancel()
            # exit cleanly
            raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    node = MockCommand()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
