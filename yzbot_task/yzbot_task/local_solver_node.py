"""Subscribe exam question text → local solve → publish /task_command."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from yzbot_task.local_solver import question_to_command


class LocalSolverNode(Node):
    def __init__(self):
        super().__init__("local_solver")
        self.declare_parameter("x_color", "red")
        self.declare_parameter("y_color", "blue")
        self.declare_parameter("threshold", 3)
        self.declare_parameter("high_zone", "A")
        self.declare_parameter("low_zone", "C")

        self.pub = self.create_publisher(String, "/task_command", 10)
        self.status_pub = self.create_publisher(String, "/task_status", 10)
        self.create_subscription(String, "/exam_question", self._on_question, 10)
        self.get_logger().info("local_solver ready: /exam_question → /task_command")

    def _on_question(self, msg: String):
        mapping = {
            "x_color": self.get_parameter("x_color").value,
            "y_color": self.get_parameter("y_color").value,
            "threshold": self.get_parameter("threshold").value,
            "high_zone": self.get_parameter("high_zone").value,
            "low_zone": self.get_parameter("low_zone").value,
        }
        try:
            cmd = question_to_command(msg.data, mapping)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"solve failed: {exc}")
            err = String()
            err.data = f"ERROR: {exc}"
            self.status_pub.publish(err)
            return

        out = String()
        out.data = cmd
        self.pub.publish(out)
        st = String()
        st.data = f"SOLVED: {cmd}"
        self.status_pub.publish(st)
        self.get_logger().info(f"solved → {cmd}")


def main(args=None):
    rclpy.init(args=args)
    node = LocalSolverNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
