"""Subscribe /exam_question → DeepSeek (or local) solve → /task_command.

Also publishes visualization-friendly topics for the 10pt LLM panel score:
  /exam_question_echo  — last question text
  /llm_xy              — e.g. x=4,y=1
  /task_info           — human summary including mapping result
"""

from __future__ import annotations

import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from yzbot_task.llm_solver import question_to_command_llm


class LlmSolverNode(Node):
    def __init__(self):
        super().__init__("llm_solver")
        self.declare_parameter("backend", "cross")  # deepseek|local|auto|cross
        self.declare_parameter("api_key", "")  # empty → env / ~/.config
        self.declare_parameter("base_url", "https://api.deepseek.com")
        self.declare_parameter("model", "deepseek-chat")
        self.declare_parameter("timeout_sec", 30.0)
        self.declare_parameter("x_color", "red")
        self.declare_parameter("y_color", "blue")
        self.declare_parameter("threshold", 3)
        self.declare_parameter("high_zone", "A")
        self.declare_parameter("low_zone", "C")

        self.pub_cmd = self.create_publisher(String, "/task_command", 10)
        self.pub_status = self.create_publisher(String, "/task_status", 10)
        self.pub_info = self.create_publisher(String, "/task_info", 10)
        self.pub_xy = self.create_publisher(String, "/llm_xy", 10)
        self.pub_q = self.create_publisher(String, "/exam_question_echo", 10)

        self.create_subscription(String, "/exam_question", self._on_question, 10)
        # coStudio「粘贴题目解题」面板也可发到此话题（与 /exam_question 同处理）
        self.create_subscription(String, "/panel_exam_question", self._on_question, 10)
        backend = self.get_parameter("backend").value
        self.get_logger().info(
            f"llm_solver ready backend={backend} "
            f"(/exam_question|/panel_exam_question → 大模型 → /task_command)"
        )

    def _mapping(self) -> dict:
        return {
            "x_color": self.get_parameter("x_color").value,
            "y_color": self.get_parameter("y_color").value,
            "threshold": self.get_parameter("threshold").value,
            "high_zone": self.get_parameter("high_zone").value,
            "low_zone": self.get_parameter("low_zone").value,
        }

    def _on_question(self, msg: String):
        q = (msg.data or "").strip()
        if not q:
            return
        echo = String()
        echo.data = q
        self.pub_q.publish(echo)

        api_key = str(self.get_parameter("api_key").value or "").strip()
        if not api_key:
            api_key = os.environ.get("DEEPSEEK_API_KEY", "") or None

        try:
            cmd, x, y = question_to_command_llm(
                q,
                self._mapping(),
                backend=str(self.get_parameter("backend").value),
                api_key=api_key,
                base_url=str(self.get_parameter("base_url").value),
                model=str(self.get_parameter("model").value),
                timeout_sec=float(self.get_parameter("timeout_sec").value),
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"LLM solve failed: {exc}")
            err = String()
            err.data = f"LLM_ERROR: {exc}"
            self.pub_status.publish(err)
            return

        out = String()
        out.data = cmd
        self.pub_cmd.publish(out)

        xy = String()
        xy.data = f"x={x},y={y}"
        self.pub_xy.publish(xy)

        info = String()
        info.data = (
            f"question={q[:120]} | xy=({x},{y}) | "
            f"map={self._mapping()} | command={cmd}"
        )
        self.pub_info.publish(info)

        st = String()
        st.data = f"LLM_SOLVED: {cmd} (x={x},y={y})"
        self.pub_status.publish(st)
        self.get_logger().info(f"LLM solved x={x} y={y} → {cmd}")


def main(args=None):
    rclpy.init(args=args)
    node = LlmSolverNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
