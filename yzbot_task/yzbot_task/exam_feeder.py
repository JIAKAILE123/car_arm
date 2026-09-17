"""Run official exam binary once and publish stdout to /exam_question."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

_DEFAULT_BIN = str(
    Path.home()
    / "桌面"
    / "全球校园人工智能算法精英大赛 大模型技术创新赛 比赛练习平台"
    / "2026"
    / "题目生成器"
    / "TMSCQtest_x86_x64.bin"
)


class ExamFeederNode(Node):
    def __init__(self):
        super().__init__("exam_feeder")
        self.declare_parameter("exam_bin", _DEFAULT_BIN)
        self.declare_parameter("auto_once", True)
        self.declare_parameter("delay_sec", 1.0)

        self.pub = self.create_publisher(String, "/exam_question", 10)
        self.status_pub = self.create_publisher(String, "/task_status", 10)

        if bool(self.get_parameter("auto_once").value):
            delay = float(self.get_parameter("delay_sec").value)
            self.create_timer(max(0.2, delay), self._fire_once)
            self._fired = False
        else:
            self._fired = True  # wait for service-style: use ros2 param trigger later

        self.get_logger().info(
            "exam_feeder ready — will run official TMSCQtest and pub /exam_question"
        )

    def _fire_once(self):
        if self._fired:
            return
        self._fired = True
        self.feed_once()

    def feed_once(self) -> bool:
        bin_path = str(self.get_parameter("exam_bin").value)
        if not os.path.isfile(bin_path):
            self.get_logger().error(f"exam_bin not found: {bin_path}")
            return False
        try:
            os.chmod(bin_path, 0o755)
        except OSError:
            pass
        try:
            proc = subprocess.run(
                [bin_path],
                capture_output=True,
                text=True,
                timeout=10.0,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"exam_bin failed: {exc}")
            return False
        text = (proc.stdout or "").strip()
        if not text:
            text = (proc.stderr or "").strip()
        if not text:
            self.get_logger().error("exam_bin produced empty output")
            return False
        # Keep first non-empty line (official printer is one line)
        line = next((ln.strip() for ln in text.splitlines() if ln.strip()), text)
        msg = String()
        msg.data = line
        self.pub.publish(msg)
        st = String()
        st.data = f"EXAM_Q: {line[:160]}"
        self.status_pub.publish(st)
        self.get_logger().info(f"published /exam_question: {line[:120]}...")
        return True


def main(args=None):
    rclpy.init(args=args)
    node = ExamFeederNode()
    try:
        # Spin briefly so the one-shot timer can fire and pubs flush
        for _ in range(30):
            rclpy.spin_once(node, timeout_sec=0.1)
            if node._fired:
                # allow a couple more cycles for publish
                rclpy.spin_once(node, timeout_sec=0.2)
                rclpy.spin_once(node, timeout_sec=0.2)
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
