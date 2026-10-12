"""Official exam binary → /exam_question.

默认：终端里按【空格】才出题（可多次）。
调试自动出题：ros2 run yzbot_task exam_feeder --ros-args -p auto_once:=true
"""

from __future__ import annotations

import os
import select
import sys
import termios
import tty
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
        # False = 按空格出题；True = 启动后自动出一题（调试用）
        self.declare_parameter("auto_once", False)
        self.declare_parameter("delay_sec", 1.0)

        self.pub = self.create_publisher(String, "/exam_question", 10)
        self.status_pub = self.create_publisher(String, "/task_status", 10)
        self._auto_fired = False

        if bool(self.get_parameter("auto_once").value):
            delay = float(self.get_parameter("delay_sec").value)
            self.create_timer(max(0.2, delay), self._auto_once_cb)
            self.get_logger().info("exam_feeder: auto_once=true，将自动出一题")
        else:
            self.get_logger().info("exam_feeder: 按【空格】出题（可多次）")

    def _auto_once_cb(self):
        if self._auto_fired:
            return
        self._auto_fired = True
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
        line = next((ln.strip() for ln in text.splitlines() if ln.strip()), text)
        msg = String()
        msg.data = line
        self.pub.publish(msg)
        st = String()
        st.data = f"EXAM_Q: {line[:160]}"
        self.status_pub.publish(st)
        self.get_logger().info(f"published /exam_question: {line[:120]}...")
        return True


def _wait_space() -> bool:
    """Block until Space (or Enter). Return False on Ctrl+C / EOF."""
    fd = sys.stdin.fileno()
    if not os.isatty(fd):
        # 非交互：读一行
        try:
            line = sys.stdin.readline()
        except Exception:
            return False
        return bool(line)

    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            r, _, _ = select.select([fd], [], [], 0.25)
            if not r:
                continue
            ch = os.read(fd, 1)
            if not ch:
                return False
            # Space or Enter
            if ch in (b" ", b"\r", b"\n"):
                return True
            # Ctrl+C
            if ch == b"\x03":
                return False
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main(args=None):
    rclpy.init(args=args)
    node = ExamFeederNode()
    auto = bool(node.get_parameter("auto_once").value)
    try:
        if auto:
            for _ in range(50):
                rclpy.spin_once(node, timeout_sec=0.1)
                if node._auto_fired:
                    rclpy.spin_once(node, timeout_sec=0.2)
                    rclpy.spin_once(node, timeout_sec=0.2)
                    break
        else:
            print()
            print("========================================")
            print("  出题终端：按【空格】出一题（可多次）")
            print("  Ctrl+C 退出")
            print("========================================")
            print()
            while rclpy.ok():
                # 让 ROS 回调有机会跑
                rclpy.spin_once(node, timeout_sec=0.05)
                sys.stdout.write("\r等待空格出题 …  ")
                sys.stdout.flush()
                if not _wait_space():
                    break
                print("\n→ 正在出题 …")
                ok = node.feed_once()
                for _ in range(5):
                    rclpy.spin_once(node, timeout_sec=0.1)
                print("→ 出题完成\n" if ok else "→ 出题失败\n")
    except KeyboardInterrupt:
        print("\n已退出出题")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
