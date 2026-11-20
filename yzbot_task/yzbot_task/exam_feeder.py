"""Official exam binary → /exam_question.

终端打开后自己输入参数再回车。
比赛模式参数 11470507：题目按 30 分钟一档，同一档内不变。
直接回车：练习模式，每次随机一题。
"""

from __future__ import annotations

import os
import sys
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

_CONTEST_ARG = "11470507"


class ExamFeederNode(Node):
    def __init__(self):
        super().__init__("exam_feeder")
        self.declare_parameter("exam_bin", _DEFAULT_BIN)

        self.pub = self.create_publisher(String, "/exam_question", 10)
        self.status_pub = self.create_publisher(String, "/task_status", 10)

    def feed_once(self, exam_arg: str = "") -> bool:
        bin_path = str(self.get_parameter("exam_bin").value)
        if not os.path.isfile(bin_path):
            self.get_logger().error(f"exam_bin not found: {bin_path}")
            return False
        try:
            os.chmod(bin_path, 0o755)
        except OSError:
            pass
        cmd = [bin_path]
        arg = str(exam_arg).strip()
        if arg:
            cmd.append(arg)
        try:
            proc = subprocess.run(
                cmd,
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
        if text:
            print(text)
            print()
        line = _question_line(text)
        if not line:
            self.get_logger().error("exam_bin produced empty output")
            return False
        msg = String()
        msg.data = line
        self.pub.publish(msg)
        st = String()
        st.data = f"EXAM_Q: {line[:160]}"
        self.status_pub.publish(st)
        mode = "比赛" if arg == _CONTEST_ARG else "练习"
        self.get_logger().info(f"published /exam_question ({mode}): {line[:120]}...")
        return True


def _question_line(text: str) -> str:
    """Keep the 题目 line. Contest mode also prints a seed header."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for ln in lines:
        if ln.startswith("题目"):
            return ln
    for ln in lines:
        if ln.startswith("使用种子") or set(ln) <= {"-"}:
            continue
        return ln
    return ""


def _read_exam_arg() -> str:
    print()
    print("========================================")
    print("  出题终端")
    print(f"  比赛模式：输入 {_CONTEST_ARG} 后回车")
    print("            （题目每 30 分钟变更一次）")
    print("  练习模式：直接回车，随机一题")
    print("========================================")
    try:
        return input("参数: ").strip()
    except EOFError:
        return ""


def main(args=None):
    rclpy.init(args=args)
    node = ExamFeederNode()
    try:
        exam_arg = _read_exam_arg()
        if exam_arg == _CONTEST_ARG:
            print("→ 比赛模式出题 …")
        elif exam_arg:
            print(f"→ 使用参数 {exam_arg} 出题 …")
        else:
            print("→ 练习模式出题 …")
        ok = node.feed_once(exam_arg)
        for _ in range(8):
            rclpy.spin_once(node, timeout_sec=0.1)
        print("→ 出题完成" if ok else "→ 出题失败")
    except KeyboardInterrupt:
        print("\n已退出出题")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
