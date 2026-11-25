"""Official exam binary → /exam_question + /exam_seed.

三种出题方式（优先级从高到低）：
1. `-p exam_arg:=11470507`（命令行预置，出题后退出）
2. `-p panel_mode:=true`（**可视化面板**：等 `/exam_arg` 话题，面板按钮/粘贴发送
   参数 11470507=比赛模式、空串=练习模式；可反复发送重新出题，节点常驻）
3. 无参数时终端输入（兼容旧用法）

比赛模式参数 11470507：题目按 30 分钟一档，同一档内不变。
练习模式：每次随机一题。
"""

from __future__ import annotations

import os
import re
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
        # 可选：预置参数（-p exam_arg:=11470507）则不弹输入，直接出题
        self.declare_parameter("exam_arg", "")
        # 面板模式：等待 /exam_arg 话题（可视化面板输入），不再读终端 stdin
        self.declare_parameter("panel_mode", False)

        self.pub = self.create_publisher(String, "/exam_question", 10)
        self.seed_pub = self.create_publisher(String, "/exam_seed", 10)
        self.status_pub = self.create_publisher(String, "/task_status", 10)
        # 面板输入出题参数：data=11470507 比赛模式；data="" 练习模式；可重复发送
        self.create_subscription(String, "/exam_arg", self._on_exam_arg, 10)

    def _on_exam_arg(self, msg: String) -> None:
        """面板 `/exam_arg`：立即用该参数重新出题。"""
        arg = (msg.data or "").strip()
        mode = "比赛模式" if arg == _CONTEST_ARG else ("练习模式" if not arg else f"参数 {arg}")
        print()
        print(f"→ 面板输入：{mode}出题 …")
        self.feed_once(arg)

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
        seed = _seed_line(text)
        if seed:
            smsg = String()
            smsg.data = seed
            self.seed_pub.publish(smsg)
            self.get_logger().info(f"published /exam_seed: {seed}")
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


def _seed_line(text: str) -> str:
    """「使用种子: 997555」→ "997555"（评分项：题目种子）。"""
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if ln.startswith("使用种子"):
            m = re.search(r"(\d+)", ln)
            if m:
                return m.group(1)
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


def _panel_banner(node: "ExamFeederNode") -> None:
    print()
    print("========================================")
    print("  出题终端（面板模式）")
    print(f"  在可视化面板「出题参数输入」里填参数并点按钮：")
    print(f"    比赛模式 {_CONTEST_ARG}（题目每 30 分钟变更一次）")
    print("    练习模式 留空 / 清空后再点一次")
    print("  也可在终端直接回车（无面板时回退到键盘输入）")
    print("========================================")


def main(args=None):
    rclpy.init(args=args)
    node = ExamFeederNode()
    try:
        exam_arg = str(node.get_parameter("exam_arg").value or "").strip()
        panel_mode = bool(node.get_parameter("panel_mode").value)
        if panel_mode and not exam_arg:
            _panel_banner(node)
            node.get_logger().info("等待面板输入 /exam_arg（比赛 11470507 / 练习留空）…")
            rclpy.spin(node)  # 面板每发一次 /exam_arg 就重新出题
            return
        if not exam_arg:
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
