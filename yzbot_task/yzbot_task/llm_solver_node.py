"""Subscribe /exam_question → DeepSeek (or local) solve → /task_command.

Also publishes visualization-friendly topics for the 10pt LLM panel score:
  /exam_question_echo  — last question text
  /llm_xy              — e.g. x=4,y=1
  /task_info           — human summary including mapping result
  /mapping_rule_text   — 当前生效的国赛映射规则原文（latched，供可视化「规则题目」面板）

可视化面板可反向输入（coStudio Publish 面板，std_msgs/String）：
  /mapping_rule_text   — 粘贴裁判给的映射规则原文 → 立即切换规则并重算上一题
  /panel_exam_question — 粘贴/输入题目 → 直接解题
"""

from __future__ import annotations

import os

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from yzbot_task.llm_solver import question_to_command_llm
from yzbot_task.local_solver import resolve_rule_id, rule_banner


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
        # 国赛三条映射规则：rule_id=0 表示按 rule_text 自动识别（默认规则①）
        self.declare_parameter("rule_id", 0)
        self.declare_parameter("rule_text", "")

        self.pub_cmd = self.create_publisher(String, "/task_command", 10)
        self.pub_status = self.create_publisher(String, "/task_status", 10)
        self.pub_info = self.create_publisher(String, "/task_info", 10)
        self.pub_xy = self.create_publisher(String, "/llm_xy", 10)
        self.pub_q = self.create_publisher(String, "/exam_question_echo", 10)
        # 出题预览：解题后先把指令发到这里供面板显示抓取顺序，不触发 task_sm。
        self.pub_preview = self.create_publisher(String, "/task_command_preview", 10)
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_rule = self.create_publisher(String, "/mapping_rule_text", latched)

        self._panel_rule_text = ""      # 面板粘贴的映射规则原文（优先于参数）
        self._last_published_rule = ""  # 防自回环：忽略自己发的 latched 规则
        self._last_question = ""        # 规则变更后自动重算
        self._pending_command = ""      # 出题解出的指令，待「开始任务」才下发执行

        self.create_subscription(String, "/exam_question", self._on_question, 10)
        # coStudio「粘贴题目解题」面板也可发到此话题（与 /exam_question 同处理）
        self.create_subscription(String, "/panel_exam_question", self._on_question, 10)
        # 面板粘贴规则原文（Publish 面板 → {"data":"<规则原文>"}）
        # 注意：订阅端用默认 volatile QoS —— 面板/终端的临时发布器都是 volatile，
        # 若这里用 TRANSIENT_LOCAL 反而收不到它们的消息（DDS 耐久性不兼容）。
        self.create_subscription(
            String, "/mapping_rule_text", self._on_panel_rule, 10
        )
        # 「开始任务」按钮（coStudio Publish 面板）：重解上一题并重新下发 /task_command。
        self.create_subscription(String, "/task_start", self._on_task_start, 10)
        backend = self.get_parameter("backend").value
        self._publish_rule()
        self.get_logger().info(f"映射规则 {self._rule_banner()}")
        self.get_logger().info(
            f"llm_solver ready backend={backend} "
            f"(/exam_question|/panel_exam_question → 大模型 → /task_command)"
        )

    def _effective_rule_text(self) -> str:
        """面板粘贴的规则原文优先于命令行参数。"""
        if self._panel_rule_text:
            return self._panel_rule_text
        return str(self.get_parameter("rule_text").value or "")

    def _on_panel_rule(self, msg: String) -> None:
        """面板输入映射规则：切换规则并重算上一题。"""
        txt = (msg.data or "").strip()
        if not txt or txt == self._last_published_rule:
            return  # 空消息 / 自己发布的 latched 回环
        self._panel_rule_text = txt
        self.get_logger().info(f"面板输入映射规则 → {self._rule_banner()}")
        self._publish_rule()
        if self._last_question:
            self.get_logger().info("规则已变更，重算上一题")
            self._on_question(String(data=self._last_question))

    def _on_task_start(self, msg: String) -> None:
        """面板「开始任务」按钮：把出题阶段解好的指令下发 /task_command，触发执行。"""
        if not self._pending_command:
            self.get_logger().warn("开始任务：尚未出题，请先点「出题程序」生成题目")
            return
        out = String()
        out.data = self._pending_command
        self.pub_cmd.publish(out)
        self.get_logger().info(
            f"面板「开始任务」→ 下发 /task_command: {self._pending_command}"
        )

    def _rule_id(self) -> int:
        return resolve_rule_id(
            self.get_parameter("rule_id").value,
            self._effective_rule_text(),
        )

    def _rule_banner(self) -> str:
        return rule_banner(self._rule_id())

    def _publish_rule(self):
        msg = String()
        msg.data = self._rule_banner()
        self._last_published_rule = msg.data
        self.pub_rule.publish(msg)

    def _mapping(self) -> dict:
        return {
            "x_color": self.get_parameter("x_color").value,
            "y_color": self.get_parameter("y_color").value,
            "threshold": self.get_parameter("threshold").value,
            "high_zone": self.get_parameter("high_zone").value,
            "low_zone": self.get_parameter("low_zone").value,
            "rule_id": self._rule_id(),
            "rule_text": self._rule_banner(),
        }

    def _on_question(self, msg: String):
        q = (msg.data or "").strip()
        if not q:
            return
        self._last_question = q
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
        # 出题阶段：只暂存指令并发预览（面板显示抓取顺序/数量），不触发 task_sm。
        # 真正执行要等面板「开始任务」按钮 → /task_start → 下发 /task_command。
        self._pending_command = cmd
        self.pub_preview.publish(out)

        xy = String()
        xy.data = f"x={x},y={y}"
        self.pub_xy.publish(xy)

        info = String()
        info.data = (
            f"question={q[:120]} | xy=({x},{y}) | "
            f"map={self._mapping()} | command={cmd}"
        )
        self.pub_info.publish(info)

        self._publish_rule()
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
