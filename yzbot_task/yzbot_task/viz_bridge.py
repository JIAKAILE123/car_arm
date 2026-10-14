"""coStudio 可视化桥：把内部状态转成赛场正式用语（不上屏调试词）。

评分面板话题：
  /viz/question      当前任务题目
  /viz/grasp_order   抓取顺序
  /viz/counts        资源数量与放置区
  /viz/progress      任务进度
  /viz/work_state    当前工作状态
  /viz/elapsed       【测试用】总用时（赛前可 -p show_elapsed:=false）
  /chat              任务总览

指示灯 Int32（/topic.data）：
  /color  0=蓝色目标 1=红色目标
  /task   0=蓝色任务 1=红色任务
  /pick   0=抓取完成 1=放置完成
  /cur    0=待机 1=前往取货 2=抓取作业 3=前往放置
"""

from __future__ import annotations

import re
import time
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32, String


def _latched_qos() -> QoSProfile:
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )


def _color_cn(c: str) -> str:
    return "红色货物" if c.lower() == "red" else "蓝色货物"


def _zone_cn(z: str) -> str:
    return f"{z.upper()} 放置区"


def _format_hms(sec: float) -> str:
    sec = max(0.0, float(sec))
    whole = int(sec)
    m, s = divmod(whole, 60)
    h, m = divmod(m, 60)
    tenths = int((sec - whole) * 10)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}.{tenths}"


def _format_command_cn(cmd: str) -> str:
    if not cmd:
        return "待生成"
    parts: List[str] = []
    for m in re.finditer(
        r"(red|blue)\s*#\s*(\d+)\s*,\s*([ABC])", cmd, flags=re.I
    ):
        parts.append(f"{_color_cn(m.group(1))}#{m.group(2)} → {_zone_cn(m.group(3))}")
    if parts:
        return "；".join(parts)
    for m in re.finditer(
        r"(red|blue)\s*,\s*(\d+)\s*,\s*([ABC])", cmd, flags=re.I
    ):
        parts.append(
            f"{_color_cn(m.group(1))} ×{m.group(2)} → {_zone_cn(m.group(3))}"
        )
    return "；".join(parts) if parts else cmd


def _format_counts(xy: str, cmd: str) -> str:
    mx = re.search(r"x\s*=\s*(\d+)", xy or "", re.I)
    my = re.search(r"y\s*=\s*(\d+)", xy or "", re.I)
    lines: List[str] = []
    if mx or my:
        x = mx.group(1) if mx else "?"
        y = my.group(1) if my else "?"
        lines.append(f"解题结果：x={x}（红色），y={y}（蓝色）")
    mapped = _format_command_cn(cmd)
    if mapped and mapped != "待生成":
        lines.append(f"抓取任务：{mapped}")
    return "\n".join(lines) if lines else "等待大模型解析"


def _format_plan_cn(plan: str, cmd: str) -> str:
    text = _format_command_cn(plan) if plan else ""
    if text and text != "待生成":
        return text
    text = _format_command_cn(cmd)
    return text if text != "待生成" else "等待规划抓取顺序"


class VizBridge(Node):
    def __init__(self):
        super().__init__("viz_bridge")
        self.declare_parameter("show_elapsed", True)  # 测试总用时；赛场改 false

        qos = _latched_qos()

        self.pub_q = self.create_publisher(String, "/viz/question", qos)
        self.pub_order = self.create_publisher(String, "/viz/grasp_order", qos)
        self.pub_counts = self.create_publisher(String, "/viz/counts", qos)
        self.pub_progress = self.create_publisher(String, "/viz/progress", qos)
        self.pub_work = self.create_publisher(String, "/viz/work_state", qos)
        self.pub_elapsed = self.create_publisher(String, "/viz/elapsed", qos)
        self.pub_chat = self.create_publisher(String, "/chat", qos)

        self.pub_color = self.create_publisher(Int32, "/color", qos)
        self.pub_task = self.create_publisher(Int32, "/task", qos)
        self.pub_pick = self.create_publisher(Int32, "/pick", qos)
        self.pub_cur = self.create_publisher(Int32, "/cur", qos)

        self._question = "等待接收比赛题目"
        self._xy = ""
        self._command = ""
        self._plan = ""
        self._work = "系统待机"
        self._progress_text = "尚未开始"
        self._done = 0
        self._total = 0
        self._cur_color: Optional[str] = None
        self._phase = "idle"

        self._t0: Optional[float] = None
        self._t_stop: Optional[float] = None
        self._elapsed_text = "测试用时：未开始"

        self.create_subscription(String, "/exam_question", self._on_question, 10)
        self.create_subscription(String, "/exam_question_echo", self._on_question, 10)
        self.create_subscription(String, "/llm_xy", self._on_xy, 10)
        self.create_subscription(String, "/task_command", self._on_command, 10)
        self.create_subscription(String, "/task_plan", self._on_plan, 10)
        self.create_subscription(String, "/task_status", self._on_status, 10)
        self.create_subscription(String, "/task_info", self._on_info, 10)

        self.create_timer(0.2, self._tick_elapsed)

        self._publish_ints(color=0, task=0, pick=-1, cur=0)
        self._flush()
        self.get_logger().info(
            "viz_bridge ready（测试总用时 /viz/elapsed；赛场 -p show_elapsed:=false）"
        )

    def _show_elapsed(self) -> bool:
        return bool(self.get_parameter("show_elapsed").value)

    def _start_timer(self, reason: str):
        if self._t0 is not None and self._t_stop is None:
            return
        # 新一轮：允许在上一轮已停表后重新开始
        self._t0 = time.monotonic()
        self._t_stop = None
        self._elapsed_text = f"测试用时：00:00.0（计时开始）"
        self.get_logger().info(f"elapsed timer START ({reason})")

    def _stop_timer(self):
        if self._t0 is None:
            return
        if self._t_stop is None:
            self._t_stop = time.monotonic()
        dt = self._t_stop - self._t0
        self._elapsed_text = f"测试总用时：{_format_hms(dt)}（已完成）"
        self.get_logger().info(f"elapsed timer STOP {dt:.1f}s")

    def _tick_elapsed(self):
        if not self._show_elapsed() or self._t0 is None:
            return
        if self._t_stop is None:
            dt = time.monotonic() - self._t0
            self._elapsed_text = f"测试用时：{_format_hms(dt)}"
        msg = String()
        msg.data = self._elapsed_text
        self.pub_elapsed.publish(msg)

    def _publish_ints(
        self,
        *,
        color: Optional[int] = None,
        task: Optional[int] = None,
        pick: Optional[int] = None,
        cur: Optional[int] = None,
    ):
        if color is not None and color >= 0:
            self.pub_color.publish(Int32(data=int(color)))
        if task is not None and task >= 0:
            self.pub_task.publish(Int32(data=int(task)))
        if pick is not None and pick >= 0:
            self.pub_pick.publish(Int32(data=int(pick)))
        if cur is not None and cur >= 0:
            self.pub_cur.publish(Int32(data=int(cur)))

    def _progress_line(self) -> str:
        if self._total <= 0:
            return self._progress_text
        return f"已完成 {self._done}/{self._total} 件货物"

    def _target_line(self) -> str:
        if not self._cur_color:
            return ""
        return f"当前目标：{_color_cn(self._cur_color)}"

    def _flush(self):
        def _s(pub, text: str):
            msg = String()
            msg.data = text
            pub.publish(msg)

        counts = _format_counts(self._xy, self._command)
        order = _format_plan_cn(self._plan, self._command)
        progress = self._progress_line()
        work = self._work
        if self._target_line() and self._phase in ("nav", "grasp", "place"):
            work = f"{self._work}（{self._target_line()}）"

        _s(self.pub_q, self._question)
        _s(self.pub_order, order)
        _s(self.pub_counts, counts)
        _s(self.pub_progress, progress)
        _s(self.pub_work, work)
        if self._show_elapsed():
            _s(self.pub_elapsed, self._elapsed_text)

        chat_lines = [
            f"【任务题目】{self._question}",
            f"【资源数量】{counts.replace(chr(10), '；')}",
            f"【抓取顺序】{order}",
            f"【任务进度】{progress}",
            f"【工作状态】{work}",
        ]
        if self._show_elapsed():
            chat_lines.append(f"【{self._elapsed_text}】")
        _s(self.pub_chat, "\n".join(chat_lines))

    def _on_question(self, msg: String):
        q = (msg.data or "").strip()
        if not q:
            return
        self._question = q
        self._phase = "solve"
        self._work = "大模型解题中"
        self._progress_text = "正在解析题目"
        self._start_timer("收到题目")
        self._publish_ints(cur=0)
        self._flush()

    def _on_xy(self, msg: String):
        raw = (msg.data or "").strip()
        if not raw:
            return
        self._xy = raw.replace(" ", "")
        self._work = "解题完成，生成抓取任务"
        self._progress_text = "解题完成"
        self._flush()

    def _on_command(self, msg: String):
        self._command = (msg.data or "").strip()
        self._phase = "ready"
        self._work = "任务指令已就绪"
        self._start_timer("收到任务指令")
        total = 0
        for m in re.finditer(r"(?:red|blue)\s*,\s*(\d+)\s*,", self._command, re.I):
            total += int(m.group(1))
        for _ in re.finditer(r"(?:red|blue)\s*#\s*\d+", self._command, re.I):
            total += 1
        if total:
            self._total = total
            self._done = 0
            self._progress_text = f"计划抓取 {self._total} 件货物"
        self._publish_ints(cur=0)
        self._flush()

    def _on_plan(self, msg: String):
        self._plan = (msg.data or "").strip()
        legs = re.findall(r"(red|blue)\s*#\s*\d+", self._plan, flags=re.I)
        if legs:
            self._total = len(legs)
            self._progress_text = f"计划抓取 {self._total} 件货物"
        self._work = "抓取顺序已确定"
        self._flush()

    def _on_info(self, msg: String):
        if "xy=" in (msg.data or "") and not self._xy:
            m = re.search(r"xy=\((\d+),(\d+)\)", msg.data or "")
            if m:
                self._xy = f"x={m.group(1)},y={m.group(2)}"
                self._flush()

    def _set_color_from_text(self, text: str):
        low = text.lower()
        m = re.search(r"\b(red|blue)[_\s#]*(\d+)?\b", low)
        if not m:
            return
        self._cur_color = m.group(1)
        c = 1 if self._cur_color == "red" else 0
        self._publish_ints(color=c, task=c)

    def _bump_done(self):
        if self._total > 0:
            self._done = min(self._total, self._done + 1)
            self._progress_text = f"已完成 {self._done}/{self._total} 件货物"

    def _on_status(self, msg: String):
        text = (msg.data or "").strip()
        if not text:
            return
        self._set_color_from_text(text)

        m = re.search(r"(?:pick|块|第)\s*(\d+)\s*/\s*(\d+)", text, re.I)
        if m:
            self._done = max(0, int(m.group(1)) - 1)
            self._total = int(m.group(2))
            self._progress_text = f"正在处理第 {m.group(1)}/{m.group(2)} 件货物"

        # 注意：勿用裸 "DONE" 子串 — PLACE_VISION_DONE / PLACE_DOCK_OK 等会误停表
        placed_one = (
            "DETACH_OK" in text
            or text.startswith("PLACED ")
            or text.startswith("PLACED\t")
        )
        round_done = (
            text == "DONE"
            or text.startswith("DONE ")
            or text == "QUEUE_DONE"
            or text.startswith("QUEUE_DONE")
            or "全部完成" in text
        ) and "DRY" not in text and "VISION" not in text

        if placed_one:
            self._phase = "done"
            self._bump_done()
            if self._total > 0 and self._done >= self._total:
                self._work = "本轮任务全部完成"
                self._stop_timer()
            else:
                self._work = "货物已放置，准备下一件"
            self._publish_ints(pick=1, cur=0)
        elif round_done:
            # 整轮结束：只停表，不再 bump（件数已由 DETACH_OK/PLACED 计过）
            self._phase = "done"
            if self._total > 0:
                self._done = self._total
            self._work = "本轮任务全部完成"
            self._stop_timer()
            self._publish_ints(pick=1, cur=0)
        elif any(k in text for k in ("DETACH", "PLACE", "LEAVE_ZONE", "PLACE_ARM")):
            self._phase = "place"
            self._work = "运往放置区并完成放置"
            self._publish_ints(cur=3, pick=0)
        elif any(
            k in text
            for k in (
                "ATTACH",
                "JOINT_GRASP",
                "PRESET_ARM",
                "OPENCV",
                "FACE_CUBE",
                "MTC_",
                "GRIPPER",
                "EE_TO_CUBE",
                "LOOK:",
            )
        ):
            self._phase = "grasp"
            self._work = "识别目标并执行抓取"
            self._publish_ints(cur=2, pick=0)
        elif any(
            k in text
            for k in (
                "NAV",
                "VIA",
                "CREEP",
                "AIM",
                "SCAN",
                "DRIVE",
                "VIS_NEAR",
                "LOCAL_FLOW",
                "PARK",
            )
        ):
            self._phase = "nav"
            self._work = "导航前往取货点"
            self._publish_ints(cur=1)
        elif any(k in text for k in ("LLM_SOLVED", "SOLVED", "PARSED", "PLAN_OK")):
            self._phase = "ready"
            self._work = "任务指令已解析"
            self._publish_ints(cur=0)
        elif any(k in text for k in ("WAITING_SERVERS", "SERVERS")):
            self._work = "系统初始化中"
            self._publish_ints(cur=0)
        elif any(k in text for k in ("ERROR", "FAIL", "ABORT")):
            self._work = "作业受阻，正在恢复"

        self._flush()


def main(args=None):
    rclpy.init(args=args)
    node = VizBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
