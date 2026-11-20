"""coStudio 可视化桥：把内部状态转成赛场正式用语（不上屏调试词）。

评分面板话题：
  /viz/question      当前任务题目
  /viz/grasp_order   抓取顺序
  /viz/counts        资源数量与放置区（含映射规则）
  /viz/mapping_rule  赛前映射规则原文
  /viz/red_count     红色数量 Int32（x）
  /viz/blue_count    蓝色数量 Int32（y）
  /viz/red_zone      红色放置区文案 如「红色 → A区」
  /viz/blue_zone     蓝色放置区文案
  /viz/progress      任务进度
  /viz/work_state    当前工作状态
  /viz/elapsed       【测试用】总用时（赛前可 -p show_elapsed:=false）
  /chat              任务总览

指示灯 Int32（/topic.data）：
  /color  0=蓝色目标 1=红色目标
  /task   0=蓝色任务 1=红色任务
  /pick   0=抓取完成 1=放置完成
  /cur    0=待机 1=前往取货 2=抓取作业 3=前往放置

默认映射规则①：
  x=红色，y=蓝色；数量≥3 → A区，否则 → C区
"""

from __future__ import annotations

import re
import time
from typing import List, Optional, Tuple

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


def _parse_xy(xy: str, cmd: str) -> Tuple[Optional[int], Optional[int]]:
    """Prefer /llm_xy；否则从 [red,n,Z;blue,m,W] 累计。"""
    mx = re.search(r"x\s*=\s*(\d+)", xy or "", re.I)
    my = re.search(r"y\s*=\s*(\d+)", xy or "", re.I)
    x = int(mx.group(1)) if mx else None
    y = int(my.group(1)) if my else None
    if x is not None and y is not None:
        return x, y
    red = blue = 0
    found = False
    for m in re.finditer(r"(red|blue)\s*,\s*(\d+)\s*,\s*[ABC]", cmd or "", re.I):
        found = True
        if m.group(1).lower() == "red":
            red += int(m.group(2))
        else:
            blue += int(m.group(2))
    for m in re.finditer(r"(red|blue)\s*#\s*\d+\s*,\s*[ABC]", cmd or "", re.I):
        found = True
        if m.group(1).lower() == "red":
            red += 1
        else:
            blue += 1
    if found:
        return (red if x is None else x), (blue if y is None else y)
    return x, y


def _zone_for(n: int, threshold: int, high: str, low: str) -> str:
    return high if int(n) >= int(threshold) else low


def _format_counts(
    xy: str,
    cmd: str,
    *,
    rule_text: str,
    threshold: int,
    high_zone: str,
    low_zone: str,
) -> str:
    x, y = _parse_xy(xy, cmd)
    lines: List[str] = [f"映射规则：{rule_text}"]
    if x is not None:
        zx = _zone_for(x, threshold, high_zone, low_zone)
        # 若命令里已写明红色区，以命令为准
        m = re.search(r"red\s*,\s*\d+\s*,\s*([ABC])", cmd or "", re.I)
        if m:
            zx = m.group(1).upper()
        lines.append(f"红色数量 x={x} → {zx}区")
    if y is not None:
        zy = _zone_for(y, threshold, high_zone, low_zone)
        m = re.search(r"blue\s*,\s*\d+\s*,\s*([ABC])", cmd or "", re.I)
        if m:
            zy = m.group(1).upper()
        lines.append(f"蓝色数量 y={y} → {zy}区")
    mapped = _format_command_cn(cmd)
    if mapped and mapped != "待生成":
        lines.append(f"抓取任务：{mapped}")
    if len(lines) == 1:
        lines.append("等待大模型解析颜色与数量")
    return "\n".join(lines)


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
        # 映射规则①（省赛自选）：x=红 y=蓝；≥threshold→high 否则→low
        self.declare_parameter("x_color", "red")
        self.declare_parameter("y_color", "blue")
        self.declare_parameter("threshold", 3)
        self.declare_parameter("high_zone", "A")
        self.declare_parameter("low_zone", "C")
        self.declare_parameter(
            "mapping_rule_text",
            "① x代表红色，y代表蓝色；数值≥3去A区，其他去C区",
        )

        qos = _latched_qos()

        self.pub_q = self.create_publisher(String, "/viz/question", qos)
        self.pub_order = self.create_publisher(String, "/viz/grasp_order", qos)
        self.pub_counts = self.create_publisher(String, "/viz/counts", qos)
        self.pub_mapping = self.create_publisher(String, "/viz/mapping_rule", qos)
        self.pub_red_count = self.create_publisher(Int32, "/viz/red_count", qos)
        self.pub_blue_count = self.create_publisher(Int32, "/viz/blue_count", qos)
        self.pub_red_zone = self.create_publisher(String, "/viz/red_zone", qos)
        self.pub_blue_zone = self.create_publisher(String, "/viz/blue_zone", qos)
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
        self._red_n: Optional[int] = None
        self._blue_n: Optional[int] = None
        self._red_zone_txt = "红色 → 待映射"
        self._blue_zone_txt = "蓝色 → 待映射"

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
            "viz_bridge ready（规则①：红/蓝数量+放置区面板；"
            "测试总用时 /viz/elapsed；赛场 -p show_elapsed:=false）"
        )

    def _rule_text(self) -> str:
        return str(self.get_parameter("mapping_rule_text").value)

    def _threshold(self) -> int:
        return int(self.get_parameter("threshold").value)

    def _high_zone(self) -> str:
        return str(self.get_parameter("high_zone").value).upper()

    def _low_zone(self) -> str:
        return str(self.get_parameter("low_zone").value).upper()

    def _refresh_xy_zones(self):
        x, y = _parse_xy(self._xy, self._command)
        self._red_n, self._blue_n = x, y
        thr = self._threshold()
        hi, lo = self._high_zone(), self._low_zone()
        # 命令里显式区优先
        zx = zy = None
        if self._command:
            mr = re.search(r"red\s*,\s*\d+\s*,\s*([ABC])", self._command, re.I)
            mb = re.search(r"blue\s*,\s*\d+\s*,\s*([ABC])", self._command, re.I)
            if mr:
                zx = mr.group(1).upper()
            if mb:
                zy = mb.group(1).upper()
        if x is not None and zx is None:
            zx = _zone_for(x, thr, hi, lo)
        if y is not None and zy is None:
            zy = _zone_for(y, thr, hi, lo)
        self._red_zone_txt = f"红色 → {zx}区" if zx else "红色 → 待映射"
        self._blue_zone_txt = f"蓝色 → {zy}区" if zy else "蓝色 → 待映射"

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
        if self._done >= self._total:
            return f"已完成 {self._done}/{self._total} 件货物"
        # 进行中不要写「已完成」，否则底部像提前结束
        return f"进度 {self._done}/{self._total} 件（进行中）"

    def _target_line(self) -> str:
        if not self._cur_color:
            return ""
        return f"当前目标：{_color_cn(self._cur_color)}"

    def _flush(self):
        def _s(pub, text: str):
            msg = String()
            msg.data = text
            pub.publish(msg)

        self._refresh_xy_zones()
        counts = _format_counts(
            self._xy,
            self._command,
            rule_text=self._rule_text(),
            threshold=self._threshold(),
            high_zone=self._high_zone(),
            low_zone=self._low_zone(),
        )
        order = _format_plan_cn(self._plan, self._command)
        progress = self._progress_line()
        work = self._work
        if self._target_line() and self._phase in ("nav", "grasp", "place"):
            work = f"{self._work}（{self._target_line()}）"

        _s(self.pub_q, self._question)
        _s(self.pub_order, order)
        _s(self.pub_counts, counts)
        _s(self.pub_mapping, self._rule_text())
        _s(self.pub_red_zone, self._red_zone_txt)
        _s(self.pub_blue_zone, self._blue_zone_txt)
        if self._red_n is not None:
            self.pub_red_count.publish(Int32(data=int(self._red_n)))
        if self._blue_n is not None:
            self.pub_blue_count.publish(Int32(data=int(self._blue_n)))
        _s(self.pub_progress, progress)
        _s(self.pub_work, work)
        if self._show_elapsed():
            _s(self.pub_elapsed, self._elapsed_text)

        chat_lines = [
            f"【任务题目】{self._question}",
            f"【映射规则】{self._rule_text()}",
            f"【资源数量】{counts.replace(chr(10), '；')}",
            f"【红色】{self._red_zone_txt}"
            + (f" 数量={self._red_n}" if self._red_n is not None else ""),
            f"【蓝色】{self._blue_zone_txt}"
            + (f" 数量={self._blue_n}" if self._blue_n is not None else ""),
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
            if self._done >= self._total:
                self._progress_text = f"已完成 {self._done}/{self._total} 件货物"
            else:
                self._progress_text = (
                    f"进度 {self._done}/{self._total} 件（进行中）"
                )

    def _on_status(self, msg: String):
        text = (msg.data or "").strip()
        if not text:
            return
        self._set_color_from_text(text)

        # STEP 1/5 — 只同步总数与当前序号；_done 只靠 PLACED 累加
        # （跳过的块不能按「第 i 步→已完成 i-1」估算，否则会虚高）
        m = re.search(r"(?:STEP|pick|块|第)\s*(\d+)\s*/\s*(\d+)", text, re.I)
        if m:
            cur_i = int(m.group(1))
            tot = int(m.group(2))
            self._total = tot
            self._progress_text = (
                f"正在处理第 {cur_i}/{tot} 件"
                f"（已放置 {self._done}/{tot}）"
            )
            # 清掉上一次 FAIL 残留的「作业受阻」文案
            self._phase = "nav"
            self._work = f"开始第 {cur_i}/{tot} 件任务"
            self._publish_ints(cur=1)

        # 1) 勿用裸 DONE — PLACE_VISION_DONE 等会误停表
        # 2) 每块先 DETACH_OK 再 PLACED —— 只计 PLACED 一次，否则翻倍提前「完成」
        # 3) 队列 DONE ≠ 全部放成功：跳过最后一块时也会发 DONE，禁止把进度拉满
        placed_one = text.startswith("PLACED ") or text.startswith("PLACED\t")
        round_done = (
            text == "DONE"
            or text.startswith("DONE ")
            or text == "QUEUE_DONE"
            or text.startswith("QUEUE_DONE")
            or "全部完成" in text
        ) and ("DRY" not in text) and ("VISION" not in text)

        if placed_one:
            self._bump_done()
            if self._total > 0 and self._done >= self._total:
                self._phase = "done"
                self._work = "本轮任务全部完成"
                self._stop_timer()
                self._publish_ints(pick=1, cur=0)
            else:
                self._phase = "nav"
                self._work = (
                    f"已放置 {self._done}/{self._total}，前往下一件"
                    if self._total > 0
                    else "货物已放置，准备下一件"
                )
                # 中途保持「前往取货」，勿 cur=0 待机（否则像结束了）
                self._publish_ints(pick=1, cur=1)
        elif round_done:
            self._phase = "done"
            self._stop_timer()
            if self._total > 0 and self._done >= self._total:
                self._progress_text = f"已完成 {self._done}/{self._total} 件货物"
                self._work = "本轮任务全部完成"
            elif self._total > 0:
                skipped = max(0, self._total - self._done)
                self._progress_text = (
                    f"流程结束：成功 {self._done}/{self._total}"
                    f"（跳过 {skipped}）"
                )
                self._work = (
                    f"队列结束，仅完成 {self._done}/{self._total} 件"
                )
            else:
                self._progress_text = f"流程结束（已放置 {self._done}）"
                self._work = "本轮流程结束"
            self._publish_ints(pick=1, cur=0)
        elif any(
            k in text
            for k in (
                # 勿用裸 "PLACE"/"DETACH"：DETACH_CLEAR、PLACE 字样会在抓取段误报「运往放置」
                "NAV_PLACE",
                "PLACE_TARGET",
                "PLACE_DOCK",
                "PLACE_ARM",
                "PLACE_VISION",
                "PLACE_LIFT",
                "PLACE_RETRY",
                "HOLDING",
                "HOLDING_BLOCK",
                "DETACH_STUCK",
                "LEAVE_ZONE",
                "DETACH →",
                "DETACH_OK",
                "DETACH_FAIL",
                "DETACH_SOFT",
            )
        ):
            self._phase = "place"
            self._work = "运往放置区并完成放置"
            self._publish_ints(cur=3, pick=0)
        elif any(
            k in text
            for k in (
                "ATTACH",
                "JOINT_GRASP",
                "PRESET_ARM",
                "DETACH_CLEAR",
                "OPENCV",
                "FACE_CUBE",
                "MTC_",
                "GRIPPER",
                "EE_TO_CUBE",
                "LOOK:",
                "LOCAL_FLOW",
                "SCAN",
                "SEE_",
                "PRE_PICK",
                "NEAR_CUBE",
                "VIS_",
                "PICK:",
                "AIM_MAP",
                "CUBE_UNDER",
                "NAV_PICK_OK",
            )
        ):
            # 导航到位后的视觉贴近/抓取 —— 勿再显示「前往取货」
            self._phase = "grasp"
            self._work = "识别目标并执行抓取"
            self._publish_ints(cur=2, pick=0)
        elif any(
            k in text
            for k in (
                "NAV",
                "VIA",
                "PARK",
                "SAFE_PICK",
            )
        ):
            self._phase = "nav"
            self._work = "导航前往取货点"
            self._publish_ints(cur=1)
        elif any(k in text for k in ("LLM_SOLVED", "SOLVED", "PARSED", "PLAN_OK")):
            self._phase = "ready"
            self._work = "任务指令已解析"
            self._publish_ints(cur=0)
        elif "WAITING_SERVERS" in text:
            self._work = "系统初始化中"
            self._publish_ints(cur=0)
        elif text.startswith("SERVERS_OK") or text.startswith("SERVERS_OK "):
            self._phase = "nav"
            self._work = "定位就绪，开始作业"
            self._publish_ints(cur=1)
        elif "LOC_WAIT" in text:
            self._phase = "ready"
            self._work = "等待仿真定位就绪"
            self._publish_ints(cur=0)
        elif any(
            k in text
            for k in (
                # 勿用裸 FAIL/ABORT；定位等待不算「作业受阻」
                "LOC_FAIL",
                "PARSE_FAIL",
                "PICK_ABORT",
                "NAV_FAIL",
                "NAV_PICK_FAIL",
                "ATTACH_FAIL",
                "SERVERS_UNAVAILABLE",
                "ERROR:",
            )
        ):
            if "LOC_" in text:
                self._work = "定位未就绪，请检查 Gazebo / Nav2"
            else:
                self._work = "作业受阻，正在恢复"
        elif "SKIP_CUBE" in text:
            self._phase = "nav"
            self._work = "本块跳过，前往下一件"
            self._publish_ints(cur=1)

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
