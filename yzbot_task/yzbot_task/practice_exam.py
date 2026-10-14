"""多套题练习库：官方只有 2 题，用本节点覆盖全部计分/映射/整机路径。

模式（-p mode:=…）:
  check   默认。本地（+可选 DeepSeek）对答案，不跑车。
  feed    按空格把自然语言题发到 /exam_question（走 llm_solver→task_sm）。
  robot   按空格把 /task_command 直接发给 task_sm（跳过解题，专测取放/导航组合）。

用法:
  # 只验解题（本地，秒级）
  ros2 run yzbot_task practice_exam --ros-args -p mode:=check -p backend:=local

  # 整机：空格出下一题（需已开 llm_solver + task_sm）
  ros2 run yzbot_task practice_exam --ros-args -p mode:=feed

  # 整机：空格发下一组任务命令（覆盖 A/C × 红/蓝 组合）
  ros2 run yzbot_task practice_exam --ros-args -p mode:=robot

  # 一键启动里用练习库代替官方 bin：
  EXAM_BANK=practice ./run.sh
"""

from __future__ import annotations

import math
import os
import random
import select
import sys
import termios
import tty
from dataclasses import dataclass
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from yzbot_task.command_parser import format_command
from yzbot_task.llm_solver import question_to_command_llm
from yzbot_task.local_solver import (
    map_xy_to_tasks,
    question_to_command,
    solve_xy_from_question,
)


@dataclass(frozen=True)
class Case:
    name: str
    question: str
    xy: Tuple[int, int]
    command: str
    note: str = ""


@dataclass(frozen=True)
class RobotCase:
    name: str
    command: str
    note: str = ""


def _ceil5(n: int) -> int:
    return int(math.ceil(max(0, int(n)) / 5.0))


def _cmd_xy(x: int, y: int) -> str:
    return format_command(map_xy_to_tasks(x, y))


def _wardrobe(body: str) -> str:
    return (
        "题目: 每个衣柜有5件衣物;有T恤、裤子两种衣柜;"
        f"{body}"
        "设T恤衣柜数量为x,裤子衣柜数量为y,计算x、y分别为多少。"
    )


def _pets(body: str) -> str:
    return (
        "题目: 每个宠物店有5只宠物;有猫、狗两种宠物店;"
        f"{body}"
        "设猫宠物店数量为x,狗宠物店数量为y,计算x、y分别为多少"
    )


def _mk(name: str, question: str, t: int, p: int, note: str = "") -> Case:
    """Build case from known type totals (ground truth, not from solver)."""
    xy = (_ceil5(t), _ceil5(p))
    return Case(name, question, xy, _cmd_xy(*xy), note)


def practice_cases() -> List[Case]:
    """覆盖：官方2题 + ceil边界 + 都需要倍增 + 区映射 AA/AC/CA/CC + 单色 + 主题。"""
    cases: List[Case] = [
        # —— 官方练习平台 2 题 ——
        _mk(
            "official_wardrobe",
            _wardrobe(
                "小美需要2件T恤,1条裤子;小帅、小丽都需要7件T恤;"
                "小娜需要2件T恤,1条裤子;小杰需要1条裤子。"
            ),
            t=2 + 7 * 2 + 2,
            p=1 + 1 + 1,
            note="官方1 T=18→4 P=3→1 → [red,4,A;blue,1,C]",
        ),
        _mk(
            "official_pets",
            _pets(
                "小爱领养2只猫,5只狗;小宠、小动都领养2只猫;"
                "小乐领养3只猫,5只狗;小欢领养4只狗。"
            ),
            t=2 + 2 * 2 + 3,
            p=5 + 5 + 4,
            note="官方2 猫9→2 狗14→3 → [red,2,C;blue,3,A]",
        ),
        # —— ceil(/5) 边界 ——
        _mk(
            "ceil_1_1",
            _wardrobe("小明需要1件T恤,1条裤子。"),
            1,
            1,
            "ceil(1/5)=1 → 双 C",
        ),
        _mk(
            "ceil_exact_5",
            _wardrobe("小明需要5件T恤,5条裤子。"),
            5,
            5,
            "恰整除 → (1,1) CC",
        ),
        _mk(
            "ceil_6_11",
            _wardrobe("小明需要6件T恤,11条裤子。"),
            6,
            11,
            "6→2, 11→3 → CA",
        ),
        _mk(
            "ceil_10_14",
            _wardrobe("小明需要10件T恤,14条裤子。"),
            10,
            14,
            "10→2, 14→3 → CA",
        ),
        _mk(
            "ceil_15_15",
            _wardrobe("小明需要15件T恤,15条裤子。"),
            15,
            15,
            "双 3 → AA",
        ),
        # —— 「都需要」人数倍增 ——
        _mk(
            "dou_2_people",
            _wardrobe("小帅、小丽都需要7件T恤;小美需要1条裤子。"),
            14,
            1,
            "7×2=14→3",
        ),
        _mk(
            "dou_3_people",
            _wardrobe("小一、小二、小三都需要4件T恤;小四需要2条裤子。"),
            12,
            2,
            "4×3=12→3",
        ),
        _mk(
            "dou_pants",
            _wardrobe("小美需要3件T恤;小帅、小丽都需要3条裤子。"),
            3,
            6,
            "裤子 3×2=6→2",
        ),
        _mk(
            "pets_dou_3",
            _pets("小A、小B、小C都领养4只猫;小D领养1只狗。"),
            12,
            1,
            "漏乘人数会错成 ceil(4/5)=1",
        ),
        # —— 区映射组合（threshold=3：≥3→A 否则 C）——
        _mk(
            "map_CC",
            _wardrobe("小明需要8件T恤,4条裤子。"),
            8,
            4,
            "(2,1) 双 C",
        ),
        _mk(
            "map_CA",
            _wardrobe("小明需要9件T恤,12条裤子。"),
            9,
            12,
            "(2,3) 红C蓝A",
        ),
        _mk(
            "map_AC",
            _wardrobe("小明需要16件T恤,7条裤子。"),
            16,
            7,
            "(4,2) 红A蓝C",
        ),
        _mk(
            "map_AA",
            _wardrobe("小明需要16件T恤;小红需要14条裤子。"),
            16,
            14,
            "(4,3) 双 A",
        ),
        _mk(
            "map_threshold_edge",
            _wardrobe("小明需要11件T恤,10条裤子。"),
            11,
            10,
            "(3,2) 恰踩 ≥3 门槛",
        ),
        # —— 单色 / 缺一项 ——
        _mk(
            "only_x_red",
            _wardrobe("小美需要8件T恤;小帅需要2件T恤。"),
            10,
            0,
            "y=0 只发红",
        ),
        _mk(
            "only_y_blue",
            _wardrobe("小美需要4条裤子;小帅需要7条裤子。"),
            0,
            11,
            "x=0 只发蓝",
        ),
        # —— 混合多人 + 宠物主题 ——
        _mk(
            "pets_mixed",
            _pets(
                "小花领养7只猫,2只狗;小草、小叶都领养3只狗;小树领养1只猫。"
            ),
            8,
            8,
            "猫8→2 狗2+6=8→2",
        ),
        _mk(
            "large_mix",
            _wardrobe(
                "小军需要23件T恤,19条裤子;小强、小刚都需要5件T恤。"
            ),
            33,
            19,
            "T=33→7 但场上最多5红——验解题；整机见 robot 模式",
        ),
        # —— 同句多项 ——
        _mk(
            "inline_both_types",
            _wardrobe(
                "小美需要2件T恤,1条裤子;小娜需要2件T恤,1条裤子;小杰需要1条裤子。"
            ),
            4,
            3,
            "无「都」句，累加正确",
        ),
        # —— 多把模拟：数量/区组合再铺一层 ——
        _mk(
            "sim_red1_blue5",
            _wardrobe("小明需要3件T恤;小红需要22条裤子。"),
            3,
            22,
            "(1,5) 红C蓝A，蓝偏多",
        ),
        _mk(
            "sim_red5_blue1",
            _wardrobe("小明需要21件T恤;小红需要2条裤子。"),
            21,
            2,
            "(5,1) 红A蓝C，红偏多",
        ),
        _mk(
            "sim_red5_blue2",
            _pets("小爱领养22只猫;小欢领养8只狗。"),
            22,
            8,
            "(5,2) 红A蓝C",
        ),
        _mk(
            "sim_red2_blue5",
            _pets("小爱领养8只猫;小欢领养21只狗。"),
            8,
            21,
            "(2,5) 红C蓝A",
        ),
        _mk(
            "sim_red4_blue4",
            _wardrobe("小明需要18件T恤,16条裤子。"),
            18,
            16,
            "(4,4) 双 A 中等满载",
        ),
        _mk(
            "sim_red3_blue5",
            _wardrobe("小一、小二、小三都需要5件T恤;小四需要23条裤子。"),
            15,
            23,
            "(3,5) 红A蓝A",
        ),
        _mk(
            "sim_red5_blue3",
            _wardrobe("小明需要24件T恤;小红、小蓝都需要7条裤子。"),
            24,
            14,
            "(5,3) 双 A",
        ),
        _mk(
            "sim_red1_blue2",
            _wardrobe("小美需要2件T恤;小帅需要6条裤子。"),
            2,
            6,
            "(1,2) 双 C 轻量",
        ),
        _mk(
            "sim_red2_blue2",
            _pets("小花领养6只猫;小草领养9只狗。"),
            6,
            9,
            "(2,2) 双 C",
        ),
        _mk(
            "sim_threshold_red3_blue1",
            _wardrobe("小明需要13件T恤,1条裤子。"),
            13,
            1,
            "(3,1) 红刚过门槛→A",
        ),
        _mk(
            "sim_dou_4_people",
            _wardrobe("小A、小B、小C、小D都需要3件T恤;小E需要8条裤子。"),
            12,
            8,
            "3×4=12→3；漏乘会错",
        ),
        _mk(
            "sim_pets_mixed_heavy",
            _pets(
                "小爱领养5只猫,4只狗;小宠、小动、小乐都领养3只猫;"
                "小欢领养6只狗。"
            ),
            14,
            10,
            "猫5+9=14→3 狗4+6=10→2",
        ),
    ]
    return cases


def robot_cases() -> List[RobotCase]:
    """整机路径组合（max_cubes=5）：区 A/C × 红/蓝 × 数量，覆盖导航/投放分支。"""
    return [
        RobotCase("R1_only_red_C", "[red,1,C]", "单红→C（短路径）"),
        RobotCase("R2_only_red_A", "[red,3,A]", "单红→A（多块同区）"),
        RobotCase("R3_only_blue_C", "[blue,1,C]", "单蓝→C"),
        RobotCase("R4_only_blue_A", "[blue,3,A]", "单蓝→A"),
        RobotCase("R5_both_CC", "[red,2,C;blue,1,C]", "双色都进 C"),
        RobotCase(
            "R6_both_CA",
            "[red,2,C;blue,3,A]",
            "官方宠物型：红C蓝A",
        ),
        RobotCase(
            "R7_both_AC",
            "[red,4,A;blue,1,C]",
            "官方衣柜型：红A蓝C + first-red 门洞",
        ),
        RobotCase("R8_both_AA", "[red,3,A;blue,3,A]", "双色都进 A，同区多槽"),
        RobotCase(
            "R9_redA_blueC",
            "[red,3,A;blue,2,C]",
            "红A蓝C 中等量",
        ),
        RobotCase(
            "R10_stress_5_5",
            "[red,5,A;blue,5,A]",
            "吃满全部立方块（最长）",
        ),
        RobotCase(
            "R11_blue_first_heavy",
            "[blue,4,A;red,1,C]",
            "命令里蓝在前（解析顺序）",
        ),
        RobotCase(
            "R12_min_both",
            "[red,1,C;blue,1,C]",
            "最小双色",
        ),
        # —— 多把：数量梯度 / 跨区 ——
        RobotCase("R13_red5_C", "[red,5,C]", "5红全进C（同区满）"),
        RobotCase("R14_blue5_C", "[blue,5,C]", "5蓝全进C"),
        RobotCase("R15_red5_A", "[red,5,A]", "5红全进A"),
        RobotCase("R16_blue5_A", "[blue,5,A]", "5蓝全进A"),
        RobotCase("R17_red1_blue4_CA", "[red,1,C;blue,4,A]", "少红多蓝跨区"),
        RobotCase("R18_red4_blue1_AC", "[red,4,A;blue,1,C]", "多红少蓝跨区"),
        RobotCase("R19_red2_blue4_CA", "[red,2,C;blue,4,A]", "红C蓝A 中重"),
        RobotCase("R20_red4_blue2_AC", "[red,4,A;blue,2,C]", "红A蓝C 中重"),
        RobotCase("R21_red5_blue1_AC", "[red,5,A;blue,1,C]", "红吃满+1蓝"),
        RobotCase("R22_red1_blue5_CA", "[red,1,C;blue,5,A]", "蓝吃满+1红"),
        RobotCase("R23_red2_blue2_AA", "[red,2,A;blue,2,A]", "双色双A（低量）"),
        RobotCase("R24_red2_blue2_CC", "[red,2,C;blue,2,C]", "双色双C"),
        RobotCase("R25_red3_blue1_AC", "[red,3,A;blue,1,C]", "门槛红A + 单蓝C"),
        RobotCase("R26_red1_blue3_CA", "[red,1,C;blue,3,A]", "单红C + 门槛蓝A"),
        RobotCase("R27_red4_blue4_AA", "[red,4,A;blue,4,A]", "双A 近满载"),
        RobotCase("R28_red5_blue3_AA", "[red,5,A;blue,3,A]", "红满+蓝3"),
        RobotCase("R29_red3_blue5_AA", "[red,3,A;blue,5,A]", "蓝满+红3"),
        RobotCase("R30_blue2_red3", "[blue,2,C;red,3,A]", "蓝前红后解析"),
        RobotCase("R31_via_B_red", "[red,2,B]", "投放 B 区（路径/停靠）"),
        RobotCase("R32_via_B_mix", "[red,2,A;blue,2,B]", "A+B 跨区"),
        RobotCase("R33_B_then_C", "[blue,2,B;red,2,C]", "B+C 跨区"),
        RobotCase("R34_all_zones", "[red,2,A;blue,2,C]", "经典 AC 对照"),
    ]


def _wait_space() -> bool:
    fd = sys.stdin.fileno()
    if not os.isatty(fd):
        try:
            return bool(sys.stdin.readline())
        except Exception:
            return False
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
            if ch in (b" ", b"\r", b"\n"):
                return True
            if ch == b"\x03":
                return False
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class PracticeExamNode(Node):
    def __init__(self):
        super().__init__("practice_exam")
        self.declare_parameter("mode", "check")  # check|feed|robot
        self.declare_parameter("backend", "local")  # local|deepseek|both
        self.declare_parameter("publish", False)  # check 时是否顺带 pub 题目
        self.declare_parameter("stop_on_fail", True)
        self.declare_parameter("wait_done", False)  # feed/robot：等 DONE 再提示下一题
        # feed/robot：打乱顺序，模拟多把随机抽题
        self.declare_parameter("shuffle", True)
        self.q_pub = self.create_publisher(String, "/exam_question", 10)
        self.cmd_pub = self.create_publisher(String, "/task_command", 10)
        self.status_pub = self.create_publisher(String, "/task_status", 10)
        self._last_status = ""
        self.create_subscription(String, "/task_status", self._on_status, 20)
        self.create_timer(0.2, self._kick)
        self._started = False

    def _on_status(self, msg: String) -> None:
        self._last_status = str(msg.data)

    def _kick(self) -> None:
        if self._started:
            return
        self._started = True
        mode = str(self.get_parameter("mode").value).strip().lower()
        if mode == "feed":
            self._run_feed()
        elif mode == "robot":
            self._run_robot()
        else:
            self._run_check()

    def _pub_status(self, text: str) -> None:
        m = String()
        m.data = text
        self.status_pub.publish(m)

    def _wait_done_or_space(self) -> bool:
        """If wait_done: spin until DONE/ERROR or Space skip. Else just Space."""
        wait = bool(self.get_parameter("wait_done").value)
        if not wait:
            return _wait_space()
        print("  （等任务 DONE，或按空格跳过）")
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            st = self._last_status.upper()
            if "DONE" in st or st.startswith("ERROR") or "SKIP" in st:
                print(f"  状态: {self._last_status}")
                return True
            # non-blocking space?
            fd = sys.stdin.fileno()
            if os.isatty(fd):
                r, _, _ = select.select([fd], [], [], 0.0)
                if r:
                    ch = os.read(fd, 1)
                    if ch in (b" ", b"\r", b"\n"):
                        print("  （空格跳过等待）")
                        return True
                    if ch == b"\x03":
                        return False
        return False

    def _run_check(self) -> None:
        backend = str(self.get_parameter("backend").value).lower()
        publish = bool(self.get_parameter("publish").value)
        stop = bool(self.get_parameter("stop_on_fail").value)
        cases = practice_cases()
        ok = fail = 0
        print(f"\n==== PRACTICE CHECK  {len(cases)} 题  backend={backend} ====")
        for i, c in enumerate(cases, 1):
            print(f"\n[{i}/{len(cases)}] {c.name}  ({c.note})")
            print(f"  Q: {c.question[:80]}...")
            if publish:
                msg = String()
                msg.data = c.question
                self.q_pub.publish(msg)

            try:
                local_xy = solve_xy_from_question(c.question)
                local_cmd = question_to_command(c.question)
            except Exception as exc:  # noqa: BLE001
                local_xy, local_cmd = None, f"ERR:{exc}"
            local_ok = local_xy == c.xy and local_cmd == c.command
            print(f"  expect  xy={c.xy} cmd={c.command}")
            print(
                f"  local   xy={local_xy} cmd={local_cmd}  "
                f"{'OK' if local_ok else 'FAIL'}"
            )

            deep_ok = True
            if backend in ("deepseek", "both"):
                try:
                    d_cmd, dx, dy = question_to_command_llm(
                        c.question, backend="deepseek", timeout_sec=45.0
                    )
                    deep_ok = (dx, dy) == c.xy and d_cmd == c.command
                    print(
                        f"  deepseek xy=({dx},{dy}) cmd={d_cmd}  "
                        f"{'OK' if deep_ok else 'FAIL'}"
                    )
                except Exception as exc:  # noqa: BLE001
                    deep_ok = False
                    print(f"  deepseek ERROR: {exc}")

            if backend == "local":
                passed = local_ok
            elif backend == "deepseek":
                passed = deep_ok
            else:
                passed = local_ok and deep_ok

            if passed:
                ok += 1
            else:
                fail += 1
                if stop:
                    print("\nSTOP on fail（-p stop_on_fail:=false 可继续）")
                    break

        print(
            f"\n==== RESULT ok={ok} fail={fail} "
            f"checked={ok + fail}/{len(cases)} ====\n"
        )
        raise SystemExit(0 if fail == 0 else 1)

    def _maybe_shuffle(self, cases: list) -> list:
        out = list(cases)
        if bool(self.get_parameter("shuffle").value):
            random.shuffle(out)
            print("  （已随机打乱顺序，模拟多把）")
        return out

    def _run_feed(self) -> None:
        cases = self._maybe_shuffle(practice_cases())
        print()
        print("========================================")
        print(f"  练习出题 feed：共 {len(cases)} 套自然语言题")
        print("  按【空格】发布下一题 → /exam_question")
        print("  （需 llm_solver + task_sm 已开）")
        print("  Ctrl+C 退出")
        print("========================================")
        print()
        for i, c in enumerate(cases, 1):
            print(f"[{i}/{len(cases)}] 就绪: {c.name}")
            print(f"  期望 xy={c.xy} → {c.command}")
            print(f"  {c.note}")
            sys.stdout.write("\r  按空格出题 …  ")
            sys.stdout.flush()
            if not _wait_space():
                break
            print()
            msg = String()
            msg.data = c.question
            self.q_pub.publish(msg)
            self._pub_status(f"PRACTICE_Q [{i}/{len(cases)}] {c.name}: {c.command}")
            print(f"  → 已发布 /exam_question")
            print(f"  Q: {c.question[:100]}...")
            for _ in range(8):
                rclpy.spin_once(self, timeout_sec=0.05)
            if i < len(cases):
                if not self._wait_done_or_space():
                    break
        print("\n练习出题结束\n")
        raise SystemExit(0)

    def _run_robot(self) -> None:
        cases = self._maybe_shuffle(robot_cases())
        print()
        print("========================================")
        print(f"  整机路径 robot：共 {len(cases)} 组 /task_command")
        print("  按【空格】直接发任务（不走大模型）")
        print("  （需 task_sm 已开；建议每题前复位场景）")
        print("  Ctrl+C 退出")
        print("========================================")
        print()
        for i, c in enumerate(cases, 1):
            print(f"[{i}/{len(cases)}] 就绪: {c.name}  {c.command}")
            print(f"  {c.note}")
            sys.stdout.write("\r  按空格发任务 …  ")
            sys.stdout.flush()
            if not _wait_space():
                break
            print()
            msg = String()
            msg.data = c.command
            self.cmd_pub.publish(msg)
            self._pub_status(f"PRACTICE_CMD [{i}/{len(cases)}] {c.name}: {c.command}")
            print(f"  → 已发布 /task_command: {c.command}")
            for _ in range(8):
                rclpy.spin_once(self, timeout_sec=0.05)
            if i < len(cases):
                if not self._wait_done_or_space():
                    break
        print("\n整机路径练习结束\n")
        raise SystemExit(0)


def main(args: Optional[list] = None):
    rclpy.init(args=args)
    node = PracticeExamNode()
    try:
        rclpy.spin(node)
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 0
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass
        raise SystemExit(code)
    except KeyboardInterrupt:
        print("\n已退出")
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
