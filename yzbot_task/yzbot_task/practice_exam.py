"""Practice exam bank + DeepSeek/local cross-check (no need to re-download exam bin).

Official TMSCQtest_*.bin only has 2 fixed questions. Use this bank to stress-test
before contest day.

Usage:
  ros2 run yzbot_task practice_exam
  ros2 run yzbot_task practice_exam --ros-args -p backend:=deepseek -p publish:=false
  ros2 run yzbot_task practice_exam --ros-args -p backend:=both -p publish:=true
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from yzbot_task.llm_solver import question_to_command_llm
from yzbot_task.local_solver import question_to_command, solve_xy_from_question


@dataclass(frozen=True)
class Case:
    name: str
    question: str
    xy: Tuple[int, int]
    command: str
    note: str = ""


def practice_cases() -> List[Case]:
    """Hand-checked cases matching official ceil(/5) + default mapping."""
    return [
        Case(
            "official_wardrobe",
            "题目: 每个衣柜有5件衣物;有T恤、裤子两种衣柜;小美需要2件T恤,1条裤子;"
            "小帅、小丽都需要7件T恤;小娜需要2件T恤,1条裤子;小杰需要1条裤子。"
            "设T恤衣柜数量为x,裤子衣柜数量为y,计算x、y分别为多少。",
            (4, 1),
            "[red,4,A;blue,1,C]",
            "T=18→4, P=3→1",
        ),
        Case(
            "official_pets",
            "题目: 每个宠物店有5只宠物;有猫、狗两种宠物店;小爱领养2只猫,5只狗;"
            "小宠、小动都领养2只猫;小乐领养3只猫,5只狗;小欢领养4只狗。"
            "设猫宠物店数量为x,狗宠物店数量为y,计算x、y分别为多少",
            (2, 3),
            "[red,2,C;blue,3,A]",
            "猫9→2, 狗14→3",
        ),
        Case(
            "exact_multiples",
            "题目: 每个衣柜有5件衣物;有T恤、裤子两种衣柜;"
            "小明需要10件T恤,5条裤子。"
            "设T恤衣柜数量为x,裤子衣柜数量为y,计算x、y分别为多少。",
            (2, 1),
            "[red,2,C;blue,1,C]",
            "10/5=2, 5/5=1 — both <3 → C",
        ),
        Case(
            "both_to_A",
            "题目: 每个衣柜有5件衣物;有T恤、裤子两种衣柜;"
            "小明需要16件T恤;小红需要14条裤子。"
            "设T恤衣柜数量为x,裤子衣柜数量为y,计算x、y分别为多少。",
            (4, 3),
            "[red,4,A;blue,3,A]",
            "both ≥3 → A",
        ),
        Case(
            "tiny_both_C",
            "题目: 每个衣柜有5件衣物;有T恤、裤子两种衣柜;"
            "小明需要1件T恤,1条裤子。"
            "设T恤衣柜数量为x,裤子衣柜数量为y,计算x、y分别为多少。",
            (1, 1),
            "[red,1,C;blue,1,C]",
            "ceil(1/5)=1",
        ),
        Case(
            "ceil_boundary_6",
            "题目: 每个衣柜有5件衣物;有T恤、裤子两种衣柜;"
            "小明需要6件T恤,11条裤子。"
            "设T恤衣柜数量为x,裤子衣柜数量为y,计算x、y分别为多少。",
            (2, 3),
            "[red,2,C;blue,3,A]",
            "ceil(6/5)=2, ceil(11/5)=3",
        ),
        Case(
            "three_people_dou",
            "题目: 每个衣柜有5件衣物;有T恤、裤子两种衣柜;"
            "小一、小二、小三都需要4件T恤;小四需要2条裤子。"
            "设T恤衣柜数量为x,裤子衣柜数量为y,计算x、y分别为多少。",
            (3, 1),
            "[red,3,A;blue,1,C]",
            "T=12→3, P=2→1",
        ),
        Case(
            "dou_pants",
            "题目: 每个衣柜有5件衣物;有T恤、裤子两种衣柜;"
            "小美需要3件T恤;小帅、小丽都需要3条裤子。"
            "设T恤衣柜数量为x,裤子衣柜数量为y,计算x、y分别为多少。",
            (1, 2),
            "[red,1,C;blue,2,C]",
            "T=3→1, P=6→2",
        ),
        Case(
            "pets_dou_multiply",
            "题目: 每个宠物店有5只宠物;有猫、狗两种宠物店;"
            "小A、小B、小C都领养4只猫;小D领养1只狗。"
            "设猫宠物店数量为x,狗宠物店数量为y,计算x、y分别为多少",
            (3, 1),
            "[red,3,A;blue,1,C]",
            "猫12→3（若漏乘人数会错成1）",
        ),
        Case(
            "pets_mixed",
            "题目: 每个宠物店有5只宠物;有猫、狗两种宠物店;"
            "小花领养7只猫,2只狗;小草、小叶都领养3只狗;小树领养1只猫。"
            "设猫宠物店数量为x,狗宠物店数量为y,计算x、y分别为多少",
            (2, 2),
            "[red,2,C;blue,2,C]",
            "猫8→2, 狗2+6=8→2",
        ),
        Case(
            "large_wardrobe",
            "题目: 每个衣柜有5件衣物;有T恤、裤子两种衣柜;"
            "小军需要23件T恤,19条裤子;小强、小刚都需要5件T恤。"
            "设T恤衣柜数量为x,裤子衣柜数量为y,计算x、y分别为多少。",
            (7, 4),
            "[red,7,A;blue,4,A]",
            "T=23+10=33→7, P=19→4",
        ),
        Case(
            "only_x_type",
            "题目: 每个衣柜有5件衣物;有T恤、裤子两种衣柜;"
            "小美需要8件T恤;小帅需要2件T恤。"
            "设T恤衣柜数量为x,裤子衣柜数量为y,计算x、y分别为多少。",
            (2, 0),
            "[red,2,C]",
            "y=0 时只发红任务",
        ),
    ]


class PracticeExamNode(Node):
    def __init__(self):
        super().__init__("practice_exam")
        self.declare_parameter("backend", "both")  # local|deepseek|both
        self.declare_parameter("publish", False)
        self.declare_parameter("stop_on_fail", True)
        self.pub = self.create_publisher(String, "/exam_question", 10)
        self.create_timer(0.3, self._run)
        self._done = False

    def _run(self):
        if self._done:
            return
        self._done = True
        backend = str(self.get_parameter("backend").value).lower()
        publish = bool(self.get_parameter("publish").value)
        stop = bool(self.get_parameter("stop_on_fail").value)

        cases = practice_cases()
        ok = 0
        fail = 0
        self.get_logger().info(
            f"practice_exam: {len(cases)} cases backend={backend} publish={publish}"
        )
        print("\n==== PRACTICE EXAM ====")
        for i, c in enumerate(cases, 1):
            print(f"\n[{i}/{len(cases)}] {c.name}  ({c.note})")
            print(f"  Q: {c.question[:72]}...")
            if publish:
                msg = String()
                msg.data = c.question
                self.pub.publish(msg)

            local_xy = solve_xy_from_question(c.question)
            local_cmd = question_to_command(c.question)
            local_ok = local_xy == c.xy and local_cmd == c.command
            print(f"  expect  xy={c.xy} cmd={c.command}")
            print(f"  local   xy={local_xy} cmd={local_cmd}  {'OK' if local_ok else 'FAIL'}")

            deep_ok = True
            if backend in ("deepseek", "both"):
                try:
                    d_cmd, dx, dy = question_to_command_llm(
                        c.question, backend="deepseek", timeout_sec=45.0
                    )
                    deep_ok = (dx, dy) == c.xy and d_cmd == c.command
                    print(
                        f"  deepseek xy=({dx}, {dy}) cmd={d_cmd}  "
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
                    print("\nSTOP on first fail (set -p stop_on_fail:=false to continue)")
                    break

        print(f"\n==== RESULT ok={ok} fail={fail} total_checked={ok + fail}/{len(cases)} ====\n")
        self.get_logger().info(f"practice_exam done ok={ok} fail={fail}")
        # Exit after one shot
        raise SystemExit(0 if fail == 0 else 1)


def main(args: Optional[list] = None):
    rclpy.init(args=args)
    node = PracticeExamNode()
    try:
        rclpy.spin(node)
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 0
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(code)
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
