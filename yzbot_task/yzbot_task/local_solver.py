"""Local word-problem solver (no cloud API).

Official pattern: capacity 5;
x = ceil(type1_total / 5), y = ceil(type2_total / 5).
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple

from yzbot_task.command_parser import PickTask, format_command


def _normalize(text: str) -> str:
    t = text.replace("，", ",").replace("；", ";").replace("：", ":")
    t = re.sub(r"\s+", "", t)
    return t


def solve_xy_from_question(question: str) -> Tuple[int, int]:
    """Sum needs for type1/type2 then ceil(/5).

    Supports T恤/裤子 and 猫/狗 style official templates.
    Handles 'A、B都需要N件X' by multiplying N by subject count.
    """
    text = _normalize(question)

    type1 = 0
    type2 = 0

    # 「小帅、小丽都需要7件T恤」/「都领养2只猫」→ subjects * N
    # names may include ASCII (小A) in practice banks
    _name = r"([\u4e00-\u9fffA-Za-z0-9、]{2,}?)"
    for m in re.finditer(_name + r"都(?:需要|领养)(\d+)件T恤", text):
        subjects = [s for s in m.group(1).split("、") if s]
        type1 += int(m.group(2)) * max(len(subjects), 1)
    for m in re.finditer(_name + r"都(?:需要|领养)(\d+)条裤子", text):
        subjects = [s for s in m.group(1).split("、") if s]
        type2 += int(m.group(2)) * max(len(subjects), 1)
    for m in re.finditer(_name + r"都(?:需要|领养)(\d+)只?猫", text):
        subjects = [s for s in m.group(1).split("、") if s]
        type1 += int(m.group(2)) * max(len(subjects), 1)
    for m in re.finditer(_name + r"都(?:需要|领养)(\d+)只?狗", text):
        subjects = [s for s in m.group(1).split("、") if s]
        type2 += int(m.group(2)) * max(len(subjects), 1)

    # Strip dou-* clauses so they are not double-counted below
    text2 = re.sub(_name + r"都(?:需要|领养)\d+件T恤", "", text)
    text2 = re.sub(_name + r"都(?:需要|领养)\d+条裤子", "", text2)
    text2 = re.sub(_name + r"都(?:需要|领养)\d+只?猫", "", text2)
    text2 = re.sub(_name + r"都(?:需要|领养)\d+只?狗", "", text2)
    for m in re.finditer(r"(\d+)件T恤", text2):
        type1 += int(m.group(1))
    for m in re.finditer(r"(\d+)只?猫", text2):
        type1 += int(m.group(1))
    for m in re.finditer(r"(\d+)条裤子", text2):
        type2 += int(m.group(1))
    for m in re.finditer(r"(\d+)只?狗", text2):
        type2 += int(m.group(1))

    if type1 == 0 and type2 == 0:
        type1 = sum(int(m.group(1)) for m in re.finditer(r"(\d+)件", text))
        type2 = sum(int(m.group(1)) for m in re.finditer(r"(\d+)条", text))

    if type1 == 0 and type2 == 0:
        raise ValueError(f"cannot extract totals from question: {question[:100]}")

    x = int(math.ceil(type1 / 5.0))
    y = int(math.ceil(type2 / 5.0))
    return x, y


def map_xy_to_tasks(
    x: int,
    y: int,
    *,
    x_color: str = "red",
    y_color: str = "blue",
    threshold: int = 3,
    high_zone: str = "A",
    low_zone: str = "C",
) -> List[PickTask]:
    def zone_for(n: int) -> str:
        return high_zone if n >= threshold else low_zone

    tasks: List[PickTask] = []
    if x > 0:
        tasks.append(PickTask(x_color.lower(), x, zone_for(x)))
    if y > 0:
        tasks.append(PickTask(y_color.lower(), y, zone_for(y)))
    return tasks


def question_to_command(question: str, mapping: Dict | None = None) -> str:
    mapping = mapping or {}
    x, y = solve_xy_from_question(question)
    tasks = map_xy_to_tasks(
        x,
        y,
        x_color=mapping.get("x_color", "red"),
        y_color=mapping.get("y_color", "blue"),
        threshold=int(mapping.get("threshold", 3)),
        high_zone=mapping.get("high_zone", "A"),
        low_zone=mapping.get("low_zone", "C"),
    )
    return format_command(tasks)
