"""Local word-problem solver (no cloud API).

Official pattern: capacity 5;
x = ceil(type1_total / 5), y = ceil(type2_total / 5).

Also accepts direct robot commands (no ceil / zone mapping), e.g.:
  「5个红色物块，去A区」→ [red,5,A]
  「[red,5,A;blue,2,C]」→ as-is
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

from yzbot_task.command_parser import PickTask, format_command, parse_command


def _normalize(text: str) -> str:
    t = text.replace("，", ",").replace("；", ";").replace("：", ":")
    t = re.sub(r"\s+", "", t)
    return t


_CN_NUM = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

# 「5个红色物块，去A区」/「抓取两个蓝色到A」/「红色5个放到C区」
_DIRECT_CLAUSE = re.compile(
    r"(?:"
    r"(?P<n1>\d+|[一二两三四五六七八九十])个?"
    r"(?P<c1>红(?:色)?|蓝(?:色)?|red|blue)"
    r"(?:物块|货物|方块|积木|块)?"
    r"|"
    r"(?P<c2>红(?:色)?|蓝(?:色)?|red|blue)"
    r"(?:物块|货物|方块|积木|块)?"
    r"(?P<n2>\d+|[一二两三四五六七八九十])个?"
    r")"
    r"[,;]?"
    r"(?:抓取|放置|投放|放到|运到|送到|去|到|→|->)?"
    r"(?P<z>[ABCabc])区?",
    re.IGNORECASE,
)


def _parse_count_token(tok: str) -> int:
    tok = (tok or "").strip()
    if tok.isdigit():
        return int(tok)
    if tok in _CN_NUM:
        return _CN_NUM[tok]
    raise ValueError(f"bad count token: {tok!r}")


def _color_token(tok: str) -> str:
    t = tok.lower()
    if t.startswith("红") or t == "red":
        return "red"
    if t.startswith("蓝") or t == "blue":
        return "blue"
    raise ValueError(f"bad color token: {tok!r}")


def try_parse_direct_command(question: str) -> Optional[Tuple[str, int, int]]:
    """If question is already a robot task, return (command, x_red, y_blue).

    Skips official ceil(/5) and ≥3→A mapping. Returns None if not direct.
    """
    raw = (question or "").strip()
    if not raw:
        return None

    # Structured command already
    try:
        if re.search(r"(red|blue)\s*[,#_]", raw, re.I) or raw.strip().startswith("["):
            tasks = parse_command(raw)
            if tasks:
                x = sum(t.count for t in tasks if t.color == "red")
                y = sum(t.count for t in tasks if t.color == "blue")
                return format_command(tasks), x, y
    except Exception:
        pass

    text = _normalize(raw)
    # Official word problems mention 衣柜/宠物店/T恤/ceil 等 → not direct
    if any(
        k in text
        for k in (
            "衣柜",
            "宠物店",
            "T恤",
            "裤子",
            "领养",
            "计算x",
            "计算X",
            "设猫",
            "设T恤",
            "ceil",
        )
    ):
        return None

    tasks: List[PickTask] = []
    for m in _DIRECT_CLAUSE.finditer(text):
        n_tok = m.group("n1") or m.group("n2")
        c_tok = m.group("c1") or m.group("c2")
        z = m.group("z").upper()
        if not n_tok or not c_tok:
            continue
        try:
            n = _parse_count_token(n_tok)
            color = _color_token(c_tok)
        except ValueError:
            continue
        if n < 1:
            continue
        tasks.append(PickTask(color, n, z))

    if not tasks:
        return None
    x = sum(t.count for t in tasks if t.color == "red")
    y = sum(t.count for t in tasks if t.color == "blue")
    return format_command(tasks), x, y


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


def is_official_xy(x: int, y: int) -> bool:
    """正式赛出题约束（讲解 PDF）：x≥1、y≥1，且 x+y=5。"""
    return int(x) >= 1 and int(y) >= 1 and int(x) + int(y) == 5


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
    """赛前映射例（规则 PDF）：数量≥threshold → high_zone，否则 → low_zone。

    默认 threshold=3、A/C，与「数量大于等于3的去A区，其他去C区」一致。
    数量门槛是 ≥3，不是 >0 / >1；>0 只表示该颜色有任务要发。
    """

    def zone_for(n: int) -> str:
        return high_zone if n >= threshold else low_zone

    tasks: List[PickTask] = []
    if x > 0:
        tasks.append(PickTask(x_color.lower(), x, zone_for(x)))
    if y > 0:
        tasks.append(PickTask(y_color.lower(), y, zone_for(y)))
    return tasks


def question_to_command(question: str, mapping: Dict | None = None) -> str:
    direct = try_parse_direct_command(question)
    if direct is not None:
        return direct[0]
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
