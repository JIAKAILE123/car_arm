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


# ===== 国赛三条映射规则（PDF《创新赛题目和映射规则》，赛前由裁判三选一）=====
# ① 颜色按变量固定，区域按数量：x=红、y=蓝；数量≥3 → A 区，否则 → C 区。
# ② 颜色按数量：≥3 的那个代表红色、否则蓝色；区域按变量：x → B 区、y → A 区。
# ③ 颜色按大小：数值大的为蓝色、数值少的为红色；区域按颜色：蓝 → B 区、红 → C 区。
OFFICIAL_RULES = {
    1: "x 代表红色,y 代表蓝色;数值大于等于 3 的去 A 区,其他颜色去 C 区",
    2: "xy 中,数值大于等于 3 值的代表红色，否则代表蓝色；x 去 B 区,y 去 A 区",
    3: "蓝色去 B 区,红色去 C 区; xy 中,数值大的是蓝色，数值少的是红色",
}
RULE_MARKS = {1: "①", 2: "②", 3: "③"}


def _norm_rule_text(text: str) -> str:
    """规则原文归一化：全角标点 → 半角，去空白，转小写。"""
    t = str(text or "").strip()
    for a, b in (
        ("，", ","), ("；", ";"), ("：", ":"), ("、", ","),
        ("（", "("), ("）", ")"), ("\u3000", " "), ("　", " "),
    ):
        t = t.replace(a, b)
    return "".join(t.split()).lower()


def detect_rule_id(text: str, default: int = 1) -> int:
    """从规则原文识别国赛规则编号（1/2/3）；识别不出回 default。"""
    t = _norm_rule_text(text)
    if not t:
        return default
    if "x去b区" in t or ("b区" in t and "y去a区" in t):
        return 2
    if "数值大的" in t or ("蓝色去b区" in t and "红色去c区" in t):
        return 3
    if "x代表红色" in t or ("大于等于3" in t and "a区" in t):
        return 1
    return default


def resolve_rule_id(rule_id=0, rule_text: str = "") -> int:
    """解析规则编号：显式 1/2/3 优先，否则按规则原文自动识别，默认 1。"""
    try:
        rid = int(rule_id)
    except (TypeError, ValueError):
        rid = 0
    if rid in (1, 2, 3):
        return rid
    return detect_rule_id(rule_text, default=1)


def rule_banner(rule_id=1) -> str:
    """规则面板文案，例如「规则②：xy 中,数值大于等于 3 ...」。"""
    rid = resolve_rule_id(rule_id)
    return f"规则{RULE_MARKS[rid]}：{OFFICIAL_RULES[rid]}"


def rule_assign(
    x: int,
    y: int,
    *,
    rule_id=1,
    x_color: str = "red",
    y_color: str = "blue",
    threshold: int = 3,
    high_zone: str = "A",
    low_zone: str = "C",
):
    """按国赛规则给出每个变量的 (变量, 数量, 颜色, 区域) 四元组列表。"""
    rid = resolve_rule_id(rule_id)
    xt, yt = int(x), int(y)
    thr = int(threshold)
    xc, yc = str(x_color).lower(), str(y_color).lower()
    out = []
    if rid == 2:
        # 数值≥threshold 的代表红色，否则蓝色；区域按变量：x→B、y→A
        for var, n in (("x", xt), ("y", yt)):
            out.append((var, n, xc if n >= thr else yc, "B" if var == "x" else "A"))
    elif rid == 3:
        # 数值大的为蓝色，小的为红色；区域按颜色：蓝→B、红→C
        for var, n, other in (("x", xt, yt), ("y", yt, xt)):
            col = yc if n >= other else xc
            out.append((var, n, col, "B" if col == yc else "C"))
    else:
        # 规则①：颜色按变量固定，区域按数量 ≥threshold → high_zone
        for var, n, col in (("x", xt, xc), ("y", yt, yc)):
            out.append((var, n, col, high_zone if n >= thr else low_zone))
    return out


def rule_view(
    x: int,
    y: int,
    *,
    rule_id=1,
    x_color: str = "red",
    y_color: str = "blue",
    threshold: int = 3,
    high_zone: str = "A",
    low_zone: str = "C",
) -> Dict:
    """可视化用：按规则算出红/蓝数量与红/蓝区域。"""
    pairs = rule_assign(
        x, y, rule_id=rule_id, x_color=x_color, y_color=y_color,
        threshold=threshold, high_zone=high_zone, low_zone=low_zone,
    )
    red_n = sum(n for _v, n, c, _z in pairs if c == "red")
    blue_n = sum(n for _v, n, c, _z in pairs if c == "blue")
    red_zone = next((z for _v, n, c, z in pairs if c == "red" and n > 0), "")
    blue_zone = next((z for _v, n, c, z in pairs if c == "blue" and n > 0), "")
    rid = resolve_rule_id(rule_id)
    return {
        "rule_id": rid,
        "red_n": red_n,
        "blue_n": blue_n,
        "red_zone": red_zone,
        "blue_zone": blue_zone,
        "text": rule_banner(rid),
    }


def map_xy_to_tasks(
    x: int,
    y: int,
    *,
    x_color: str = "red",
    y_color: str = "blue",
    threshold: int = 3,
    high_zone: str = "A",
    low_zone: str = "C",
    rule_id=1,
) -> List[PickTask]:
    """按国赛映射规则把 (x, y) 变成抓取任务列表。

    rule_id=1（默认）与旧行为完全一致：颜色按变量、区域按数量≥threshold。
    rule_id=2/3 见 OFFICIAL_RULES 注释；数量门槛是 ≥3（x+y=5 时不会同为 ≥3）。
    """
    tasks: List[PickTask] = []
    for _var, n, col, zone in rule_assign(
        x, y, rule_id=rule_id, x_color=x_color, y_color=y_color,
        threshold=threshold, high_zone=high_zone, low_zone=low_zone,
    ):
        if n > 0:
            tasks.append(PickTask(col, n, zone))
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
        rule_id=resolve_rule_id(mapping.get("rule_id", 0), mapping.get("rule_text", "")),
    )
    return format_command(tasks)
