"""Parse fixed task command strings.

Supported formats:
  Count mode (解题/批量): [red,4,A;blue,1,C]  → 4 个红块、1 个蓝块（自动按序号取）
  Cube-id mode (指定编号): [red#1,A;blue#2,B;red#3,C]  → 指定 red_1 / blue_2 / red_3
  Also accepts: red_1,A  (underscore form)

Ambiguous form [red,1,A;blue,2,B;...] is count by default. If that would need more
than max_cubes, it is auto-reinterpreted as cube ids (1→cube#1, not “one cube”).
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple


@dataclass
class PickTask:
    color: str  # red | blue
    count: int
    zone: str  # A | B | C
    cube_id: Optional[int] = None  # if set, pick this numbered cube once

    def __str__(self) -> str:
        if self.cube_id is not None:
            return f"{self.color}#{self.cube_id},{self.zone}"
        return f"{self.color},{self.count},{self.zone}"


# Explicit cube: red#1,A  or  red_1,A
_TOKEN_ID = re.compile(
    r"(red|blue)\s*[#_|]\s*([1-5])\s*,\s*([ABC])",
    re.IGNORECASE,
)

# Count mode: red,4,A  (also used for ambiguous id-without-#)
_TOKEN_COUNT = re.compile(
    r"(red|blue)\s*,\s*(\d+)\s*,\s*([ABC])",
    re.IGNORECASE,
)


def _need_by_color(tasks: List[PickTask]) -> Dict[str, int]:
    need = {"red": 0, "blue": 0}
    for t in tasks:
        c = t.color.lower()
        if t.cube_id is not None:
            need[c] += 1
        else:
            need[c] += int(t.count)
    return need


def _parse_parts(parts: List[str]) -> List[PickTask]:
    tasks: List[PickTask] = []
    for part in parts:
        # Prefer id form when # or _ present
        if re.search(r"(red|blue)\s*[#_]", part, re.I):
            m_id = _TOKEN_ID.search(part)
            if not m_id:
                raise ValueError(f"bad cube-id token: {part!r}")
            color, cid, zone = m_id.group(1), int(m_id.group(2)), m_id.group(3)
            tasks.append(
                PickTask(
                    color=color.lower(),
                    count=1,
                    zone=zone.upper(),
                    cube_id=cid,
                )
            )
            continue
        m_c = _TOKEN_COUNT.search(part)
        if not m_c:
            raise ValueError(f"cannot parse token: {part!r}")
        color, count, zone = m_c.group(1), int(m_c.group(2)), m_c.group(3)
        if count < 1:
            raise ValueError(f"invalid count: {count}")
        tasks.append(
            PickTask(color=color.lower(), count=count, zone=zone.upper())
        )
    return tasks


def _reinterpret_count_as_ids(parts: List[str]) -> Optional[List[PickTask]]:
    """Treat color,N,zone as cube#N → zone when N is in 1..5."""
    tasks: List[PickTask] = []
    seen: Dict[str, set] = {"red": set(), "blue": set()}
    for part in parts:
        if re.search(r"(red|blue)\s*[#_]", part, re.I):
            m_id = _TOKEN_ID.search(part)
            if not m_id:
                return None
            color, cid, zone = (
                m_id.group(1).lower(),
                int(m_id.group(2)),
                m_id.group(3).upper(),
            )
        else:
            m_c = _TOKEN_COUNT.search(part)
            if not m_c:
                return None
            color = m_c.group(1).lower()
            cid = int(m_c.group(2))
            zone = m_c.group(3).upper()
            if cid < 1 or cid > 5:
                return None
        if cid in seen[color]:
            continue
        seen[color].add(cid)
        tasks.append(
            PickTask(color=color, count=1, zone=zone, cube_id=cid)
        )
    return tasks or None


def parse_command(text: str, *, max_cubes: int = 5) -> List[PickTask]:
    if not text or not str(text).strip():
        raise ValueError("empty command")

    raw = str(text).strip()
    m = re.search(r"\[([^\]]+)\]", raw)
    payload = m.group(1) if m else raw

    parts = [p.strip() for p in payload.split(";") if p.strip()]
    if not parts:
        raise ValueError(f"cannot parse command: {text!r}")

    tasks = _parse_parts(parts)
    need = _need_by_color(tasks)
    over = any(need[c] > int(max_cubes) for c in ("red", "blue"))
    if over:
        as_ids = _reinterpret_count_as_ids(parts)
        if as_ids is not None:
            need2 = _need_by_color(as_ids)
            if all(need2[c] <= int(max_cubes) for c in ("red", "blue")):
                return as_ids
        raise ValueError(
            f"need {need} cubes but only {max_cubes} each — "
            f"use count e.g. [red,4,A;blue,1,C] or ids e.g. [red#1,A;blue#2,B]"
        )
    return tasks


def format_command(tasks: List[PickTask]) -> str:
    inner = ";".join(str(t) for t in tasks)
    return f"[{inner}]"
