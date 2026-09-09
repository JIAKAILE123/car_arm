"""Parse fixed task command strings.

Supported formats:
  Count mode (解题/批量): [red,4,A;blue,1,C]  → 4 个红块、1 个蓝块（自动按序号取）
  Cube-id mode (指定编号): [red#1,A;blue#2,B;red#3,C]  → 指定 red_1 / blue_2 / red_3
  Also accepts: red_1,A  (underscore form)
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Optional


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

# Count mode: red,4,A
_TOKEN_COUNT = re.compile(
    r"(red|blue)\s*,\s*(\d+)\s*,\s*([ABC])",
    re.IGNORECASE,
)


def parse_command(text: str) -> List[PickTask]:
    if not text or not str(text).strip():
        raise ValueError("empty command")

    raw = str(text).strip()
    m = re.search(r"\[([^\]]+)\]", raw)
    payload = m.group(1) if m else raw

    tasks: List[PickTask] = []
    # Split on ; and parse each segment (avoids count-regex eating id forms wrongly)
    parts = [p.strip() for p in payload.split(";") if p.strip()]
    for part in parts:
        mid = _TOKEN_ID.fullmatch(part.replace(" ", "")) or _TOKEN_ID.search(part)
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

    if not tasks:
        raise ValueError(f"cannot parse command: {text!r}")
    return tasks


def format_command(tasks: List[PickTask]) -> str:
    inner = ";".join(str(t) for t in tasks)
    return f"[{inner}]"
