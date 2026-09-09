"""Parse fixed task command strings.

Supported formats:
  [red,4,A;blue,1,C]
  red,4,A;blue,1,C
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List


@dataclass
class PickTask:
    color: str  # red | blue
    count: int
    zone: str  # A | B | C

    def __str__(self) -> str:
        return f"{self.color},{self.count},{self.zone}"


_TOKEN = re.compile(
    r"(red|blue)\s*,\s*(\d+)\s*,\s*([ABC])",
    re.IGNORECASE,
)


def parse_command(text: str) -> List[PickTask]:
    if not text or not str(text).strip():
        raise ValueError("empty command")

    raw = str(text).strip()
    # Prefer bracketed payload if present
    m = re.search(r"\[([^\]]+)\]", raw)
    payload = m.group(1) if m else raw

    tasks: List[PickTask] = []
    for color, count, zone in _TOKEN.findall(payload):
        n = int(count)
        if n < 1:
            raise ValueError(f"invalid count: {n}")
        tasks.append(
            PickTask(color=color.lower(), count=n, zone=zone.upper())
        )

    if not tasks:
        raise ValueError(f"cannot parse command: {text!r}")
    return tasks


def format_command(tasks: List[PickTask]) -> str:
    inner = ";".join(str(t) for t in tasks)
    return f"[{inner}]"
