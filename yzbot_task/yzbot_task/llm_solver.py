"""LLM / hybrid solver: question → (x,y) → mapped task command string."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from yzbot_task.command_parser import format_command
from yzbot_task.deepseek_client import solve_xy_deepseek
from yzbot_task.local_solver import (
    is_official_xy,
    map_xy_to_tasks,
    solve_xy_from_question,
    try_parse_direct_command,
)

_OFFICIAL_XY_HINT = (
    "\n请仔细复核：正式赛 x≥1、y≥1 且 x+y=5；"
    "都需要/都领养必须×人数；ceil(14/5)=3。只输出 JSON。"
)


def solve_xy(
    question: str,
    *,
    backend: str = "deepseek",
    api_key: Optional[str] = None,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    timeout_sec: float = 30.0,
) -> Tuple[int, int]:
    """backend: deepseek | local | auto | cross.

    cross: call DeepSeek, also try local; if both ok and disagree → prefer local
    (deterministic for official templates) after one DeepSeek retry.
    """
    b = (backend or "deepseek").strip().lower()
    if b == "local":
        return solve_xy_from_question(question)
    if b == "deepseek":
        xy = solve_xy_deepseek(
            question,
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_sec=timeout_sec,
        )
        if is_official_xy(*xy):
            return xy
        # 正式赛约束不满足时再催一次
        try:
            xy2 = solve_xy_deepseek(
                question + _OFFICIAL_XY_HINT,
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout_sec=timeout_sec,
            )
            if is_official_xy(*xy2):
                return xy2
        except Exception:
            pass
        return xy
    if b in ("auto", "cross"):
        local_xy = None
        try:
            local_xy = solve_xy_from_question(question)
        except Exception:
            local_xy = None
        try:
            llm_xy = solve_xy_deepseek(
                question,
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout_sec=timeout_sec,
            )
        except Exception:
            if local_xy is not None:
                return local_xy
            raise
        if local_xy is None:
            if is_official_xy(*llm_xy):
                return llm_xy
            try:
                llm2 = solve_xy_deepseek(
                    question + _OFFICIAL_XY_HINT,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    timeout_sec=timeout_sec,
                )
                return llm2
            except Exception:
                return llm_xy
        if llm_xy == local_xy:
            return llm_xy
        # Mismatch: one retry, then prefer local (尤其本地已满足正式赛约束时)
        try:
            llm2 = solve_xy_deepseek(
                question + _OFFICIAL_XY_HINT,
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout_sec=timeout_sec,
            )
            if llm2 == local_xy:
                return llm2
            if is_official_xy(*local_xy) and not is_official_xy(*llm2):
                return local_xy
            if is_official_xy(*llm2) and not is_official_xy(*local_xy):
                return llm2
        except Exception:
            pass
        if is_official_xy(*local_xy) and not is_official_xy(*llm_xy):
            return local_xy
        return local_xy
    raise ValueError(f"unknown backend: {backend}")


def question_to_command_llm(
    question: str,
    mapping: Optional[Dict] = None,
    *,
    backend: str = "deepseek",
    api_key: Optional[str] = None,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    timeout_sec: float = 30.0,
) -> Tuple[str, int, int]:
    """Return (task_command, x, y). Mapping is always applied locally.

    Direct robot phrases (「5个红色物块，去A区」) bypass ceil/zone mapping.
    """
    direct = try_parse_direct_command(question)
    if direct is not None:
        return direct
    mapping = mapping or {}
    x, y = solve_xy(
        question,
        backend=backend,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_sec=timeout_sec,
    )
    tasks = map_xy_to_tasks(
        x,
        y,
        x_color=str(mapping.get("x_color", "red")),
        y_color=str(mapping.get("y_color", "blue")),
        threshold=int(mapping.get("threshold", 3)),
        high_zone=str(mapping.get("high_zone", "A")),
        low_zone=str(mapping.get("low_zone", "C")),
    )
    return format_command(tasks), x, y
