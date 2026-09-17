"""LLM / hybrid solver: question → (x,y) → mapped task command string."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from yzbot_task.command_parser import format_command
from yzbot_task.deepseek_client import solve_xy_deepseek
from yzbot_task.local_solver import map_xy_to_tasks, solve_xy_from_question


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
        return solve_xy_deepseek(
            question,
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_sec=timeout_sec,
        )
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
            return llm_xy
        if llm_xy == local_xy:
            return llm_xy
        # Mismatch: one retry with explicit reminder, then trust local
        try:
            retry_q = (
                question
                + "\n请仔细复核：都需要/都领养必须×人数；ceil(14/5)=3。"
                + "只输出 JSON。"
            )
            llm2 = solve_xy_deepseek(
                retry_q,
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout_sec=timeout_sec,
            )
            if llm2 == local_xy:
                return llm2
        except Exception:
            pass
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
    """Return (task_command, x, y). Mapping is always applied locally."""
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
