"""DeepSeek OpenAI-compatible chat client for contest word problems.

API key (never commit):
  export DEEPSEEK_API_KEY=sk-...
  or put the key in ~/.config/yzbot/deepseek_api_key
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

SYSTEM_PROMPT = """你是赛题应用题解题器。容量一律为 5：
  x = ceil(第一种物品总数 / 5)
  y = ceil(第二种物品总数 / 5)
规则：
1. 「甲、乙都需要N / 都领养N」= 每人各 N，必须 ×人数。
2. 同一人句子里逗号分隔的多种物品分别累加。
3. ceil 是向上取整：14/5→3，不是 2 也不是 4。
4. 正式出题程序：x≥1、y≥1，且 x+y=5（不会出现 0）。
只输出一行 JSON，不要解释：
{"x": <整数>, "y": <整数>}
不要输出颜色或放置区（映射由赛前规则+本地代码完成）。

例（宠物店）：猫=2+2+2+3=9→x=2；狗=5+5+4=14→y=3 → {"x":2,"y":3}
例（衣柜）：T恤=2+7+7+2=18→x=4；裤子=1+1+1=3→y=1 → {"x":4,"y":1}
"""


def load_api_key(explicit: Optional[str] = None) -> str:
    if explicit:
        return str(explicit).strip()
    env = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if env:
        return env
    path = Path.home() / ".config" / "yzbot" / "deepseek_api_key"
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    raise RuntimeError(
        "missing DeepSeek API key: set DEEPSEEK_API_KEY or "
        f"write key to {path}"
    )


def _extract_xy_json(text: str) -> Tuple[int, int]:
    raw = text.strip()
    # Prefer fenced or bare JSON object
    m = re.search(r"\{[^{}]*\}", raw, flags=re.S)
    if m:
        raw = m.group(0)
    data = json.loads(raw)
    x = int(data["x"])
    y = int(data["y"])
    if x < 0 or y < 0:
        raise ValueError(f"negative xy in model output: {data}")
    return x, y


def solve_xy_deepseek(
    question: str,
    *,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    timeout_sec: float = 30.0,
) -> Tuple[int, int]:
    """Call DeepSeek chat completions → (x, y)."""
    key = load_api_key(api_key)
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question.strip()},
        ],
        "temperature": 0.0,
        "max_tokens": 64,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek HTTP {exc.code}: {detail[:300]}") from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"DeepSeek request failed: {exc}") from exc

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected DeepSeek response: {payload!r}") from exc

    return _extract_xy_json(str(content))
