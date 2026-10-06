"""Robust extraction of JSON objects and numbers from LLM output."""
from __future__ import annotations

import json
import re
from typing import Any, Optional

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _balanced_spans(text: str, open_ch: str, close_ch: str):
    """Yield top-level balanced substrings starting with open_ch (string-literal aware)."""
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth > 0:
            in_str = True
        elif ch == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_ch and depth > 0:
            depth -= 1
            if depth == 0:
                yield text[start : i + 1]


def _loads_lenient(s: str) -> Optional[Any]:
    for candidate in (s, re.sub(r",\s*([}\]])", r"\1", s)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return None


def extract_json(text: str) -> Optional[Any]:
    """Return the first parseable JSON object (preferred) or array in `text`, else None."""
    if not text:
        return None
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for chunk in candidates:
        parsed = _loads_lenient(chunk.strip())
        if parsed is not None:
            return parsed
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            for span in _balanced_spans(chunk, open_ch, close_ch):
                parsed = _loads_lenient(span)
                if parsed is not None:
                    return parsed
    return None


def to_float(value: Any, default: float = 0.0) -> float:
    """Coerce numbers or strings like '120 kcal' / '~2.5' to float."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = _NUM.search(value.replace(",", ""))
        if m:
            return float(m.group(0))
    return default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
