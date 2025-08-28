from __future__ import annotations

from typing import Any, Dict


def combined_text(rec: Dict[str, Any]) -> str:
    q = rec.get("question", "") or ""
    a = rec.get("answer_text", "") or ""
    return (q + "\n" + a).strip()

