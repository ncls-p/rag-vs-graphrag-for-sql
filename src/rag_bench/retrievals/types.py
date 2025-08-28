from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Hit:
    id: int
    score: float
    components: Dict[str, float]
    payload: Optional[Dict[str, Any]] = None

