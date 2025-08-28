from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

from ..retrievals.types import Hit


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return float(min(values))
    if p >= 1:
        return float(max(values))
    xs = sorted(values)
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(xs[int(k)])
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return float(d0 + d1)


def find_rank(hits: List[Hit], target_id: int) -> Tuple[int, Optional[Hit]]:
    for idx, h in enumerate(hits, start=1):
        if h.id == target_id:
            return idx, h
    return 0, None


def recall_at_k(hits: List[Hit], target_id: int, k: int) -> float:
    rank, _ = find_rank(hits[:k], target_id)
    return 1.0 if rank > 0 else 0.0


def mrr_at_k(hits: List[Hit], target_id: int, k: int) -> float:
    rank, _ = find_rank(hits[:k], target_id)
    return 1.0 / float(rank) if rank > 0 else 0.0


@dataclass
class BackendMetrics:
    recall_at_5: float = 0.0
    mrr_at_5: float = 0.0
    query_latency_ms_mean: float = 0.0
    query_latency_ms_p95: float = 0.0
    index_time_ms: float = 0.0
    corrections_applied: int = 0
    corrections_helped: int = 0


@dataclass
class PerQueryRecord:
    id: int
    rank: int
    score: float
    latency_ms: float
    correction_applied: bool
    components: dict
