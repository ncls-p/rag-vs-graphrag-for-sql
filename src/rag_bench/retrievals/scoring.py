from __future__ import annotations

import math
from typing import List, Set


def cosine(u: List[float], v: List[float]) -> float:
    if not u or not v or len(u) != len(v):
        return 0.0
    dot = 0.0
    nu = 0.0
    nv = 0.0
    for a, b in zip(u, v):
        dot += a * b
        nu += a * a
        nv += b * b
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (math.sqrt(nu) * math.sqrt(nv))


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = a.intersection(b)
    union = a.union(b)
    if not union:
        return 0.0
    return float(len(inter)) / float(len(union))
