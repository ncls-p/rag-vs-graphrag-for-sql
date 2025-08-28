from __future__ import annotations

from typing import List, Optional

from ..config import Config
from ..qdrant_io import QdrantIO
from .types import Hit


class QdrantRetriever:
    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or Config()
        self.qio = QdrantIO(self.cfg)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        source_format: Optional[str] = None,
        source_formats: Optional[List[str]] = None,
    ) -> List[Hit]:
        raw = self.qio.search(
            query=query,
            top_k=top_k,
            source_format=source_format,
            source_formats=source_formats,
        )
        out: List[Hit] = []
        for r in raw:
            out.append(
                Hit(
                    id=int(r["id"]),
                    score=float(r["score"]),
                    components={"semantic": float(r["score"])},
                    payload=r.get("payload"),
                )
            )
        return out
