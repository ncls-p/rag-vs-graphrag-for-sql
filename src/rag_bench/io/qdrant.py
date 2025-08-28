from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ..config import Config
from ..dataloader import load_records
from ..embeddings import OllamaEmbedder
from ..utils.text import combined_text


class QdrantIO:
    def __init__(
        self, cfg: Optional[Config] = None, client: Optional[QdrantClient] = None
    ) -> None:
        self.cfg = cfg or Config()
        self.base_collection = self.cfg.qdrant_collection
        self.client = client or QdrantClient(
            url=self.cfg.qdrant_url,
            api_key=self.cfg.qdrant_api_key,
            prefer_grpc=True,
        )

    def _collection_name(self, source_format: str) -> str:
        sf = (source_format or "unknown").lower()
        return f"{self.base_collection}_{sf}"

    def ensure_collection(
        self,
        vector_size: int,
        distance: qmodels.Distance = qmodels.Distance.COSINE,
        collection_name: Optional[str] = None,
    ) -> None:
        name = collection_name or self.base_collection
        try:
            _ = self.client.get_collection(name)
            return
        except Exception:
            pass

        self.client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=distance),
        )

    def upsert_points(
        self, points: List[qmodels.PointStruct], collection_name: str
    ) -> None:
        if not points:
            return
        self.client.upsert(collection_name=collection_name, points=points)

    def index_dataset(
        self,
        data_path: Path,
        batch_size: int = 32,
        warmup: bool = True,
        progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        records = load_records(data_path)
        if not records:
            return {"indexed": 0, "batches": 0}

        embedder = OllamaEmbedder()
        if warmup:
            try:
                _ = embedder.embed_one("warmup")
            except Exception:
                pass

        dim: Optional[int] = None

        total = 0
        batches = 0
        per_format_batches: Dict[str, List[qmodels.PointStruct]] = {}
        per_format_counts: Dict[str, int] = {}
        per_format_dim: Dict[str, int] = {}
        skipped = 0

        # Progress metadata
        total_records = len(records)
        files_order: List[str] = []
        seen_files: set[str] = set()
        for rec in records:
            sp = str(rec.get("source_path") or "")
            if sp and sp not in seen_files:
                seen_files.add(sp)
                files_order.append(sp)
        total_files = len(files_order) if files_order else 1
        file_index = 0
        current_file: Optional[str] = None
        processed = 0
        if progress:
            try:
                progress(
                    {
                        "backend": "qdrant",
                        "total_files": total_files,
                        "total_records": total_records,
                        "file_index": 0,
                        "file_path": None,
                        "record_index": 0,
                    }
                )
            except Exception:
                pass

        for rec in records:
            rid = rec.get("id")
            if rid is None:
                continue
            # Progress update on file boundary
            spath = str(rec.get("source_path") or "")
            if spath and spath != current_file:
                current_file = spath
                file_index += 1
            processed += 1
            if progress:
                try:
                    progress(
                        {
                            "backend": "qdrant",
                            "total_files": total_files,
                            "total_records": total_records,
                            "file_index": file_index,
                            "file_path": current_file,
                            "record_index": processed,
                        }
                    )
                except Exception:
                    pass
            text = combined_text(rec)
            vec: Optional[List[float]] = None
            # Outer retry around embedder (which already retries internally)
            try:
                vec = embedder.embed_one(text).vector
            except Exception:
                try:
                    vec = embedder.embed_one(text).vector
                except Exception:
                    skipped += 1
                    continue
            if dim is None:
                dim = len(vec)
                if not dim or dim <= 0:
                    raise AssertionError(
                        "Embedding dimension is 0; check Ollama embeddings response"
                    )
            payload = {
                "id": rec.get("id"),
                "question": rec.get("question"),
                "answer_text": rec.get("answer_text"),
                "entities": rec.get("entities", []),
                "doc_type": rec.get("doc_type"),
                "tags": rec.get("tags", []),
                "source_format": rec.get("source_format", "unknown"),
            }
            point = qmodels.PointStruct(id=rid, vector=vec, payload=payload)

            sf = str(rec.get("source_format", "unknown")).lower()
            col = self._collection_name(sf)
            if sf not in per_format_batches:
                per_format_batches[sf] = []
                per_format_counts[sf] = 0
                per_format_dim[sf] = dim
                self.ensure_collection(
                    dim, qmodels.Distance.COSINE, collection_name=col
                )

            per_format_batches[sf].append(point)

            if len(per_format_batches[sf]) >= batch_size:
                self.upsert_points(per_format_batches[sf], collection_name=col)
                total += len(per_format_batches[sf])
                per_format_counts[sf] += len(per_format_batches[sf])
                batches += 1
                per_format_batches[sf] = []

        for sf, b in per_format_batches.items():
            if not b:
                continue
            col = self._collection_name(sf)
            self.upsert_points(b, collection_name=col)
            total += len(b)
            per_format_counts[sf] += len(b)
            batches += 1

        return {
            "indexed": total,
            "batches": batches,
            "base_collection": self.base_collection,
            "per_format_counts": per_format_counts,
            "vector_size": dim or 0,
            "skipped": skipped,
        }

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        source_format: Optional[str] = None,
        source_formats: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        k = top_k or self.cfg.top_k
        embedder = OllamaEmbedder()
        qvec = embedder.embed_one(query).vector
        out: List[Dict[str, Any]] = []
        try:
            cols = self.client.get_collections()
            names = [c.name for c in getattr(cols, "collections", [])]
        except Exception:
            names = []
        # Normalize allowed formats
        allowed: Optional[List[str]] = None
        if source_formats:
            allowed = [str(sf).lower() for sf in source_formats if sf]
        elif source_format:
            allowed = [str(source_format).lower()]

        if allowed:
            col = self._collection_name(str(source_format).lower())
            allowed_names = {self._collection_name(sf) for sf in allowed}
            target_cols = [n for n in names if n in allowed_names]
        else:
            target_cols = [n for n in names if n.startswith(self.base_collection + "_")]
        if not target_cols:
            target_cols = [self.base_collection]

        for col in target_cols:
            try:
                results = self.client.search(
                    collection_name=col,
                    query_vector=qvec,
                    limit=k,
                    with_payload=with_payload,
                    with_vectors=with_vectors,
                )
            except Exception:
                continue
            for sp in results:
                out.append(
                    {
                        "id": sp.id,
                        "score": sp.score,
                        "payload": sp.payload if with_payload else None,
                    }
                )
        out.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return out[:k]
