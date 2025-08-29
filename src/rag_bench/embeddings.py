from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from .config import Config


@dataclass
class EmbeddingResult:
    vector: List[float]
    elapsed_ms: float


class OllamaEmbedder:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        retries: Optional[int] = None,
        backoff_seconds: float = 0.5,
    ):
        cfg = Config()
        self.base_url = base_url or cfg.ollama_base_url
        self.model = model or cfg.ollama_model
        self.timeout = timeout or cfg.http_timeout_seconds
        self.retries = retries if retries is not None else cfg.request_retries
        self.backoff_seconds = backoff_seconds

    def _post_embeddings(self, text: str) -> Dict[str, Any]:
        base = self.base_url.rstrip("/")
        # Try both common Ollama paths for compatibility
        candidate_paths = ["/api/embeddings", "/api/embed"]
        payload = {"model": self.model, "input": text}
        last_err: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            for path in candidate_paths:
                url = f"{base}{path}"
                try:
                    resp = requests.post(url, json=payload, timeout=self.timeout)
                    resp.raise_for_status()
                    data = resp.json()

                    # Accept known response shapes only if a non-empty vector is present.
                    # Some Ollama builds return {"embedding": []} from /api/embeddings;
                    # in that case, fall through and try /api/embed.
                    if "embedding" in data and isinstance(data["embedding"], list):
                        if len(data["embedding"]) > 0:
                            return data
                        # empty -> try next path
                        continue
                    if "embeddings" in data and isinstance(data["embeddings"], list):
                        embs = data["embeddings"]
                        # Either a flat vector or list of vectors
                        if embs and (
                            (isinstance(embs[0], list) and len(embs[0]) > 0)
                            or (not isinstance(embs[0], list) and len(embs) > 0)
                        ):
                            return data
                        continue
                    if (
                        "data" in data
                        and isinstance(data["data"], list)
                        and data["data"]
                        and isinstance(data["data"][0], dict)
                        and isinstance(data["data"][0].get("embedding"), list)
                        and len(data["data"][0]["embedding"]) > 0
                    ):
                        return data
                except Exception as e:
                    last_err = e
            # Backoff before retrying
            if attempt < self.retries:
                time.sleep(self.backoff_seconds * attempt)
        if last_err:
            raise last_err
        raise RuntimeError("Embeddings endpoint returned no valid data")

    def embed_one(self, text: str) -> EmbeddingResult:
        t0 = time.perf_counter()
        data = self._post_embeddings(text)
        # Support multiple response shapes:
        # - {"embedding": [...]}
        # - {"embeddings": [[...]]} or {"embeddings": [...]}
        # - {"data": [{"embedding": [...]}]}
        if "embedding" in data and isinstance(data["embedding"], list):
            vec = data["embedding"]
        elif "embeddings" in data and isinstance(data["embeddings"], list):
            embs = data["embeddings"]
            if embs and isinstance(embs[0], list):
                vec = embs[0]
            else:
                vec = embs
        elif "data" in data and data["data"] and "embedding" in data["data"][0]:
            vec = data["data"][0]["embedding"]
        else:
            raise ValueError(
                f"Unexpected embeddings response: {json.dumps(data)[:200]}"
            )
        # Normalize to float list
        vec = [float(x) for x in vec]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return EmbeddingResult(vector=vec, elapsed_ms=elapsed_ms)

    def embed_many(self, texts: List[str]) -> List[EmbeddingResult]:
        out: List[EmbeddingResult] = []
        for idx, t in enumerate(texts):
            out.append(self.embed_one(t))
        return out

    def warmup_dimension(self) -> int:
        res = self.embed_one("warmup")
        return len(res.vector)


if __name__ == "__main__":
    # Lightweight CLI to probe the model and print dimension
    emb = OllamaEmbedder()
    try:
        dim = emb.warmup_dimension()
        print(json.dumps({"model": emb.model, "dimension": dim}))
    except Exception as e:
        print(json.dumps({"model": emb.model, "error": str(e)}))
