from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import urlparse


@dataclass(frozen=True)
class Config:
    # Embeddings
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.environ.get(
        "OLLAMA_EMBED_MODEL", "dengcao/Qwen3-Embedding-0.6B:q8_0"
    )
    http_timeout_seconds: float = float(os.environ.get("HTTP_TIMEOUT_SECONDS", "30"))
    request_retries: int = int(os.environ.get("REQUEST_RETRIES", "3"))

    # Qdrant
    qdrant_url: str = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.environ.get("QDRANT_API_KEY") or None
    qdrant_collection: str = os.environ.get("QDRANT_COLLECTION", "qa_demo")

    # Neo4j
    neo4j_uri: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    # Defaults align with docker-compose.yml (neo4j/neo4jtest) to improve DX.
    neo4j_user: Optional[str] = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password: Optional[str] = os.environ.get("NEO4J_PASSWORD", "neo4jtest")
    neo4j_read_only_user: Optional[str] = os.environ.get(
        "NEO4J_RO_USER", os.environ.get("NEO4J_USER", "neo4j")
    )
    neo4j_read_only_password: Optional[str] = os.environ.get(
        "NEO4J_RO_PASSWORD", os.environ.get("NEO4J_PASSWORD", "neo4jtest")
    )

    # Retrieval / scoring
    top_k: int = int(os.environ.get("TOP_K", "5"))
    shortlist_size: int = int(os.environ.get("SHORTLIST_SIZE", "50"))
    alpha: float = float(os.environ.get("NEO4J_SCORE_ALPHA", "0.7"))  # semantic cosine
    beta: float = float(os.environ.get("NEO4J_SCORE_BETA", "0.2"))  # entity jaccard
    gamma: float = float(os.environ.get("NEO4J_SCORE_GAMMA", "0.1"))  # neighbor boost

    # Agentic/self-correction
    enable_self_correction: bool = os.environ.get(
        "ENABLE_SELF_CORRECTION", "true"
    ).lower() in {"1", "true", "yes"}

    # Safety toggles
    allow_destructive_ops: bool = os.environ.get(
        "ALLOW_DESTRUCTIVE_OPS", "false"
    ).lower() in {"1", "true", "yes"}

    @staticmethod
    def _socket_check(host: str, port: int, timeout: float = 1.5) -> Tuple[bool, str]:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True, f"tcp:{host}:{port} reachable"
        except Exception as e:
            return False, f"tcp:{host}:{port} unreachable: {e}"

    @classmethod
    def _parse_host_port(
        cls, url: str, default_port: Optional[int] = None
    ) -> Tuple[Optional[str], Optional[int]]:
        try:
            parsed = urlparse(url)
            if parsed.scheme and parsed.netloc:
                host = parsed.hostname
                port = parsed.port or default_port
                return host, port
            # If raw "host:port"
            if ":" in url and "://" not in url:
                host, p = url.split(":", 1)
                return host, int(p)
            return None, default_port
        except Exception:
            return None, default_port

    def health_ollama(self) -> Tuple[bool, str]:
        host, port = self._parse_host_port(self.ollama_base_url, 11434)
        if not host or not port:
            return False, f"Invalid OLLAMA_BASE_URL: {self.ollama_base_url}"
        return self._socket_check(host, port)

    def health_qdrant(self) -> Tuple[bool, str]:
        host, port = self._parse_host_port(self.qdrant_url, 6333)
        if not host or not port:
            return False, f"Invalid QDRANT_URL: {self.qdrant_url}"
        return self._socket_check(host, port)

    def health_neo4j(self) -> Tuple[bool, str]:
        # Bolt port 7687 by default; we only do a TCP check here
        host, port = self._parse_host_port(self.neo4j_uri, 7687)
        if not host or not port:
            return False, f"Invalid NEO4J_URI: {self.neo4j_uri}"
        return self._socket_check(host, port)

    # OpenAI-compatible Chat API
    openai_base_url: str = os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1")
    openai_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY") or None
    openai_model: str = os.environ.get("OPENAI_MODEL", os.environ.get("OPENAI_CHAT_MODEL", "gpt-3.5-turbo"))
