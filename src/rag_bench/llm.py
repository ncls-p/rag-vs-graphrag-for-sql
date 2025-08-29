from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from .config import Config


@dataclass
class ChatResponse:
    content: str
    model: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    elapsed_ms: float


class OpenAIChat:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        cfg = Config()
        self.base_url = (base_url or cfg.openai_base_url).rstrip("/")
        self.api_key = api_key or cfg.openai_api_key or ""
        self.model = model or cfg.openai_model
        self.timeout = timeout or cfg.http_timeout_seconds

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ChatResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if extra:
            body.update(extra)
        t0 = time.perf_counter()
        resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        # OpenAI compatible response
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage") or {}
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return ChatResponse(
            content=content,
            model=data.get("model") or self.model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            elapsed_ms=elapsed_ms,
        )

