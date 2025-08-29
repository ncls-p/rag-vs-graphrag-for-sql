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
        url_chat = f"{self.base_url}/chat/completions"
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

        def _parse_openai_like(d: Dict[str, Any]) -> str:
            # Standard Chat Completions
            if isinstance(d.get("choices"), list) and d["choices"]:
                msg = d["choices"][0].get("message") or {}
                if isinstance(msg, dict):
                    return str(msg.get("content") or "")
            # Responses API content list (some providers)
            if isinstance(d.get("content"), list) and d["content"]:
                parts = []
                for it in d["content"]:
                    if isinstance(it, dict) and it.get("type") == "output_text":
                        parts.append(str(it.get("text") or ""))
                    elif isinstance(it, dict) and "text" in it:
                        parts.append(str(it.get("text") or ""))
                if parts:
                    return "".join(parts)
            # Fallback keys used by some gateways
            if isinstance(d.get("output_text"), str):
                return d["output_text"]
            if isinstance(d.get("text"), str):
                return d["text"]
            return ""

        # Try Chat Completions
        resp = requests.post(url_chat, headers=headers, data=json.dumps(body), timeout=self.timeout)
        if not resp.ok:
            # Improve error message; include API response body if available
            try:
                errj = resp.json()
                emsg = errj.get("error", {}).get("message") or errj.get("message") or resp.text
            except Exception:
                emsg = resp.text
            # Fallback: try /responses for providers that don't support /chat/completions
            url_resp = f"{self.base_url}/responses"
            payload = {
                "model": self.model,
                # Join messages to a single prompt
                "input": "\n".join([f"{m.get('role')}: {m.get('content')}" for m in messages]),
                "temperature": temperature,
            }
            if max_tokens is not None:
                payload["max_output_tokens"] = max_tokens
            try:
                r2 = requests.post(url_resp, headers=headers, data=json.dumps(payload), timeout=self.timeout)
                r2.raise_for_status()
                data2 = r2.json()
                content2 = _parse_openai_like(data2)
                usage2 = data2.get("usage") or {}
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return ChatResponse(
                    content=content2,
                    model=data2.get("model") or self.model,
                    prompt_tokens=usage2.get("prompt_tokens"),
                    completion_tokens=usage2.get("completion_tokens"),
                    total_tokens=usage2.get("total_tokens"),
                    elapsed_ms=elapsed_ms,
                )
            except Exception:
                raise RuntimeError(f"OpenAI chat error {resp.status_code}: {emsg}")

        data = resp.json()
        content = _parse_openai_like(data)
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
