from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from http.client import IncompleteRead, RemoteDisconnected
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def call_openai_compatible_chat(
    config: dict[str, Any],
    prompt: str,
    system_prompt: str = "You are a careful educational learner-simulation agent.",
) -> str:
    api_key = _resolve_api_key(config)
    base_url = str(config["base_url"]).rstrip("/")
    url = f"{base_url}/chat/completions"
    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": config.get("temperature", 0.2),
        "max_tokens": config.get("max_tokens", 600),
        "stream": bool(config.get("stream", False)),
    }
    if isinstance(config.get("extra_body"), dict):
        payload.update(config["extra_body"])

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    max_retries = int(config.get("max_retries", 2))
    retry_sleep = float(config.get("retry_sleep_seconds", 2))
    last_error: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(
                request,
                timeout=float(config.get("timeout_seconds", 60)),
            ) as response:
                if payload["stream"]:
                    return _read_streaming_chat_response(response)
                body = json.loads(response.read().decode("utf-8"))
                return _read_non_streaming_chat_response(body)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if not _is_retryable_http_error(exc.code) or attempt >= max_retries:
                raise RuntimeError(f"LLM API HTTP {exc.code}: {error_body}") from exc
            last_error = RuntimeError(f"LLM API HTTP {exc.code}: {error_body[:500]}")
        except TRANSIENT_CONNECTION_ERRORS as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    "LLM API connection failed after "
                    f"{attempt + 1} attempt(s). Last error: {type(exc).__name__}: {exc}"
                ) from exc
            last_error = exc
        time.sleep(retry_sleep * (attempt + 1))

    raise RuntimeError(f"LLM API request failed: {last_error}")


def _resolve_api_key(config: dict[str, Any]) -> str:
    api_key = config.get("api_key")
    api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
    if not api_key:
        api_key = os.environ.get(api_key_env)
        if not api_key and _looks_like_api_key(api_key_env):
            api_key = api_key_env
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set config field 'api_key' or environment variable: {api_key_env}"
        )
    return str(api_key)


def _read_non_streaming_chat_response(body: dict[str, Any]) -> str:
    message = body["choices"][0].get("message", {})
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        raise RuntimeError(
            "LLM returned reasoning_content but empty content. "
            "Use stream=true for thinking models or increase max_tokens."
        )
    raise RuntimeError(f"LLM returned empty content: {json.dumps(body, ensure_ascii=False)[:1000]}")


def _read_streaming_chat_response(response: Any) -> str:
    content_parts: list[str] = []
    reasoning_seen = False
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if delta.get("reasoning_content"):
            reasoning_seen = True
        content = delta.get("content")
        if content:
            content_parts.append(str(content))
    content = "".join(content_parts).strip()
    if content:
        return content
    if reasoning_seen:
        raise RuntimeError(
            "LLM stream returned reasoning_content but no final content. "
            "Increase max_tokens or check whether the model finished its answer."
        )
    raise RuntimeError("LLM stream returned empty content.")


def _looks_like_api_key(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.startswith(("sk-", "sk_")) and len(value) > 20


TRANSIENT_CONNECTION_ERRORS = (
    TimeoutError,
    ConnectionError,
    IncompleteRead,
    RemoteDisconnected,
    socket.timeout,
    urllib.error.URLError,
)


def _is_retryable_http_error(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504}
