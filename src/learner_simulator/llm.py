from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from http.client import IncompleteRead, RemoteDisconnected
from pathlib import Path
from typing import Any


# Bound provider-facing concurrency so variant and learner fan-out cannot
# silently become a 200-way request burst.
_REQUEST_GATE = threading.BoundedSemaphore(40)


class RetryableLLMOutputError(RuntimeError):
    """The request succeeded, but the model did not produce usable final content."""


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

    max_retries = int(config.get("max_retries", 2))
    output_retries = int(config.get("output_retries", 0))
    output_retry_max_tokens = int(
        config.get("output_retry_max_tokens", payload["max_tokens"])
    )
    retry_sleep = float(config.get("retry_sleep_seconds", 2))
    transient_attempt = 0
    output_attempt = 0
    access_denied_attempt = 0
    while True:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with _REQUEST_GATE:
                with urllib.request.urlopen(
                    request,
                    timeout=float(config.get("timeout_seconds", 60)),
                ) as response:
                    if payload["stream"]:
                        return _read_streaming_chat_response(response)
                    body = json.loads(response.read().decode("utf-8"))
                    return _read_non_streaming_chat_response(body)
        except RetryableLLMOutputError:
            if output_attempt >= output_retries:
                raise
            output_attempt += 1
            payload["max_tokens"] = max(
                int(payload["max_tokens"]),
                output_retry_max_tokens,
            )
            time.sleep(retry_sleep * output_attempt)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            # Quota allocation failures are not transient; retrying multiplies
            # token consumption and delays every other request.
            if exc.code == 429 and "insufficient_quota" in error_body:
                raise RuntimeError(
                    f"LLM API HTTP {exc.code}: quota allocation exceeded; "
                    "reduce total API concurrency or increase provider quota"
                ) from exc
            transient_access_denied = (
                exc.code == 403
                and "AccessDenied.Unpurchased" in error_body
                and access_denied_attempt
                < int(config.get("transient_access_denied_retries", 0))
            )
            if transient_access_denied:
                access_denied_attempt += 1
                time.sleep(
                    float(config.get("transient_access_denied_sleep_seconds", 15))
                    * access_denied_attempt
                )
                continue
            if (
                not _is_retryable_http_error(exc.code)
                or transient_attempt >= max_retries
            ):
                raise RuntimeError(f"LLM API HTTP {exc.code}: {error_body}") from exc
            transient_attempt += 1
            backoff = retry_sleep * (2 ** (transient_attempt - 1))
            backoff = min(backoff, float(config.get("retry_max_sleep_seconds", 30)))
            time.sleep(backoff)
        except TRANSIENT_CONNECTION_ERRORS as exc:
            if transient_attempt >= max_retries:
                raise RuntimeError(
                    "LLM API connection failed after "
                    f"{transient_attempt + 1} attempt(s). "
                    f"Last error: {type(exc).__name__}: {exc}"
                ) from exc
            transient_attempt += 1
            time.sleep(retry_sleep * transient_attempt)


def _resolve_api_key(config: dict[str, Any]) -> str:
    api_key = config.get("api_key")
    api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
    if not api_key:
        api_key = os.environ.get(api_key_env)
        if not api_key and _looks_like_api_key(api_key_env):
            api_key = api_key_env
    if not api_key and config.get("api_key_file"):
        key_path = Path(str(config["api_key_file"])).expanduser()
        if not key_path.is_absolute():
            key_path = Path.cwd() / key_path
        try:
            key_lines = key_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise RetryableLLMOutputError(
                f"Unable to read configured API-key file: {key_path}"
            ) from exc
        # A shared key file may deliberately contain one credential per line.
        # `api_key_file_line` is 1-based so its value matches what a user sees
        # in an editor; the default preserves the historical single-key-file
        # behavior by using the first non-empty line.
        line_number = config.get("api_key_file_line")
        if line_number is None:
            api_key = next((line.strip() for line in key_lines if line.strip()), "")
        else:
            try:
                requested_line = int(line_number)
            except (TypeError, ValueError) as exc:
                raise RetryableLLMOutputError(
                    "api_key_file_line must be a positive 1-based line number"
                ) from exc
            if requested_line <= 0 or requested_line > len(key_lines):
                raise RetryableLLMOutputError(
                    f"Configured API-key line {requested_line} is unavailable in {key_path}"
                )
            api_key = key_lines[requested_line - 1].strip()
    if not api_key:
        raise RetryableLLMOutputError(
            "Missing API key. Set config field 'api_key', 'api_key_file', "
            f"or environment variable: {api_key_env}"
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
    raise RetryableLLMOutputError(
        f"LLM returned empty content: {json.dumps(body, ensure_ascii=False)[:1000]}"
    )


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
        raise RetryableLLMOutputError(
            "LLM stream returned reasoning_content but no final content. "
            "Increase max_tokens or check whether the model finished its answer."
        )
    raise RetryableLLMOutputError("LLM stream returned empty content.")


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
