from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import learner_simulator.llm as module  # noqa: E402


class FakeResponse:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self.lines)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        key_file = Path(tmpdir) / "keys.txt"
        key_file.write_text(
            "first-key" + chr(10) + "second-key" + chr(10), encoding="utf-8"
        )
        assert module._resolve_api_key(
            {"api_key_file": str(key_file), "api_key_file_line": 2}
        ) == "second-key"
        assert module._resolve_api_key({"api_key_file": str(key_file)}) == "first-key"

    responses = iter(
        [
            FakeResponse(
                [
                    b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n',
                    b"data: [DONE]\n",
                ]
            ),
            FakeResponse(
                [
                    b'data: {"choices":[{"delta":{"content":"final"}}]}\n',
                    b"data: [DONE]\n",
                ]
            ),
        ]
    )
    token_budgets = []
    original = module.urllib.request.urlopen

    def fake_urlopen(request, timeout):
        token_budgets.append(json.loads(request.data)["max_tokens"])
        return next(responses)

    module.urllib.request.urlopen = fake_urlopen
    try:
        result = module.call_openai_compatible_chat(
            {
                "api_key": "test",
                "base_url": "https://example.invalid/v1",
                "model": "fake",
                "stream": True,
                "max_tokens": 4096,
                "output_retries": 1,
                "output_retry_max_tokens": 8192,
                "retry_sleep_seconds": 0,
            },
            "prompt",
        )
    finally:
        module.urllib.request.urlopen = original

    assert result == "final"
    assert token_budgets == [4096, 8192]
    print("llm_retry_test_ok")


if __name__ == "__main__":
    main()
