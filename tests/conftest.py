"""Pytest fixtures for the migrated k3_features API contract tests."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest


# Moonshot official API endpoints. Tests that need to distinguish vendor vs.
# official deployments use this list via the `is_vendor` fixture.
MOONSHOT_APIS = [
    "https://api.moonshot.cn/v1",
    "https://api.moonshot.ai/v1"
]

# Default sampling parameters sent with --default-sampling. All but top_k are
# the fixed values from the Kimi K2.7 Code quickstart
# (https://platform.kimi.ai/docs/guide/kimi-k2-7-code-quickstart).
DEFAULT_SAMPLING_PARAMS: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 50,
    "n": 1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}


def sampling_params(overrides: str) -> dict[str, Any]:
    """DEFAULT_SAMPLING_PARAMS updated with a JSON object of overrides.

    A null value is sent as JSON null (e.g. {"seed": null}).
    """
    parsed = json.loads(overrides) if overrides.strip() else {}
    if not isinstance(parsed, dict):
        raise ValueError(f"sampling params overrides must be a JSON object, got: {overrides}")
    return {**DEFAULT_SAMPLING_PARAMS, **parsed}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--base-url",
        default=os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
        help="API base URL (default: KIMI_BASE_URL or https://api.moonshot.cn/v1)",
    )
    parser.addoption(
        "--api-key",
        default=os.environ.get("KIMI_API_KEY", ""),
        help="API key (default: KIMI_API_KEY)",
    )
    parser.addoption(
        "--smoke-model",
        default=os.environ.get("MODEL_NAME", ""),
        help="Model name to test (default: MODEL_NAME)",
    )
    parser.addoption(
        "--think-mode",
        choices=["kimi", "opensource", "none"],
        default=os.environ.get("THINK_MODE", "kimi"),
        help="Thinking parameter format (default: THINK_MODE or kimi)",
    )
    parser.addoption(
        "--case-dir",
        default=os.environ.get(
            "WALLE_CASE_DIR",
            str(Path(__file__).parent.parent / "testdata" / "walle_validator_cases" / "validator_cases"),
        ),
        help="walle validator_cases directory (default: testdata/walle_validator_cases/validator_cases)",
    )
    parser.addoption(
        "--selection",
        choices=["all", "explicit", "object"],
        default=os.environ.get("WALLE_SELECTION", "all"),
        help="Case selection mode: all, explicit, or object (default: all)",
    )
    parser.addoption(
        "--max-cases",
        type=int,
        default=None,
        help="Maximum number of selected cases to run",
    )
    parser.addoption(
        "--thinking",
        action="store_true",
        default=False,
        help="Enable thinking mode for tool-call schema requests",
    )
    parser.addoption(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("WALLE_MAX_TOKENS", "2048")),
        help="Maximum output tokens for each tool-call response (default: 2048)",
    )
    parser.addoption(
        "--tool-choice",
        choices=["required", "auto"],
        default=os.environ.get("TOOL_CHOICE", "required"),
        help="tool_choice for tool-call schema requests; use auto for vendors "
             "without guided decoding (default: TOOL_CHOICE or required)",
    )
    parser.addoption(
        "--prompt-tokens-reference",
        default=os.environ.get("PROMPT_TOKENS_REFERENCE", "cases"),
        help="Source of expected prompt_tokens: 'cases' for the constants in the case "
             "files (K3), or a Hugging Face repo id / local directory whose tokenizer and "
             "chat template compute the expected count (default: PROMPT_TOKENS_REFERENCE or cases)",
    )
    parser.addoption(
        "--strip-reasoning-effort",
        action="store_true",
        default=os.environ.get("STRIP_REASONING_EFFORT", "") not in ("", "0", "false"),
        help="Remove reasoning_effort from prompt-token requests, for models without "
             "thinking effort (default: STRIP_REASONING_EFFORT)",
    )
    parser.addoption(
        "--default-sampling",
        action="store_true",
        default=os.environ.get("DEFAULT_SAMPLING", "") not in ("", "0", "false", "no"),
        help="Add the default sampling params (temperature 1.0, top_p 0.95, top_k 50, "
             "n 1, presence/frequency_penalty 0) to every chat completion request that "
             "does not set them (default: DEFAULT_SAMPLING)",
    )
    parser.addoption(
        "--sampling-params",
        default=os.environ.get("SAMPLING_PARAMS", ""),
        help="JSON object overriding or adding to the --default-sampling params, "
             "e.g. '{\"top_k\": 0, \"seed\": null}' (default: SAMPLING_PARAMS)",
    )
    parser.addoption(
        "--cache-bypass",
        action="store_true",
        default=os.environ.get("CACHE_BYPASS", "") not in ("", "0", "false", "no"),
        help="Prefix every chat completion prompt with a unique request id (first tool "
             "description, else first message) so no request can hit the server's prefix "
             "cache. Changes prompt token counts (default: CACHE_BYPASS)",
    )


def _bust_prefix_cache(body: dict[str, Any]) -> None:
    """Put a unique id at the very start of the rendered prompt.

    Kimi's chat template renders tool declarations before the messages, so the
    id goes into the first tool's description when there are tools.
    """
    nonce = f"[request {uuid.uuid4().hex}] "
    tools = body.get("tools") or []
    function = tools[0].get("function") if tools and isinstance(tools[0], dict) else None
    if isinstance(function, dict):
        function["description"] = nonce + (function.get("description") or "")
        return
    messages = body.get("messages") or []
    if not messages or not isinstance(messages[0], dict):
        return
    content = messages[0].get("content")
    if isinstance(content, str):
        messages[0]["content"] = nonce + content
    elif isinstance(content, list):
        content.insert(0, {"type": "text", "text": nonce})


def _rewrite_request(request: httpx.Request, params: dict[str, Any] | None,
                     cache_bypass: bool) -> httpx.Request:
    """Return `request` with default sampling params added to its JSON body
    and, with `cache_bypass`, a unique id at the start of its prompt.

    Sampling values the request already sets are kept, so tests that send
    specific (or deliberately wrong) sampling params are unaffected.
    """
    if request.method != "POST" or not request.url.path.endswith("/chat/completions"):
        return request
    try:
        body = json.loads(request.content)
    except (httpx.RequestNotRead, ValueError):
        return request
    if not isinstance(body, dict):
        return request
    missing = {k: v for k, v in (params or {}).items() if k not in body}
    if not missing and not cache_bypass:
        return request
    body.update(missing)
    if cache_bypass:
        _bust_prefix_cache(body)
    headers = httpx.Headers(request.headers)
    headers.pop("content-length", None)
    return httpx.Request(
        request.method,
        request.url,
        headers=headers,
        content=json.dumps(body, ensure_ascii=False).encode(),
        extensions=request.extensions,
    )


def pytest_configure(config: pytest.Config) -> None:
    # Patch at the transport boundary so every client (openai SDK and raw
    # httpx, in every suite) is rewritten without per-test changes.
    cache_bypass = config.getoption("cache_bypass")
    params = (sampling_params(config.getoption("sampling_params"))
              if config.getoption("default_sampling") else None)
    if params is None and not cache_bypass:
        return
    original_send = httpx.Client.send

    def send(self: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        return original_send(self, _rewrite_request(request, params, cache_bypass), **kwargs)

    httpx.Client.send = send


@pytest.fixture(scope="session")
def key(pytestconfig: pytest.Config) -> str:
    return pytestconfig.getoption("api_key")


@pytest.fixture(scope="session")
def think_mode(pytestconfig: pytest.Config) -> str:
    return pytestconfig.getoption("think_mode")


@pytest.fixture(scope="session")
def case_dir(pytestconfig: pytest.Config) -> str:
    return pytestconfig.getoption("case_dir")


@pytest.fixture(scope="session")
def selection(pytestconfig: pytest.Config) -> str:
    return pytestconfig.getoption("selection")


@pytest.fixture(scope="session")
def max_cases(pytestconfig: pytest.Config) -> int | None:
    return pytestconfig.getoption("max_cases")


@pytest.fixture(scope="session")
def thinking_flag(pytestconfig: pytest.Config) -> bool:
    return bool(pytestconfig.getoption("thinking"))


@pytest.fixture(scope="session")
def max_tokens_for_tool_schema(pytestconfig: pytest.Config) -> int:
    return pytestconfig.getoption("max_tokens")


@pytest.fixture(scope="session")
def tool_choice_for_tool_schema(pytestconfig: pytest.Config) -> str:
    return pytestconfig.getoption("tool_choice")


@pytest.fixture(scope="function")
def is_vendor(pytestconfig: pytest.Config) -> bool:
    base_url = pytestconfig.getoption("base_url")
    return base_url not in MOONSHOT_APIS


@pytest.fixture(scope="function")
def model(pytestconfig: pytest.Config) -> str:
    return pytestconfig.getoption("smoke_model")


@pytest.fixture(scope="function")
def client(pytestconfig: pytest.Config, key: str) -> openai.Client:
    return openai.Client(
        api_key=key,
        base_url=str(pytestconfig.getoption("base_url")).rstrip("/"),
        timeout=120,
    )


@pytest.fixture(scope="function")
def hclient(pytestconfig: pytest.Config, key: str) -> Any:
    http_client = httpx.Client(
        base_url=str(pytestconfig.getoption("base_url")).rstrip("/"),
        headers={"Authorization": f"Bearer {key}"},
        timeout=120,
    )
    yield http_client
    http_client.close()
