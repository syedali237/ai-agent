"""Model access behind one interface.

Every stage calls `client.messages.create(model=..., max_tokens=..., messages=..., system=..., tools=...)`
and reads Anthropic-shaped responses (`.content` blocks of type text / tool_use, `.usage`). `LLMRouter`
keeps that interface but sends each request to the right provider by model id:
  - ids listed in config.PROVIDERS -> OpenAI-compatible /chat/completions (gpt-oss, DeepSeek, ...)
  - anything else                  -> the Anthropic SDK (needs ANTHROPIC_API_KEY)
Provider failures surface as LLMError, which the agent loop turns into needs_human_review.
"""
import json
import os
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import httpx

import config


class LLMError(Exception):
    pass


RETRY_STATUSES = {429, 500, 502, 503, 504}
ROLES = ("INTAKE_MODEL", "ANALYSIS_MODEL", "HARNESS_MODEL", "REVIEW_MODEL", "EXECUTION_MODEL", "VERIFY_MODEL")


def _to_openai_messages(system: Optional[str], messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = [{"role": "system", "content": system}] if system else []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        if m["role"] == "assistant":
            text = "".join(b["text"] for b in content if b["type"] == "text")
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                     for b in content if b["type"] == "tool_use"]
            msg: Dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:  # user turn made of tool_result / text blocks
            for b in content:
                if b["type"] == "tool_result":
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": str(b["content"])})
                elif b["type"] == "text":
                    out.append({"role": "user", "content": b["text"]})
    return out


def _to_openai_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"type": "function",
             "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t["input_schema"]}}
            for t in tools]


def _from_openai_response(data: Dict[str, Any]) -> SimpleNamespace:
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"malformed response from provider: {str(data)[:300]}") from e
    blocks: List[SimpleNamespace] = []
    if message.get("content"):
        blocks.append(SimpleNamespace(type="text", text=message["content"]))
    for i, call in enumerate(message.get("tool_calls") or []):
        fn = call.get("function", {})
        raw = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(args, dict):
                raise ValueError
        except ValueError:
            # Surfaces as "unexpected arguments" in the executor, so the model gets to retry; never executed.
            args = {"__invalid_arguments__": str(raw)[:200]}
        blocks.append(SimpleNamespace(type="tool_use", id=call.get("id") or f"call_{i}",
                                      name=fn.get("name", ""), input=args))
    usage = data.get("usage") or {}
    return SimpleNamespace(
        content=blocks,
        usage=SimpleNamespace(input_tokens=usage.get("prompt_tokens", 0), output_tokens=usage.get("completion_tokens", 0)),
    )


class OpenAICompatClient:
    """Minimal Anthropic-shaped facade over an OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0,
                 transport: Optional[httpx.BaseTransport] = None, backoff: Tuple[float, ...] = (1.0, 3.0, 8.0, 15.0)):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.http = httpx.Client(timeout=timeout, transport=transport)
        self.backoff = backoff
        self.messages = self  # lets callers write client.messages.create(...)

    def create(self, *, model: str, max_tokens: int, messages: List[Dict[str, Any]],
               system: Optional[str] = None, tools: Optional[List[Dict[str, Any]]] = None, **_ignored) -> SimpleNamespace:
        payload: Dict[str, Any] = {"model": model, "max_tokens": max_tokens,
                                   "messages": _to_openai_messages(system, messages)}
        if tools:
            payload["tools"] = _to_openai_tools(tools)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        last = "no attempt made"
        for attempt in range(len(self.backoff) + 1):
            try:
                resp = self.http.post(self.url, json=payload, headers=headers)
            except httpx.TransportError as e:
                last = f"{type(e).__name__}: {e}"
            else:
                if resp.status_code == 200:
                    try:
                        return _from_openai_response(resp.json())
                    except ValueError as e:  # body was not JSON
                        raise LLMError(f"provider returned non-JSON body: {resp.text[:200]}") from e
                last = f"HTTP {resp.status_code}: {resp.text[:300]}"
                # The model emitted a malformed or imaginary tool call (Groq reports this as a 400).
                # Sampling is random, so asking again usually works, and no waiting is needed.
                bad_generation = resp.status_code == 400 and "tool_use_failed" in resp.text
                if resp.status_code not in RETRY_STATUSES and not bad_generation:
                    raise LLMError(f"{self.url} {last}")
                if bad_generation:
                    continue
            if attempt < len(self.backoff):
                time.sleep(self.backoff[attempt])
        raise LLMError(f"{self.url} failed after {len(self.backoff) + 1} attempts: {last}")


class LLMRouter:
    """Drop-in `client` for the whole pipeline. Clients are created lazily, one per provider."""

    def __init__(self, transport: Optional[httpx.BaseTransport] = None, backoff: Tuple[float, ...] = (1.0, 3.0, 8.0, 15.0)):
        self.messages = self
        self._transport, self._backoff = transport, backoff
        self._compat: Dict[str, OpenAICompatClient] = {}
        self._anthropic: Optional[anthropic.Anthropic] = None

    def create(self, **kwargs) -> Any:
        model = kwargs["model"]
        spec = config.PROVIDERS.get(model)
        if spec is not None:
            return self._compat_client(model, spec).create(**kwargs)
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise LLMError(f"no provider configured for model '{model}' and ANTHROPIC_API_KEY is not set")
        if self._anthropic is None:
            self._anthropic = anthropic.Anthropic()
        try:
            return self._anthropic.messages.create(**kwargs)
        except anthropic.APIError as e:
            raise LLMError(f"Anthropic API error: {e}") from e

    def _compat_client(self, model: str, spec: Dict[str, str]) -> OpenAICompatClient:
        key = os.getenv(spec["api_key_env"])
        if not key:
            raise LLMError(f"{spec['api_key_env']} is not set (needed for model '{model}')")
        if model not in self._compat:
            self._compat[model] = OpenAICompatClient(spec["base_url"], key, transport=self._transport,
                                                     backoff=self._backoff)
        return self._compat[model]


def make_client() -> LLMRouter:
    return LLMRouter()


def required_key_names() -> List[str]:
    """Names (never values) of the env vars the configured roles need."""
    names = set()
    for role in ROLES:
        model = getattr(config, role)
        spec = config.PROVIDERS.get(model)
        names.add(spec["api_key_env"] if spec else "ANTHROPIC_API_KEY")
    return sorted(names)


def missing_keys() -> List[str]:
    return [n for n in required_key_names() if not os.getenv(n)]


def check_role_independence() -> None:
    """Raise if the configuration would let a model grade its own work."""
    if config.EXECUTION_MODEL == config.VERIFY_MODEL:
        raise ValueError("EXECUTION_MODEL and VERIFY_MODEL must differ: the executor must not judge its own step")
    if config.ANALYSIS_MODEL == config.REVIEW_MODEL or config.HARNESS_MODEL == config.REVIEW_MODEL:
        raise ValueError("REVIEW_MODEL must differ from ANALYSIS_MODEL and HARNESS_MODEL: the plan's author must not review it")
