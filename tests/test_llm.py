import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import helpers.execution as execution  # noqa: E402
from helpers.analyse_input import TaskObject  # noqa: E402
from helpers.execution import execute_step  # noqa: E402
from helpers.json_utils import call_json  # noqa: E402
from helpers.llm import (  # noqa: E402
    LLMError, LLMRouter, OpenAICompatClient, _to_openai_messages, check_role_independence, missing_keys,
)
from helpers.session_memory import SessionMemory, SessionState  # noqa: E402
from helpers.tool_registry import BUILTIN_TOOLS  # noqa: E402


def reply(message, usage=None):
    return httpx.Response(200, json={"choices": [{"message": message}],
                                     "usage": usage or {"prompt_tokens": 7, "completion_tokens": 3}})


def client_with(handler, backoff=(0.0, 0.0)):
    return OpenAICompatClient("https://host.example/v1/", "sk-test", transport=httpx.MockTransport(handler), backoff=backoff)


def test_request_translation_system_tools_and_tool_results():
    sent = {}

    def handler(request):
        sent["url"], sent["auth"], sent["body"] = str(request.url), request.headers["authorization"], json.loads(request.content)
        return reply({"content": "ok"})

    messages = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": [{"type": "text", "text": "calling"},
                                          {"type": "tool_use", "id": "c1", "name": "write_file", "input": {"file_path": "a"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "done", "is_error": True}]},
    ]
    tools = [{"name": "write_file", "description": "w", "input_schema": {"type": "object", "properties": {}}}]
    client_with(handler).create(model="m", max_tokens=9, system="be careful", messages=messages, tools=tools)

    assert sent["url"] == "https://host.example/v1/chat/completions" and sent["auth"] == "Bearer sk-test"
    body = sent["body"]
    assert body["max_tokens"] == 9 and body["model"] == "m"
    assert body["messages"][0] == {"role": "system", "content": "be careful"}
    assistant = body["messages"][2]
    assert assistant["tool_calls"][0]["function"] == {"name": "write_file", "arguments": '{"file_path": "a"}'}
    assert body["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": "done"}
    assert body["tools"][0] == {"type": "function", "function": {"name": "write_file", "description": "w",
                                                                  "parameters": {"type": "object", "properties": {}}}}


def test_response_translation_text_tool_calls_and_usage():
    msg = {"content": "thinking", "tool_calls": [
        {"id": "c9", "type": "function", "function": {"name": "get_file_content", "arguments": '{"file_path": "x"}'}}]}
    out = client_with(lambda r: reply(msg)).create(model="m", max_tokens=1, messages=[{"role": "user", "content": "x"}])
    assert [b.type for b in out.content] == ["text", "tool_use"]
    assert out.content[1].input == {"file_path": "x"} and out.content[1].id == "c9"
    assert (out.usage.input_tokens, out.usage.output_tokens) == (7, 3)


def test_unparseable_tool_arguments_become_a_rejectable_call_not_an_execution():
    msg = {"content": None, "tool_calls": [{"id": "c1", "function": {"name": "write_file", "arguments": "{oops"}}]}
    out = client_with(lambda r: reply(msg)).create(model="m", max_tokens=1, messages=[{"role": "user", "content": "x"}])
    assert list(out.content[0].input) == ["__invalid_arguments__"]


def test_retries_transient_errors_then_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, text="slow down") if len(calls) < 3 else reply({"content": "fine"})

    out = client_with(handler).create(model="m", max_tokens=1, messages=[{"role": "user", "content": "x"}])
    assert out.content[0].text == "fine" and len(calls) == 3


def test_gives_up_after_retries_and_does_not_retry_client_errors():
    n = []
    with pytest.raises(LLMError, match="after 3 attempts"):
        client_with(lambda r: (n.append(1), httpx.Response(503, text="down"))[1]).create(
            model="m", max_tokens=1, messages=[{"role": "user", "content": "x"}])
    assert len(n) == 3
    n.clear()
    with pytest.raises(LLMError, match="HTTP 401"):
        client_with(lambda r: (n.append(1), httpx.Response(401, text="bad key"))[1]).create(
            model="m", max_tokens=1, messages=[{"role": "user", "content": "x"}])
    assert len(n) == 1


def test_malformed_tool_generation_is_retried_immediately():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(400, json={"error": {"code": "tool_use_failed", "message": "Tool choice is none, but model called a tool"}})
        return reply({"content": "fine"})

    c = client_with(handler, backoff=(999.0,))  # would hang the test if it slept
    assert c.create(model="m", max_tokens=1, messages=[{"role": "user", "content": "x"}]).content[0].text == "fine"
    assert len(calls) == 2


def test_json_calls_tell_the_model_it_has_no_tools():
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        return reply({"content": '{"a": 1}'})

    call_json(client_with(handler), model="m", max_tokens=5, prompt="p")
    assert sent["messages"][0]["role"] == "system" and "no tools" in sent["messages"][0]["content"]
    assert "tools" not in sent


def test_network_failure_and_malformed_body_raise_llmerror():
    def boom(request):
        raise httpx.ConnectError("no route")

    with pytest.raises(LLMError):
        client_with(boom).create(model="m", max_tokens=1, messages=[{"role": "user", "content": "x"}])
    with pytest.raises(LLMError, match="malformed"):
        client_with(lambda r: httpx.Response(200, json={"error": "nope"})).create(
            model="m", max_tokens=1, messages=[{"role": "user", "content": "x"}])


def test_call_json_works_through_the_adapter_including_fenced_json():
    c = client_with(lambda r: reply({"content": '```json\n{"a": 1}\n```'}))
    assert call_json(c, model="m", max_tokens=10, prompt="p") == {"a": 1}


def test_router_sends_each_model_to_its_own_host_with_its_own_key(monkeypatch):
    seen = []

    def handler(request):
        seen.append((request.url.host, request.headers["authorization"], json.loads(request.content)["model"]))
        return reply({"content": "ok"})

    monkeypatch.setenv("GPT_OSS_API_KEY", "key-oss")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key-ds")
    router = LLMRouter(transport=httpx.MockTransport(handler), backoff=())
    for model in (config.GPT_OSS_MODEL, config.DEEPSEEK_MODEL):
        router.messages.create(model=model, max_tokens=1, messages=[{"role": "user", "content": "x"}])
    assert seen[0][1:] == ("Bearer key-oss", config.GPT_OSS_MODEL)
    assert seen[1][1:] == ("Bearer key-ds", config.DEEPSEEK_MODEL)
    assert seen[0][0] == config.PROVIDERS[config.GPT_OSS_MODEL]["base_url"].split("//")[1].split("/")[0]


def test_router_reports_missing_keys_as_llmerror(monkeypatch):
    monkeypatch.delenv("GPT_OSS_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    router = LLMRouter(transport=httpx.MockTransport(lambda r: reply({"content": "x"})))
    with pytest.raises(LLMError, match="GPT_OSS_API_KEY"):
        router.create(model=config.GPT_OSS_MODEL, max_tokens=1, messages=[])
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        router.create(model="claude-haiku-4-5", max_tokens=1, messages=[])


def test_missing_keys_lists_names_only(monkeypatch):
    for name in ("GPT_OSS_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert set(missing_keys()) == {"GPT_OSS_API_KEY", "DEEPSEEK_API_KEY"}  # no claude-* role configured
    monkeypatch.setenv("GPT_OSS_API_KEY", "x")
    assert missing_keys() == ["DEEPSEEK_API_KEY"]


def test_role_independence_is_enforced(monkeypatch):
    check_role_independence()  # shipped config is valid
    monkeypatch.setattr(config, "VERIFY_MODEL", config.EXECUTION_MODEL)
    with pytest.raises(ValueError, match="judge its own"):
        check_role_independence()
    monkeypatch.setattr(config, "VERIFY_MODEL", config.DEEPSEEK_MODEL)
    monkeypatch.setattr(config, "REVIEW_MODEL", config.ANALYSIS_MODEL)
    with pytest.raises(ValueError, match="author"):
        check_role_independence()


def test_executor_loop_runs_over_the_openai_adapter(tmp_path):
    """The whole execute_step loop (tool call -> guarded write -> final text) through the real adapter."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("old")
    turns = [
        reply({"content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "write_file", "arguments": json.dumps({"file_path": "src/a.py", "content": "new"})}}]}),
        reply({"content": "wrote it"}),
    ]
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return turns.pop(0)

    harness = {"step_id": 1, "goal": "edit", "success_criteria": ["done"], "confidence_score": 0.95, "guardrails": {
        "scope": {"allowed_tools": ["write_file"], "allowed_read": [], "allowed_write": ["src/*.py"], "forbidden_write": []},
        "limits": {"tool_limits": {"write_file": {"max_per_step": 2}}, "max_duration_seconds": 30},
        "content": {"must_not_modify": [], "must_preserve": ["api"]}}}
    memory = SessionMemory()
    sid = memory.start_session("x")
    for s in (SessionState.ANALYZED, SessionState.REVIEWED, SessionState.APPROVED):
        memory.transition(sid, s)
    memory.set(f"session:{sid}:plan", {"version": 1, "analysis": {"substeps": [{"step_id": 1, "goal": "edit", "dependencies": []}]}})
    memory.set(f"session:{sid}:harness", {1: harness})
    memory.set(f"session:{sid}:review", {"final_approved": True})
    task = TaskObject(session_id=sid, user_prompt="x", task_category="bug_fix")

    log = execute_step(1, task, memory, client_with(handler), str(tmp_path), dict(BUILTIN_TOOLS))
    assert log["status"] == "completed" and (tmp_path / "src" / "a.py").read_text() == "new"
    # the second request replayed the tool call and its result in OpenAI format
    roles = [m["role"] for m in requests[1]["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert "working_dir" not in requests[0]["tools"][0]["function"]["parameters"]["properties"]
    assert execution  # module imported for patching parity


def test_message_translation_handles_plain_string_turns():
    assert _to_openai_messages(None, [{"role": "user", "content": "hi"}]) == [{"role": "user", "content": "hi"}]
