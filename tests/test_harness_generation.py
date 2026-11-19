import copy
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.analyse_input import AnalysisError, TaskObject  # noqa: E402
from helpers.generate_harness import run_harness_generation, validate_harness  # noqa: E402
from helpers.session_memory import SessionMemory, SessionState  # noqa: E402
from helpers.tool_registry import BUILTIN_TOOLS  # noqa: E402

REG = dict(BUILTIN_TOOLS)
STEP = {"step_id": 1, "goal": "edit foo", "type": "refactor", "dependencies": [],
        "estimated_effort": 2, "rollback_hint": "restore snapshot", "confidence_score": 0.8}
READ_STEP = {**STEP, "step_id": 2, "type": "research", "goal": "read foo"}
PLAN = {"must_not_break": [], "substeps": [STEP, READ_STEP]}

GOOD = {
    "guardrails": {
        "scope": {"allowed_tools": ["get_file_content", "write_file"], "allowed_read": ["src/**"],
                  "allowed_write": ["src/foo.py"], "forbidden_write": ["tests/**"]},
        "limits": {"tool_limits": {"get_file_content": {"max_per_step": 5}, "write_file": {"max_per_step": 2},
                                   "run_python_file": {"max_per_step": 9, "timeout_seconds": 5}},
                   "max_duration_seconds": 60},
        "content": {"must_not_modify": ["bar()"], "must_preserve": []},
    },
    "validation_checks": [{"name": "syntax", "check": "python -m py_compile src/foo.py"}],
    "rollback_plan": {"actions": ["restore src/foo.py from snapshot"], "test_rollback": "diff against snapshot"},
    "success_criteria": ["foo() returns 1"],
    "confidence_score": 0.95,
}
READ_ONLY = {
    "guardrails": {
        "scope": {"allowed_tools": ["get_file_content"], "allowed_read": ["src/**"], "allowed_write": [], "forbidden_write": []},
        "limits": {"tool_limits": {"get_file_content": {"max_per_step": 5}}, "max_duration_seconds": 30},
    },
    "validation_checks": [{"name": "read", "check": "file content returned"}],
    "rollback_plan": {"actions": [], "test_rollback": ""},
    "success_criteria": ["foo located"],
    "confidence_score": 0.9,
}


def good():
    return copy.deepcopy(GOOD)


def v(data, step=STEP):
    return validate_harness(data, step, REG)


def test_valid_harness_passes_and_caps_confidence_at_step_confidence():
    out = v(good())
    assert out["confidence_score"] == 0.8  # min(0.95, step 0.8)
    assert set(out["guardrails"]["limits"]["tool_limits"]) == {"get_file_content", "write_file"}
    assert set(out["guardrails"]) == {"scope", "limits", "content"}


def test_cannot_exceed_three_guardrail_layers():
    d = good()
    d["guardrails"]["extra"] = {}
    with pytest.raises(AnalysisError):
        v(d)


def _mutate(path, value):
    d = good()
    target = d
    for key in path[:-1]:
        target = target[key]
    if value is DELETE:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return d


DELETE = object()


@pytest.mark.parametrize("bad", [
    _mutate(["guardrails", "scope", "allowed_tools"], ["get_file_content", "rm_rf"]),
    _mutate(["guardrails", "scope", "allowed_tools"], []),
    _mutate(["guardrails", "scope"], DELETE),
    _mutate(["guardrails", "limits"], DELETE),
    _mutate(["guardrails", "content"], DELETE),                       # writer without content layer
    _mutate(["guardrails", "limits", "max_duration_seconds"], 0),
    _mutate(["validation_checks"], []),
    _mutate(["success_criteria"], []),
    _mutate(["rollback_plan"], {"actions": []}),
    _mutate(["confidence_score"], 1.5),
])
def test_invalid_harnesses_rejected(bad):
    with pytest.raises(AnalysisError):
        v(bad)


@pytest.mark.parametrize("pattern", ["/etc/passwd", "C:/Windows", "../secrets", "src/../../x", "**", "*", "~/x"])
def test_unsafe_write_patterns_rejected(pattern):
    with pytest.raises(AnalysisError):
        v(_mutate(["guardrails", "scope", "allowed_write"], [pattern]))


def test_write_without_explicit_scope_rejected():
    with pytest.raises(AnalysisError):
        v(_mutate(["guardrails", "scope", "allowed_write"], []))


def test_research_step_cannot_write():
    with pytest.raises(AnalysisError):
        v(good(), READ_STEP)


def test_read_only_step_needs_no_rollback_actions_or_content_layer():
    out = v(copy.deepcopy(READ_ONLY), READ_STEP)
    assert set(out["guardrails"]) == {"scope", "limits"}


def test_run_python_needs_timeout():
    d = copy.deepcopy(READ_ONLY)
    d["guardrails"]["scope"]["allowed_tools"] = ["run_python_file"]
    d["guardrails"]["limits"]["tool_limits"] = {"run_python_file": {"max_per_step": 1}}
    d["rollback_plan"] = {"actions": ["none"], "test_rollback": ""}
    with pytest.raises(AnalysisError):
        v(d, STEP)
    d["guardrails"]["limits"]["tool_limits"]["run_python_file"]["timeout_seconds"] = 5
    assert v(d, STEP)["guardrails"]["limits"]["tool_limits"]["run_python_file"]["timeout_seconds"] == 5


class FakeClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.messages = self

    def create(self, **kwargs):
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.replies.pop(0))])


def setup():
    memory = SessionMemory()
    sid = memory.start_session("x")
    memory.transition(sid, SessionState.ANALYZED)
    return memory, sid, TaskObject(session_id=sid, user_prompt="x", task_category="bug_fix")


def test_run_generation_stores_harness_per_step():
    memory, sid, task = setup()
    client = FakeClient(json.dumps(GOOD), json.dumps(READ_ONLY))
    result = run_harness_generation(task, PLAN, memory, client, REG)
    assert result["status"] == "generated"
    stored = memory.get(f"session:{sid}:harness")
    assert set(stored) == {1, 2}
    assert result["harnesses"][1].allowed_tools == ["get_file_content", "write_file"]


def test_run_generation_escalates_on_invalid_harness_after_one_corrective_retry():
    memory, sid, task = setup()
    bad = json.dumps(_mutate(["validation_checks"], []))
    client = FakeClient(bad, bad, bad)
    assert run_harness_generation(task, PLAN, memory, client, REG)["status"] == "needs_human_review"
    assert memory.get_state(sid) is SessionState.NEEDS_HUMAN_REVIEW


def test_invalid_harness_is_corrected_when_the_model_sees_its_error():
    memory, sid, task = setup()
    prompts = []

    class Spy(FakeClient):
        def create(self, **kwargs):
            prompts.append(kwargs["messages"][0]["content"])
            return super().create(**kwargs)

    no_checks = _mutate(["validation_checks"], [])
    client = Spy(json.dumps(no_checks), json.dumps(GOOD), json.dumps(READ_ONLY))
    result = run_harness_generation(task, PLAN, memory, client, REG)
    assert result["status"] == "generated"
    assert "previous answer was rejected" in prompts[1] and "at least one validation check" in prompts[1]


def test_empty_or_missing_content_layer_is_filled_from_the_plan():
    plan = {**PLAN, "must_not_break": ["existing add()"]}
    for content in ({"must_not_modify": [], "must_preserve": []}, DELETE):
        memory, sid, task = setup()
        client = FakeClient(json.dumps(_mutate(["guardrails", "content"], content)), json.dumps(READ_ONLY))
        result = run_harness_generation(task, plan, memory, client, REG)
        assert result["status"] == "generated"
        assert result["harnesses"][1].guardrails["content"]["must_preserve"] == ["existing add()"]


def test_model_supplied_content_is_kept_over_the_plan_default():
    plan = {**PLAN, "must_not_break": ["from plan"]}
    memory, sid, task = setup()
    result = run_harness_generation(task, plan, memory, FakeClient(json.dumps(GOOD), json.dumps(READ_ONLY)), REG)
    assert result["harnesses"][1].guardrails["content"]["must_not_modify"] == ["bar()"]


def test_must_not_modify_cannot_overlap_what_the_step_must_write():
    with pytest.raises(AnalysisError, match="overlaps allowed_write"):
        v(_mutate(["guardrails", "content"], {"must_not_modify": ["src/foo.py"], "must_preserve": []}))
    # function names and unrelated files are fine
    assert v(_mutate(["guardrails", "content"], {"must_not_modify": ["bar()", "src/other.py"], "must_preserve": []}))


def test_review_findings_reach_harness_generation_prompt():
    memory, sid, task = setup()
    seen = []

    class Spy(FakeClient):
        def create(self, **kwargs):
            seen.append(kwargs["messages"][0]["content"])
            return super().create(**kwargs)

    findings = [{"severity": "critical", "issue": "must_not_modify blocks the edit", "recommendation": "remove it"}]
    run_harness_generation(task, {**PLAN, "substeps": [STEP]}, memory, Spy(json.dumps(GOOD)), REG, review_findings=findings)
    assert "REJECTED by an independent reviewer" in seen[0] and "blocks the edit" in seen[0]


def test_step_duration_has_a_floor():
    d = good()
    d["guardrails"]["limits"]["max_duration_seconds"] = 5
    assert v(d)["guardrails"]["limits"]["max_duration_seconds"] == 60


def test_run_generation_escalates_on_low_confidence():
    memory, sid, task = setup()
    client = FakeClient(json.dumps(_mutate(["confidence_score"], 0.4)), json.dumps(READ_ONLY))
    result = run_harness_generation(task, PLAN, memory, client, REG)
    assert result["status"] == "needs_human_review"
    assert memory.get_state(sid) is SessionState.NEEDS_HUMAN_REVIEW
