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

STEP = {"step_id": 1, "goal": "edit foo", "type": "refactor", "dependencies": [],
        "estimated_effort": 2, "rollback_hint": "restore snapshot", "confidence_score": 0.8}
READ_STEP = {**STEP, "step_id": 2, "type": "research", "goal": "read foo"}
PLAN = {"must_not_break": [], "substeps": [STEP, READ_STEP]}

GOOD = {
    "allowed_tools": ["get_file_content", "write_file"],
    "tool_limits": {"get_file_content": {"max_per_step": 5}, "write_file": {"max_per_step": 2},
                    "run_python_file": {"max_per_step": 9, "timeout_seconds": 5}},
    "guardrails": {
        "file_access": {"allowed_read": ["src/**"], "allowed_write": ["src/foo.py"], "forbidden_write": ["tests/**"]},
        "resource_limits": {"max_duration_seconds": 60},
    },
    "validation_checks": [{"name": "syntax", "check": "python -m py_compile src/foo.py"}],
    "rollback_plan": {"actions": ["restore src/foo.py from snapshot"], "test_rollback": "diff against snapshot"},
    "success_criteria": ["foo() returns 1"],
    "confidence_score": 0.95,
}
READ_ONLY = {**GOOD, "allowed_tools": ["get_file_content"],
             "guardrails": {**GOOD["guardrails"],
                            "file_access": {"allowed_read": ["src/**"], "allowed_write": [], "forbidden_write": []}},
             "rollback_plan": {"actions": [], "test_rollback": ""}}


def variant(**overrides):
    d = copy.deepcopy(GOOD)
    d.update(overrides)
    return d


def test_valid_harness_passes_and_caps_confidence_at_step_confidence():
    out = validate_harness(copy.deepcopy(GOOD), STEP)
    assert out["confidence_score"] == 0.8  # min(0.95, step 0.8)
    assert set(out["tool_limits"]) == {"get_file_content", "write_file"}  # ungranted limit dropped


@pytest.mark.parametrize("bad", [
    variant(allowed_tools=["get_file_content", "rm_rf"]),
    variant(allowed_tools=[]),
    variant(validation_checks=[]),
    variant(success_criteria=[]),
    variant(rollback_plan={"actions": []}),
    variant(confidence_score=1.5),
])
def test_invalid_harnesses_rejected(bad):
    with pytest.raises(AnalysisError):
        validate_harness(bad, STEP)


@pytest.mark.parametrize("pattern", ["/etc/passwd", "C:/Windows", "../secrets", "src/../../x", "**", "*", "~/x"])
def test_unsafe_write_patterns_rejected(pattern):
    bad = copy.deepcopy(GOOD)
    bad["guardrails"]["file_access"]["allowed_write"] = [pattern]
    with pytest.raises(AnalysisError):
        validate_harness(bad, STEP)


def test_write_without_explicit_scope_rejected():
    bad = copy.deepcopy(GOOD)
    bad["guardrails"]["file_access"]["allowed_write"] = []
    with pytest.raises(AnalysisError):
        validate_harness(bad, STEP)


def test_research_step_cannot_write():
    with pytest.raises(AnalysisError):
        validate_harness(copy.deepcopy(GOOD), READ_STEP)


def test_read_only_step_needs_no_rollback_actions():
    assert validate_harness(copy.deepcopy(READ_ONLY), READ_STEP)["allowed_tools"] == ["get_file_content"]


def test_run_python_needs_timeout():
    bad = variant(allowed_tools=["run_python_file"], tool_limits={"run_python_file": {"max_per_step": 1}})
    bad["guardrails"]["file_access"]["allowed_write"] = []
    with pytest.raises(AnalysisError):
        validate_harness(bad, STEP)


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
    result = run_harness_generation(task, PLAN, memory, client)
    assert result["status"] == "generated"
    assert set(memory.get(f"session:{sid}:harness")) == {1, 2}


def test_run_generation_escalates_on_invalid_harness():
    memory, sid, task = setup()
    client = FakeClient(json.dumps(variant(validation_checks=[])))
    assert run_harness_generation(task, PLAN, memory, client)["status"] == "needs_human_review"
    assert memory.get_state(sid) is SessionState.NEEDS_HUMAN_REVIEW


def test_run_generation_escalates_on_low_confidence():
    memory, sid, task = setup()
    client = FakeClient(json.dumps(variant(confidence_score=0.4)), json.dumps(READ_ONLY))
    result = run_harness_generation(task, PLAN, memory, client)
    assert result["status"] == "needs_human_review"
    assert memory.get_state(sid) is SessionState.NEEDS_HUMAN_REVIEW
