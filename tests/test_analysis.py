import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.analyse_input import (  # noqa: E402
    AnalysisError, confidence_policy, run_analysis, validate_plan,
)
from helpers.json_utils import parse_json  # noqa: E402
from helpers.session_memory import SessionMemory, SessionState  # noqa: E402

TASK = {
    "task_category": "bug_fix", "target_files": ["a.py"], "success_criteria": ["tests pass"],
    "constraints": [], "risk_level": "low", "estimated_complexity": 2,
    "assumptions": [], "confidence_score": 0.9,
}
PLAN = {
    "task_summary": "fix", "risks_identified": [], "dependencies": [], "must_not_break": [],
    "substeps": [
        {"step_id": 1, "goal": "read", "type": "research", "dependencies": [],
         "estimated_effort": 1, "rollback_hint": "none needed", "confidence_score": 0.9},
        {"step_id": 2, "goal": "edit", "type": "refactor", "dependencies": [1],
         "estimated_effort": 2, "rollback_hint": "restore snapshot", "confidence_score": 0.8},
    ],
}


class FakeClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        text = self.replies.pop(0)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def test_parse_json_fenced_and_prose():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": 1} hope that helps') == {"a": 1}


def test_confidence_policy():
    assert confidence_policy(0.9) == "execute"
    assert confidence_policy(0.7) == "execute_tight_validation"
    assert confidence_policy(0.59) == "human_review"


def test_validate_plan_rejects_forward_dependency():
    bad = json.loads(json.dumps(PLAN))
    bad["substeps"][0]["dependencies"] = [2]
    with pytest.raises(AnalysisError):
        validate_plan(bad)


def test_validate_plan_requires_rollback_hint():
    bad = json.loads(json.dumps(PLAN))
    bad["substeps"][1]["rollback_hint"] = ""
    with pytest.raises(AnalysisError):
        validate_plan(bad)


def test_run_analysis_happy_path_with_retry_on_bad_json():
    memory = SessionMemory()
    sid = memory.start_session("fix the bug")
    client = FakeClient("not json at all", json.dumps(TASK), "```json\n" + json.dumps(PLAN) + "\n```")
    result = run_analysis("fix the bug", sid, memory, client)
    assert result["status"] == "analyzed"
    assert memory.get_state(sid) is SessionState.ANALYZED
    assert memory.get(f"session:{sid}:plan")["version"] == 1
    assert client.calls == 3


def test_run_analysis_low_confidence_escalates():
    memory = SessionMemory()
    sid = memory.start_session("???")
    client = FakeClient(json.dumps({**TASK, "confidence_score": 0.3}))
    result = run_analysis("???", sid, memory, client)
    assert result["status"] == "needs_human_review"
    assert memory.get_state(sid) is SessionState.NEEDS_HUMAN_REVIEW


def test_run_analysis_invalid_plan_escalates():
    memory = SessionMemory()
    sid = memory.start_session("x")
    client = FakeClient(json.dumps(TASK), json.dumps({"substeps": []}))
    assert run_analysis("x", sid, memory, client)["status"] == "needs_human_review"
