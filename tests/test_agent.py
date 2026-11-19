"""End-to-end agent loop with a scripted model. Replies are routed by which stage is calling."""
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.agent import run_agent_task  # noqa: E402
from helpers.session_memory import SessionMemory, SessionState  # noqa: E402
from helpers.tool_registry import BUILTIN_TOOLS  # noqa: E402

REG = dict(BUILTIN_TOOLS)

INTAKE = {"task_category": "bug_fix", "target_files": ["src/a.py"], "success_criteria": ["a.py updated"],
          "constraints": [], "risk_level": "low", "estimated_complexity": 1, "assumptions": [], "confidence_score": 0.9}
PLAN = {"task_summary": "edit", "risks_identified": [], "dependencies": [], "must_not_break": [],
        "substeps": [{"step_id": 1, "goal": "edit src/a.py", "type": "refactor", "dependencies": [],
                      "estimated_effort": 1, "rollback_hint": "restore snapshot", "confidence_score": 0.9}]}
HARNESS = {
    "guardrails": {
        "scope": {"allowed_tools": ["get_file_content", "write_file"], "allowed_read": ["src/**"],
                  "allowed_write": ["src/*.py"], "forbidden_write": []},
        "limits": {"tool_limits": {"get_file_content": {"max_per_step": 3}, "write_file": {"max_per_step": 2}},
                   "max_duration_seconds": 30},
        "content": {"must_not_modify": [], "must_preserve": ["public api"]},
    },
    "validation_checks": [{"name": "diff", "check": "a.py contains the change"}],
    "rollback_plan": {"actions": ["restore a.py"], "test_rollback": "compare"},
    "success_criteria": ["a.py updated"], "confidence_score": 0.95,
}
APPROVE = {"approved": True, "findings": [], "confidence_adjustment": 0.0}
REJECT = {"approved": False, "findings": [{"severity": "warning", "issue": "too vague", "recommendation": "x"}],
          "confidence_adjustment": 0.0}
MET = {"criteria": [{"index": 0, "met": True, "evidence": "diff shows it"}], "confidence": 0.9, "findings": []}
UNMET = {"criteria": [{"index": 0, "met": False, "evidence": "not in diff"}], "confidence": 0.9,
         "findings": [], "next_action_if_failed": "retry"}


def text(t):
    return SimpleNamespace(type="text", text=t)


def write(content, n=1):
    return [SimpleNamespace(type="tool_use", id=f"t{n}", name="write_file",
                            input={"file_path": "src/a.py", "content": content})]


class RoutingClient:
    ROLES = (("Parse this user request", "intake"), ("analyzing this task", "analysis"),
             ("Generate a harness", "harness"), ("independent safety reviewer", "review"),
             ("independent verifier", "verify"))

    def __init__(self, **roles):
        self.roles = {k: list(v) for k, v in roles.items()}
        self.seen = []
        self.messages = self

    def create(self, **kw):
        prompt = kw["messages"][0]["content"]
        role = "execution" if "tools" in kw else next(r for needle, r in self.ROLES if needle in prompt)
        self.seen.append(role)
        reply = self.roles[role].pop(0)
        blocks = reply if role == "execution" else [text(json.dumps(reply))]
        return SimpleNamespace(content=blocks, usage=SimpleNamespace(input_tokens=1, output_tokens=1))


@pytest.fixture
def wd(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("old")
    return str(tmp_path)


def exec_turns(content, n=1):
    return [write(content, n), [text(f"wrote {content}")]]


def test_full_pipeline_completes_and_cleans_up(wd):
    client = RoutingClient(intake=[INTAKE], analysis=[PLAN], harness=[HARNESS], review=[APPROVE],
                           execution=exec_turns("new"), verify=[MET])
    memory = SessionMemory()
    result = run_agent_task("fix a.py", client, wd, REG, memory)
    assert result["status"] == "completed"
    assert open(os.path.join(wd, "src", "a.py")).read() == "new"
    assert memory.get_state(result["session_id"]) is SessionState.FINALIZED
    assert memory.get(f"session:{result['session_id']}:snapshot_root") is None
    # each stage ran in order, execution before its verification
    assert client.seen == ["intake", "analysis", "harness", "review", "execution", "execution", "verify"]


def test_failed_verification_retries_with_feedback_then_succeeds(wd):
    client = RoutingClient(intake=[INTAKE], analysis=[PLAN], harness=[HARNESS], review=[APPROVE],
                           execution=exec_turns("bad", 1) + exec_turns("good", 2), verify=[UNMET, MET])
    result = run_agent_task("fix a.py", client, wd, REG)
    assert result["status"] == "completed" and result["steps"][1]["attempts"] == 2
    assert open(os.path.join(wd, "src", "a.py")).read() == "good"


def test_retry_prompt_carries_failure_and_previous_attempt_was_undone(wd):
    seen_prompts = []
    client = RoutingClient(intake=[INTAKE], analysis=[PLAN], harness=[HARNESS], review=[APPROVE],
                           execution=exec_turns("bad", 1) + [[text("gave up")]], verify=[UNMET, UNMET, UNMET])
    original_create = client.create

    def spy(**kw):
        if "tools" in kw:
            seen_prompts.append((kw["messages"][0]["content"], open(os.path.join(wd, "src", "a.py")).read()))
        return original_create(**kw)

    client.create = spy
    client.messages = client
    # second attempt only asks to look; third has nothing left, so the loop ends in rollback anyway
    client.roles["execution"] += [[text("still nothing")], [text("still nothing")]]
    run_agent_task("fix a.py", client, wd, REG)
    retry_prompt, file_at_retry = seen_prompts[2]  # first call of attempt 2
    assert "PREVIOUS ATTEMPT" in retry_prompt and "criterion not met" in retry_prompt
    assert file_at_retry == "old"  # attempt 1's write was rolled back before the retry


def test_retries_exhausted_rolls_back_everything(wd):
    turns = exec_turns("v1", 1) + exec_turns("v2", 2) + exec_turns("v3", 3)
    client = RoutingClient(intake=[INTAKE], analysis=[PLAN], harness=[HARNESS], review=[APPROVE],
                           execution=turns, verify=[UNMET, UNMET, UNMET])
    memory = SessionMemory()
    result = run_agent_task("fix a.py", client, wd, REG, memory)
    assert result["status"] == "rolled_back" and result["steps"][1]["attempts"] == 3
    assert open(os.path.join(wd, "src", "a.py")).read() == "old"
    assert memory.get_state(result["session_id"]) is SessionState.ROLLED_BACK


def test_escalating_verdict_stops_without_undoing(wd):
    unsure = {**MET, "confidence": 0.2}
    client = RoutingClient(intake=[INTAKE], analysis=[PLAN], harness=[HARNESS], review=[APPROVE],
                           execution=exec_turns("new"), verify=[unsure])
    memory = SessionMemory()
    result = run_agent_task("fix a.py", client, wd, REG, memory)
    assert result["status"] == "needs_human_review" and result["pending_journal_entries"] == 1
    assert open(os.path.join(wd, "src", "a.py")).read() == "new"  # left in place for the human to inspect
    assert memory.get_state(result["session_id"]) is SessionState.NEEDS_HUMAN_REVIEW


def test_plan_rejected_three_times_never_executes(wd):
    client = RoutingClient(intake=[INTAKE], analysis=[PLAN, PLAN, PLAN], harness=[HARNESS] * 3,
                           review=[REJECT] * 3, execution=[], verify=[])
    result = run_agent_task("fix a.py", client, wd, REG)
    assert result["status"] == "needs_human_review"
    assert "execution" not in client.seen
    assert open(os.path.join(wd, "src", "a.py")).read() == "old"


def test_plan_rejected_once_then_approved_runs(wd):
    client = RoutingClient(intake=[INTAKE], analysis=[PLAN, PLAN], harness=[HARNESS] * 2,
                           review=[REJECT, APPROVE], execution=exec_turns("new"), verify=[MET])
    assert run_agent_task("fix a.py", client, wd, REG)["status"] == "completed"
    assert client.seen.count("analysis") == 2


def test_low_confidence_intake_never_reaches_execution(wd):
    client = RoutingClient(intake=[{**INTAKE, "confidence_score": 0.2}])
    assert run_agent_task("???", client, wd, REG)["status"] == "needs_human_review"
    assert client.seen == ["intake"]
