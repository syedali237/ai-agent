import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.analyse_input import TaskObject  # noqa: E402
from helpers.session_memory import SessionMemory, SessionState  # noqa: E402
from helpers.verify_step import hard_checks, validate_judgement, verify_step  # noqa: E402
from helpers.workspace_diff import diff_trees, snapshot_tree  # noqa: E402
from helpers.analyse_input import AnalysisError  # noqa: E402


def test_workspace_diff_sees_create_modify_delete_and_skips_caches(tmp_path):
    (tmp_path / "a.txt").write_text("1")
    (tmp_path / "b.txt").write_text("1")
    (tmp_path / "__pycache__").mkdir()
    before = snapshot_tree(str(tmp_path))
    (tmp_path / "a.txt").write_text("22")
    (tmp_path / "b.txt").unlink()
    (tmp_path / "c.txt").write_text("new")
    (tmp_path / "__pycache__" / "x.pyc").write_text("ignored")
    changed = diff_trees(before, snapshot_tree(str(tmp_path)))
    assert (changed["modified"], changed["deleted"], changed["created"]) == (["a.txt"], ["b.txt"], ["c.txt"])
    assert changed["truncated"] is False


HARNESS = {
    "step_id": 1, "goal": "edit a.py", "success_criteria": ["a.py returns 1", "tests still pass"],
    "validation_checks": [{"name": "t", "check": "run tests"}], "confidence_score": 0.95,
    "guardrails": {
        "scope": {"allowed_tools": ["write_file"], "allowed_read": ["src/**"], "allowed_write": ["src/*.py"],
                  "forbidden_write": ["src/secret.py"]},
        "limits": {"tool_limits": {"write_file": {"max_per_step": 2}}, "max_duration_seconds": 30},
        "content": {"must_not_modify": ["src/legacy.py"], "must_preserve": ["api"]},
    },
}


def changed(created=(), modified=(), deleted=(), truncated=False):
    return {"created": list(created), "modified": list(modified), "deleted": list(deleted), "truncated": truncated}


def test_hard_checks_pass_for_in_scope_changes():
    assert all(c["passed"] for c in hard_checks(HARNESS, changed(modified=["src/a.py"])))


@pytest.mark.parametrize("ch", [
    changed(modified=["README.md"]),            # outside allowed_write
    changed(modified=["src/secret.py"]),        # forbidden_write
    changed(modified=["src/legacy.py"]),        # must_not_modify (also fails scope? no: in allowed_write)
    changed(deleted=["src/other/x.py"]),        # deletions count
])
def test_hard_checks_fail_for_out_of_bounds_changes(ch):
    assert not all(c["passed"] for c in hard_checks(HARNESS, ch))


def test_read_only_step_must_change_nothing():
    ro = json.loads(json.dumps(HARNESS))
    ro["guardrails"]["scope"]["allowed_write"] = []
    assert all(c["passed"] for c in hard_checks(ro, changed()))
    assert not all(c["passed"] for c in hard_checks(ro, changed(created=["src/new.py"])))


def judgement(met=(True, True), conf=0.9, action="retry", evidence="shown in diff"):
    return {"criteria": [{"index": i, "met": m, "evidence": evidence} for i, m in enumerate(met)],
            "confidence": conf, "findings": [], "next_action_if_failed": action}


@pytest.mark.parametrize("bad", [
    {**judgement(), "criteria": judgement()["criteria"][:1]},                       # missing a criterion
    {**judgement(), "criteria": judgement()["criteria"] + judgement()["criteria"][:1]},  # duplicate
    judgement(evidence=""),                                                          # no evidence
    judgement(conf=1.5),
    {**judgement(), "criteria": "all good"},
])
def test_invalid_judgements_rejected(bad):
    with pytest.raises(AnalysisError):
        validate_judgement(bad, HARNESS["success_criteria"])


class FakeClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.replies.pop(0))])


def session(wd, ch, harness=HARNESS, status="completed", mode="normal", journal=None, final_text="done"):
    memory = SessionMemory()
    sid = memory.start_session("x")
    for s in (SessionState.ANALYZED, SessionState.REVIEWED, SessionState.APPROVED,
              SessionState.EXECUTING, SessionState.VERIFYING):
        memory.transition(sid, s)
    memory.set(f"session:{sid}:harness", {1: harness})
    memory.set(f"session:{sid}:execution_log", {1: {
        "step_id": 1, "status": status, "changed_files": ch, "tool_calls": [], "final_text": final_text,
        "validation_mode": mode, "reason": ""}})
    memory.set(f"session:{sid}:journal", journal or [])
    return memory, TaskObject(session_id=sid, user_prompt="x", task_category="bug_fix")


def test_all_criteria_met_proceeds(tmp_path):
    memory, task = session(str(tmp_path), changed())
    client = FakeClient(json.dumps(judgement()))
    result = verify_step(1, task, memory, client, str(tmp_path))
    assert result["verified"] is True and result["next_action"] == "proceed"
    assert memory.get(f"session:{task.session_id}:verification")[1]["verified"] is True


def test_judge_saying_verified_is_not_enough_when_a_criterion_is_unmet(tmp_path):
    memory, task = session(str(tmp_path), changed())
    reply = {**judgement(met=(True, False), action="rollback"), "verified": True}
    result = verify_step(1, task, memory, FakeClient(json.dumps(reply)), str(tmp_path))
    assert result["verified"] is False and result["next_action"] == "rollback"
    assert "tests still pass" in result["findings"][0]


def test_unmet_criterion_cannot_be_waved_through_with_proceed(tmp_path):
    memory, task = session(str(tmp_path), changed())
    result = verify_step(1, task, memory, FakeClient(json.dumps(judgement(met=(True, False), action="proceed"))), str(tmp_path))
    assert result["next_action"] == "retry"


def test_low_judge_confidence_escalates_and_tight_mode_is_stricter(tmp_path):
    memory, task = session(str(tmp_path), changed())
    assert verify_step(1, task, memory, FakeClient(json.dumps(judgement(conf=0.5))), str(tmp_path))["next_action"] == "escalate"
    memory, task = session(str(tmp_path), changed(), mode="normal")
    assert verify_step(1, task, memory, FakeClient(json.dumps(judgement(conf=0.8))), str(tmp_path))["next_action"] == "proceed"
    memory, task = session(str(tmp_path), changed(), mode="tight")
    assert verify_step(1, task, memory, FakeClient(json.dumps(judgement(conf=0.8))), str(tmp_path))["next_action"] == "escalate"


def test_out_of_bounds_change_rolls_back_without_asking_the_judge(tmp_path):
    memory, task = session(str(tmp_path), changed(modified=["README.md"]))
    client = FakeClient()
    result = verify_step(1, task, memory, client, str(tmp_path))
    assert result["next_action"] == "rollback" and client.calls == []


def test_unusable_judge_output_escalates(tmp_path):
    memory, task = session(str(tmp_path), changed())
    result = verify_step(1, task, memory, FakeClient("nope", "still nope"), str(tmp_path))
    assert result["next_action"] == "escalate" and not result["verified"]


def test_truncated_diff_with_unjournaled_tool_escalates(tmp_path):
    entry = {"seq": 1, "step_id": 1, "tool": "run_python_file", "recoverable": False, "snapshots": []}
    memory, task = session(str(tmp_path), changed(truncated=True), journal=[entry])
    client = FakeClient()
    assert verify_step(1, task, memory, client, str(tmp_path))["next_action"] == "escalate" and client.calls == []


def test_diff_evidence_reaches_the_judge_and_executor_claims_are_labelled(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("new line\n")
    backup = tmp_path / "backup"
    backup.write_text("old line\n")
    entry = {"seq": 1, "step_id": 1, "tool": "write_file", "recoverable": True,
             "snapshots": [{"rel": "src/a.py", "existed": True, "kind": "file", "backup": str(backup)}]}
    memory, task = session(str(tmp_path), changed(modified=["src/a.py"]), journal=[entry], final_text="I fixed everything")
    client = FakeClient(json.dumps(judgement()))
    verify_step(1, task, memory, client, str(tmp_path))
    prompt = client.calls[0]["messages"][0]["content"]
    assert "-old line" in prompt and "+new line" in prompt and "unverified claim" in prompt


def test_cannot_verify_a_step_that_did_not_complete_or_wrong_state(tmp_path):
    memory, task = session(str(tmp_path), changed(), status="failed")
    with pytest.raises(ValueError):
        verify_step(1, task, memory, FakeClient(), str(tmp_path))
    memory, task = session(str(tmp_path), changed())
    memory.transition(task.session_id, SessionState.EXECUTING)
    with pytest.raises(ValueError):
        verify_step(1, task, memory, FakeClient(), str(tmp_path))
