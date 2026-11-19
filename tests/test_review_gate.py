import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.analyse_input import TaskObject, analyze_task  # noqa: E402
from helpers.session_memory import SessionMemory, SessionState  # noqa: E402
from helpers.verify_harness import plan_confidence, review_gate  # noqa: E402

PLAN = {
    "task_summary": "fix", "risks_identified": [], "dependencies": [], "must_not_break": [],
    "substeps": [
        {"step_id": 1, "goal": "read", "type": "research", "dependencies": [],
         "estimated_effort": 1, "rollback_hint": "none", "confidence_score": 0.9},
        {"step_id": 2, "goal": "edit", "type": "refactor", "dependencies": [1],
         "estimated_effort": 2, "rollback_hint": "restore snapshot", "confidence_score": 0.8},
    ],
}


class FakeClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []
        self.messages = self

    def create(self, **kwargs):
        self.prompts.append(kwargs["messages"][0]["content"])
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.replies.pop(0))])


def setup():
    memory = SessionMemory()
    sid = memory.start_session("fix it")
    memory.transition(sid, SessionState.ANALYZED)
    task = TaskObject(session_id=sid, user_prompt="fix it", task_category="bug_fix")
    return memory, sid, task


def review(approved=True, findings=None, adj=0.0):
    return json.dumps({"approved": approved, "findings": findings or [], "confidence_adjustment": adj})


def test_clean_review_approves():
    memory, sid, task = setup()
    result = review_gate(task, PLAN, memory, FakeClient(review()))
    assert result["status"] == "approved"
    assert memory.get_state(sid) is SessionState.APPROVED
    assert memory.get(f"session:{sid}:review")["final_approved"] is True


def test_critical_finding_blocks_even_if_model_says_approved():
    memory, sid, task = setup()
    findings = [{"severity": "critical", "issue": "deletes prod data", "recommendation": "drop step"}]
    result = review_gate(task, PLAN, memory, FakeClient(review(approved=True, findings=findings)))
    assert result["status"] == "rejected"
    assert memory.get_state(sid) is SessionState.REJECTED


def test_low_confidence_goes_to_human_review():
    memory, sid, task = setup()
    result = review_gate(task, PLAN, memory, FakeClient(review(adj=-0.4)))
    assert result["status"] == "needs_human_review"
    assert memory.get_state(sid) is SessionState.NEEDS_HUMAN_REVIEW


def test_extreme_adjustment_on_a_fixable_plan_is_clamped_so_replanning_still_happens():
    memory, sid, task = setup()
    findings = [{"severity": "critical", "issue": "contradictory guardrail", "recommendation": "fix it"}]
    confident = {**PLAN, "substeps": [{**s, "confidence_score": 0.98} for s in PLAN["substeps"]]}
    result = review_gate(task, confident, memory, FakeClient(review(approved=False, findings=findings, adj=-1.0)))
    assert result["status"] == "rejected"  # not needs_human_review
    assert result["review"]["confidence_adjustment"] == -0.3
    assert memory.get_state(sid) is SessionState.REJECTED


def test_garbage_review_escalates():
    memory, sid, task = setup()
    result = review_gate(task, PLAN, memory, FakeClient("nope", "still nope"))
    assert result["status"] == "needs_human_review"
    assert memory.get_state(sid) is SessionState.NEEDS_HUMAN_REVIEW


def test_replan_budget_exhausted_escalates():
    memory, sid, task = setup()
    bad = review(approved=False, findings=[{"severity": "warning", "issue": "vague", "recommendation": "x"}])
    statuses = []
    for i in range(3):
        statuses.append(review_gate(task, PLAN, memory, FakeClient(bad))["status"])
        if statuses[-1] == "rejected":
            memory.transition(sid, SessionState.ANALYZED)
    assert statuses == ["rejected", "rejected", "needs_human_review"]


def test_plan_confidence_uses_weakest_step_and_clamps():
    assert plan_confidence(PLAN, 0.0) == 0.8
    assert plan_confidence(PLAN, 0.5) == 1.0
    assert plan_confidence(PLAN, -0.9) == 0.0


def test_rejection_findings_reach_reanalysis_prompt():
    memory, sid, task = setup()
    client = FakeClient(json.dumps(PLAN))
    findings = [{"severity": "critical", "issue": "step 2 too broad", "recommendation": "split it"}]
    analyze_task(task, memory, client, review_findings=findings)
    assert "step 2 too broad" in client.prompts[0]
