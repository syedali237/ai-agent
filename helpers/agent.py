"""The staged agent loop: intake -> analyze -> harness -> review -> (execute -> verify) per step -> finalize.

Policy implemented here (see .claude/skills/dynamic-harness/SKILL.md):
  - plan rejected: re-analyze with the reviewer's findings, max MAX_PLAN_ATTEMPTS, then human review
  - step fails (execution failed/blocked, or verification says retry): undo that step, retry up to
    MAX_STEP_RETRIES with the failure as feedback; then undo the whole session
  - verification says rollback: undo the whole session. If anything cannot be undone -> human review
  - verification says escalate / runaway tool / low confidence: stop, change nothing more, human review
  - FINALIZED only after the last step passed verification
"""
import logging
from dataclasses import asdict
from typing import Any, Dict, Optional

from helpers.analyse_input import AnalysisError, analyze_task, run_analysis
from helpers.execution import cleanup_session, execute_step, rollback_journal
from helpers.generate_harness import run_harness_generation
from helpers.session_memory import MAX_PLAN_ATTEMPTS, MAX_STEP_RETRIES, SessionMemory, SessionState
from helpers.tool_registry import ToolSpec, load_registry
from helpers.verify_harness import review_gate
from helpers.verify_step import verify_step

log = logging.getLogger(__name__)


def _escalate(memory: SessionMemory, sid: str, reason: str, **extra) -> Dict[str, Any]:
    """Stop for a human. Deliberately does NOT undo anything: the snapshots and journal are kept."""
    memory.transition(sid, SessionState.NEEDS_HUMAN_REVIEW)
    log.warning("session %s needs human review: %s", sid, reason)
    return {"status": "needs_human_review", "session_id": sid, "reason": reason,
            "pending_journal_entries": len(memory.get(f"session:{sid}:journal") or []), **extra}


def _rollback_all(task, memory: SessionMemory, working_dir: str, reason: str, **extra) -> Dict[str, Any]:
    sid = task.session_id
    reports = rollback_journal(task, memory, working_dir)
    if all(r["ok"] for r in reports):
        memory.transition(sid, SessionState.ROLLED_BACK)
        cleanup_session(memory, sid)
        log.warning("session %s rolled back: %s", sid, reason)
        return {"status": "rolled_back", "session_id": sid, "reason": reason, "rollback": reports, **extra}
    memory.transition(sid, SessionState.NEEDS_HUMAN_REVIEW)  # something could not be undone
    return {"status": "needs_human_review", "session_id": sid, "rollback": reports, **extra,
            "reason": f"{reason}; rollback incomplete: " + "; ".join(r["detail"] for r in reports if not r["ok"])}


def run_agent_task(user_prompt: str, client, working_dir: str, registry: Optional[Dict[str, ToolSpec]] = None,
                   memory: Optional[SessionMemory] = None) -> Dict[str, Any]:
    memory = memory if memory is not None else SessionMemory()
    registry = registry if registry is not None else load_registry()
    sid = memory.start_session(user_prompt)

    # Stages 1-2: intake + analysis
    analysis = run_analysis(user_prompt, sid, memory, client)
    if analysis["status"] != "analyzed":
        return {"status": "needs_human_review", "session_id": sid, "reason": analysis["reason"]}
    task, plan = analysis["task"], analysis["plan"]

    # Stages 3-4: harness generation + plan review, re-planning on rejection
    findings = None
    for _ in range(MAX_PLAN_ATTEMPTS):
        if findings:
            try:
                plan = analyze_task(task, memory, client, review_findings=findings)
            except (AnalysisError, ValueError) as e:
                return _escalate(memory, sid, f"re-analysis failed: {e}")
        generated = run_harness_generation(task, plan, memory, client, registry)
        if generated["status"] != "generated":
            return {"status": "needs_human_review", "session_id": sid, "reason": generated["reason"]}
        gate = review_gate(task, plan, memory, client, {s: asdict(h) for s, h in generated["harnesses"].items()})
        if gate["status"] == "approved":
            break
        if gate["status"] == "needs_human_review":
            return {"status": "needs_human_review", "session_id": sid, "reason": gate["reason"]}
        findings = gate["review"]["findings"]
        memory.transition(sid, SessionState.ANALYZED)
    else:
        return _escalate(memory, sid, "plan was never approved")  # unreachable: the gate escalates first

    # Stages 5-6: execute and verify one step at a time
    steps: Dict[int, Dict[str, Any]] = {}
    for step in plan["substeps"]:
        step_id = step["step_id"]
        feedback: Optional[str] = None
        steps[step_id] = {"attempts": 0}
        for attempt in range(MAX_STEP_RETRIES + 1):
            steps[step_id]["attempts"] = attempt + 1
            run = execute_step(step_id, task, memory, client, working_dir, registry, feedback)

            if run["status"] in ("blocked_low_confidence", "blocked_dependency"):
                return _escalate(memory, sid, run["reason"], failed_step=step_id, steps=steps)
            if run.get("possible_runaway_tool"):
                return _escalate(memory, sid, run["reason"], failed_step=step_id, steps=steps)

            if run["status"] == "completed":
                verdict = verify_step(step_id, task, memory, client, working_dir)
                steps[step_id]["verification"] = verdict
                action = verdict["next_action"]
                if action == "proceed":
                    break
                feedback = "; ".join(verdict["findings"]) or "verification failed"
            else:
                action = "retry"
                feedback = f"execution {run['status']}: {run['reason']}"

            if action == "escalate":
                return _escalate(memory, sid, feedback, failed_step=step_id, steps=steps)
            if action == "retry" and attempt < MAX_STEP_RETRIES:
                undone = rollback_journal(task, memory, working_dir, step_id)  # clean slate for the retry
                if not all(r["ok"] for r in undone):
                    return _escalate(memory, sid, "cannot cleanly undo the failed attempt: "
                                     + "; ".join(r["detail"] for r in undone if not r["ok"]),
                                     failed_step=step_id, steps=steps)
                continue
            return _rollback_all(task, memory, working_dir, feedback, failed_step=step_id, steps=steps)

    memory.transition(sid, SessionState.FINALIZED)  # only reachable after the last step verified
    cleanup_session(memory, sid)
    return {"status": "completed", "session_id": sid, "steps": steps}
