# Phase 6: per-step verification (plan review lives in verify_harness.py; this verifies what actually ran).
#
# Two layers, in this order:
#   1. hard checks in code: the files the step really changed vs. the harness scope / content layer
#   2. an independent judge model (config.VERIFY_MODEL, never the executor) that must cite evidence
#      for every success criterion. Code derives the verdict; the judge's own opinion is advisory.
# If success cannot be proven the result is "escalate", never a guess.
import difflib
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import config
from helpers.analyse_input import AnalysisError, TaskObject
from helpers.guardrails import path_matches
from helpers.json_utils import call_json
from helpers.session_memory import SessionMemory, SessionState

JUDGE_CONFIDENCE = {"normal": 0.70, "tight": 0.85}   # tighter validation for mid-confidence harnesses
FAILURE_ACTIONS = {"retry", "rollback", "escalate"}
MAX_FILE_DIFF_CHARS = 4000
MAX_EVIDENCE_CHARS = 20000
MAX_DIFFABLE_BYTES = 200_000


def hard_checks(harness: Dict[str, Any], changed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deterministic checks on the files the step really changed (created + modified + deleted)."""
    scope = harness["guardrails"]["scope"]
    content = harness["guardrails"].get("content") or {}
    touched = list(changed["created"]) + list(changed["modified"]) + list(changed["deleted"])
    out_of_scope = [p for p in touched
                    if not path_matches(scope["allowed_write"], p) or path_matches(scope["forbidden_write"], p)]
    protected = [p for p in touched if path_matches(content.get("must_not_modify", []), p)]
    return [
        {"check": "writes_within_scope", "passed": not out_of_scope,
         "detail": f"changed files outside this step's write scope: {out_of_scope}" if out_of_scope else "ok"},
        {"check": "protected_files_untouched", "passed": not protected,
         "detail": f"must_not_modify files were changed: {protected}" if protected else "ok"},
    ]


def _read_text(path: str) -> Optional[str]:
    try:
        if os.path.getsize(path) > MAX_DIFFABLE_BYTES:
            return None
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def build_diffs(entries: List[Dict[str, Any]], changed: Dict[str, Any], working_dir: str) -> str:
    """Unified diffs for changed files, using the earliest snapshot of each file as 'before'."""
    root = os.path.realpath(working_dir)
    earliest: Dict[str, Dict[str, Any]] = {}
    for entry in sorted(entries, key=lambda e: e["seq"]):
        for snap in entry.get("snapshots", []):
            earliest.setdefault(snap["rel"], snap)

    parts: List[str] = []
    for rel in changed["created"] + changed["modified"]:
        after = _read_text(os.path.join(root, rel))
        snap = earliest.get(rel)
        if after is None:
            parts.append(f"--- {rel}: changed (binary or too large to show)")
        elif snap and snap["existed"] and snap["kind"] == "file":
            before = _read_text(snap["backup"]) or ""
            diff = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                                f"a/{rel}", f"b/{rel}"))
            parts.append(diff[:MAX_FILE_DIFF_CHARS] or f"--- {rel}: touched, content identical")
        elif rel in changed["created"]:
            parts.append(f"+++ NEW FILE {rel}\n{after[:MAX_FILE_DIFF_CHARS]}")
        else:
            parts.append(f"--- {rel}: modified, no snapshot to diff against (side effect of an unjournaled tool)")
    parts += [f"--- DELETED {rel}" for rel in changed["deleted"]]
    return "\n".join(parts)[:MAX_EVIDENCE_CHARS] or "(no files changed)"


def validate_judgement(data: Dict[str, Any], criteria: List[str]) -> Dict[str, Any]:
    rows = data.get("criteria")
    if not isinstance(rows, list):
        raise AnalysisError("judgement 'criteria' must be a list")
    seen: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("index"), int) or not isinstance(row.get("met"), bool):
            raise AnalysisError(f"invalid criterion row: {row!r}")
        if not str(row.get("evidence", "")).strip():
            raise AnalysisError(f"criterion {row['index']} has no evidence")
        seen[row["index"]] = row
    if set(seen) != set(range(len(criteria))) or len(rows) != len(criteria):
        raise AnalysisError("judgement must cover every success criterion exactly once")
    conf = data.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
        raise AnalysisError("judgement 'confidence' must be in [0, 1]")
    findings = data.get("findings", [])
    if not isinstance(findings, list):
        raise AnalysisError("judgement 'findings' must be a list")
    action = data.get("next_action_if_failed")
    return {
        "criteria": [{"criterion": criteria[i], "met": seen[i]["met"], "evidence": seen[i]["evidence"]}
                     for i in range(len(criteria))],
        "confidence": float(conf), "findings": [str(f) for f in findings],
        "failure_action": action if action in FAILURE_ACTIONS else "retry",
    }


def _store(memory: SessionMemory, sid: str, step_id: int, result: Dict[str, Any]) -> Dict[str, Any]:
    result["verified_at"] = datetime.now().isoformat()
    all_results = memory.get(f"session:{sid}:verification") or {}
    all_results[step_id] = result
    memory.set(f"session:{sid}:verification", all_results)
    return result


def _tool_call_lines(log: Dict[str, Any]) -> str:
    lines = []
    for c in log["tool_calls"]:
        outcome = f"BLOCKED ({c['blocked']})" if c["blocked"] else c["result"]
        lines.append(f"- {c['tool']} {json.dumps(c['args'])} -> {outcome}")
    return "\n".join(lines) or "(none)"


def verify_step(step_id: int, task: TaskObject, memory: SessionMemory, client, working_dir: str) -> Dict[str, Any]:
    """Verify a completed step right after it ran (rule 5).

    Returns {verified, next_action: proceed|retry|rollback|escalate, findings, checks, criteria, confidence}.
    """
    sid = task.session_id
    state = memory.get_state(sid)
    if state is not SessionState.VERIFYING:
        raise ValueError(f"cannot verify step {step_id} from state '{state.value}'")
    harness = memory.get(f"session:{sid}:harness")[step_id]
    log = memory.get(f"session:{sid}:execution_log")[step_id]
    if log["status"] != "completed":
        raise ValueError(f"step {step_id} did not complete (status {log['status']}); nothing to verify")
    changed = log["changed_files"]
    entries = [e for e in (memory.get(f"session:{sid}:journal") or []) if e["step_id"] == step_id]

    result: Dict[str, Any] = {"step_id": step_id, "verified": False, "next_action": "escalate",
                              "findings": [], "criteria": [], "confidence": None,
                              "checks": hard_checks(harness, changed)}

    failed = [c for c in result["checks"] if not c["passed"]]
    if failed:  # out-of-bounds change: no model opinion can make this acceptable
        result.update(next_action="rollback", findings=[c["detail"] for c in failed])
        return _store(memory, sid, step_id, result)

    if changed["truncated"] and any(not e["recoverable"] for e in entries):
        result["findings"] = ["workspace too large to diff, so side effects of an unjournaled tool are unproven"]
        return _store(memory, sid, step_id, result)

    criteria = harness["success_criteria"]
    content = harness["guardrails"].get("content") or {}
    prompt = f"""You are an independent verifier. You did not execute this step. Be skeptical:
the executor's own summary is a claim, not evidence. A criterion is met ONLY if the tool results or
diffs below show it. If the evidence does not show it, it is not met.

STEP GOAL: {harness['goal']}
SUCCESS CRITERIA (by index):
{chr(10).join(f'{i}: {c}' for i, c in enumerate(criteria))}
VALIDATION CHECKS THE PLAN CALLED FOR: {json.dumps(harness['validation_checks'])}
MUST NOT MODIFY: {content.get('must_not_modify', [])}
MUST PRESERVE: {content.get('must_preserve', [])}

TOOL CALLS (blocked = refused by guardrails, not executed):
{_tool_call_lines(log)}

FILES ACTUALLY CHANGED (diffs):
{build_diffs(entries, changed, working_dir)}

EXECUTOR'S SUMMARY (unverified claim): {log['final_text']}

Return pure JSON:
{{
  "criteria": [{{"index": 0, "met": true | false, "evidence": "what in the results/diffs shows this"}}],
  "confidence": 0.0-1.0,
  "findings": ["problems or concerns, if any"],
  "next_action_if_failed": "retry" | "rollback" | "escalate"
}}
Include exactly one entry per criterion index."""
    try:
        data = call_json(client, model=config.VERIFY_MODEL, max_tokens=2000, prompt=prompt)
        judged = validate_judgement(data, criteria)
    except (AnalysisError, ValueError) as e:
        result["findings"] = [f"verification unusable: {e}"]  # inconclusive -> escalate, never guess
        return _store(memory, sid, step_id, result)

    needed = JUDGE_CONFIDENCE[log["validation_mode"]]
    result.update(criteria=judged["criteria"], confidence=judged["confidence"], findings=judged["findings"])
    unmet = [c["criterion"] for c in judged["criteria"] if not c["met"]]
    if unmet:
        result.update(next_action=judged["failure_action"],
                      findings=[f"criterion not met: {c}" for c in unmet] + judged["findings"])
    elif judged["confidence"] < needed:
        result["findings"] = judged["findings"] + [
            f"judge confidence {judged['confidence']:.2f} below the {needed:.2f} required in {log['validation_mode']} mode"]
    else:
        result.update(verified=True, next_action="proceed")
    return _store(memory, sid, step_id, result)
