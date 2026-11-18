# Handover the harnesses from the first high model to another good model for verifying if
# the harnesses are correctly implemented and functioning as expected and covering all the cases

# Phase 3: plan review gate. An independent model (config.REVIEW_MODEL, never the analysis
# model) checks the plan before anything may execute. The model's verdict is advisory:
# critical findings or low confidence block approval regardless of what it says.
import json
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Optional

import config
from helpers.analyse_input import CONFIDENCE_MIN, AnalysisError, TaskObject
from helpers.json_utils import call_json
from helpers.session_memory import MAX_PLAN_ATTEMPTS, SessionMemory, SessionState

SEVERITIES = {"critical", "warning", "info"}


def validate_review(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data.get("approved"), bool):
        raise AnalysisError("review 'approved' must be a boolean")
    findings = data.get("findings", [])
    if not isinstance(findings, list):
        raise AnalysisError("review 'findings' must be a list")
    for f in findings:
        if not isinstance(f, dict) or f.get("severity") not in SEVERITIES:
            raise AnalysisError(f"invalid review finding: {f!r}")
        if not str(f.get("issue", "")).strip():
            raise AnalysisError("review finding has no issue text")
        f.setdefault("recommendation", "")
    adjustment = data.get("confidence_adjustment", 0.0)
    if isinstance(adjustment, bool) or not isinstance(adjustment, (int, float)) or not -1 <= adjustment <= 1:
        raise AnalysisError(f"invalid confidence_adjustment: {adjustment!r}")
    return {"approved": data["approved"], "findings": findings, "confidence_adjustment": float(adjustment)}


def plan_confidence(plan: Dict[str, Any], adjustment: float) -> float:
    """A plan is only as confident as its weakest step, shifted by the reviewer's adjustment."""
    weakest = min(step["confidence_score"] for step in plan["substeps"])
    return max(0.0, min(1.0, weakest + adjustment))


def review_plan(
    task: TaskObject,
    plan: Dict[str, Any],
    memory: SessionMemory,
    client,
    harnesses: Optional[Dict[Any, Any]] = None,
) -> Dict[str, Any]:
    """Second model reviews the plan (and harnesses, once phase 4 provides them).

    Returns the validated review plus 'final_approved' and 'plan_confidence'.
    """
    harness_block = (
        f"\nHARNESSES:\n{json.dumps(harnesses, indent=2)}\n" if harnesses else
        "\n(No harnesses generated yet: review the plan only.)\n"
    )
    prompt = f"""You are an independent safety reviewer. You did not write this plan. Be skeptical.

TASK:
{json.dumps(asdict(task), indent=2)}

PLAN:
{json.dumps(plan, indent=2)}
{harness_block}
Review for:
1. Missing dependencies between steps
2. Steps with too broad a scope or over-permissive file access
3. Weak or missing validation / success criteria
4. Missing or unrealistic rollback paths
5. Steps that mutate things in the task's constraints or must_not_break list
6. Hidden assumptions and impossible success criteria
7. Confidence scores that look inflated

Return pure JSON:
{{
  "approved": true | false,
  "findings": [
    {{"severity": "critical" | "warning" | "info", "issue": "...", "recommendation": "..."}}
  ],
  "confidence_adjustment": -1.0 to 1.0
}}
Use "critical" for anything that makes execution unsafe. Do not approve a plan with a critical finding."""
    data = call_json(client, model=config.REVIEW_MODEL, max_tokens=2000, prompt=prompt)
    review = validate_review(data)

    has_critical = any(f["severity"] == "critical" for f in review["findings"])
    confidence = plan_confidence(plan, review["confidence_adjustment"])
    review["plan_confidence"] = confidence
    # Enforced in code: the model cannot approve past a critical finding or low confidence.
    review["final_approved"] = review["approved"] and not has_critical and confidence >= CONFIDENCE_MIN
    review["reviewed_at"] = datetime.now().isoformat()

    memory.set(f"session:{task.session_id}:review", review)
    return review


def review_gate(
    task: TaskObject,
    plan: Dict[str, Any],
    memory: SessionMemory,
    client,
    harnesses: Optional[Dict[Any, Any]] = None,
) -> Dict[str, Any]:
    """Run the review and move the session state machine accordingly.

    status: "approved" | "rejected" (caller may re-analyze with the findings) |
            "needs_human_review" (unusable review, low confidence, or re-plan budget spent).
    Execution must only start when status == "approved".
    """
    sid = task.session_id
    try:
        review = review_plan(task, plan, memory, client, harnesses)
    except (AnalysisError, ValueError) as e:
        memory.transition(sid, SessionState.NEEDS_HUMAN_REVIEW)
        return {"status": "needs_human_review", "reason": f"review unusable: {e}"}

    memory.transition(sid, SessionState.REVIEWED)

    if review["final_approved"]:
        memory.transition(sid, SessionState.APPROVED)
        return {"status": "approved", "review": review}

    if review["plan_confidence"] < CONFIDENCE_MIN:
        memory.transition(sid, SessionState.REJECTED)
        memory.transition(sid, SessionState.NEEDS_HUMAN_REVIEW)
        return {"status": "needs_human_review", "review": review,
                "reason": f"plan confidence {review['plan_confidence']:.2f} below {CONFIDENCE_MIN}"}

    memory.transition(sid, SessionState.REJECTED)
    attempts = memory.get(f"session:{sid}:state")["plan_attempts"]
    if attempts >= MAX_PLAN_ATTEMPTS:
        memory.transition(sid, SessionState.NEEDS_HUMAN_REVIEW)
        return {"status": "needs_human_review", "review": review,
                "reason": f"plan rejected {attempts} times"}
    return {"status": "rejected", "review": review}
