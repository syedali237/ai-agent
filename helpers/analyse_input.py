# Analyse tasks, what needs to be done, what edge cases could be there , break it into smaller subtasks,
# and determine the dependencies and order of execution for each subtask.
# Each subtask to have their own harness, guardrails, and verification steps.
# The analysis should also consider potential risks, failure points, and how to mitigate them.
# Finally, it should provide a clear plan for execution, including timelines, resource allocation, and checkpoints for review.
# Should also plan for rollback planning in case of failures or unexpected issues, the agent should plan "if this fails, here's how I would undo it."
# For each step agent should give a confidence score, low confidence score triggers deep analysis and additional verification steps or seek human intervention.

# Phase 2: structured JSON output for stage 1 (intake -> task object) and stage 2 (analysis -> plan).
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List

import config
from helpers.json_utils import call_json
from helpers.session_memory import SessionMemory, SessionState

TASK_CATEGORIES = {"code_refactor", "feature_add", "bug_fix", "research", "other"}
RISK_LEVELS = {"low", "medium", "high", "critical"}
STEP_TYPES = {"research", "design", "code_addition", "testing", "validation", "refactor"}

# Confidence policy: >= 0.85 normal, 0.60-0.84 tighter validation, < 0.60 do not execute.
CONFIDENCE_NORMAL = 0.85
CONFIDENCE_MIN = 0.60


class AnalysisError(Exception):
    """Model output was unparseable or failed schema validation. Escalate, don't guess."""


@dataclass
class TaskObject:
    session_id: str
    user_prompt: str
    task_category: str
    target_files: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    risk_level: str = "medium"
    estimated_complexity: int = 3
    assumptions: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


def confidence_policy(score: float) -> str:
    if score >= CONFIDENCE_NORMAL:
        return "execute"
    if score >= CONFIDENCE_MIN:
        return "execute_tight_validation"
    return "human_review"


def _require_list(data: Dict[str, Any], key: str) -> List[Any]:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise AnalysisError(f"'{key}' must be a list")
    return value


def _require_number(data: Dict[str, Any], key: str, lo: float, hi: float) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not lo <= value <= hi:
        raise AnalysisError(f"'{key}' must be a number in [{lo}, {hi}], got {value!r}")
    return value


def validate_task_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    if data.get("task_category") not in TASK_CATEGORIES:
        raise AnalysisError(f"invalid task_category: {data.get('task_category')!r}")
    if data.get("risk_level") not in RISK_LEVELS:
        raise AnalysisError(f"invalid risk_level: {data.get('risk_level')!r}")
    return {
        "task_category": data["task_category"],
        "risk_level": data["risk_level"],
        "estimated_complexity": int(_require_number(data, "estimated_complexity", 1, 5)),
        "confidence_score": float(_require_number(data, "confidence_score", 0, 1)),
        "target_files": [str(f) for f in _require_list(data, "target_files")],
        "success_criteria": [str(c) for c in _require_list(data, "success_criteria")],
        "constraints": [str(c) for c in _require_list(data, "constraints")],
        "assumptions": [str(a) for a in _require_list(data, "assumptions")],
    }


def intake_task(user_prompt: str, session_id: str, memory: SessionMemory, client) -> TaskObject:
    """Stage 1 (small model): normalize the request into a task object."""
    prompt = f"""Parse this user request into a structured task.

User request: {json.dumps(user_prompt)}

Return pure JSON with exactly these fields:
{{
  "task_category": "code_refactor" | "feature_add" | "bug_fix" | "research" | "other",
  "target_files": ["files or directories mentioned or implied"],
  "success_criteria": ["how we will know this succeeded"],
  "constraints": ["what must not be done"],
  "risk_level": "low" | "medium" | "high" | "critical",
  "estimated_complexity": 1-5,
  "assumptions": ["what we assume about the codebase"],
  "confidence_score": 0.0-1.0  // how well you understand the request
}}"""
    data = call_json(client, model=config.INTAKE_MODEL, max_tokens=1000, prompt=prompt)
    task = TaskObject(session_id=session_id, user_prompt=user_prompt, **validate_task_fields(data))
    memory.set(f"session:{session_id}:task", asdict(task))
    return task


def validate_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    substeps = plan.get("substeps")
    if not isinstance(substeps, list) or not substeps:
        raise AnalysisError("plan must contain a non-empty 'substeps' list")

    ids = []
    for step in substeps:
        if not isinstance(step, dict) or not isinstance(step.get("step_id"), int):
            raise AnalysisError(f"substep needs an integer step_id: {step!r}")
        if not str(step.get("goal", "")).strip():
            raise AnalysisError(f"step {step['step_id']} has no goal")
        if step.get("type") not in STEP_TYPES:
            raise AnalysisError(f"step {step['step_id']} has invalid type {step.get('type')!r}")
        _require_number(step, "estimated_effort", 1, 5)
        _require_number(step, "confidence_score", 0, 1)
        if not isinstance(step.get("dependencies", []), list):
            raise AnalysisError(f"step {step['step_id']} dependencies must be a list")
        step.setdefault("dependencies", [])
        # every step needs a rollback idea before it can ever run
        if not str(step.get("rollback_hint", "")).strip():
            raise AnalysisError(f"step {step['step_id']} has no rollback_hint")
        ids.append(step["step_id"])

    if len(set(ids)) != len(ids):
        raise AnalysisError("duplicate step_id in plan")
    for step in substeps:
        for dep in step["dependencies"]:
            if dep not in ids:
                raise AnalysisError(f"step {step['step_id']} depends on unknown step {dep}")
            if dep >= step["step_id"]:
                raise AnalysisError(f"step {step['step_id']} must depend only on earlier steps (got {dep})")

    for key in ("risks_identified", "dependencies", "must_not_break"):
        if not isinstance(plan.get(key, []), list):
            raise AnalysisError(f"'{key}' must be a list")
        plan.setdefault(key, [])
    plan.setdefault("task_summary", "")
    return plan


def analyze_task(task: TaskObject, memory: SessionMemory, client) -> Dict[str, Any]:
    """Stage 2 (large model): decompose into substeps with risks, dependencies and rollback hints."""
    prompt = f"""You are analyzing this task deeply before any execution happens.

TASK:
{json.dumps(asdict(task), indent=2)}

Return pure JSON:
{{
  "task_summary": "one-line summary",
  "risks_identified": ["potential failure modes"],
  "dependencies": ["what must exist or happen first"],
  "must_not_break": ["existing functionality to preserve"],
  "substeps": [
    {{
      "step_id": 1,
      "goal": "precise goal for this step",
      "type": "research" | "design" | "code_addition" | "testing" | "validation" | "refactor",
      "dependencies": [earlier step_ids only],
      "estimated_effort": 1-5,
      "rollback_hint": "if this fails, how to undo it",
      "confidence_score": 0.0-1.0
    }}
  ]
}}
Number steps from 1. A step may depend only on lower-numbered steps."""
    data = call_json(client, model=config.ANALYSIS_MODEL, max_tokens=4000, prompt=prompt)
    plan = validate_plan(data)

    prior = memory.get(f"session:{task.session_id}:plan")
    memory.set(f"session:{task.session_id}:plan", {
        "version": (prior["version"] + 1) if prior else 1,
        "analysis": plan,
        "created_at": datetime.now().isoformat(),
    })
    return plan


def run_analysis(user_prompt: str, session_id: str, memory: SessionMemory, client) -> Dict[str, Any]:
    """Intake + analysis, driving the session state machine.

    Returns {"status": "analyzed", "task": ..., "plan": ...} or
    {"status": "needs_human_review", "reason": ...}. Never raises on model/schema errors.
    """
    try:
        task = intake_task(user_prompt, session_id, memory, client)
        if confidence_policy(task.confidence_score) == "human_review":
            raise AnalysisError(f"intake confidence {task.confidence_score} below {CONFIDENCE_MIN}")
        plan = analyze_task(task, memory, client)
    except (AnalysisError, ValueError) as e:
        memory.transition(session_id, SessionState.NEEDS_HUMAN_REVIEW)
        return {"status": "needs_human_review", "reason": str(e)}

    memory.transition(session_id, SessionState.ANALYZED)
    return {"status": "analyzed", "task": task, "plan": plan}
