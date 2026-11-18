# Should generate harnesses, guardrails, agent loop, verify, how to manage context,
# tool refining according to the individiual task

# Phase 4: one harness per substep. The model proposes guardrails; code validates them and
# tightens them where the plan implies it. Nothing here executes tools (that is phase 5),
# but the stored harness is the single source of truth the executor will enforce.
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List

import config
from helpers.analyse_input import CONFIDENCE_MIN, AnalysisError, TaskObject
from helpers.json_utils import call_json
from helpers.session_memory import SessionMemory, SessionState

# Tools that actually exist in tools/. A harness may only grant these.
KNOWN_TOOLS = {"get_files_info", "get_file_content", "write_file", "run_python_file"}
MUTATING_TOOLS = {"write_file", "run_python_file"}
# Step types that must never write files, regardless of what the model proposes.
READ_ONLY_STEP_TYPES = {"research", "design", "validation"}


@dataclass
class Harness:
    step_id: int
    goal: str
    execution_model: str
    allowed_tools: List[str]
    tool_limits: Dict[str, Dict[str, int]]
    guardrails: Dict[str, Any]
    validation_checks: List[Dict[str, str]]
    rollback_plan: Dict[str, Any]
    success_criteria: List[str]
    confidence_score: float
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


def _bad_pattern(pattern: Any) -> str:
    """Return a reason if a file pattern is unsafe, else ''. Patterns are workspace-relative globs."""
    if not isinstance(pattern, str) or not pattern.strip():
        return "empty or non-string pattern"
    p = pattern.replace("\\", "/")
    if p.startswith("/") or (len(p) > 1 and p[1] == ":") or p.startswith("~"):
        return f"absolute path {pattern!r}"
    if ".." in p.split("/"):
        return f"path escapes workspace {pattern!r}"
    if p.strip("*/") == "":
        return f"pattern {pattern!r} matches everything"
    return ""


def _str_list(value: Any, name: str) -> List[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise AnalysisError(f"'{name}' must be a list of strings")
    return value


def validate_harness(data: Dict[str, Any], step: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and tighten a model-proposed harness. Raises AnalysisError if unusable."""
    sid = step["step_id"]

    tools = _str_list(data.get("allowed_tools"), "allowed_tools")
    unknown = set(tools) - KNOWN_TOOLS
    if unknown:
        raise AnalysisError(f"step {sid}: unknown tools {sorted(unknown)}")
    if not tools:
        raise AnalysisError(f"step {sid}: no tools allowed")
    if step["type"] in READ_ONLY_STEP_TYPES and "write_file" in tools:
        raise AnalysisError(f"step {sid}: '{step['type']}' steps must not be granted write_file")

    limits = data.get("tool_limits")
    if not isinstance(limits, dict):
        raise AnalysisError(f"step {sid}: tool_limits must be an object")
    for tool in tools:
        lim = limits.get(tool)
        if not isinstance(lim, dict) or not isinstance(lim.get("max_per_step"), int) or lim["max_per_step"] < 1:
            raise AnalysisError(f"step {sid}: tool_limits.{tool}.max_per_step must be an integer >= 1")
        if tool == "run_python_file" and not (isinstance(lim.get("timeout_seconds"), int) and lim["timeout_seconds"] > 0):
            raise AnalysisError(f"step {sid}: run_python_file needs a positive timeout_seconds")
    limits = {t: limits[t] for t in tools}  # drop limits for tools not granted

    guard = data.get("guardrails")
    if not isinstance(guard, dict) or not isinstance(guard.get("file_access"), dict):
        raise AnalysisError(f"step {sid}: guardrails.file_access is required")
    fa = guard["file_access"]
    for key in ("allowed_read", "allowed_write", "forbidden_write"):
        fa[key] = _str_list(fa.get(key, []), f"file_access.{key}")
        for pattern in fa[key]:
            reason = _bad_pattern(pattern)
            if reason:
                raise AnalysisError(f"step {sid}: file_access.{key}: {reason}")
    if "write_file" in tools and not fa["allowed_write"]:
        raise AnalysisError(f"step {sid}: write_file granted but allowed_write is empty (must be explicit)")
    if "write_file" not in tools:
        fa["allowed_write"] = []  # no write tool, no write scope
    res = guard.get("resource_limits")
    if not isinstance(res, dict) or not (isinstance(res.get("max_duration_seconds"), (int, float)) and res["max_duration_seconds"] > 0):
        raise AnalysisError(f"step {sid}: guardrails.resource_limits.max_duration_seconds must be positive")
    guard.setdefault("code_patterns", {"must_not_modify": [], "must_preserve": []})

    # Rule 2: never execute a step with no validation strategy.
    checks = data.get("validation_checks")
    if not isinstance(checks, list) or not checks:
        raise AnalysisError(f"step {sid}: at least one validation check is required")
    for c in checks:
        if not isinstance(c, dict) or not str(c.get("name", "")).strip() or not str(c.get("check", "")).strip():
            raise AnalysisError(f"step {sid}: each validation check needs 'name' and 'check'")

    # Rule 4: know how to undo before acting. Required whenever the step can mutate anything.
    rollback = data.get("rollback_plan")
    actions = rollback.get("actions") if isinstance(rollback, dict) else None
    if set(tools) & MUTATING_TOOLS and (not isinstance(actions, list) or not actions):
        raise AnalysisError(f"step {sid}: mutating step needs rollback_plan.actions")
    if not isinstance(rollback, dict):
        rollback = {"actions": [], "test_rollback": ""}
    rollback.setdefault("actions", [])
    rollback.setdefault("test_rollback", "")

    criteria = _str_list(data.get("success_criteria"), "success_criteria")
    if not criteria:
        raise AnalysisError(f"step {sid}: success_criteria must not be empty")

    conf = data.get("confidence_score")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
        raise AnalysisError(f"step {sid}: confidence_score must be in [0, 1]")
    # A harness can't be more confident than the plan step it implements.
    conf = min(float(conf), float(step["confidence_score"]))

    return {
        "allowed_tools": tools, "tool_limits": limits, "guardrails": guard,
        "validation_checks": checks, "rollback_plan": rollback,
        "success_criteria": criteria, "confidence_score": conf,
    }


def generate_harness(task: TaskObject, plan: Dict[str, Any], step: Dict[str, Any], client) -> Harness:
    prompt = f"""Generate a harness (guardrails) for ONE step of an approved-in-principle plan.
Be as restrictive as the goal allows: least tools, narrowest file scope.

TASK: {task.user_prompt}
TASK CONSTRAINTS: {json.dumps(task.constraints)}
MUST NOT BREAK: {json.dumps(plan.get('must_not_break', []))}
STEP {step['step_id']} ({step['type']}): {step['goal']}
STEP ROLLBACK HINT: {step['rollback_hint']}
TARGET FILES: {json.dumps(task.target_files)}

Available tools (use only these): get_files_info, get_file_content, write_file, run_python_file.
All paths are workspace-relative globs. No absolute paths, no "..", no bare "*" or "**".

Return pure JSON:
{{
  "allowed_tools": ["..."],
  "tool_limits": {{"<tool>": {{"max_per_step": N}}, "run_python_file": {{"max_per_step": N, "timeout_seconds": S}}}},
  "guardrails": {{
    "file_access": {{"allowed_read": ["..."], "allowed_write": ["..."], "forbidden_write": ["..."]}},
    "code_patterns": {{"must_not_modify": ["..."], "must_preserve": ["..."]}},
    "resource_limits": {{"max_duration_seconds": N}}
  }},
  "validation_checks": [{{"name": "...", "check": "how to validate"}}],
  "rollback_plan": {{"actions": ["steps to undo"], "test_rollback": "how to verify the undo worked"}},
  "success_criteria": ["explicit, checkable criteria"],
  "confidence_score": 0.0-1.0
}}"""
    data = call_json(client, model=config.REVIEW_MODEL, max_tokens=2000, prompt=prompt)
    checked = validate_harness(data, step)
    return Harness(
        step_id=step["step_id"], goal=step["goal"], execution_model=config.EXECUTION_MODEL, **checked
    )


def generate_harnesses(task: TaskObject, plan: Dict[str, Any], memory: SessionMemory, client) -> Dict[int, Harness]:
    """One harness per substep, stored at session:{id}:harness as {step_id: harness dict}."""
    harnesses = {step["step_id"]: generate_harness(task, plan, step, client) for step in plan["substeps"]}
    memory.set(f"session:{task.session_id}:harness", {sid: asdict(h) for sid, h in harnesses.items()})
    return harnesses


def run_harness_generation(task: TaskObject, plan: Dict[str, Any], memory: SessionMemory, client) -> Dict[str, Any]:
    """Generate all harnesses; escalate (never guess) if any is unusable or too uncertain."""
    try:
        harnesses = generate_harnesses(task, plan, memory, client)
    except (AnalysisError, ValueError) as e:
        memory.transition(task.session_id, SessionState.NEEDS_HUMAN_REVIEW)
        return {"status": "needs_human_review", "reason": f"harness generation failed: {e}"}

    weak = [h.step_id for h in harnesses.values() if h.confidence_score < CONFIDENCE_MIN]
    if weak:
        memory.transition(task.session_id, SessionState.NEEDS_HUMAN_REVIEW)
        return {"status": "needs_human_review", "harnesses": harnesses,
                "reason": f"harness confidence below {CONFIDENCE_MIN} for steps {weak}"}
    return {"status": "generated", "harnesses": harnesses}
