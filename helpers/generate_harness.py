# Should generate harnesses, guardrails, agent loop, verify, how to manage context,
# tool refining according to the individiual task

# Phase 4: one harness per substep. The model proposes guardrails; code validates them and
# tightens them where the plan implies it. Nothing here executes tools (that is phase 5),
# but the stored harness is the single source of truth the executor will enforce.
#
# Guardrails are layered, at most 3 per step, each checked separately by helpers/guardrails.py:
#   scope   - which tools, which files may be read / written
#   limits  - per-tool call counts and timeouts, total step duration
#   content - what must not be modified / must be preserved (required when the step writes files)
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import config
from helpers.analyse_input import CONFIDENCE_MIN, AnalysisError, TaskObject
from helpers.json_utils import call_json
from helpers.session_memory import SessionMemory, SessionState
from helpers.tool_registry import ToolSpec, load_registry

GUARDRAIL_LAYERS = ("scope", "limits", "content")  # hard cap: 3 layers per step
# Step types that must never write files, regardless of what the model proposes.
READ_ONLY_STEP_TYPES = {"research", "design", "validation"}


@dataclass
class Harness:
    step_id: int
    goal: str
    execution_model: str
    guardrails: Dict[str, Dict[str, Any]]   # keys are a subset of GUARDRAIL_LAYERS
    validation_checks: List[Dict[str, str]]
    rollback_plan: Dict[str, Any]
    success_criteria: List[str]
    confidence_score: float
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def allowed_tools(self) -> List[str]:
        return self.guardrails["scope"]["allowed_tools"]


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


def _validate_scope(sid: int, step: Dict[str, Any], scope: Any, registry: Dict[str, ToolSpec]) -> Dict[str, Any]:
    if not isinstance(scope, dict):
        raise AnalysisError(f"step {sid}: guardrails.scope is required")
    tools = _str_list(scope.get("allowed_tools"), "scope.allowed_tools")
    if not tools:
        raise AnalysisError(f"step {sid}: no tools allowed")
    unknown = set(tools) - set(registry)
    if unknown:
        raise AnalysisError(f"step {sid}: unknown or disabled tools {sorted(unknown)}")
    writers = [t for t in tools if registry[t].writes_files]
    if step["type"] in READ_ONLY_STEP_TYPES and writers:
        raise AnalysisError(f"step {sid}: '{step['type']}' steps must not be granted file-writing tools {writers}")

    out: Dict[str, Any] = {"allowed_tools": tools}
    for key in ("allowed_read", "allowed_write", "forbidden_write"):
        out[key] = _str_list(scope.get(key, []), f"scope.{key}")
        for pattern in out[key]:
            reason = _bad_pattern(pattern)
            if reason:
                raise AnalysisError(f"step {sid}: scope.{key}: {reason}")
    if writers and not out["allowed_write"]:
        raise AnalysisError(f"step {sid}: {writers} granted but allowed_write is empty (must be explicit)")
    if not writers:
        out["allowed_write"] = []  # no writing tool, no write scope
    return out


def _validate_limits(sid: int, tools: List[str], limits: Any, registry: Dict[str, ToolSpec]) -> Dict[str, Any]:
    if not isinstance(limits, dict) or not isinstance(limits.get("tool_limits"), dict):
        raise AnalysisError(f"step {sid}: guardrails.limits.tool_limits is required")
    per_tool: Dict[str, Any] = {}
    for tool in tools:
        lim = limits["tool_limits"].get(tool)
        if not isinstance(lim, dict) or not isinstance(lim.get("max_per_step"), int) or lim["max_per_step"] < 1:
            raise AnalysisError(f"step {sid}: limits.tool_limits.{tool}.max_per_step must be an integer >= 1")
        if registry[tool].needs_timeout and not (isinstance(lim.get("timeout_seconds"), int) and lim["timeout_seconds"] > 0):
            raise AnalysisError(f"step {sid}: {tool} needs a positive timeout_seconds")
        per_tool[tool] = lim  # limits for tools not granted are dropped
    duration = limits.get("max_duration_seconds")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
        raise AnalysisError(f"step {sid}: limits.max_duration_seconds must be positive")
    return {"tool_limits": per_tool, "max_duration_seconds": duration}


def _validate_content(sid: int, content: Any, needed: bool) -> Optional[Dict[str, Any]]:
    if content is None:
        if needed:
            raise AnalysisError(f"step {sid}: a file-writing step needs a guardrails.content layer")
        return None
    if not isinstance(content, dict):
        raise AnalysisError(f"step {sid}: guardrails.content must be an object")
    out = {k: _str_list(content.get(k, []), f"content.{k}") for k in ("must_not_modify", "must_preserve")}
    if needed and not (out["must_not_modify"] or out["must_preserve"]):
        raise AnalysisError(f"step {sid}: content layer must name what to preserve or not modify")
    return out


def validate_harness(data: Dict[str, Any], step: Dict[str, Any], registry: Optional[Dict[str, ToolSpec]] = None) -> Dict[str, Any]:
    """Validate and tighten a model-proposed harness. Raises AnalysisError if unusable."""
    registry = registry if registry is not None else load_registry()
    sid = step["step_id"]

    layers = data.get("guardrails")
    if not isinstance(layers, dict):
        raise AnalysisError(f"step {sid}: guardrails must be an object with layers {list(GUARDRAIL_LAYERS)}")
    extra = set(layers) - set(GUARDRAIL_LAYERS)
    if extra:
        raise AnalysisError(f"step {sid}: unknown guardrail layers {sorted(extra)} (max 3: {list(GUARDRAIL_LAYERS)})")

    scope = _validate_scope(sid, step, layers.get("scope"), registry)
    tools = scope["allowed_tools"]
    limits = _validate_limits(sid, tools, layers.get("limits"), registry)
    writes = any(registry[t].writes_files for t in tools)
    content = _validate_content(sid, layers.get("content"), needed=writes)
    guardrails: Dict[str, Any] = {"scope": scope, "limits": limits}
    if content is not None:
        guardrails["content"] = content

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
    if any(registry[t].mutating for t in tools) and (not isinstance(actions, list) or not actions):
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
        "guardrails": guardrails, "validation_checks": checks, "rollback_plan": rollback,
        "success_criteria": criteria, "confidence_score": conf,
    }


def _tool_catalog(registry: Dict[str, ToolSpec]) -> str:
    lines = []
    for spec in registry.values():
        paths = f", path args: {', '.join(spec.path_params)}" if spec.path_params else ""
        lines.append(f"- {spec.name} [{spec.access}{paths}]: {spec.description}")
    return "\n".join(lines)


def generate_harness(task: TaskObject, plan: Dict[str, Any], step: Dict[str, Any], client,
                     registry: Optional[Dict[str, ToolSpec]] = None) -> Harness:
    registry = registry if registry is not None else load_registry()
    prompt = f"""Generate a harness (guardrails) for ONE step of a plan.
Be as restrictive as the goal allows: least tools, narrowest file scope.

TASK: {task.user_prompt}
TASK CONSTRAINTS: {json.dumps(task.constraints)}
MUST NOT BREAK: {json.dumps(plan.get('must_not_break', []))}
STEP {step['step_id']} ({step['type']}): {step['goal']}
STEP ROLLBACK HINT: {step['rollback_hint']}
TARGET FILES: {json.dumps(task.target_files)}

Available tools (grant only these):
{_tool_catalog(registry)}
All paths are workspace-relative globs. No absolute paths, no "..", no bare "*" or "**".
Guardrails have at most 3 layers: scope, limits, content. The content layer is required when any
granted tool writes files and must name something to preserve or not modify.

Return pure JSON:
{{
  "guardrails": {{
    "scope": {{"allowed_tools": ["..."], "allowed_read": ["..."], "allowed_write": ["..."], "forbidden_write": ["..."]}},
    "limits": {{
      "tool_limits": {{"<tool>": {{"max_per_step": N, "timeout_seconds": S}}}},
      "max_duration_seconds": N
    }},
    "content": {{"must_not_modify": ["..."], "must_preserve": ["..."]}}
  }},
  "validation_checks": [{{"name": "...", "check": "how to validate"}}],
  "rollback_plan": {{"actions": ["steps to undo"], "test_rollback": "how to verify the undo worked"}},
  "success_criteria": ["explicit, checkable criteria"],
  "confidence_score": 0.0-1.0
}}
(timeout_seconds is required for tools that run code.)"""
    data = call_json(client, model=config.REVIEW_MODEL, max_tokens=2500, prompt=prompt)
    checked = validate_harness(data, step, registry)
    return Harness(
        step_id=step["step_id"], goal=step["goal"], execution_model=config.EXECUTION_MODEL, **checked
    )


def generate_harnesses(task: TaskObject, plan: Dict[str, Any], memory: SessionMemory, client,
                       registry: Optional[Dict[str, ToolSpec]] = None) -> Dict[int, Harness]:
    """One harness per substep, stored at session:{id}:harness as {step_id: harness dict}."""
    registry = registry if registry is not None else load_registry()
    harnesses = {
        step["step_id"]: generate_harness(task, plan, step, client, registry) for step in plan["substeps"]
    }
    memory.set(f"session:{task.session_id}:harness", {sid: asdict(h) for sid, h in harnesses.items()})
    return harnesses


def run_harness_generation(task: TaskObject, plan: Dict[str, Any], memory: SessionMemory, client,
                           registry: Optional[Dict[str, ToolSpec]] = None) -> Dict[str, Any]:
    """Generate all harnesses; escalate (never guess) if any is unusable or too uncertain."""
    try:
        harnesses = generate_harnesses(task, plan, memory, client, registry)
    except (AnalysisError, ValueError) as e:
        memory.transition(task.session_id, SessionState.NEEDS_HUMAN_REVIEW)
        return {"status": "needs_human_review", "reason": f"harness generation failed: {e}"}

    weak = [h.step_id for h in harnesses.values() if h.confidence_score < CONFIDENCE_MIN]
    if weak:
        memory.transition(task.session_id, SessionState.NEEDS_HUMAN_REVIEW)
        return {"status": "needs_human_review", "harnesses": harnesses,
                "reason": f"harness confidence below {CONFIDENCE_MIN} for steps {weak}"}
    return {"status": "generated", "harnesses": harnesses}
