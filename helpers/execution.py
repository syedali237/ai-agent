# Should execute while monitoring , does the pre-checks meets constraints, executes, with guards, validate results.

# Phase 5: bounded execution. The execution model sees one step and only the tools its harness
# grants. Every tool call goes through one dispatcher that:
#   1. ignores any model-supplied working_dir (the workspace is fixed by the caller)
#   2. checks the scope and limits guardrail layers (helpers/guardrails.py) before running anything
#   3. journals an undo (helpers/snapshots.py) BEFORE any mutating call
#   4. enforces timeouts
# Verification and the session-level rollback decision are phase 6; this module only executes.
import copy
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import config
from helpers.analyse_input import CONFIDENCE_MIN, CONFIDENCE_NORMAL, TaskObject
from helpers.guardrails import check_call
from helpers.session_memory import SessionMemory, SessionState
from helpers.snapshots import SnapshotError, record_before_mutation, rollback_entries
from helpers.tool_registry import ToolSpec, load_registry, resolve_entrypoint, to_tool_param
from helpers.workspace_diff import diff_trees, snapshot_tree
from tools.get_file_content import get_file_content, schema_get_file_content
from tools.get_files_info import get_files_info, schema_get_files_info
from tools.run_python_file import run_python_file, schema_run_python_file
from tools.write_file import schema_write_file, write_file

MAX_VIOLATIONS = 3          # blocked calls tolerated before the step is abandoned
MAX_RESULT_CHARS = config.MAX_CHARS
BUILTIN_SCHEMAS = {
    "get_files_info": schema_get_files_info, "get_file_content": schema_get_file_content,
    "write_file": schema_write_file, "run_python_file": schema_run_python_file,
}
JSON_TYPES = {"string": str, "array": list, "object": dict, "boolean": bool, "integer": int, "number": (int, float)}
SYSTEM_PROMPT = (
    "You are executing exactly ONE step of an already-approved plan inside a sandboxed workspace. "
    "Use only the tools you are given and stay inside the stated file scope. Paths are relative to the "
    "workspace. If a call is blocked, do not try to work around it: stop and explain what you needed. "
    "Know your tools' limits: run_python_file runs ONE .py file (file_path must end in .py; arguments go in "
    "args), so it cannot run `python -m pytest`. If you cannot do something with your tools, such as running "
    "the tests, do not keep trying: finish and say so in your summary."
)


class ExecutionRefused(Exception):
    """Execution was attempted when policy forbids it (rule 1). A caller bug, not a runtime failure."""


def _model_schema(spec: ToolSpec) -> Dict[str, Any]:
    """Tool definition shown to the model, with working_dir removed (the caller fixes it)."""
    schema = copy.deepcopy(BUILTIN_SCHEMAS[spec.name] if spec.builtin else to_tool_param(spec))
    schema["input_schema"]["properties"].pop("working_dir", None)
    required = schema["input_schema"].get("required", [])
    schema["input_schema"]["required"] = [r for r in required if r != "working_dir"]
    return schema


def _check_args(schema: Dict[str, Any], args: Dict[str, Any]) -> Optional[str]:
    props = schema["input_schema"]["properties"]
    unknown = set(args) - set(props)
    if unknown:
        return f"unexpected arguments {sorted(unknown)}"
    missing = [r for r in schema["input_schema"]["required"] if r not in args]
    if missing:
        return f"missing required arguments {missing}"
    for key, value in args.items():
        expected = JSON_TYPES.get(props[key].get("type"))
        if expected and (not isinstance(value, expected) or (props[key]["type"] in ("integer", "number") and isinstance(value, bool))):
            return f"argument '{key}' must be of type {props[key]['type']}"
    return None


def _run_tool(spec: ToolSpec, working_dir: str, args: Dict[str, Any], timeout: float) -> str:
    if spec.builtin:
        if spec.name == "get_files_info":
            return get_files_info(working_dir, args.get("directory", "."))
        if spec.name == "get_file_content":
            return get_file_content(working_dir, args["file_path"])
        if spec.name == "write_file":
            return write_file(working_dir, args["file_path"], args["content"])
        return run_python_file(working_dir, args["file_path"], args.get("args", []), timeout=timeout)

    func = resolve_entrypoint(spec.entrypoint)
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(func, working_dir, **args)
    try:
        return str(future.result(timeout=timeout))
    finally:
        pool.shutdown(wait=False)  # a timed-out thread cannot be killed; the caller flags it


def _trim(text: Any, limit: int = MAX_RESULT_CHARS) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "\n[...truncated]"


def _short_args(args: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (v if not isinstance(v, str) or len(v) <= 200 else v[:200] + "...") for k, v in args.items()}


def _build_prompt(task: TaskObject, harness: Dict[str, Any], prior: List[str], feedback: Optional[str] = None) -> str:
    g = harness["guardrails"]
    scope, limits, content = g["scope"], g["limits"], g.get("content") or {}
    prior_block = "\n".join(prior) if prior else "(none)"
    retry_block = (
        f"\nYOUR PREVIOUS ATTEMPT AT THIS STEP FAILED. Its changes were undone. Problems found:\n{feedback}\n"
        if feedback else ""
    )
    return f"""OVERALL TASK: {task.user_prompt}
{retry_block}

YOUR STEP ({harness['step_id']}): {harness['goal']}
SUCCESS CRITERIA:
{chr(10).join('- ' + c for c in harness['success_criteria'])}

BOUNDS
- Tools: {', '.join(scope['allowed_tools'])}
- May read: {scope['allowed_read'] or 'nothing'}
- May write: {scope['allowed_write'] or 'nothing'}   Never write: {scope['forbidden_write'] or 'nothing extra'}
- Must not modify: {content.get('must_not_modify', [])}   Must preserve: {content.get('must_preserve', [])}
- Per-tool call limits: { {t: v['max_per_step'] for t, v in limits['tool_limits'].items()} }, time budget {limits['max_duration_seconds']}s

RESULTS OF EARLIER STEPS:
{prior_block}

Do only this step. When finished, reply in plain text with: what you did, which files you changed, anything unexpected."""


def execute_step(step_id: int, task: TaskObject, memory: SessionMemory, client, working_dir: str,
                 registry: Optional[Dict[str, ToolSpec]] = None, feedback: Optional[str] = None) -> Dict[str, Any]:
    """Run one step inside its harness. Returns the execution log entry (also stored in memory).

    status: "completed" (session moves to verifying) | "blocked" (too many guardrail violations) |
            "failed" (timeout, turn or duration budget) | "blocked_low_confidence" | "blocked_dependency".
    Failed/blocked steps leave the session in `executing`; phase 6 decides retry vs rollback.
    """
    sid = task.session_id
    registry = registry if registry is not None else load_registry()
    working_dir = os.path.realpath(working_dir)

    # Rule 1: never execute before a plan exists and has been approved.
    review = memory.get(f"session:{sid}:review")
    if not review or review.get("final_approved") is not True:
        raise ExecutionRefused("plan has not been approved by the review gate")
    state = memory.get_state(sid)
    if state not in (SessionState.APPROVED, SessionState.VERIFYING, SessionState.EXECUTING):
        raise ExecutionRefused(f"cannot execute from session state '{state.value}'")
    harnesses = memory.get(f"session:{sid}:harness") or {}
    if step_id not in harnesses:
        raise ExecutionRefused(f"no harness for step {step_id}")  # rule 2: no harness, no execution
    harness = harnesses[step_id]
    plan = memory.get(f"session:{sid}:plan")["analysis"]
    step_def = next(s for s in plan["substeps"] if s["step_id"] == step_id)

    if harness["confidence_score"] < CONFIDENCE_MIN:
        return {"status": "blocked_low_confidence", "step_id": step_id, "action_needed": "human_review",
                "reason": f"harness confidence {harness['confidence_score']:.2f} below {CONFIDENCE_MIN}"}
    logs = memory.get(f"session:{sid}:execution_log") or {}
    pending = [d for d in step_def["dependencies"] if logs.get(d, {}).get("status") != "completed"]
    if pending:
        return {"status": "blocked_dependency", "step_id": step_id, "reason": f"steps {pending} have not completed"}

    if state is not SessionState.EXECUTING:
        memory.transition(sid, SessionState.EXECUTING)

    snapshot_root = memory.get(f"session:{sid}:snapshot_root")
    if not snapshot_root or not os.path.isdir(snapshot_root):
        snapshot_root = tempfile.mkdtemp(prefix=f"agent-{sid}-")
        memory.set(f"session:{sid}:snapshot_root", snapshot_root)

    allowed = harness["guardrails"]["scope"]["allowed_tools"]
    limits = harness["guardrails"]["limits"]
    tool_params = [_model_schema(registry[t]) for t in allowed]
    prior = [f"Step {d}: {_trim(logs[d].get('final_text', ''), 1000)}" for d in step_def["dependencies"]]
    messages: List[Dict[str, Any]] = [{"role": "user", "content": _build_prompt(task, harness, prior, feedback)}]
    tree_before = snapshot_tree(working_dir)

    calls_made: Dict[str, int] = {}
    tool_log: List[Dict[str, Any]] = []
    violations = 0
    tokens_in = tokens_out = 0
    status, reason, final_text, runaway = "failed", "turn budget exhausted", "", False
    started = time.monotonic()
    max_turns = sum(v["max_per_step"] for v in limits["tool_limits"].values()) + 3

    for _ in range(max_turns):
        elapsed = time.monotonic() - started
        if elapsed > limits["max_duration_seconds"]:
            reason = f"exceeded max_duration_seconds ({limits['max_duration_seconds']})"
            break
        message = client.messages.create(
            model=config.EXECUTION_MODEL, max_tokens=2048, system=SYSTEM_PROMPT,
            tools=tool_params, messages=messages,
        )
        usage = getattr(message, "usage", None)
        tokens_in += getattr(usage, "input_tokens", 0) or 0
        tokens_out += getattr(usage, "output_tokens", 0) or 0

        assistant_blocks, texts, results, abort = [], [], [], False
        for block in message.content:
            if block.type == "text":
                assistant_blocks.append({"type": "text", "text": block.text})
                texts.append(block.text)
                continue
            if block.type != "tool_use":
                continue
            assistant_blocks.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
            args = dict(block.input or {})
            args.pop("working_dir", None)  # never trust a model-supplied workspace
            entry = {"tool": block.name, "args": _short_args(args), "blocked": None, "result": ""}
            is_error = True
            spec = registry.get(block.name)

            if abort:
                entry["blocked"], result = "aborted", "Error: step aborted, call not executed."
            elif spec is None or block.name not in allowed:
                violations += 1
                entry["blocked"] = "scope"
                result = f"Blocked by guardrails (scope): tool '{block.name}' is not available in this step."
            elif (arg_error := _check_args(tool_params[allowed.index(block.name)], args)):
                entry["blocked"] = "arguments"
                result = f"Error: invalid arguments: {arg_error}"
            elif (found := check_call(harness, spec, args, working_dir, calls_made, time.monotonic() - started)):
                violations += 1
                entry["blocked"] = found[0][0]
                result = "Blocked by guardrails: " + "; ".join(f"({layer}) {msg}" for layer, msg in found)
            else:
                try:
                    if spec.mutating:  # rule 4: the undo is recorded before the mutation
                        journal = memory.get(f"session:{sid}:journal") or []
                        journal.append(record_before_mutation(spec, args, working_dir, snapshot_root, len(journal) + 1, step_id))
                        memory.set(f"session:{sid}:journal", journal)
                    calls_made[spec.name] = calls_made.get(spec.name, 0) + 1
                    remaining = max(1.0, limits["max_duration_seconds"] - (time.monotonic() - started))
                    per_tool = limits["tool_limits"][spec.name].get("timeout_seconds", remaining)
                    result = _run_tool(spec, working_dir, args, min(per_tool, remaining))
                    is_error = result.startswith("Error:")
                except SnapshotError as e:
                    violations += 1
                    entry["blocked"] = "rollback"
                    result = f"Blocked: cannot record an undo for this call ({e})."
                except FutureTimeout:
                    abort, runaway = True, True
                    status, reason = "failed", f"tool '{spec.name}' timed out; it may still be running"
                    result = "Error: tool timed out."
                except Exception as e:
                    result = f"Error: {type(e).__name__}: {e}"
            entry["result"] = _trim(result, 500)
            tool_log.append(entry)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": _trim(result),
                            **({"is_error": True} if is_error else {})})

        messages.append({"role": "assistant", "content": assistant_blocks})
        if abort:
            break
        if not results:
            status, reason, final_text = "completed", "", "\n".join(texts).strip()
            break
        messages.append({"role": "user", "content": results})
        if violations >= MAX_VIOLATIONS:
            status, reason = "blocked", f"{violations} guardrail violations"
            break

    log = {
        "step_id": step_id, "status": status, "reason": reason, "final_text": final_text,
        "tool_calls": tool_log, "calls_made": calls_made, "violations": violations,
        "changed_files": diff_trees(tree_before, snapshot_tree(working_dir)),
        "validation_mode": "normal" if harness["confidence_score"] >= CONFIDENCE_NORMAL else "tight",
        "possible_runaway_tool": runaway, "duration_seconds": round(time.monotonic() - started, 3),
        "input_tokens": tokens_in, "output_tokens": tokens_out, "model": config.EXECUTION_MODEL,
        "executed_at": datetime.now().isoformat(),
    }
    logs = memory.get(f"session:{sid}:execution_log") or {}
    logs[step_id] = log
    memory.set(f"session:{sid}:execution_log", logs)

    if status == "completed":
        memory.transition(sid, SessionState.VERIFYING)
    return log


def rollback_journal(task: TaskObject, memory: SessionMemory, working_dir: str,
                     step_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Undo journaled mutations newest-first (one step, or the whole session). Phase 6 decides when."""
    entries = memory.get(f"session:{task.session_id}:journal") or []
    selected = [e for e in entries if step_id is None or e["step_id"] == step_id]

    # Files this undo can restore, vs. files the steps really changed (from the workspace diff).
    covered = {snap["rel"] for e in selected for snap in e.get("snapshots", [])}
    logs = memory.get(f"session:{task.session_id}:execution_log") or {}
    touched, uncertain = set(), False
    for sid_ in {e["step_id"] for e in selected}:
        changed = (logs.get(sid_) or {}).get("changed_files")
        if changed is None:  # step crashed before it could record what it changed
            uncertain = True
            continue
        touched |= set(changed["created"]) | set(changed["modified"]) | set(changed["deleted"])
        uncertain = uncertain or changed["truncated"]

    reports = rollback_entries(selected, os.path.realpath(working_dir))
    if not uncertain and touched <= covered:
        # An unjournaled tool (e.g. run_python_file) cannot be undone from a snapshot, but if every
        # file change in the workspace is one we just restored, there is nothing left to undo.
        for report in reports:
            if report["strategy"] is None and not report["ok"]:
                report["ok"] = True
                report["detail"] = ("no workspace changes beyond journaled files; side effects outside the "
                                    "workspace (processes, network) cannot be checked")
    done = {r["seq"] for r in reports if r["ok"]}
    memory.set(f"session:{task.session_id}:journal", [e for e in entries if e["seq"] not in done])
    return reports


def cleanup_session(memory: SessionMemory, session_id: str) -> None:
    """Delete the snapshot directory. Call only after the session is finalized or rolled back."""
    import shutil
    root = memory.get(f"session:{session_id}:snapshot_root")
    if root and os.path.isdir(root):
        shutil.rmtree(root, ignore_errors=True)
    memory.delete(f"session:{session_id}:snapshot_root")
