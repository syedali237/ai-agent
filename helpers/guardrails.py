"""Pure guardrail checks, one function per layer, used by the executor (phase 5).

Each check takes the stored harness dict and returns a violation message or None. Nothing
here runs a tool. The `content` layer (must_not_modify / must_preserve) needs the file diff
produced by an actual write, so it is checked after the mutation by the executor/verifier, not here.
"""
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from config import resolve_path_in_workdir
from helpers.tool_registry import ToolSpec

_FLAGS = re.IGNORECASE if os.name == "nt" else 0


def _glob_regex(pattern: str) -> "re.Pattern[str]":
    """`**` crosses directories, `*` and `?` do not. `dir/**` also matches `dir` itself."""
    p = pattern.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    p = p.strip("/") if p != "/" else p
    suffix = ""
    if p.endswith("/**"):
        p, suffix = p[:-3], "(?:/.*)?"
    out, i = [], 0
    while i < len(p):
        if p.startswith("**", i):
            out.append(".*")
            i += 2
        elif p[i] == "*":
            out.append("[^/]*")
            i += 1
        elif p[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(p[i]))
            i += 1
    return re.compile("^" + "".join(out) + suffix + "$", _FLAGS)


def path_matches(patterns: List[str], rel_path: str) -> bool:
    return any(_glob_regex(p).match(rel_path) for p in patterns)


def _resolve(working_dir: str, value: Any) -> Tuple[Optional[str], Optional[str]]:
    """Return (workspace-relative posix path, None) or (None, error). Follows symlinks."""
    if not isinstance(value, str):
        return None, f"path argument must be a string, got {type(value).__name__}"
    root, target = resolve_path_in_workdir(working_dir, value)
    root, target = os.path.realpath(root), os.path.realpath(target)
    try:
        inside = os.path.commonpath([root, target]) == root
    except ValueError:  # different drives
        inside = False
    if not inside:
        return None, f'"{value}" is outside the workspace'
    return os.path.relpath(target, root).replace(os.sep, "/"), None


def check_scope(harness: Dict[str, Any], spec: ToolSpec, args: Dict[str, Any], working_dir: str) -> Optional[str]:
    scope = harness["guardrails"]["scope"]
    if spec.name not in scope["allowed_tools"]:
        return f"tool '{spec.name}' is not allowed in this step"
    for param in spec.path_params:
        # A missing path argument means the tool falls back to its default, so check "." (deny by default).
        rel, error = _resolve(working_dir, args.get(param, "."))
        if error:
            return error
        if spec.writes_files:
            if not path_matches(scope["allowed_write"], rel):
                return f'write to "{rel}" is outside allowed_write'
            if path_matches(scope["forbidden_write"], rel):
                return f'write to "{rel}" is forbidden'
        elif not path_matches(scope["allowed_read"] + scope["allowed_write"], rel):
            return f'access to "{rel}" is outside allowed_read'
    return None


def check_limits(harness: Dict[str, Any], tool: str, calls_made: Dict[str, int], elapsed_seconds: float) -> Optional[str]:
    limits = harness["guardrails"]["limits"]
    per_tool = limits["tool_limits"].get(tool)
    if per_tool is None:
        return f"no limits defined for tool '{tool}'"
    if calls_made.get(tool, 0) >= per_tool["max_per_step"]:
        return f"'{tool}' exceeded max_per_step ({per_tool['max_per_step']})"
    if elapsed_seconds > limits["max_duration_seconds"]:
        return f"step exceeded max_duration_seconds ({limits['max_duration_seconds']})"
    return None


def check_call(harness: Dict[str, Any], spec: ToolSpec, args: Dict[str, Any], working_dir: str,
               calls_made: Dict[str, int], elapsed_seconds: float) -> List[Tuple[str, str]]:
    """Run every pre-call layer separately. Returns [(layer, message)]; empty means allowed."""
    violations = []
    for layer, message in (
        ("scope", check_scope(harness, spec, args, working_dir)),
        ("limits", check_limits(harness, spec.name, calls_made, elapsed_seconds)),
    ):
        if message:
            violations.append((layer, message))
    return violations
