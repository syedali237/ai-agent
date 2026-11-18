"""Tool registry: built-in tools plus human-authored custom tools.

Custom tools are declared by a JSON manifest in custom_tools/ (see custom_tools/README.md).
Only humans add manifests; the model can be *granted* a registered, enabled tool by a harness
but can never create or register one. Manifests are validated at load time and fail loudly.
"""
import importlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

CUSTOM_TOOLS_DIR = Path(__file__).resolve().parent.parent / "custom_tools"
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,40}$")
ENTRYPOINT_RE = re.compile(r"^custom_tools\.[a-z_][a-z0-9_.]*:[A-Za-z_]\w*$")
ACCESS_LEVELS = {"read_only", "mutating"}
ROLLBACK_STRATEGIES = {"snapshot_paths", "custom_undo"}


class ToolRegistryError(Exception):
    pass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    access: str                        # "read_only" | "mutating"
    writes_files: bool                 # needs an explicit allowed_write scope in its harness
    path_params: Tuple[str, ...]       # argument names holding file paths, so scope can be enforced
    needs_timeout: bool                # harness must set tool_limits[name].timeout_seconds
    builtin: bool
    input_schema: Optional[Dict[str, Any]] = None
    entrypoint: Optional[str] = None   # "custom_tools.module:function" (custom tools only)
    rollback_strategy: Optional[str] = None
    undo_entrypoint: Optional[str] = None

    @property
    def mutating(self) -> bool:
        return self.access == "mutating"


BUILTIN_TOOLS: Dict[str, ToolSpec] = {
    "get_files_info": ToolSpec("get_files_info", "List files in a directory.", "read_only",
                               False, ("directory",), False, True),
    "get_file_content": ToolSpec("get_file_content", "Read a file's content.", "read_only",
                                 False, ("file_path",), False, True),
    "write_file": ToolSpec("write_file", "Create or overwrite a file.", "mutating",
                           True, ("file_path",), False, True),
    "run_python_file": ToolSpec("run_python_file", "Run a Python file and return its output.", "mutating",
                                False, ("file_path",), True, True),
}


def _validate_manifest(path: Path, m: Dict[str, Any]) -> ToolSpec:
    def fail(msg: str):
        raise ToolRegistryError(f"{path.name}: {msg}")

    name = m.get("name")
    if not isinstance(name, str) or not NAME_RE.match(name):
        fail(f"invalid name {name!r} (lowercase letters, digits, underscores; 3-41 chars)")
    if name in BUILTIN_TOOLS:
        fail(f"'{name}' collides with a built-in tool")
    if not str(m.get("description", "")).strip():
        fail("description is required")
    if m.get("access") not in ACCESS_LEVELS:
        fail(f"access must be one of {sorted(ACCESS_LEVELS)}")
    entry = m.get("entrypoint")
    if not isinstance(entry, str) or not ENTRYPOINT_RE.match(entry):
        fail("entrypoint must look like 'custom_tools.module:function'")

    schema = m.get("input_schema")
    if not isinstance(schema, dict) or schema.get("type") != "object" or not isinstance(schema.get("properties"), dict):
        fail("input_schema must be a JSON-schema object with 'properties'")
    params = m.get("path_params", [])
    if not isinstance(params, list) or not all(isinstance(p, str) for p in params):
        fail("path_params must be a list of argument names")
    for p in params:
        if p not in schema["properties"]:
            fail(f"path_param '{p}' is not in input_schema.properties")

    strategy = m.get("rollback_strategy")
    undo = m.get("undo_entrypoint")
    if m["access"] == "mutating":
        # Mutating tools must be scopable and undoable (execution rules 3 and 4).
        if not params:
            fail("mutating tools must declare path_params so guardrails can scope them")
        if strategy not in ROLLBACK_STRATEGIES:
            fail(f"mutating tools need rollback_strategy in {sorted(ROLLBACK_STRATEGIES)}")
        if strategy == "custom_undo" and not (isinstance(undo, str) and ENTRYPOINT_RE.match(undo)):
            fail("rollback_strategy 'custom_undo' needs a valid undo_entrypoint")

    return ToolSpec(
        name=name, description=m["description"].strip(), access=m["access"],
        writes_files=m["access"] == "mutating",   # conservative: custom mutators need allowed_write
        path_params=tuple(params), needs_timeout=True,  # custom code can hang
        builtin=False, input_schema=schema, entrypoint=entry,
        rollback_strategy=strategy, undo_entrypoint=undo,
    )


def load_registry(custom_dir: Optional[Path] = None) -> Dict[str, ToolSpec]:
    """Built-in tools plus every *enabled* manifest in custom_dir. Invalid manifests raise.

    A manifest without "enabled": true is ignored, so adding a file never grants anything by accident.
    """
    registry = dict(BUILTIN_TOOLS)
    directory = Path(custom_dir) if custom_dir is not None else CUSTOM_TOOLS_DIR
    if not directory.is_dir():
        return registry
    for path in sorted(directory.glob("*.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ToolRegistryError(f"{path.name}: invalid JSON: {e}") from e
        if not isinstance(manifest, dict):
            raise ToolRegistryError(f"{path.name}: manifest must be a JSON object")
        if manifest.get("enabled") is not True:
            continue
        spec = _validate_manifest(path, manifest)
        if spec.name in registry:
            raise ToolRegistryError(f"{path.name}: duplicate tool name '{spec.name}'")
        registry[spec.name] = spec
    return registry


def resolve_entrypoint(entrypoint: str) -> Callable:
    """Import a custom tool function lazily (only the executor needs this)."""
    module_name, func_name = entrypoint.split(":")
    return getattr(importlib.import_module(module_name), func_name)


def to_tool_param(spec: ToolSpec) -> Dict[str, Any]:
    """Anthropic tool definition for a custom tool. Built-ins keep their own schema_* objects."""
    if spec.input_schema is None:
        raise ValueError(f"'{spec.name}' is built in; use its schema from tools/")
    return {"name": spec.name, "description": spec.description, "input_schema": spec.input_schema}
