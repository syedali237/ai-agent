import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.analyse_input import AnalysisError  # noqa: E402
from helpers.generate_harness import validate_harness  # noqa: E402
from helpers.tool_registry import BUILTIN_TOOLS, ToolRegistryError, load_registry, to_tool_param  # noqa: E402

MANIFEST = {
    "name": "format_python_file", "description": "Format one file.", "enabled": True,
    "access": "mutating", "entrypoint": "custom_tools.formatting:format_python_file",
    "path_params": ["file_path"], "rollback_strategy": "snapshot_paths",
    "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
}


def write(tmp_path, manifest, name="tool.json"):
    (tmp_path / name).write_text(json.dumps(manifest), encoding="utf-8")


def with_(**kw):
    m = copy.deepcopy(MANIFEST)
    m.update(kw)
    return m


def test_no_directory_means_builtins_only(tmp_path):
    assert set(load_registry(tmp_path / "missing")) == set(BUILTIN_TOOLS)


def test_shipped_custom_tools_dir_loads():
    assert set(BUILTIN_TOOLS) <= set(load_registry())


def test_valid_manifest_registers(tmp_path):
    write(tmp_path, MANIFEST)
    spec = load_registry(tmp_path)["format_python_file"]
    assert spec.mutating and spec.writes_files and spec.needs_timeout and not spec.builtin
    assert to_tool_param(spec)["name"] == "format_python_file"


def test_disabled_or_unflagged_manifest_is_ignored(tmp_path):
    write(tmp_path, with_(enabled=False), "a.json")
    m = copy.deepcopy(MANIFEST)
    del m["enabled"]
    write(tmp_path, m, "b.json")
    assert set(load_registry(tmp_path)) == set(BUILTIN_TOOLS)


@pytest.mark.parametrize("bad", [
    with_(name="write_file"),                                   # collides with built-in
    with_(name="Bad Name"),
    with_(entrypoint="os:system"),                              # outside custom_tools
    with_(access="root"),
    with_(path_params=[]),                                      # mutating but unscopable
    with_(path_params=["nope"]),                                # not in schema
    with_(rollback_strategy=None),                              # mutating but not undoable
    with_(rollback_strategy="custom_undo"),                     # missing undo_entrypoint
    with_(input_schema={"type": "string"}),
    with_(description=""),
])
def test_invalid_manifests_raise(tmp_path, bad):
    write(tmp_path, bad)
    with pytest.raises(ToolRegistryError):
        load_registry(tmp_path)


def test_duplicate_names_raise(tmp_path):
    write(tmp_path, MANIFEST, "a.json")
    write(tmp_path, MANIFEST, "b.json")
    with pytest.raises(ToolRegistryError):
        load_registry(tmp_path)


def test_bad_json_raises(tmp_path):
    (tmp_path / "x.json").write_text("{nope", encoding="utf-8")
    with pytest.raises(ToolRegistryError):
        load_registry(tmp_path)


def test_read_only_custom_tool_needs_no_rollback(tmp_path):
    write(tmp_path, with_(access="read_only", rollback_strategy=None, path_params=[]))
    assert not load_registry(tmp_path)["format_python_file"].mutating


# --- a harness may only grant registered custom tools, with the same scoping rules as write_file ---

STEP = {"step_id": 1, "goal": "fmt", "type": "refactor", "dependencies": [],
        "estimated_effort": 1, "rollback_hint": "snapshot", "confidence_score": 0.9}


def harness_granting(tool, timeout=True):
    limit = {"max_per_step": 1}
    if timeout:
        limit["timeout_seconds"] = 10
    return {
        "guardrails": {
            "scope": {"allowed_tools": [tool], "allowed_read": ["src/**"], "allowed_write": ["src/**"], "forbidden_write": []},
            "limits": {"tool_limits": {tool: limit}, "max_duration_seconds": 30},
            "content": {"must_not_modify": [], "must_preserve": ["public API"]},
        },
        "validation_checks": [{"name": "c", "check": "x"}],
        "rollback_plan": {"actions": ["restore snapshot"], "test_rollback": ""},
        "success_criteria": ["formatted"], "confidence_score": 0.9,
    }


def test_harness_can_grant_enabled_custom_tool(tmp_path):
    write(tmp_path, MANIFEST)
    reg = load_registry(tmp_path)
    assert validate_harness(harness_granting("format_python_file"), STEP, reg)


def test_harness_cannot_grant_unregistered_tool(tmp_path):
    with pytest.raises(AnalysisError):
        validate_harness(harness_granting("format_python_file"), STEP, load_registry(tmp_path))


def test_custom_tool_requires_timeout_and_write_scope(tmp_path):
    write(tmp_path, MANIFEST)
    reg = load_registry(tmp_path)
    with pytest.raises(AnalysisError):
        validate_harness(harness_granting("format_python_file", timeout=False), STEP, reg)
    no_write = harness_granting("format_python_file")
    no_write["guardrails"]["scope"]["allowed_write"] = []
    with pytest.raises(AnalysisError):
        validate_harness(no_write, STEP, reg)
    research = {**STEP, "type": "research"}
    with pytest.raises(AnalysisError):
        validate_harness(harness_granting("format_python_file"), research, reg)
