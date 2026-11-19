"""Undo journal for mutating tool calls (execution rule 4: record the undo *before* mutating).

A journal entry is a plain dict (JSON-serializable, so it can move to Redis/Postgres later):
    {seq, step_id, tool, args, strategy, undo_entrypoint, snapshots: [...], recoverable}
Strategies:
    snapshot_paths - copy every declared path param aside first; rollback restores or deletes it
    custom_undo    - rollback calls the tool's undo_entrypoint(working_dir, **args)
    None           - e.g. run_python_file: side effects are unknown and NOT recoverable by snapshot
"""
import os
import shutil
from typing import Any, Dict, List

from helpers.guardrails import resolve_in_workspace
from helpers.tool_registry import ToolSpec, resolve_entrypoint

MAX_SNAPSHOT_BYTES = 50 * 1024 * 1024


class SnapshotError(Exception):
    pass


def _size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _remove(path: str) -> None:
    if os.path.islink(path) or os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def record_before_mutation(spec: ToolSpec, args: Dict[str, Any], working_dir: str, snapshot_root: str,
                           seq: int, step_id: int) -> Dict[str, Any]:
    """Snapshot whatever the call may overwrite and return the journal entry. Raises SnapshotError."""
    strategy = spec.rollback_strategy or ("snapshot_paths" if spec.writes_files else None)
    entry: Dict[str, Any] = {
        "seq": seq, "step_id": step_id, "tool": spec.name, "args": dict(args), "strategy": strategy,
        "undo_entrypoint": spec.undo_entrypoint, "snapshots": [], "recoverable": strategy is not None,
    }
    if strategy != "snapshot_paths":
        return entry

    root = os.path.realpath(working_dir)
    for idx, param in enumerate(spec.path_params):
        rel, error = resolve_in_workspace(working_dir, args.get(param, "."))
        if error:
            raise SnapshotError(error)
        target = os.path.join(root, rel)
        if not os.path.lexists(target):
            entry["snapshots"].append({"rel": rel, "existed": False, "kind": None, "backup": None})
            continue
        if _size(target) > MAX_SNAPSHOT_BYTES:
            raise SnapshotError(f'"{rel}" is too large to snapshot (> {MAX_SNAPSHOT_BYTES} bytes)')
        backup = os.path.join(snapshot_root, f"{seq}-{idx}")
        if os.path.isdir(target):
            shutil.copytree(target, backup, symlinks=True)
            kind = "dir"
        else:
            shutil.copy2(target, backup)
            kind = "file"
        entry["snapshots"].append({"rel": rel, "existed": True, "kind": kind, "backup": backup})
    return entry


def rollback_entries(entries: List[Dict[str, Any]], working_dir: str) -> List[Dict[str, Any]]:
    """Undo entries newest-first. Returns one report per entry; never raises."""
    root = os.path.realpath(working_dir)
    reports = []
    for entry in sorted(entries, key=lambda e: e["seq"], reverse=True):
        report = {"seq": entry["seq"], "tool": entry["tool"], "strategy": entry["strategy"], "ok": True, "detail": ""}
        try:
            if entry["strategy"] == "snapshot_paths":
                for snap in entry["snapshots"]:
                    target = os.path.join(root, snap["rel"])
                    _remove(target)
                    if snap["existed"]:
                        os.makedirs(os.path.dirname(target), exist_ok=True)
                        if snap["kind"] == "dir":
                            shutil.copytree(snap["backup"], target, symlinks=True)
                        else:
                            shutil.copy2(snap["backup"], target)
            elif entry["strategy"] == "custom_undo":
                resolve_entrypoint(entry["undo_entrypoint"])(working_dir, **entry["args"])
            else:
                report["ok"] = False
                report["detail"] = "side effects are not recoverable by snapshot; follow the harness rollback_plan"
        except Exception as e:  # report and keep unwinding the rest
            report["ok"] = False
            report["detail"] = f"{type(e).__name__}: {e}"
        reports.append(report)
    return reports
