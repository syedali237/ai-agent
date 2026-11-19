"""Before/after view of the workspace, so verification sees what a step *actually* changed.

This catches side effects the tool journal cannot (e.g. a script run by run_python_file writing
files). Cheap signature per file: (size, mtime_ns). Directories that are never part of the work
(VCS, virtualenvs, caches) are skipped, so writes there are NOT detected.
"""
import os
from typing import Any, Dict

EXCLUDED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
                 ".mypy_cache", ".idea", ".vscode"}
MAX_TREE_ENTRIES = 50_000


def snapshot_tree(working_dir: str) -> Dict[str, Any]:
    root = os.path.realpath(working_dir)
    files: Dict[str, Any] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for name in filenames:
            if len(files) >= MAX_TREE_ENTRIES:
                return {"files": files, "truncated": True}
            path = os.path.join(dirpath, name)
            try:
                st = os.lstat(path)
            except OSError:
                continue
            files[os.path.relpath(path, root).replace(os.sep, "/")] = (st.st_size, st.st_mtime_ns)
    return {"files": files, "truncated": False}


def diff_trees(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    b, a = before["files"], after["files"]
    return {
        "created": sorted(set(a) - set(b)),
        "deleted": sorted(set(b) - set(a)),
        "modified": sorted(p for p in set(a) & set(b) if a[p] != b[p]),
        "truncated": before["truncated"] or after["truncated"],
    }
