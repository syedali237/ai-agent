import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.guardrails import check_call, check_limits, check_scope, path_matches  # noqa: E402
from helpers.tool_registry import BUILTIN_TOOLS  # noqa: E402

HARNESS = {"guardrails": {
    "scope": {"allowed_tools": ["get_file_content", "write_file"], "allowed_read": ["src/**", "README.md"],
              "allowed_write": ["src/*.py"], "forbidden_write": ["src/secret.py"]},
    "limits": {"tool_limits": {"write_file": {"max_per_step": 2}, "get_file_content": {"max_per_step": 5}},
               "max_duration_seconds": 10},
}}
READ, WRITE, RUN = BUILTIN_TOOLS["get_file_content"], BUILTIN_TOOLS["write_file"], BUILTIN_TOOLS["run_python_file"]


@pytest.mark.parametrize("patterns,path,expected", [
    (["src/**"], "src/a/b.py", True),
    (["src/**"], "src", True),
    (["src/**"], "srcx/a.py", False),
    (["src/*.py"], "src/a.py", True),
    (["src/*.py"], "src/sub/a.py", False),   # single * does not cross directories
    (["README.md"], "README.md", True),
    (["./src/**"], "src/a.py", True),
    (["a?.py"], "ab.py", True),
])
def test_path_matches(patterns, path, expected):
    assert path_matches(patterns, path) is expected


@pytest.fixture
def wd(tmp_path):
    (tmp_path / "src").mkdir()
    return str(tmp_path)


def test_scope_allows_in_scope_read_and_write(wd):
    assert check_scope(HARNESS, READ, {"file_path": "src/a/b.py"}, wd) is None
    assert check_scope(HARNESS, WRITE, {"file_path": "src/a.py"}, wd) is None


@pytest.mark.parametrize("tool,path", [
    (WRITE, "src/sub/a.py"),        # not matched by src/*.py
    (WRITE, "src/secret.py"),       # forbidden_write
    (WRITE, "README.md"),           # readable but not writable
    (READ, "other/x.py"),           # outside allowed_read
    (READ, "../outside.txt"),       # escapes workspace
])
def test_scope_blocks(wd, tool, path):
    assert check_scope(HARNESS, tool, {"file_path": path}, wd)


def test_scope_blocks_tool_not_granted(wd):
    assert "not allowed" in check_scope(HARNESS, RUN, {"file_path": "src/a.py"}, wd)


def test_missing_path_arg_is_denied_by_default(wd):
    assert check_scope(HARNESS, READ, {}, wd)  # falls back to "." which is not in allowed_read


def test_non_string_path_blocked(wd):
    assert check_scope(HARNESS, READ, {"file_path": ["src/a.py"]}, wd)


def test_symlink_escape_blocked(wd, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    (outside / "x.txt").write_text("secret")
    link = os.path.join(wd, "src", "link")
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this system")
    assert check_scope(HARNESS, READ, {"file_path": "src/link/x.txt"}, wd)


def test_limits():
    assert check_limits(HARNESS, "write_file", {}, 1) is None
    assert "max_per_step" in check_limits(HARNESS, "write_file", {"write_file": 2}, 1)
    assert "max_duration" in check_limits(HARNESS, "write_file", {}, 11)
    assert check_limits(HARNESS, "run_python_file", {}, 1)  # no limits defined -> blocked


def test_check_call_reports_each_layer_separately(wd):
    result = check_call(HARNESS, WRITE, {"file_path": "README.md"}, wd, {"write_file": 2}, 1)
    assert [layer for layer, _ in result] == ["scope", "limits"]
    assert check_call(HARNESS, WRITE, {"file_path": "src/a.py"}, wd, {}, 1) == []
