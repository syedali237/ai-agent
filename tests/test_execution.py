import json
import os
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import helpers.execution as execution  # noqa: E402
from helpers.analyse_input import TaskObject  # noqa: E402
from helpers.execution import ExecutionRefused, cleanup_session, execute_step, rollback_journal  # noqa: E402
from helpers.session_memory import SessionMemory, SessionState  # noqa: E402
from helpers.snapshots import MAX_SNAPSHOT_BYTES, SnapshotError, record_before_mutation, rollback_entries  # noqa: E402
from helpers.tool_registry import BUILTIN_TOOLS, load_registry  # noqa: E402

REG = dict(BUILTIN_TOOLS)


def text(t):
    return SimpleNamespace(type="text", text=t)


def call(name, n=1, **inp):
    return SimpleNamespace(type="tool_use", id=f"tu{n}", name=name, input=inp)


class FakeClient:
    """Scripted model: each entry in `turns` is the list of content blocks for one response."""

    def __init__(self, *turns):
        self.turns = list(turns)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=self.turns.pop(0), usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def harness(step_id=1, tools=("get_file_content", "write_file"), write=("src/*.py",), conf=0.95, limits=None, extra=None):
    h = {
        "step_id": step_id, "goal": "edit src/a.py", "success_criteria": ["a.py updated"], "confidence_score": conf,
        "guardrails": {
            "scope": {"allowed_tools": list(tools), "allowed_read": ["src/**"], "allowed_write": list(write),
                      "forbidden_write": ["src/secret.py"]},
            "limits": {"tool_limits": limits or {t: {"max_per_step": 3} for t in tools}, "max_duration_seconds": 30},
            "content": {"must_not_modify": [], "must_preserve": ["api"]},
        },
    }
    if extra:
        h.update(extra)
    return h


@pytest.fixture
def wd(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("old")
    (tmp_path / "src" / "secret.py").write_text("s3cret")
    return str(tmp_path)


def setup(harnesses, deps=None, approved=True):
    memory = SessionMemory()
    sid = memory.start_session("do it")
    for s in (SessionState.ANALYZED, SessionState.REVIEWED, SessionState.APPROVED):
        memory.transition(sid, s)
    substeps = [{"step_id": h["step_id"], "goal": h["goal"], "dependencies": (deps or {}).get(h["step_id"], [])}
                for h in harnesses]
    memory.set(f"session:{sid}:plan", {"version": 1, "analysis": {"substeps": substeps}})
    memory.set(f"session:{sid}:harness", {h["step_id"]: h for h in harnesses})
    memory.set(f"session:{sid}:review", {"final_approved": approved})
    return memory, TaskObject(session_id=sid, user_prompt="do it", task_category="bug_fix")


def read(wd, rel):
    return open(os.path.join(wd, rel)).read()


def test_happy_path_writes_journals_and_moves_to_verifying(wd):
    memory, task = setup([harness()])
    client = FakeClient([call("write_file", file_path="src/a.py", content="new")], [text("done")])
    log = execute_step(1, task, memory, client, wd, REG)
    assert log["status"] == "completed" and log["final_text"] == "done"
    assert read(wd, "src/a.py") == "new"
    assert memory.get_state(task.session_id) is SessionState.VERIFYING
    journal = memory.get(f"session:{task.session_id}:journal")
    assert len(journal) == 1 and journal[0]["snapshots"][0]["existed"] is True
    assert log["input_tokens"] == 20 and log["validation_mode"] == "normal"


def test_rollback_restores_and_removes_new_files(wd):
    memory, task = setup([harness()])
    client = FakeClient([call("write_file", 1, file_path="src/a.py", content="new"),
                         call("write_file", 2, file_path="src/b.py", content="created")], [text("ok")])
    execute_step(1, task, memory, client, wd, REG)
    assert read(wd, "src/b.py") == "created"
    reports = rollback_journal(task, memory, wd)
    assert all(r["ok"] for r in reports)
    assert read(wd, "src/a.py") == "old" and not os.path.exists(os.path.join(wd, "src", "b.py"))
    assert memory.get(f"session:{task.session_id}:journal") == []
    cleanup_session(memory, task.session_id)


def test_out_of_scope_write_is_blocked_and_not_executed(wd):
    memory, task = setup([harness()])
    client = FakeClient([call("write_file", file_path="src/secret.py", content="pwned")], [text("gave up")])
    log = execute_step(1, task, memory, client, wd, REG)
    assert read(wd, "src/secret.py") == "s3cret"
    assert log["violations"] == 1 and log["tool_calls"][0]["blocked"] == "scope"
    assert memory.get(f"session:{task.session_id}:journal") is None  # nothing mutated, nothing journaled


def test_repeated_violations_abandon_the_step(wd):
    memory, task = setup([harness()])
    bad = lambda n: [call("write_file", n, file_path="../escape.txt", content="x")]  # noqa: E731
    client = FakeClient(bad(1), bad(2), bad(3), [text("never reached")])
    log = execute_step(1, task, memory, client, wd, REG)
    assert log["status"] == "blocked" and log["violations"] == 3
    assert memory.get_state(task.session_id) is SessionState.EXECUTING  # phase 6 decides what next
    assert not os.path.exists(os.path.join(os.path.dirname(wd), "escape.txt"))


def test_model_supplied_working_dir_is_ignored(wd, tmp_path_factory):
    other = str(tmp_path_factory.mktemp("other"))
    memory, task = setup([harness()])
    client = FakeClient([call("write_file", working_dir=other, file_path="src/a.py", content="new")], [text("ok")])
    execute_step(1, task, memory, client, wd, REG)
    assert read(wd, "src/a.py") == "new" and os.listdir(other) == []


def test_per_tool_call_limit(wd):
    memory, task = setup([harness(limits={"get_file_content": {"max_per_step": 3}, "write_file": {"max_per_step": 1}})])
    client = FakeClient([call("write_file", 1, file_path="src/a.py", content="1")],
                        [call("write_file", 2, file_path="src/a.py", content="2")], [text("ok")])
    log = execute_step(1, task, memory, client, wd, REG)
    assert read(wd, "src/a.py") == "1" and log["tool_calls"][1]["blocked"] == "limits"


def test_only_granted_tools_are_exposed_without_working_dir(wd):
    memory, task = setup([harness(tools=("get_file_content",), write=())])
    client = FakeClient([text("nothing to do")])
    execute_step(1, task, memory, client, wd, REG)
    tools = client.calls[0]["tools"]
    assert [t["name"] for t in tools] == ["get_file_content"]
    assert "working_dir" not in tools[0]["input_schema"]["properties"]
    assert "working_dir" not in tools[0]["input_schema"]["required"]
    assert client.calls[0]["model"]


def test_ungranted_tool_call_is_blocked(wd):
    memory, task = setup([harness(tools=("get_file_content",), write=())])
    client = FakeClient([call("write_file", file_path="src/a.py", content="x")], [text("ok")])
    log = execute_step(1, task, memory, client, wd, REG)
    assert read(wd, "src/a.py") == "old" and log["tool_calls"][0]["blocked"] == "scope"


def test_bad_arguments_do_not_run_the_tool(wd):
    memory, task = setup([harness()])
    client = FakeClient([call("write_file", file_path="src/a.py")], [text("ok")])  # missing content
    log = execute_step(1, task, memory, client, wd, REG)
    assert read(wd, "src/a.py") == "old" and log["tool_calls"][0]["blocked"] == "arguments"


def test_refuses_when_plan_not_approved_or_wrong_state(wd):
    memory, task = setup([harness()], approved=False)
    with pytest.raises(ExecutionRefused):
        execute_step(1, task, memory, FakeClient(), wd, REG)
    memory, task = setup([harness()])
    memory.transition(task.session_id, SessionState.EXECUTING)
    memory.transition(task.session_id, SessionState.ROLLED_BACK)
    with pytest.raises(ExecutionRefused):
        execute_step(1, task, memory, FakeClient(), wd, REG)


def test_refuses_step_without_harness(wd):
    memory, task = setup([harness()])
    with pytest.raises(ExecutionRefused):
        execute_step(99, task, memory, FakeClient(), wd, REG)


def test_low_confidence_blocks_without_calling_model(wd):
    memory, task = setup([harness(conf=0.4)])
    client = FakeClient()
    assert execute_step(1, task, memory, client, wd, REG)["status"] == "blocked_low_confidence"
    assert client.calls == [] and memory.get_state(task.session_id) is SessionState.APPROVED


def test_dependency_must_have_completed(wd):
    memory, task = setup([harness(1), harness(2)], deps={2: [1]})
    client = FakeClient()
    assert execute_step(2, task, memory, client, wd, REG)["status"] == "blocked_dependency"
    assert client.calls == []


def test_second_step_runs_after_first_and_sees_its_summary(wd):
    memory, task = setup([harness(1), harness(2)], deps={2: [1]})
    execute_step(1, task, memory, FakeClient([text("step one summary")]), wd, REG)
    client = FakeClient([text("done")])
    log = execute_step(2, task, memory, client, wd, REG)
    assert log["status"] == "completed"
    assert "step one summary" in client.calls[0]["messages"][0]["content"]


def test_tighter_validation_mode_for_mid_confidence(wd):
    memory, task = setup([harness(conf=0.7)])
    assert execute_step(1, task, memory, FakeClient([text("ok")]), wd, REG)["validation_mode"] == "tight"


def test_custom_tool_dispatch_and_timeout(wd, tmp_path_factory, monkeypatch):
    manifests = tmp_path_factory.mktemp("manifests")
    (manifests / "slow.json").write_text(json.dumps({
        "name": "slow_tool", "description": "x", "enabled": True, "access": "mutating",
        "entrypoint": "custom_tools.slow:run", "path_params": ["file_path"], "rollback_strategy": "snapshot_paths",
        "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
    }))
    reg = load_registry(manifests)
    seen = {}

    def fake_tool(working_dir, file_path):
        seen["wd"] = working_dir
        if file_path == "src/slow.py":
            time.sleep(0.4)
        open(os.path.join(working_dir, file_path), "w").write("touched")
        return "touched"

    monkeypatch.setattr(execution, "resolve_entrypoint", lambda entry: fake_tool)
    h = harness(tools=("slow_tool",), write=("src/*.py",), limits={"slow_tool": {"max_per_step": 2, "timeout_seconds": 0.1}})

    memory, task = setup([h])
    log = execute_step(1, task, memory, FakeClient([call("slow_tool", file_path="src/a.py")], [text("ok")]), wd, reg)
    assert log["status"] == "completed" and read(wd, "src/a.py") == "touched"
    assert seen["wd"] == os.path.realpath(wd)

    memory, task = setup([h])
    log = execute_step(1, task, memory, FakeClient([call("slow_tool", file_path="src/slow.py")]), wd, reg)
    assert log["status"] == "failed" and log["possible_runaway_tool"] is True
    assert memory.get(f"session:{task.session_id}:journal")  # undo was recorded before the call
    time.sleep(0.5)  # let the lingering worker finish before the tmp dir is cleaned up


def _journal_with_script_run(memory, sid, wd, changed):
    """A step that wrote src/a.py (journaled) and also ran a script (unjournaled)."""
    root = os.path.join(wd, "..")
    write = record_before_mutation(BUILTIN_TOOLS["write_file"], {"file_path": "src/a.py"}, wd, root, 1, 1)
    run = record_before_mutation(BUILTIN_TOOLS["run_python_file"], {"file_path": "src/a.py"}, wd, root, 2, 1)
    memory.set(f"session:{sid}:journal", [write, run])
    memory.set(f"session:{sid}:execution_log", {1: {"changed_files": changed}})


def test_unjournaled_tool_is_not_a_blocker_when_every_file_change_was_restored(wd):
    memory, task = setup([harness()])
    _journal_with_script_run(memory, task.session_id, wd,   # snapshot first, as the executor does
                             {"created": [], "deleted": [], "modified": ["src/a.py"], "truncated": False})
    open(os.path.join(wd, "src", "a.py"), "w").write("changed by the step")
    reports = rollback_journal(task, memory, wd)
    assert all(r["ok"] for r in reports) and read(wd, "src/a.py") == "old"
    assert memory.get(f"session:{task.session_id}:journal") == []


def test_unjournaled_tool_still_blocks_when_it_changed_files_nothing_can_restore(wd):
    memory, task = setup([harness()])
    open(os.path.join(wd, "src", "stray.txt"), "w").write("made by a script")
    _journal_with_script_run(memory, task.session_id, wd,
                             {"created": ["src/stray.txt"], "deleted": [], "modified": [], "truncated": False})
    reports = rollback_journal(task, memory, wd)
    assert [r["ok"] for r in sorted(reports, key=lambda r: r["seq"])] == [True, False]
    assert len(memory.get(f"session:{task.session_id}:journal")) == 1  # the unrecoverable entry stays


def test_unjournaled_tool_blocks_when_the_diff_is_unreliable(wd):
    memory, task = setup([harness()])
    for changed in ({"created": [], "deleted": [], "modified": [], "truncated": True}, None):
        _journal_with_script_run(memory, task.session_id, wd,
                                 changed or {"created": [], "deleted": [], "modified": [], "truncated": False})
        if changed is None:
            memory.set(f"session:{task.session_id}:execution_log", {})  # step crashed, nothing recorded
        assert not all(r["ok"] for r in rollback_journal(task, memory, wd))


# --- snapshots in isolation ---

def test_snapshot_too_large_is_refused(wd, monkeypatch):
    import helpers.snapshots as snaps
    monkeypatch.setattr(snaps, "MAX_SNAPSHOT_BYTES", 1)
    with pytest.raises(SnapshotError):
        record_before_mutation(BUILTIN_TOOLS["write_file"], {"file_path": "src/a.py"}, wd, os.path.join(wd, ".."), 1, 1)
    assert MAX_SNAPSHOT_BYTES > 1


def test_run_python_file_is_journaled_as_unrecoverable(wd, tmp_path_factory):
    root = str(tmp_path_factory.mktemp("snaps"))
    entry = record_before_mutation(BUILTIN_TOOLS["run_python_file"], {"file_path": "src/a.py"}, wd, root, 1, 1)
    assert entry["recoverable"] is False
    report = rollback_entries([entry], wd)[0]
    assert report["ok"] is False and "rollback_plan" in report["detail"]


def test_directory_snapshot_and_reverse_order_unwind(wd, tmp_path_factory):
    root = str(tmp_path_factory.mktemp("snaps"))
    spec = BUILTIN_TOOLS["write_file"]
    first = record_before_mutation(spec, {"file_path": "src/a.py"}, wd, root, 1, 1)
    open(os.path.join(wd, "src", "a.py"), "w").write("v2")
    second = record_before_mutation(spec, {"file_path": "src/a.py"}, wd, root, 2, 1)
    open(os.path.join(wd, "src", "a.py"), "w").write("v3")
    reports = rollback_entries([first, second], wd)
    assert [r["seq"] for r in reports] == [2, 1] and read(wd, "src/a.py") == "old"
