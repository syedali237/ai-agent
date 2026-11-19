"""Manual end-to-end run of the full agent loop (plan -> review -> execute -> verify -> finalize/rollback).

    uv run python scripts/run_agent.py                         # demo task in a fresh throwaway sandbox
    uv run python scripts/run_agent.py "your task" --keep      # your own task, keep the sandbox to inspect
    uv run python scripts/run_agent.py "task" --workdir PATH   # run on a real folder (asks to confirm)
    add --dump session.json to save everything the agent recorded (task, plan, harnesses, review,
    execution log, undo journal, verification) for inspection.

The agent can only touch files inside the working folder, within each step's harness. Use the default
sandbox first. Costs real API calls (about 3 planning calls + 2 per step + 1 verification per step).
"""
import argparse
import json
import logging
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402  (loads .env)
from helpers.agent import run_agent_task  # noqa: E402
from helpers.llm import LLMError, ROLES, check_role_independence, make_client, missing_keys  # noqa: E402
from helpers.session_memory import SessionMemory  # noqa: E402

DEFAULT_PROMPT = (
    "Add a function slugify(text) to src/utils.py that lowercases text and replaces spaces "
    "with hyphens, and add a unit test for it in tests/test_utils.py."
)
SEED_FILES = {
    "README.md": "# Sandbox project\n",
    "src/utils.py": "def add(a, b):\n    return a + b\n",
    "tests/test_utils.py": "from src.utils import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
}


def seed(path):
    for rel, content in SEED_FILES.items():
        full = os.path.join(path, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)


def listing(path, show_content):
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", ".venv"}]
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, path)
            print(f"  {rel}")
            if show_content:
                with open(full, encoding="utf-8", errors="replace") as f:
                    print("    | " + f.read().rstrip().replace("\n", "\n    | "))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    ap.add_argument("--workdir", help="existing folder to work in (default: a new seeded sandbox)")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation for --workdir")
    ap.add_argument("--keep", action="store_true", help="keep the sandbox folder afterwards")
    ap.add_argument("--dump", help="write the full session record to this JSON file")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="  [log] %(message)s")

    print("Models:", {k: getattr(config, k) for k in ROLES})
    check_role_independence()
    if missing_keys():
        sys.exit(f"Missing environment variables (set them in .env, see .env.example): {', '.join(missing_keys())}")

    sandbox = None
    if args.workdir:
        workdir = os.path.abspath(args.workdir)
        if not os.path.isdir(workdir):
            sys.exit(f"{workdir} is not a folder")
        if not args.yes and input(f"The agent may create/modify/delete files in:\n  {workdir}\nContinue? [y/N] ").strip().lower() != "y":
            sys.exit("aborted")
    else:
        workdir = sandbox = tempfile.mkdtemp(prefix="agent-sandbox-")
        seed(workdir)
    print(f"Workspace: {workdir}\nTask: {args.prompt}\n")

    memory = SessionMemory()
    try:
        result = run_agent_task(args.prompt, make_client(), workdir, memory=memory)
    except LLMError as e:  # only raised before a session exists
        sys.exit(f"API error: {e}")
    sid = result["session_id"]

    print("=" * 8, "RESULT", "=" * 8)
    print("status:", result["status"], "| session state:", memory.get_state(sid).value)
    if result.get("reason"):
        print("reason:", result["reason"])
    for step_id, info in (result.get("steps") or {}).items():
        verdict = info.get("verification") or {}
        print(f"step {step_id}: attempts={info['attempts']} verified={verdict.get('verified')} "
              f"next_action={verdict.get('next_action')} confidence={verdict.get('confidence')}")
        for finding in verdict.get("findings", []):
            print("   finding:", finding)

    print("\n" + "=" * 8, "WHAT THE EXECUTOR DID", "=" * 8)
    for step_id, log in (memory.get(f"session:{sid}:execution_log") or {}).items():
        print(f"step {step_id}: {log['status']} ({log['duration_seconds']}s, {log['violations']} blocked calls, "
              f"mode={log['validation_mode']})")
        for call in log["tool_calls"]:
            print(f"   {call['tool']} {json.dumps(call['args'])[:100]} -> {call['blocked'] or call['result'][:70]!r}")
        print("   changed:", log["changed_files"])

    print("\n" + "=" * 8, "WORKSPACE NOW", "=" * 8)
    listing(workdir, show_content=sandbox is not None)

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump(memory.get_session(sid), f, indent=2, default=str)
        print("\nfull session record written to", args.dump)

    if sandbox and not args.keep:
        shutil.rmtree(sandbox, ignore_errors=True)
    elif sandbox:
        print("\nsandbox kept at", sandbox)


if __name__ == "__main__":
    main()
