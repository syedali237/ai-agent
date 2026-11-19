"""Live smoke test for the planning half of the pipeline (no tools run, no files touched).

    uv run python scripts/smoke_plan.py ["your task prompt"] [--out result.json]

Runs intake -> analysis -> harness generation -> plan review against the real models, re-planning
on rejection (max 3 attempts), and prints what each stage produced. Costs a handful of API calls.
"""
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

import config  # noqa: E402
from helpers.analyse_input import analyze_task, run_analysis  # noqa: E402
from helpers.generate_harness import run_harness_generation  # noqa: E402
from helpers.session_memory import MAX_PLAN_ATTEMPTS, SessionMemory, SessionState  # noqa: E402
from helpers.verify_harness import review_gate  # noqa: E402

DEFAULT_PROMPT = (
    "Add a function slugify(text) to src/utils.py that lowercases text and replaces spaces "
    "with hyphens, and add a unit test for it in tests/test_utils.py."
)


def banner(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def main():
    args = sys.argv[1:]
    out_path = None
    if "--out" in args:
        i = args.index("--out")
        out_path = args[i + 1]
        del args[i:i + 2]
    prompt = args[0] if args else DEFAULT_PROMPT

    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set (put it in .env)")
    client = anthropic.Anthropic()

    print("Models:", {k: getattr(config, k) for k in
                      ("INTAKE_MODEL", "ANALYSIS_MODEL", "REVIEW_MODEL", "EXECUTION_MODEL", "VERIFY_MODEL")})
    print("Prompt:", prompt)

    memory = SessionMemory()
    sid = memory.start_session(prompt)
    report = {"prompt": prompt, "session_id": sid, "attempts": []}

    try:
        banner("1-2. INTAKE + ANALYSIS")
        analysis = run_analysis(prompt, sid, memory, client)
        if analysis["status"] != "analyzed":
            print("STOPPED:", analysis["reason"])
            report["final"] = analysis
            return finish(report, memory, sid, out_path)
        task, plan = analysis["task"], analysis["plan"]
        print(json.dumps(asdict(task), indent=2))
        print(json.dumps(plan, indent=2))

        findings = None
        for attempt in range(1, MAX_PLAN_ATTEMPTS + 1):
            record = {"attempt": attempt}
            report["attempts"].append(record)
            if findings:  # re-plan using the reviewer's objections
                banner(f"RE-ANALYSIS (attempt {attempt})")
                plan = analyze_task(task, memory, client, review_findings=findings)
                print(json.dumps(plan, indent=2))
            record["plan"] = plan

            banner(f"3. HARNESSES (attempt {attempt})")
            generated = run_harness_generation(task, plan, memory, client)
            if generated["status"] != "generated":
                print("STOPPED:", generated["reason"])
                report["final"] = generated
                return finish(report, memory, sid, out_path)
            harness_view = {s: asdict(h) for s, h in generated["harnesses"].items()}
            record["harnesses"] = harness_view
            print(json.dumps(harness_view, indent=2))

            banner(f"4. PLAN REVIEW (attempt {attempt})")
            gate = review_gate(task, plan, memory, client, harness_view)
            record["review"] = gate.get("review")
            print("status:", gate["status"], gate.get("reason", ""))
            print(json.dumps(gate.get("review"), indent=2))
            if gate["status"] != "rejected":
                report["final"] = {"status": gate["status"], "reason": gate.get("reason", "")}
                break
            findings = gate["review"]["findings"]
            memory.transition(sid, SessionState.ANALYZED)
    except anthropic.APIError as e:
        print(f"\nAPI ERROR: {type(e).__name__}: {e}")
        report["final"] = {"status": "api_error", "reason": str(e)}
    finish(report, memory, sid, out_path)


def finish(report, memory, sid, out_path):
    banner("RESULT")
    print("final:", report.get("final"))
    print("session state:", memory.get_state(sid).value)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print("full report written to", out_path)


if __name__ == "__main__":
    main()
