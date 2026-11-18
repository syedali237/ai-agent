---
name: dynamic-harness
description: Use when building or extending this repo's staged coding-agent workflow (analyze, harness/guardrails, plan review, bounded execution, verification, rollback), or when changing helpers/analyse_input.py, generate_harness.py, execution.py, verify_harness.py, or the session/temporal/vector storage layers. Defines the task object, confidence policy, execution rules, and build order.
---

# Dynamic Harness Agent

This repo is evolving from a simple tool-calling CLI (`main.py`) into a controlled agent runtime: **plan first, act second, validate last**. Full vision is in `README.md`; this skill is the working contract for implementing it.

## Lifecycle

```
intake -> analyze -> generate harness -> review plan -> (approved?) -> execute step -> verify step -> next step -> finalize
                          ^                      | rejected (max 3 attempts, then human review)
                          +----------------------+            verify fails -> retry (max 2) -> rollback / escalate
```

| Stage | Model role | Repo location |
|---|---|---|
| Intake: normalize request into a task object | small/cheap | `helpers/analyse_input.py` |
| Analyze: decompose into substeps with risks and dependencies | large | `helpers/analyse_input.py` |
| Harness: per-step guardrails, checks, rollback, confidence | large | `helpers/generate_harness.py` |
| Plan review: independent safety check | second large | `helpers/verify_harness.py` |
| Execute: one step inside its harness | small/fast | `helpers/execution.py`, `tools/` |
| Verify: compare plan vs. what actually happened | large | `helpers/verify_harness.py` |

Model IDs live in `config.py` only. Different models for different jobs is the point: never let the model that executed a step be the only one to judge it.

## Task object (stage 1 output)

`session_id`, `user_prompt`, `task_category`, `target_files`, `success_criteria`, `constraints`, `risk_level` (low/medium/high/critical), `estimated_complexity` (1-5), `assumptions`, `confidence_score`.

## Harness (one per substep)

`step_id`, `goal`, `allowed_tools`, `tool_limits`, `guardrails` (file read/write allow and forbid lists, resource limits), `validation_checks`, `rollback_plan`, `success_criteria`, `confidence_score`.

## Execution policy (hard rules)

1. Never execute before a plan exists and has been approved by the review stage.
2. Never execute a step with no validation strategy.
3. Never let the execution model exceed the harness tool and file scope. **Enforce in code** (`tools/`, `call_function.py`), not just in the prompt.
4. Record how to undo each mutation *before* performing it (e.g. snapshot a file before `write_file`).
5. Verify each step right after it runs; never batch verification to the end.
6. Never finalize without a verification pass. If verification can't prove success, stop and escalate; don't guess.
7. The execution model gets only: approved plan, current step, allowed tools/files, expected output format, needed session state. No unrestricted context.

## Confidence policy (tunable per task type)

- `>= 0.85`: execute normally
- `0.60 - 0.84`: execute with tighter validation
- `< 0.60`: deeper analysis or human review; do not execute

## Human review is required when

confidence stays low after a second analysis pass; the task is ambiguous; rollback would be risky or destructive; the plan touches an untrusted dependency or environment; verification is inconclusive.

## Session state machine

`received -> analyzed -> reviewed -> approved -> executing -> verifying -> finalized`, with terminal branches `rejected`, `rolled_back`, `needs_human_review`.

## Build order (do not skip ahead)

1. Session schema and state machine (in-memory `SessionMemory`)
2. Structured JSON analysis output
3. Plan review gate
4. Harness generation with per-step guardrails
5. Bounded execution with tool allowlists and timeouts
6. Per-step verification and rollback
7. Persist events (temporal DB, e.g. SQLite/Postgres)
8. Vector retrieval of prior tasks
9. Tune thresholds and retry policy

Phases 1-6 need no external database. Do not add Redis/Postgres/Pinecone until 1-6 work end-to-end.

## Implementation notes

- All LLM calls that must return JSON go through a tolerant `parse_json()` (handles markdown fences); retry once, then escalate.
- Keep path safety in `config.py`; every tool resolves paths inside the workspace boundary.
- Cap retries (plan: 3 attempts, step: 2 retries) so the loop always terminates.

## References (load only when needed)

- [references/implementation.md](references/implementation.md): code patterns for every stage plus the main loop
- [references/database.md](references/database.md): Redis / Postgres / vector DB / analytics setup (optional, later phases)
