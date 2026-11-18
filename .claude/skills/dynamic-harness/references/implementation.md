# Dynamic Harness Agent - Implementation Guide

Concrete code patterns for the agent described in `../SKILL.md`. These are reference sketches, not drop-in code: read each stage, then adapt it to this repo (`main.py`, `call_function.py`, `config.py`, `tools/`, `helpers/`).

**Conventions used below**
- Model IDs come from `config.py` (`ANALYSIS_MODEL`, `REVIEW_MODEL`, `EXECUTION_MODEL`, `VERIFY_MODEL`), never hard-coded. `main.py` currently uses `claude-haiku-4-5` for execution.
- Every LLM call that must return JSON goes through `parse_json()` (section 0), because models sometimes wrap JSON in markdown fences.
- Start with the in-memory `SessionMemory`. Redis, Postgres and vectors are later phases (see `database.md`).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Runtime                             │
├─────────────────────────────────────────────────────────────┤
│  1. TaskIntake    2. Analyzer    3. HarnessGen              │
│  4. Reviewer      5. Executor    6. Verifier                │
│  7. Persister     8. VectorIndexer                          │
└─────────────────────────────────────────────────────────────┘
           ↓              ↓              ↓
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ Session Memory   │ │  Temporal DB     │ │  Vector DB       │
│  (Redis/Memory)  │ │  (PostgreSQL)    │ │ (Pinecone/Qdrant)│
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

---

## 0. Shared Helper: Robust JSON Parsing

```python
# helpers/json_utils.py
import json, re

def parse_json(text: str) -> dict:
    """Parse model output as JSON, tolerating ```json fences and stray prose."""
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise
```

On failure, retry the call once with "Return only valid JSON", then escalate. Never execute on unparsed output.

---

## 1. Session Memory Layer

### Technology: Redis or In-Memory Dict

**For local development (single-process):**

```python
# session_memory.py
import json
from typing import Dict, Any
from datetime import datetime, timedelta

class SessionMemory:
    def __init__(self, ttl_hours=24):
        self.store: Dict[str, Dict[str, Any]] = {}
        self.ttl = timedelta(hours=ttl_hours)
        self.created_at = {}
    
    def set(self, key: str, value: Any) -> None:
        """Set a key-value pair with TTL."""
        self.store[key] = value
        self.created_at[key] = datetime.now()
    
    def get(self, key: str) -> Any:
        """Get a key, return None if expired."""
        if key not in self.store:
            return None
        
        if datetime.now() - self.created_at[key] > self.ttl:
            del self.store[key]
            del self.created_at[key]
            return None
        
        return self.store[key]
    
    def delete(self, key: str) -> None:
        """Delete a key."""
        self.store.pop(key, None)
        self.created_at.pop(key, None)
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get all keys for a session."""
        prefix = f"session:{session_id}:"
        return {k: v for k, v in self.store.items() if k.startswith(prefix)}

# Usage in agent
memory = SessionMemory(ttl_hours=24)

# Store task
memory.set(f"session:{session_id}:task", task_object)

# Retrieve task
task = memory.get(f"session:{session_id}:task")

# Store harness
memory.set(f"session:{session_id}:harness", harness_dict)
```

**For production (distributed):**

```python
# session_memory.py with Redis
import redis
import json
from typing import Any

class RedisSessionMemory:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.ttl_seconds = 24 * 3600  # 24 hours
    
    def set(self, key: str, value: Any) -> None:
        """Set with TTL."""
        self.redis_client.setex(
            key,
            self.ttl_seconds,
            json.dumps(value)
        )
    
    def get(self, key: str) -> Any:
        """Get and auto-deserialize."""
        data = self.redis_client.get(key)
        return json.loads(data) if data else None
    
    def get_session(self, session_id: str) -> dict:
        """Get all keys for a session."""
        prefix = f"session:{session_id}:"
        keys = self.redis_client.keys(f"{prefix}*")
        return {k.decode(): json.loads(self.redis_client.get(k)) for k in keys}
```

---

## 2. Task Intake

### Function: Parse & Normalize

```python
# task_intake.py
import uuid
from dataclasses import dataclass, asdict
from typing import List, Optional
from datetime import datetime

@dataclass
class TaskObject:
    session_id: str
    user_prompt: str
    task_category: str  # code_refactor, feature_add, bug_fix, etc.
    target_files: List[str]
    success_criteria: List[str]
    risk_level: str  # low, medium, high, critical
    estimated_complexity: int  # 1-5
    constraints: List[str]
    assumptions: List[str]
    confidence_score: float = 0.0
    created_at: str = None
    
    def __post_init__(self):
        if not self.session_id:
            self.session_id = f"task-{datetime.now().strftime('%Y-%m-%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

def intake_task(user_prompt: str, session_memory) -> TaskObject:
    """
    Use LLM to parse user prompt into structured task.
    """
    from anthropic import Anthropic
    
    client = Anthropic()
    
    # Parsing prompt (cheap LLM call)
    response = client.messages.create(
        model=config.EXECUTION_MODEL,  # small/cheap model is enough for parsing
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": f"""Parse this user request into a structured task.

User request: "{user_prompt}"

Output JSON (no markdown, pure JSON):
{{
  "task_category": "code_refactor" | "feature_add" | "bug_fix" | "research" | "other",
  "target_files": [...files or directories mentioned or implied...],
  "success_criteria": [...how will we know this succeeded...],
  "risk_level": "low" | "medium" | "high" | "critical",
  "estimated_complexity": 1-5,
  "constraints": [...what must not be done...],
  "assumptions": [...what we're assuming about the codebase...]
}}"""
            }
        ]
    )
    
    # Parse response
    import json
    task_data = parse_json(response.content[0].text)
    
    # Create task object
    task = TaskObject(
        session_id="",  # Generated in __post_init__
        user_prompt=user_prompt,
        **task_data
    )
    
    # Store in session memory
    session_memory.set(f"session:{task.session_id}:task", asdict(task))
    
    return task
```

---

## 3. Deep Analysis Stage

### Function: Structured Planning

```python
# analyzer.py
import json
from typing import Dict, Any
from anthropic import Anthropic

def analyze_task(task: TaskObject, session_memory) -> Dict[str, Any]:
    """
    Large model analyzes task deeply and creates a plan.
    """
    client = Anthropic()
    
    task_dict = asdict(task) if hasattr(task, '__dataclass_fields__') else task
    
    # Analysis prompt (use extended thinking if available)
    analysis_prompt = f"""You are analyzing this task deeply before execution.

TASK:
{json.dumps(task_dict, indent=2)}

Analyze and output a JSON plan with these fields:
{{
  "task_summary": "one-line summary of what's being asked",
  "risks_identified": ["list of potential failure modes"],
  "dependencies": ["what must happen first"],
  "must_not_break": ["existing functionality that must be preserved"],
  "substeps": [
    {{
      "step_id": 1,
      "goal": "precise goal for this step",
      "dependencies": [list of step_ids this depends on],
      "estimated_effort": 1-5,
      "type": "research|design|code_addition|testing|validation|refactor"
    }}
  ]
}}"""
    
    response = client.messages.create(
        model=config.ANALYSIS_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": analysis_prompt}]
    )
    
    plan = parse_json(response.content[0].text)
    
    # Store plan in session memory
    plan_object = {
        "version": 1,
        "analysis": plan,
        "created_at": datetime.now().isoformat()
    }
    session_memory.set(f"session:{task.session_id}:plan", plan_object)
    
    return plan
```

---

## 4. Harness Generation

### Function: Per-Step Guardrails

```python
# harness_generator.py
import json
from typing import Dict, List

@dataclass
class Harness:
    step_id: int
    goal: str
    execution_model: str
    allowed_tools: List[str]
    tool_limits: Dict[str, Dict[str, int]]
    guardrails: Dict[str, Any]
    validation_checks: List[Dict[str, str]]
    rollback_plan: Dict[str, Any]
    success_criteria: List[str]
    confidence_score: float

def generate_harnesses(task: TaskObject, plan: Dict[str, Any], session_memory) -> Dict[int, Harness]:
    """
    Generate harness for each substep.
    """
    client = Anthropic()
    
    harnesses = {}
    
    for substep in plan["substeps"]:
        step_id = substep["step_id"]
        goal = substep["goal"]
        
        # Harness generation prompt
        harness_prompt = f"""Generate a detailed harness for this step.

TASK: {task.user_prompt}
STEP {step_id}: {goal}
ESTIMATED EFFORT: {substep['estimated_effort']} (1-5)

Output JSON:
{{
  "allowed_tools": [...which tools can this step use?...],
  "tool_limits": {{
    "read_file": {{"max_per_step": N, "max_size_kb": K}},
    "write_file": {{"max_per_step": N}},
    "run_command": {{"max_per_step": N, "timeout_seconds": S}}
  }},
  "guardrails": {{
    "file_access": {{
      "allowed_read": [...directories or patterns...],
      "allowed_write": [...specific files...],
      "forbidden_write": [...what must not be modified...],
      "forbidden_delete": ["any" or [...specific files...]]
    }},
    "code_patterns": {{
      "must_not_modify": [...function/class names...],
      "must_preserve": [...exports/APIs...]
    }},
    "resource_limits": {{
      "max_duration_seconds": N,
      "max_file_write_kb": K
    }}
  }},
  "validation_checks": [
    {{"name": "check_name", "check": "how to validate"}},
    ...
  ],
  "rollback_plan": {{
    "actions": [...steps to undo...],
    "test_rollback": "how to verify undo worked"
  }},
  "success_criteria": [...explicit success criteria...],
  "confidence_score": 0.0-1.0
}}"""
        
        response = client.messages.create(
            model=config.REVIEW_MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": harness_prompt}]
        )
        
        harness_data = parse_json(response.content[0].text)
        
        harness = Harness(
            step_id=step_id,
            goal=goal,
            execution_model=config.EXECUTION_MODEL,
            **harness_data
        )
        
        harnesses[step_id] = harness
    
    # Store all harnesses
    harness_dict = {
        step_id: {
            "step_id": harness.step_id,
            "goal": harness.goal,
            "allowed_tools": harness.allowed_tools,
            "tool_limits": harness.tool_limits,
            "guardrails": harness.guardrails,
            "validation_checks": harness.validation_checks,
            "rollback_plan": harness.rollback_plan,
            "success_criteria": harness.success_criteria,
            "confidence_score": harness.confidence_score
        }
        for step_id, harness in harnesses.items()
    }
    
    session_memory.set(f"session:{task.session_id}:harness", harness_dict)
    
    return harnesses
```

---

## 5. Plan Review Gate

### Function: Second Model Approval

```python
# plan_reviewer.py
import json

def review_plan(task: TaskObject, plan: Dict, harnesses: Dict, session_memory) -> bool:
    """
    Second model reviews plan before execution.
    Returns True if approved, False if rejected.
    """
    client = Anthropic()
    
    review_prompt = f"""You are a safety reviewer. Examine this plan and harnesses.

TASK: {task.user_prompt}

PLAN:
{json.dumps(plan, indent=2)}

HARNESSES:
{json.dumps(harnesses, indent=2)}

Review for:
1. Missing dependencies between steps
2. Over-permissive file access
3. Weak validation checks
4. Missing rollback paths
5. Impossible success criteria
6. Resource miscalibration
7. Hidden assumptions

Output JSON:
{{
  "approved": true|false,
  "findings": [
    {{"severity": "critical|warning|info", "issue": "...", "recommendation": "..."}}
  ],
  "confidence_adjustment": 0.0
}}"""
    
    response = client.messages.create(
        model=config.REVIEW_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": review_prompt}]
    )
    
    review = parse_json(response.content[0].text)
    
    # Store review decision
    review_object = {
        "approved": review["approved"],
        "findings": review["findings"],
        "confidence_adjustment": review["confidence_adjustment"],
        "reviewed_at": datetime.now().isoformat()
    }
    session_memory.set(f"session:{task.session_id}:review", review_object)
    
    return review["approved"]
```

---

## 6. Bounded Execution

### Function: Tool Execution with Guardrails

```python
# executor.py
import json
from typing import Dict, Any

class ExecutionMonitor:
    """Enforces guardrails during execution."""
    
    def __init__(self, harness: Harness):
        self.harness = harness
        self.tool_calls_made = {}
        self.total_duration_ms = 0
    
    def pre_check_tool_call(self, tool_name: str, args: Dict) -> bool:
        """
        Pre-execution check: is this tool call within bounds?
        Returns True if allowed, False if forbidden.
        """
        # Check tool is in allowlist
        if tool_name not in self.harness.allowed_tools:
            return False
        
        # Check tool call limits
        limit = self.harness.tool_limits.get(tool_name, {})
        max_calls = limit.get("max_per_step", 999)
        current_calls = self.tool_calls_made.get(tool_name, 0)
        if current_calls >= max_calls:
            return False
        
        # Check file guardrails (if relevant)
        if tool_name == "write_file":
            filepath = args.get("path", "")
            allowed_write = self.harness.guardrails["file_access"]["allowed_write"]
            forbidden_write = self.harness.guardrails["file_access"]["forbidden_write"]
            
            # Check against forbidden list
            for pattern in forbidden_write:
                if pattern == "any":
                    return False
                if pattern in filepath:
                    return False
            
            # Check against allowed list (if not empty)
            if allowed_write and not any(pattern in filepath for pattern in allowed_write):
                return False
        
        return True
    
    def log_tool_call(self, tool_name: str, args: Dict, result: Any, duration_ms: int):
        """Log a successful tool call."""
        self.tool_calls_made[tool_name] = self.tool_calls_made.get(tool_name, 0) + 1
        self.total_duration_ms += duration_ms
        
        # Check resource limits
        if self.total_duration_ms > self.harness.guardrails["resource_limits"]["max_duration_seconds"] * 1000:
            raise Exception(f"Step exceeded max duration ({self.harness.guardrails['resource_limits']['max_duration_seconds']}s)")

def execute_step(step_id: int, task: TaskObject, session_memory) -> Dict[str, Any]:
    """
    Execute a single step within its harness.
    """
    # Retrieve harness
    harnesses = session_memory.get(f"session:{task.session_id}:harness")
    harness_data = harnesses[step_id]
    
    # Check confidence gate
    if harness_data["confidence_score"] < 0.60:
        return {
            "status": "blocked_low_confidence",
            "reason": f"Confidence {harness_data['confidence_score']} below 0.60 threshold",
            "action_needed": "human_review"
        }
    
    # Create monitor
    monitor = ExecutionMonitor(harness_data)
    
    client = Anthropic()
    
    # Execute step
    execution_prompt = f"""Execute this step within the given bounds.

GOAL: {harness_data['goal']}
ALLOWED TOOLS: {harness_data['allowed_tools']}
GUARDRAILS: {json.dumps(harness_data['guardrails'], indent=2)}

Do not exceed these bounds. Output what you did."""
    
    # (This is simplified—in practice, you'd use tool_use or structured output)
    response = client.messages.create(
        model=config.EXECUTION_MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": execution_prompt}]
    )
    
    execution_log = {
        "step_id": step_id,
        "status": "completed",
        "output": response.content[0].text,
        "tool_calls_made": monitor.tool_calls_made,
        "total_duration_ms": monitor.total_duration_ms,
        "executed_at": datetime.now().isoformat()
    }
    
    # Store execution log
    execution_logs = session_memory.get(f"session:{task.session_id}:execution_log") or {}
    execution_logs[step_id] = execution_log
    session_memory.set(f"session:{task.session_id}:execution_log", execution_logs)
    
    return execution_log
```

---

## 7. Verification

### Function: Outcome Validation

```python
# verifier.py
import json

def verify_step(step_id: int, task: TaskObject, session_memory) -> Dict[str, Any]:
    """
    Verify that step execution matches expectations.
    """
    client = Anthropic()
    
    # Get step harness
    harnesses = session_memory.get(f"session:{task.session_id}:harness")
    harness = harnesses[step_id]
    
    # Get execution log
    execution_logs = session_memory.get(f"session:{task.session_id}:execution_log")
    execution = execution_logs[step_id]
    
    # Get plan
    plan = session_memory.get(f"session:{task.session_id}:plan")
    
    verification_prompt = f"""Verify this step execution.

STEP GOAL: {harness['goal']}
SUCCESS CRITERIA:
{json.dumps(harness['success_criteria'], indent=2)}

WHAT WAS EXECUTED:
{execution['output']}

TOOL CALLS:
{json.dumps(execution['tool_calls_made'])}

Did the execution meet the success criteria?
Output JSON:
{{
  "verified": true|false,
  "findings": ["..."],
  "passed_criteria": ["criterion 1", ...],
  "failed_criteria": ["criterion 2", ...],
  "next_action": "proceed|retry|rollback|escalate"
}}"""
    
    response = client.messages.create(
        model=config.VERIFY_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": verification_prompt}]
    )
    
    verification = parse_json(response.content[0].text)
    
    # Store verification result
    verification_result = {
        "step_id": step_id,
        "verified": verification["verified"],
        "findings": verification["findings"],
        "next_action": verification["next_action"],
        "verified_at": datetime.now().isoformat()
    }
    
    session_memory.set(f"session:{task.session_id}:verification_{step_id}", verification_result)
    
    return verification_result
```

---

## 8. Temporal DB Persistence

### Schema & Storage

```python
# persistence.py
import json
import psycopg2
from datetime import datetime

class TemporalDB:
    def __init__(self, connection_string: str):
        self.conn = psycopg2.connect(connection_string)
    
    def persist_session(self, session_id: str, session_memory) -> None:
        """
        Persist entire session to temporal DB.
        """
        # Get all session data from memory
        task = session_memory.get(f"session:{session_id}:task")
        plan = session_memory.get(f"session:{session_id}:plan")
        review = session_memory.get(f"session:{session_id}:review")
        harnesses = session_memory.get(f"session:{session_id}:harness")
        execution_logs = session_memory.get(f"session:{session_id}:execution_log")
        
        cursor = self.conn.cursor()
        
        # Insert summary
        cursor.execute("""
            INSERT INTO task_summaries
            (session_id, user_prompt, task_category, status, started_at, completed_at, duration_seconds, final_confidence)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            session_id,
            task["user_prompt"],
            task["task_category"],
            "completed",
            task["created_at"],
            datetime.now().isoformat(),
            0,  # Calculate from timestamps
            0.82  # Final confidence
        ))
        
        # Insert events (one per stage)
        events = [
            ("task_received", task),
            ("task_analyzed", plan),
            ("plan_reviewed", review),
            ("harnesses_generated", {"harness_count": len(harnesses)}),
            ("execution_completed", {"execution_count": len(execution_logs)})
        ]
        
        for event_type, event_data in events:
            cursor.execute("""
                INSERT INTO agent_events
                (session_id, event_type, timestamp, event_data)
                VALUES (%s, %s, %s, %s)
            """, (
                session_id,
                event_type,
                datetime.now().isoformat(),
                json.dumps(event_data)
            ))
        
        self.conn.commit()
        cursor.close()

# Usage
db = TemporalDB("postgresql://user:password@localhost/agent_db")
db.persist_session(session_id, session_memory)
```

---

## 9. Vector DB Embedding

### Indexing for Future Reuse

```python
# vector_indexing.py
from pinecone import Pinecone

class VectorIndexer:
    def __init__(self, api_key: str):
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index("agent-tasks")
    
    def embed_task(self, session_id: str, task: Dict, session_memory) -> None:
        """
        Embed task and harnesses for future retrieval.
        """
        client = Anthropic()
        
        # Create task summary for embedding
        summary = f"{task['task_category']}: {task['user_prompt']}"
        
        # Get embedding from Claude (or use external service)
        embedding = self._get_embedding(summary)  # Implement with your embedding service
        
        # Prepare metadata
        plan = session_memory.get(f"session:{session_id}:plan")
        harnesses = session_memory.get(f"session:{session_id}:harness")
        
        metadata = {
            "task_type": task["task_category"],
            "summary": summary,
            "complexity": task["estimated_complexity"],
            "success": True,  # If verification passed
            "harness_count": len(harnesses) if harnesses else 0,
            "tools_used": list(set(
                tool for h in (harnesses or {}).values() 
                for tool in h.get("allowed_tools", [])
            ))
        }
        
        # Upsert to vector DB
        self.index.upsert(
            vectors=[{
                "id": session_id,
                "values": embedding,
                "metadata": metadata
            }]
        )
    
    def find_similar_tasks(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Find similar past tasks.
        """
        query_embedding = self._get_embedding(query)
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return results["matches"]
```

---

## 10. Main Agent Loop

### Orchestrating All Stages

Key properties: plan rejection loops back to analysis (bounded), each step is verified immediately after it executes (so rollback is per-step), and nothing is persisted unless verification passed.

```python
# agent.py
import logging
from dataclasses import asdict

MAX_PLAN_ATTEMPTS = 3
MAX_STEP_RETRIES = 2

def run_agent_task(user_prompt: str) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    memory = SessionMemory(ttl_hours=24)      # swap for Redis later
    db = None        # optional phase: TemporalDB(...)
    indexer = None   # optional phase: VectorIndexer(...)

    # STAGE 1: INTAKE
    task = intake_task(user_prompt, memory)

    # STAGES 2-4: ANALYZE -> HARNESS -> REVIEW (loop until approved)
    approved = False
    for attempt in range(1, MAX_PLAN_ATTEMPTS + 1):
        plan = analyze_task(task, memory)
        harnesses = generate_harnesses(task, plan, memory)
        harness_view = {sid: asdict(h) for sid, h in harnesses.items()}
        approved = review_plan(task, plan, harness_view, memory)
        if approved:
            break
        logger.warning("Plan rejected (attempt %d/%d)", attempt, MAX_PLAN_ATTEMPTS)
    if not approved:
        return {"status": "needs_human_review", "session_id": task.session_id}

    # STAGES 5-6: EXECUTE + VERIFY, one step at a time
    for substep in plan["substeps"]:
        step_id = substep["step_id"]
        for retry in range(MAX_STEP_RETRIES + 1):
            execution = execute_step(step_id, task, memory)
            if execution["status"] == "blocked_low_confidence":
                return {"status": "needs_human_review", "session_id": task.session_id,
                        "reason": execution["reason"]}

            verification = verify_step(step_id, task, memory)
            action = verification["next_action"]
            if action == "proceed":
                break
            if action == "retry" and retry < MAX_STEP_RETRIES:
                continue
            rollback_step(step_id, task, memory)   # run the harness rollback_plan
            return {"status": "needs_human_review" if action == "escalate" else "rolled_back",
                    "failed_step": step_id, "session_id": task.session_id}

    # STAGES 7-8: PERSIST + EMBED (only after everything verified)
    if db:
        db.persist_session(task.session_id, memory)
    if indexer:
        indexer.embed_task(task.session_id, asdict(task), memory)
    return {"status": "completed", "session_id": task.session_id}
```

`analyze_task` returns the plan dict directly (`plan["substeps"]`); the versioned wrapper lives in session memory under `session:{id}:plan`. `rollback_step` is not sketched here: it executes the harness's `rollback_plan.actions` through the same guarded tools.

---

## Deployment Checklist

- [ ] Redis/in-memory session store with TTL
- [ ] PostgreSQL temporal DB with JSON event storage
- [ ] Vector DB (Pinecone/Qdrant) set up and indexed
- [ ] Anthropic API keys configured
- [ ] Logging configured (file + stdout)
- [ ] Error handling + rollback logic tested
- [ ] Model routing verified (analysis/review/verify on the large model, execution on the small model)
- [ ] Guardrail enforcement tested (file access, tool limits)
- [ ] Verification gate tested (can reject bad plans)
- [ ] Persistence tested (events written to DB)

---

## Monitoring & Debugging

### Key Metrics to Track

```python
metrics = {
    "task_completion_rate": num_completed / num_total,
    "avg_execution_time": sum(durations) / len(durations),
    "avg_confidence_score": sum(scores) / len(scores),
    "retry_rate": num_retries / num_total,
    "rollback_rate": num_rollbacks / num_total,
    "avg_harness_revisions": sum(revisions) / len(revisions),
}
```

### Debugging a Failed Task

```python
def debug_session(session_id: str, session_memory, db):
    """Retrieve full session for debugging."""
    
    # Get all session data
    task = session_memory.get(f"session:{session_id}:task")
    plan = session_memory.get(f"session:{session_id}:plan")
    harnesses = session_memory.get(f"session:{session_id}:harness")
    execution_logs = session_memory.get(f"session:{session_id}:execution_log")
    
    # Get temporal events
    cursor = db.conn.cursor()
    cursor.execute("SELECT event_type, event_data FROM agent_events WHERE session_id=%s ORDER BY timestamp", (session_id,))
    events = cursor.fetchall()
    
    print(f"Task: {task['user_prompt']}")
    print(f"Plan: {len(plan['substeps'])} steps")
    print(f"Execution logs: {len(execution_logs or {})} steps completed")
    print(f"Events: {len(events)}")
    
    # Show execution summary
    for step_id, log in (execution_logs or {}).items():
        print(f"  Step {step_id}: {log['status']}")
```

---

This implementation guide provides the patterns you need to build each stage. Start with Stage 1 (Task Intake) and incrementally add stages, testing each one before moving to the next.
