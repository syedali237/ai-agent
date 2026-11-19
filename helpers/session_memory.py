"""Phase 1: in-memory session store and session state machine.

Key layout follows .claude/skills/dynamic-harness/references/database.md:
    session:{session_id}:task | plan | harness | execution_log | review | state
"""
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional

MAX_PLAN_ATTEMPTS = 3
MAX_STEP_RETRIES = 2


class SessionState(str, Enum):
    RECEIVED = "received"
    ANALYZED = "analyzed"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    FINALIZED = "finalized"
    # terminal branches
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


S = SessionState

# Allowed transitions. REJECTED -> ANALYZED is the bounded re-plan loop;
# VERIFYING -> EXECUTING covers both "next step" and "retry step".
TRANSITIONS: Dict[SessionState, set] = {
    S.RECEIVED: {S.ANALYZED, S.NEEDS_HUMAN_REVIEW},
    S.ANALYZED: {S.REVIEWED, S.NEEDS_HUMAN_REVIEW},
    S.REVIEWED: {S.APPROVED, S.REJECTED},
    S.APPROVED: {S.EXECUTING, S.NEEDS_HUMAN_REVIEW},
    S.EXECUTING: {S.VERIFYING, S.ROLLED_BACK, S.NEEDS_HUMAN_REVIEW},
    S.VERIFYING: {S.EXECUTING, S.FINALIZED, S.ROLLED_BACK, S.NEEDS_HUMAN_REVIEW},
    S.REJECTED: {S.ANALYZED, S.NEEDS_HUMAN_REVIEW},
    S.FINALIZED: set(),
    S.ROLLED_BACK: set(),
    S.NEEDS_HUMAN_REVIEW: set(),
}


class InvalidTransition(Exception):
    pass


def new_session_id() -> str:
    return f"task-{datetime.now().strftime('%Y-%m-%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


class SessionMemory:
    """Single-process key/value store with TTL, plus the session state machine."""

    def __init__(self, ttl_hours: int = 24):
        self.ttl = timedelta(hours=ttl_hours)
        self._store: Dict[str, Any] = {}
        self._created_at: Dict[str, datetime] = {}

    # -- generic KV ---------------------------------------------------------
    def set(self, key: str, value: Any) -> None:
        self._store[key] = value
        self._created_at[key] = datetime.now()

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._store:
            return default
        if datetime.now() - self._created_at[key] >= self.ttl:
            self.delete(key)
            return default
        return self._store[key]

    def delete(self, key: str) -> None:
        self._store.pop(key, None)
        self._created_at.pop(key, None)

    def get_session(self, session_id: str) -> Dict[str, Any]:
        prefix = f"session:{session_id}:"
        return {k: self.get(k) for k in list(self._store) if k.startswith(prefix)}

    # -- state machine ------------------------------------------------------
    def start_session(self, user_prompt: str, session_id: Optional[str] = None) -> str:
        session_id = session_id or new_session_id()
        self.set(f"session:{session_id}:state", {
            "state": S.RECEIVED.value,
            "plan_attempts": 0,
            "history": [(S.RECEIVED.value, datetime.now().isoformat())],
        })
        self.set(f"session:{session_id}:task", {"session_id": session_id, "user_prompt": user_prompt})
        return session_id

    def get_state(self, session_id: str) -> SessionState:
        record = self.get(f"session:{session_id}:state")
        if record is None:
            raise KeyError(f"unknown or expired session: {session_id}")
        return SessionState(record["state"])

    def transition(self, session_id: str, new_state: SessionState) -> SessionState:
        record = self.get(f"session:{session_id}:state")
        if record is None:
            raise KeyError(f"unknown or expired session: {session_id}")
        current = SessionState(record["state"])
        new_state = SessionState(new_state)

        if new_state not in TRANSITIONS[current]:
            raise InvalidTransition(f"{current.value} -> {new_state.value} is not allowed")

        # Re-planning is bounded so the loop always terminates.
        if current is S.REJECTED and new_state is S.ANALYZED and record["plan_attempts"] >= MAX_PLAN_ATTEMPTS:
            raise InvalidTransition(
                f"plan rejected {record['plan_attempts']} times; must go to needs_human_review"
            )
        if new_state is S.REJECTED:
            record["plan_attempts"] += 1

        record["state"] = new_state.value
        record["history"].append((new_state.value, datetime.now().isoformat()))
        self.set(f"session:{session_id}:state", record)
        return new_state

    def is_terminal(self, session_id: str) -> bool:
        return not TRANSITIONS[self.get_state(session_id)]
