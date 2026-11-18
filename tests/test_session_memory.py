import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.session_memory import (  # noqa: E402
    MAX_PLAN_ATTEMPTS, InvalidTransition, SessionMemory, SessionState as S,
)


def test_happy_path():
    m = SessionMemory()
    sid = m.start_session("fix bug")
    for s in (S.ANALYZED, S.REVIEWED, S.APPROVED, S.EXECUTING, S.VERIFYING, S.FINALIZED):
        m.transition(sid, s)
    assert m.is_terminal(sid)


def test_cannot_execute_before_approval():
    m = SessionMemory()
    sid = m.start_session("x")
    m.transition(sid, S.ANALYZED)
    with pytest.raises(InvalidTransition):
        m.transition(sid, S.EXECUTING)


def test_replan_is_bounded():
    m = SessionMemory()
    sid = m.start_session("x")
    m.transition(sid, S.ANALYZED)
    for _ in range(MAX_PLAN_ATTEMPTS):
        m.transition(sid, S.REVIEWED)
        m.transition(sid, S.REJECTED)
        if _ < MAX_PLAN_ATTEMPTS - 1:
            m.transition(sid, S.ANALYZED)
    with pytest.raises(InvalidTransition):
        m.transition(sid, S.ANALYZED)
    m.transition(sid, S.NEEDS_HUMAN_REVIEW)


def test_ttl_expiry():
    m = SessionMemory(ttl_hours=0)
    m.set("k", 1)
    assert m.get("k") is None
