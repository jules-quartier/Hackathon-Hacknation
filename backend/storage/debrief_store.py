"""The spoken debrief of an expert flight: data/sessions/<id>/debrief.json.

{session_id, phase ("questions" | "teachback" | "done"), intro, items [gap questions with their
status, answer and what was learned], teach_back {speech, slots, round, confirmed, history}}.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from backend.core.config import SESSIONS_DIR
from backend.storage.session_recorder import valid_session_id

_lock = threading.RLock()


def _path(session_id: str):
    return SESSIONS_DIR / session_id / "debrief.json"


def load(session_id: str) -> dict[str, Any] | None:
    if not valid_session_id(session_id):
        return None
    with _lock:
        p = _path(session_id)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None


def save(state: dict[str, Any]) -> None:
    with _lock:
        p = _path(state["session_id"])
        if p.parent.is_dir():
            p.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
