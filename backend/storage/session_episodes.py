"""A recorded flight cut into competence-grid task episodes, from its telemetry and events.

The live FlightLog is reset at every Start Flight; the debrief, the Work Map and the novice's
mastery report work on any past session, so they replay the recording through a fresh FlightLog:
the same task detection, the same measured signatures as during the flight.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from backend.core.config import SESSIONS_DIR
from backend.sim.flight_log import FlightLog
from backend.sim.tasks import MIN_EPISODE_S, Episode
from backend.storage.session_recorder import load_jsonl, session_meta, valid_session_id

_lock = threading.Lock()
_cache: dict[str, tuple[float, "SessionFlight"]] = {}


@dataclass
class SessionFlight:
    session_id: str
    mode: str
    episodes: list[Episode] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    duration: float = 0.0
    off_record: list[list[float]] = field(default_factory=list)  # [start, end] sim-time windows

    def is_off_record(self, t0: float, t1: float | None = None) -> bool:
        t1 = t0 if t1 is None else t1
        return any(a <= t1 and t0 <= b for a, b in self.off_record)

    def episodes_of(self, task: str) -> list[Episode]:
        return [ep for ep in self.episodes if ep.task == task]

    def events_of(self, *types: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") in types]


def load_flight(session_id: str) -> SessionFlight | None:
    """The session's episodes (cached until its telemetry file changes)."""
    if not valid_session_id(session_id):
        return None
    root = SESSIONS_DIR / session_id
    tel_path = root / "telemetry.jsonl"
    if not root.is_dir():
        return None
    stamp = max((p.stat().st_mtime for p in (tel_path, root / "events.jsonl", root / "transcript.jsonl", root / "meta.json")
                 if p.exists()), default=0.0)
    with _lock:
        hit = _cache.get(session_id)
        if hit and hit[0] == stamp:
            return hit[1]

    meta = session_meta(session_id)
    telemetry = load_jsonl(tel_path)
    events = [e for e in load_jsonl(root / "events.jsonl") if e.get("type") != "defects_placed"]
    events.sort(key=lambda e: float(e.get("t", 0.0)))
    log = FlightLog(record_hz=10.5)  # recorded at 10 Hz with rounded times: keep every row
    i = 0
    for row in telemetry:
        t = float(row.get("t", 0.0))
        while i < len(events) and float(events[i].get("t", 0.0)) <= t:
            log.record_event(events[i])
            i += 1
        try:
            log.record(row)
        except KeyError:
            continue  # an older recording without some field
    episodes = [ep for ep in log.tasks.finished if ep.duration >= MIN_EPISODE_S]
    cur = log.current_episode()
    if cur and cur.duration >= MIN_EPISODE_S:
        episodes.append(cur)

    off = [list(w) for w in meta.get("off_record", [])]
    flight = SessionFlight(
        session_id=session_id,
        mode=meta.get("mode", "expert"),
        episodes=episodes,
        events=events,
        transcript=load_jsonl(root / "transcript.jsonl"),
        duration=float(telemetry[-1]["t"]) if telemetry else 0.0,
        off_record=off,
    )
    with _lock:
        _cache[session_id] = (stamp, flight)
    return flight
