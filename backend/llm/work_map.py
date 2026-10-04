"""Module 2, the Work Map: an expert flight as a clickable timeline of steps.

Each step is a task episode the expert flew (replayed from the recording, see
storage/session_episodes.py) with:
- the screen moment (sim time of the key event or the middle of the step; the camera frame
  recorded closest to it is served by /session/{id}/frame),
- the decision (what telemetry measured, in words),
- the reason in the expert's words (the answer that taught the rule, with its own moment),
- the guardrails around it (threshold, abort and condition rules of that task, each with the
  expert's words and the moment they were taught).
Built without an LLM: every number comes from the telemetry and every quote from the expert, so
the map can never say something the expert did not.
"""

from __future__ import annotations

from typing import Any

from backend.sim.tasks import TASK_METRICS, Episode
from backend.storage import competence_store, debrief_store
from backend.storage.competence_store import _HABIT_WORDS
from backend.storage.session_episodes import SessionFlight, load_flight
from backend.storage.session_recorder import frame_path

GUARDRAIL_TYPES = {"threshold", "abort", "condition"}
KEY_EVENTS = {"defect_spotted", "road_crossed", "insulator_inspected", "very_close_cable", "collision",
              "conflict_predicted", "compass_interference", "sudden_deceleration", "takeoff", "mission_complete"}
JUDGMENT_EVENTS = {"defect_spotted", "road_crossed", "very_close_cable", "conflict_predicted", "sudden_deceleration"}
MIN_STEP_S = 3.0
MAX_STEPS = 14
EXTRA_WORDS = {"cable_dist_min": lambda v: f"came within {v:.1f} metres of the cable"}


def _merge(flight: SessionFlight) -> list[dict[str, Any]]:
    """Episodes -> steps: short transits without events dropped, repeats of the same task merged."""
    steps: list[dict[str, Any]] = []
    for ep in flight.episodes:
        events = [e for e in flight.events if ep.start - 0.5 <= float(e.get("t", 0)) <= ep.end + 0.5
                  and e.get("type") in KEY_EVENTS]
        if ep.duration < MIN_STEP_S and not events:
            continue
        last = steps[-1] if steps else None
        if last and last["task"] == ep.task and last["target"] == ep.target:
            last["end"] = ep.end
            last["events"] += [e for e in events if e not in last["events"]]
            if ep.duration > last["main"].duration:
                last["main"] = ep
            continue
        steps.append({"task": ep.task, "target": ep.target, "start": ep.start, "end": ep.end, "main": ep, "events": events})
    while len(steps) > MAX_STEPS:  # keep the map readable: drop the shortest uneventful step
        quiet = [s for s in steps if not s["events"]] or steps
        steps.remove(min(quiet, key=lambda s: s["end"] - s["start"]))
    return steps


def _decision(task: str, ep: Episode, events: list[dict[str, Any]], who: str = "The expert") -> str:
    parts = []
    for m in TASK_METRICS.get(task, []):
        if m in ep.signature:
            words = (_HABIT_WORDS.get(m) or EXTRA_WORDS.get(m))
            if words:
                parts.append(words(ep.signature[m]))
    text = (f"{who} " + ", ".join(parts) + ".") if parts else f"{who} flew this part."
    for e in events:
        kind = e.get("type")
        if kind == "road_crossed":
            checked = f" after checking the traffic for {e['checked_before_s']:.0f} s" if e.get("checked_before_s", 0) >= 1.5 else ""
            text += f" Crossed the road at {e.get('min_altitude', 0):.0f} m{checked}."
        elif kind == "defect_spotted":
            text += f" Spotted: {e.get('label')} ({e.get('target')})."
        elif kind == "insulator_inspected":
            text += f" Insulator {e.get('insulator_id')} inspected."
        elif kind == "very_close_cable":
            text += f" Close call: {e.get('cable_dist', 0):.1f} m from a cable."
    return text


def _moment(session: str | None, t: float | None) -> dict[str, Any] | None:
    if not session or t is None:
        return None
    return {"session": session, "t": round(float(t), 1), "frame": frame_path(session, float(t)) is not None}


def _rule_view(slot: str, entry: dict[str, Any]) -> dict[str, Any]:
    spec = competence_store.SLOTS[slot]
    said = competence_store.quote(entry)
    learned = entry.get("learned_in") or {}
    return {
        "slot": slot,
        "label": spec["label"],
        "type": spec["type"],
        "guardrail": spec["type"] in GUARDRAIL_TYPES,
        "rule": entry["rule"],
        "conditions": entry.get("conditions") or [],
        "reason": entry.get("reason", ""),
        "confidence": entry.get("confidence", "once"),
        "confirmations": entry.get("confirmations", 0),
        "teachback": bool((entry.get("teachback") or {}).get("confirmed")),
        "words": said and {"text": said["text"], "question": said["question"], "phase": said["phase"]},
        "moment": _moment(said["session"] if said else learned.get("session"), said["t"] if said else learned.get("t")),
    }


def _step_slots(task: str, events: list[dict[str, Any]]) -> list[str]:
    tasks = [task]
    kinds = {e.get("type") for e in events}
    if "defect_spotted" in kinds:
        tasks.append("defect_assessment")
    if "compass_interference" in kinds:
        tasks.append("interference_wind")
    if kinds & {"very_close_cable", "collision"} and task != "emergency":
        tasks.append("emergency")
    return [s for s in competence_store.SLOTS if competence_store.SLOTS[s]["task"] in tasks]


def build(session_id: str) -> dict[str, Any] | None:
    flight = load_flight(session_id)
    if flight is None:
        return None
    entries = competence_store.load()
    steps_out: list[dict[str, Any]] = []
    guardrails: dict[str, dict[str, Any]] = {}
    for i, s in enumerate(_merge(flight), start=1):
        task = s["task"]
        title = competence_store.TASKS[task]["short"] + (f" · {s['target']}" if s["target"] else "")
        if flight.is_off_record(s["start"], s["end"]):
            steps_out.append({"id": i, "task": task, "title": "Off the record", "off_record": True,
                              "t_start": round(s["start"], 1), "t_end": round(s["end"], 1)})
            continue
        key = next((e for e in s["events"] if e.get("type") in JUDGMENT_EVENTS), None)
        t_moment = float(key["t"]) if key else (s["start"] + s["end"]) / 2
        slots = _step_slots(task, s["events"])
        rules = [_rule_view(k, entries[k]) for k in slots if k in entries]
        for r in rules:
            if r["guardrail"]:
                guardrails[r["slot"]] = r
        # the reason: the expert's words behind the step's main rule (this flight's answers first)
        spoken = [r for r in rules if r["words"]]
        spoken.sort(key=lambda r: (r["moment"] or {}).get("session") != session_id)
        steps_out.append({
            "id": i,
            "task": task,
            "title": title,
            "t_start": round(s["start"], 1),
            "t_end": round(s["end"], 1),
            "moment": _moment(session_id, t_moment),
            "decision": _decision(task, s["main"], s["events"], "The expert" if flight.mode == "expert" else "The pilot"),
            "events": [{"type": e["type"], "t": round(float(e.get("t", 0)), 1),
                        **{k: e[k] for k in ("label", "target", "insulator_id", "min_altitude") if k in e}} for e in s["events"]],
            "reason": spoken[0] if spoken else None,
            "rules": rules,
            "guardrails": [r for r in rules if r["guardrail"]],
            "judgment_call": bool(key) or any(r["conditions"] for r in rules),
            "open": [competence_store.SLOTS[k]["label"] for k in slots if k not in entries],
        })

    debrief = debrief_store.load(session_id) or {}
    flown = {s["task"] for s in steps_out}
    return {
        "session_id": session_id,
        "mode": flight.mode,
        "title": "Power line inspection: three towers, six insulators, one road",
        "duration_s": round(flight.duration, 1),
        "steps": steps_out,
        "guardrails": list(guardrails.values()),
        "counts": {
            "steps": len(steps_out),
            "judgment_calls": sum(1 for s in steps_out if s.get("judgment_call")),
            "guardrails": len(guardrails),
            "rules_in_expert_words": len({r["slot"] for s in steps_out for r in s.get("rules", []) if r["words"]}),
            "open_gaps": sum(1 for k in competence_store.SLOTS
                             if competence_store.SLOTS[k]["task"] in flown and k not in entries),
        },
        "debrief": {
            "phase": debrief.get("phase"),
            "questions": len(debrief.get("items", [])),
            "answered": sum(1 for it in debrief.get("items", []) if it.get("status") == "answered"),
            "teach_back": debrief.get("teach_back"),
        },
    }
