"""The apprentice's structured knowledge: one entry per competence-grid slot.

data/knowledge/competence.json maps "task.slot" to
  {rule, conditions, reason, confidence ("once" | "confirmed"), confirmations,
   evidence {metric: value}, learned_in {session, t}, answers [{q, a, t, session, phase}],
   teachback {session, confirmed}}.
answers keep the expert's own words with the moment they refer to (phase: live question, note,
debrief or teach-back), so the Work Map and the tutor can quote them and replay that moment.
A filled slot with evidence is checked against later episodes of the same task: the same
behaviour confirms it, a different one becomes a deviation the observer asks about.
The filled slots are also written into knowledge.md so the tutor and the debrief use them.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from backend.core.config import KNOWLEDGE_DIR
from backend.storage.knowledge_store import get_knowledge, save_knowledge

GRID_PATH = Path(__file__).resolve().parents[1] / "llm" / "prompts" / "competence_grid.json"
STORE_PATH = KNOWLEDGE_DIR / "competence.json"
KB_SECTION = "## 5. Expert Competence Grid"
MAX_ANSWERS_KEPT = 3

GRID: dict[str, Any] = json.loads(GRID_PATH.read_text(encoding="utf-8"))
TASKS: dict[str, dict[str, Any]] = {t["id"]: t for t in GRID["tasks"]}
SLOTS: dict[str, dict[str, Any]] = {
    f"{t['id']}.{s['id']}": {**s, "key": f"{t['id']}.{s['id']}", "task": t["id"]}
    for t in GRID["tasks"]
    for s in t["slots"]
}

_lock = threading.RLock()
_coverage: dict[str, int] | None = None


def load() -> dict[str, dict[str, Any]]:
    with _lock:
        if not STORE_PATH.exists():
            return {}
        data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if k in SLOTS}


def _save(entries: dict[str, dict[str, Any]]) -> None:
    global _coverage
    with _lock:
        STORE_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        _coverage = None
        _write_markdown(entries)


def reset() -> None:
    _save({})


def forget(key: str) -> bool:
    """Take one rule off the record (the expert asked to forget it). True if it existed."""
    with _lock:
        entries = load()
        if key not in entries:
            return False
        del entries[key]
        _save(entries)
        return True


def mark_teachback(keys: list[str], session: str | None) -> None:
    """The expert confirmed the apprentice's explanation of these rules."""
    with _lock:
        entries = load()
        for key in keys:
            if key in entries:
                entries[key]["teachback"] = {"session": session, "confirmed": True}
        _save(entries)


def quote(entry: dict[str, Any]) -> dict[str, Any] | None:
    """The expert's own words behind a rule: the latest real answer, with its moment."""
    for said in reversed(entry.get("answers") or []):
        if str(said.get("a", "")).strip() and not str(said.get("a")).startswith("("):
            learned = entry.get("learned_in") or {}
            return {
                "text": said["a"],
                "question": said.get("q", ""),
                "t": said.get("t", learned.get("t")),
                "session": said.get("session", learned.get("session")),
                "phase": said.get("phase", "live"),
            }
    return None


def coverage() -> dict[str, int]:
    """Gauge numbers, cached: the state broadcast reads them 30 times a second."""
    global _coverage
    with _lock:
        if _coverage is None:
            entries = load()
            _coverage = {
                "filled": len(entries),
                "confirmed": sum(1 for e in entries.values() if e.get("confidence") == "confirmed"),
                "total": len(SLOTS),
            }
        return dict(_coverage)


def slot_name(key: str) -> str:
    slot = SLOTS[key]
    return f"{TASKS[slot['task']]['short']} → {slot['label']}"


def open_slots(task: str, entries: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    """Empty slots of a task, with an open example question (the full description lists answers)."""
    return [
        {"slot": s["key"], "type": s["type"], "example_question": s["ask_example"]}
        for s in SLOTS.values()
        if s["task"] == task and s["key"] not in entries
    ]


def describe(key: str, entry: dict[str, Any]) -> str:
    """One line for prompts: rule, conditions, confidence."""
    text = entry["rule"]
    if entry.get("conditions"):
        text += " Conditions: " + "; ".join(entry["conditions"])
    if entry.get("confidence") == "confirmed":
        text += f" (seen again in flight x{entry.get('confirmations', 0)})"
    return text


def fill(
    key: str,
    rule: str,
    conditions: list[str],
    reason: str,
    evidence: dict[str, float],
    question: str,
    answer: str,
    session: str | None,
    t: float,
    about_deviation: bool,
    observed_times: int = 0,
    phase: str = "live",
) -> dict[str, Any]:
    """Store the merged rule of a slot. observed_times > 1: the pilot confirmed a habit the apprentice
    had already seen that many times (a hypothesis), so the rule starts out confirmed."""
    with _lock:
        entries = load()
        entry = entries.get(key) or {"confidence": "once", "confirmations": 0, "evidence": {}, "answers": []}
        entry.update({"rule": rule, "conditions": conditions, "reason": reason})
        # A deviation answer explains an exception (a condition): the base rule's evidence stays.
        if evidence and not (about_deviation and entry["evidence"]):
            entry["evidence"] = evidence
            entry["learned_in"] = {"session": session, "t": round(t, 1)}
        entry.setdefault("learned_in", {"session": session, "t": round(t, 1)})  # the moment it was first taught
        if observed_times > 1:
            entry["confirmations"] = max(entry.get("confirmations", 0), observed_times - 1)
            entry["confidence"] = "confirmed"
            entry["induced"] = True
        said = {"q": question, "a": answer, "t": round(t, 1), "session": session, "phase": phase}
        entry["answers"] = (entry.get("answers", []) + [said])[-MAX_ANSWERS_KEPT:]
        entry.pop("teachback", None)  # changed since the last teach-back: to be explained back again
        entries[key] = entry
        _save(entries)
        return entry


# ---- closing the loop: later episodes against learned rules ------------------------

def _tolerance(metric: str, expected: float) -> float:
    if metric == "duration_s":
        return max(3.0, abs(expected))  # within a factor of 2
    if metric.startswith("speed"):
        return max(0.8, 0.35 * abs(expected))
    if metric.startswith("altitude"):
        return max(3.0, 0.25 * abs(expected))
    return max(1.5, 0.3 * abs(expected))  # distances and height relative to the cable


def review_episode(task: str, measured: dict[str, float], start: float, session: str | None) -> list[dict[str, Any]]:
    """Compare a finished episode with the learned rules of its task.

    Returns one result per slot with evidence: {"slot", "status": "confirmed" | "deviation",
    "expected", "now"}. Confirmations are saved; deviations are for the observer to ask about.
    """
    results: list[dict[str, Any]] = []
    with _lock:
        entries = load()
        changed = False
        for key, entry in entries.items():
            evidence = entry.get("evidence") or {}
            if SLOTS[key]["task"] != task or not evidence:
                continue
            learned = entry.get("learned_in") or {}
            if learned.get("session") == session and start <= learned.get("t", 0.0):
                continue  # the episode the rule was learned from
            metrics = [m for m in evidence if m in measured]
            if not metrics:
                continue
            expected = {m: evidence[m] for m in metrics}
            now = {m: measured[m] for m in metrics}
            if all(abs(now[m] - expected[m]) <= _tolerance(m, expected[m]) for m in metrics):
                entry["confirmations"] = entry.get("confirmations", 0) + 1
                entry["confidence"] = "confirmed"
                changed = True
                results.append({"slot": key, "status": "confirmed", "expected": expected, "now": now})
            else:
                results.append({"slot": key, "status": "deviation", "expected": expected, "now": now,
                                "rule": describe(key, entry)})
        if changed:
            _save(entries)
    return results


# ---- rule induction: a habit seen several times becomes a hypothesis to confirm ---------

_HABIT_WORDS = {
    "insulator_dist_median": lambda v: f"held about {v:.0f} metres from the insulator",
    "cable_dz_median": lambda v: f"stayed about {abs(v):.0f} metres {'above' if v >= 0 else 'below'} the cable" if abs(v) >= 0.5 else "stayed level with the cable",
    "cable_dist_median": lambda v: f"kept about {v:.0f} metres from the cable",
    "duration_s": lambda v: f"held about {v:.0f} seconds",
    "speed_median": lambda v: f"flew at about {v:.0f} metres per second",
    "pylon_dist_min": lambda v: f"came no closer than about {v:.0f} metres to the tower",
    "tree_dist_min": lambda v: f"kept at least {v:.0f} metres from the tree",
    "altitude_median": lambda v: f"flew at about {v:.0f} metres",
    "altitude_max": lambda v: f"climbed to about {v:.0f} metres",
}
_TASK_PLURAL = {
    "preflight_takeoff": "take-offs", "corridor_transit": "transits along the line",
    "structure_approach": "tower approaches", "insulator_inspection": "insulator inspections",
    "conductor_inspection": "cable inspections", "road_crossing": "road crossings",
    "vegetation": "passes near trees", "emergency": "close calls",
}
MIN_HABIT_EPISODES = 2


def hypothesis(key: str, episodes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """If the expert did the slot's measurable part the same way in every episode, the habit as a
    rule to confirm: {"slot", "text", "evidence", "n"}. None if too few episodes or not consistent."""
    metrics = SLOTS[key].get("evidence") or []
    sigs = [e["signature"] for e in episodes if all(m in e["signature"] for m in metrics)]
    if not metrics or len(sigs) < MIN_HABIT_EPISODES:
        return None
    evidence: dict[str, float] = {}
    for m in metrics:
        values = sorted(float(sig[m]) for sig in sigs)
        median = values[len(values) // 2]
        if any(abs(v - median) > _tolerance(m, median) for v in values):
            return None  # not a habit (yet): the open question is better
        evidence[m] = round(median, 1)
    habit = " and ".join(_HABIT_WORDS[m](v) for m, v in evidence.items() if m in _HABIT_WORDS)
    plural = _TASK_PLURAL.get(SLOTS[key]["task"], "times")
    return {
        "slot": key,
        "text": f"In {len(sigs)} {plural} the pilot {habit}.",
        "example_question": f"I noticed that in your last {len(sigs)} {plural} you {habit}: is that your rule, and what decides it?",
        "evidence": evidence,
        "n": len(sigs),
    }


# ---- views ---------------------------------------------------------------------------

def grid_view() -> dict[str, Any]:
    """Every task with its slots and what was learned, for the frontend."""
    entries = load()
    return {
        "coverage": coverage(),
        "tasks": [
            {
                "id": t["id"],
                "name": t["short"],
                "slots": [
                    {"key": f"{t['id']}.{s['id']}", "label": s["label"], "type": s["type"], "learn": s["learn"],
                     "entry": entries.get(f"{t['id']}.{s['id']}")}
                    for s in t["slots"]
                ],
            }
            for t in GRID["tasks"]
        ],
    }


_METRIC_WORDS = {
    "duration_s": ("held", "s"), "speed_median": ("speed", "m/s"), "altitude_median": ("height", "m"),
    "altitude_max": ("max height", "m"), "cable_dist_median": ("cable distance", "m"),
    "cable_dz_median": ("height vs cable", "m"), "insulator_dist_median": ("insulator distance", "m"),
    "pylon_dist_min": ("closest to tower", "m"), "tree_dist_min": ("closest to tree", "m"),
}


def _evidence_text(evidence: dict[str, float]) -> str:
    parts = []
    for m, v in evidence.items():
        word, unit = _METRIC_WORDS.get(m, (m, ""))
        parts.append(f"{word} {v:+.1f} {unit}" if m == "cable_dz_median" else f"{word} {v:.1f} {unit}")
    return ", ".join(parts)


def _write_markdown(entries: dict[str, dict[str, Any]]) -> None:
    cov = {"filled": len(entries), "total": len(SLOTS),
           "confirmed": sum(1 for e in entries.values() if e.get("confidence") == "confirmed")}
    lines = [
        KB_SECTION,
        f"Knowledge: {cov['filled']}/{cov['total']} slots filled, {cov['confirmed']} confirmed in later flights.",
    ]
    for t in GRID["tasks"]:
        filled = [(s, entries[f"{t['id']}.{s['id']}"]) for s in t["slots"] if f"{t['id']}.{s['id']}" in entries]
        if not filled:
            continue
        lines.append(f"\n### {t['name']}")
        for s, e in filled:
            status = f"confirmed x{e.get('confirmations', 0)}" if e.get("confidence") == "confirmed" else "heard once"
            lines.append(f"- **{s['label']}** ({status}): {e['rule']}")
            for c in e.get("conditions") or []:
                lines.append(f"  - Condition: {c}")
            if e.get("evidence"):
                lines.append(f"  - Measured: {_evidence_text(e['evidence'])}")

    # replace only this section, keep anything written after it
    current = get_knowledge()
    start = current.find(KB_SECTION)
    if start < 0:
        head, tail = current.rstrip(), ""
    else:
        end = current.find("\n## ", start + len(KB_SECTION))
        head, tail = current[:start].rstrip(), (current[end:] if end >= 0 else "")
    save_knowledge(f"{head}\n\n" + "\n".join(lines) + "\n" + tail)
