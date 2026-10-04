"""Module 3, the tutor that teaches the expert's way: the expert's moment, predictions, mastery.

- expert_moment(): the rule the expert taught for a situation, in their own words, with the moment
  of their flight it was taught at, so the tutor can replay that frame when it steps in.
- Prediction questions: before a decision point (road, insulator, tower, tree, defect) the tutor
  asks the novice what the expert would do; after a caught mistake it asks why the expert would
  not have done it. The answer is judged against the expert's rule and reason (one Haiku call,
  a local fallback without a key).
- mastery(): after a novice flight, every expert rule the novice practised, mastered or not,
  measured against the expert's own numbers. No LLM.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.core.config import ROAD_MIN_CROSSING_ALT_M, TUTOR_MODEL
from backend.llm.client import LLMClient
from backend.storage import competence_store
from backend.storage.competence_store import _HABIT_WORDS, _tolerance
from backend.storage.session_episodes import load_flight
from backend.storage.session_recorder import frame_path

logger = logging.getLogger("robot-apprentice.teach")

# Which expert rules explain a situation, best first
TOPIC_SLOTS: dict[str, list[str]] = {
    "road": ["road_crossing.crossing_rule", "road_crossing.traffic_check"],
    "cable": ["interference_wind.emi_distance", "insulator_inspection.inspection_standoff",
              "conductor_inspection.scan_technique", "corridor_transit.transit_position", "emergency.escape_manoeuvre"],
    "insulator": ["insulator_inspection.inspection_standoff", "insulator_inspection.viewpoints", "insulator_inspection.dwell"],
    "tower": ["structure_approach.structure_clearance", "structure_approach.approach_path"],
    "tree": ["vegetation.tree_clearance_flight"],
    "speed": ["corridor_transit.transit_speed", "structure_approach.approach_path"],
    "ground": ["emergency.escape_manoeuvre"],
    "defect": ["defect_assessment.severity", "defect_assessment.confirmation", "defect_assessment.report_content"],
}
HAZARD_TOPIC = {"cable": "cable", "tower": "tower", "tree": "tree", "ground": "ground"}

# "What would the expert do here?" before a decision point; "why?" after a caught mistake
PREDICT_QUESTIONS = {
    "road": "Road ahead. Before you cross: what would the expert do here, and why?",
    "insulator": "Insulator {target} coming up. How close would the expert hold, and why?",
    "tower": "Tower ahead. How would the expert approach it, and how close would they go?",
    "tree": "Tree ahead. How much room would the expert give it, and why?",
    "defect": "You found {target}. What would the expert do with it now?",
}
WHY_QUESTIONS = {
    "road": "The expert would have climbed before that road. Why do you think?",
    "cable": "The expert would never get that close to the cable. Why do you think?",
    "tower": "The expert keeps more room from the tower. Why do you think?",
    "tree": "The expert gives trees more room than that. Why do you think?",
    "speed": "The expert would have slowed down much earlier there. Why do you think?",
}


def _short(text: str, words: int = 22) -> str:
    w = text.strip().rstrip(".").split()
    return " ".join(w[:words]) + ("..." if len(w) > words else "")


def expert_rule_for(topic: str) -> tuple[str, dict[str, Any]] | None:
    """The first rule the expert taught for a topic: (slot, entry)."""
    entries = competence_store.load()
    for slot in TOPIC_SLOTS.get(topic, []):
        if slot in entries:
            return slot, entries[slot]
    return None


def expert_moment(topic: str) -> dict[str, Any] | None:
    """What the tutor shows when it steps in: the expert's rule, their words and the screen moment."""
    found = expert_rule_for(topic)
    if not found:
        return None
    slot, entry = found
    said = competence_store.quote(entry) or {}
    session, t = said.get("session"), said.get("t")
    if session is None:
        learned = entry.get("learned_in") or {}
        session, t = learned.get("session"), learned.get("t")
    has_frame = bool(session and t is not None and frame_path(session, float(t)))
    return {
        "slot": slot,
        "slot_name": competence_store.slot_name(slot),
        "rule": entry["rule"],
        "reason": entry.get("reason", ""),
        "quote": said.get("text"),
        "question": said.get("question"),
        "phase": said.get("phase"),
        "session": session,
        "t": t,
        "frame": has_frame,
    }


def reason_words(topic: str) -> str | None:
    """The expert's reason, short enough for a spoken alert ("because the compass goes wild")."""
    found = expert_rule_for(topic)
    if not found:
        return None
    _, entry = found
    reason = (entry.get("reason") or "").strip().rstrip(".")
    if not reason:
        rule = entry["rule"]
        reason = rule.split(" because ", 1)[1] if " because " in rule else ""
    return _short(reason, 14) if reason else None


def caught_explanation(topic: str) -> str:
    """One sentence appended to an alert: what the expert does there and why, in their terms."""
    found = expert_rule_for(topic)
    if not found:
        return ""
    slot, entry = found
    why = reason_words(topic)
    if topic == "road" and "altitude_median" in (entry.get("evidence") or {}):
        alt = entry["evidence"]["altitude_median"]
        return f" The expert crosses at about {alt:.0f} metres" + (f", because {why}." if why else ".")
    rule = _short(entry["rule"].split(" because ")[0], 16)
    return f" The expert's rule: {rule}" + (f", because {why}." if why else ".")


# ---- prediction questions ------------------------------------------------------------

def prediction_question(topic: str, target: str = "", why: bool = False) -> dict[str, Any] | None:
    """The question to ask, if the expert taught a rule the answer can be judged against."""
    found = expert_rule_for(topic)
    if not found:
        return None
    text = (WHY_QUESTIONS if why else PREDICT_QUESTIONS).get(topic)
    if not text:
        return None
    slot, _ = found
    return {"topic": topic, "slot": slot, "slot_name": competence_store.slot_name(slot),
            "kind": "why" if why else "predict", "question": text.format(target=target or "it")}


class PredictionVerdict(BaseModel):
    verdict: Literal["right", "partly", "wrong"] = Field(description="How well the novice's answer matches the expert's rule and reason.")
    speech: str = Field(description="At most 2 short sentences, 35 words, spoken to the novice: confirm or correct, "
                                    "and give the expert's reason in the expert's own words.")


PREDICT_SYSTEM = """You are a flight instructor teaching a novice drone pilot to inspect a high-voltage line the way one expert pilot does. You asked the novice what the expert would do (or why the expert would not have done what the novice just did). Judge the novice's answer against the expert's rule, conditions and reason:
- right: the same decision and a sensible reason; partly: the right idea but a wrong number, or no reason; wrong: a different decision.
- speech: spoken aloud to a pilot who is flying. Start with "Right", "Almost" or "Not quite". Then give the expert's decision and reason, quoting the expert's own words when they fit ("The expert says: ..."). Never invent a number the expert did not give. Plain spoken English, no markdown."""


def _numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]


def _fallback_verdict(answer: str, entry: dict[str, Any], quote: str | None) -> PredictionVerdict:
    """Without the model: the same numbers as the expert's rule count as right."""
    expert_nums = _numbers(entry["rule"]) + [float(v) for v in (entry.get("evidence") or {}).values()]
    said = _numbers(answer)
    close = any(abs(a - b) <= max(1.5, 0.25 * abs(b)) for a in said for b in expert_nums)
    verdict = "right" if close else ("partly" if said or len(answer.split()) > 6 else "wrong")
    lead = {"right": "Right.", "partly": "Almost.", "wrong": "Not quite."}[verdict]
    words = f" The expert says: \"{_short(quote, 20)}\"" if quote else f" The expert's rule: {_short(entry['rule'], 24)}."
    return PredictionVerdict(verdict=verdict, speech=lead + words)


class PredictionCoach:
    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient(model=TUTOR_MODEL, timeout=15.0, purpose="tutor")

    def judge(self, question: dict[str, Any], answer: str) -> dict[str, Any]:
        entry = competence_store.load().get(question["slot"])
        if not entry:
            return {"verdict": "partly", "speech": "I have no rule from the expert on this yet."}
        said = competence_store.quote(entry)
        verdict: PredictionVerdict | None = None
        if self.client.is_available:
            prompt = json.dumps({
                "question": question["question"],
                "novice_answer": answer,
                "expert_rule": entry["rule"],
                "expert_conditions": entry.get("conditions") or [],
                "expert_reason": entry.get("reason", ""),
                "expert_own_words": said and said["text"],
                "measured_on_the_expert": entry.get("evidence") or {},
            }, ensure_ascii=False)
            try:
                verdict = self.client.parse(prompt=prompt, system=PREDICT_SYSTEM, output_format=PredictionVerdict, max_tokens=250)
            except Exception:
                logger.exception("Prediction judging failed, using the local check")
        if verdict is None or not verdict.speech.strip():
            verdict = _fallback_verdict(answer, entry, said and said["text"])
        return verdict.model_dump()


# ---- mastery after a novice flight ---------------------------------------------------

def _measure_words(metric: str, value: float) -> str:
    fn = _HABIT_WORDS.get(metric)
    return fn(value) if fn else f"{metric} {value}"


# Safety events that show a rule was not applied, and the topic they belong to
BREACH_TOPICS = {"very_close_cable": "cable", "cannot_stop": "speed"}


def mastery(session_id: str) -> dict[str, Any]:
    """Which expert rules the novice applied this flight, which to practise next, and why."""
    flight = load_flight(session_id)
    entries = competence_store.load()
    if not flight:
        return {"mastered": [], "practice": [], "not_practised": [], "quiz": {}}

    results: dict[str, dict[str, Any]] = {}

    def practice(slot: str, why: str) -> None:
        if slot not in entries:
            return
        cur = results.get(slot)
        if cur and cur["status"] == "practice":
            cur["why"].append(why)
            return
        results[slot] = {"status": "practice", "why": [why]}

    # 1. the expert's measured rules against the novice's episodes of the same task
    for slot, entry in entries.items():
        evidence = entry.get("evidence") or {}
        task = competence_store.SLOTS[slot]["task"]
        eps = [ep for ep in flight.episodes_of(task) if not flight.is_off_record(ep.start, ep.end)]
        metrics = [m for m in evidence if eps and m in eps[0].signature]
        if not metrics:
            continue
        misses = []
        for ep in eps:
            for m in metrics:
                if abs(ep.signature[m] - evidence[m]) > _tolerance(m, evidence[m]):
                    misses.append(f"you {_measure_words(m, ep.signature[m])}; the expert {_measure_words(m, evidence[m])}")
        if misses:
            practice(slot, misses[0])
        else:
            results[slot] = {"status": "mastered",
                             "why": [f"{len(eps)} time{'s' if len(eps) > 1 else ''} like the expert"]}

    # 2. safety breaches and the Guardian taking over
    for e in flight.events:
        kind = e.get("type")
        if kind == "road_crossed" and e.get("low"):
            practice("road_crossing.crossing_rule", f"you crossed the road at {e.get('min_altitude', 0):.0f} m "
                                                    f"(minimum {ROAD_MIN_CROSSING_ALT_M:.0f} m)")
        elif kind in BREACH_TOPICS or kind in ("guardian_engaged", "conflict_predicted", "collision"):
            topic = BREACH_TOPICS.get(kind) or HAZARD_TOPIC.get(str(e.get("hazard") or e.get("with") or ""), "")
            found = expert_rule_for(topic) if topic else None
            if found:
                what = {"guardian_engaged": "the Guardian had to take over", "conflict_predicted": "you were on course to hit it",
                        "collision": "you crashed", "very_close_cable": f"you came within {e.get('cable_dist', 2):.1f} m of a cable",
                        "cannot_stop": "you were too fast to stop"}[kind]
                practice(found[0], what)

    # 3. prediction questions: understanding the rule
    quiz = {"asked": 0, "right": 0, "partly": 0, "wrong": 0}
    for r in flight.transcript:
        if r.get("role") != "tutor_quiz" or r.get("slot") not in entries:
            continue
        quiz["asked"] += 1
        v = r.get("verdict", "wrong")
        quiz[v] = quiz.get(v, 0) + 1
        if v == "wrong":
            practice(r["slot"], "you could not say what the expert would do")
        elif r["slot"] not in results:
            results[r["slot"]] = {"status": "mastered", "why": ["you predicted the expert's decision"]}

    def view(slot: str, res: dict[str, Any]) -> dict[str, Any]:
        entry = entries[slot]
        said = competence_store.quote(entry)
        return {"slot": slot, "name": competence_store.slot_name(slot), "why": res["why"][:2], "expert_rule": entry["rule"],
                "expert_words": said and said["text"], "moment": said and {"session": said["session"], "t": said["t"]}}

    flown = {ep.task for ep in flight.episodes}
    return {
        "mastered": [view(k, r) for k, r in results.items() if r["status"] == "mastered"],
        "practice": [view(k, r) for k, r in results.items() if r["status"] == "practice"],
        "not_practised": sorted({competence_store.TASKS[competence_store.SLOTS[k]["task"]]["short"]
                                 for k in entries if competence_store.SLOTS[k]["task"] not in flown
                                 and competence_store.SLOTS[k]["task"] not in ("defect_assessment", "interference_wind")}),
        "quiz": quiz,
    }
