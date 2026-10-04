"""Module 2, the spoken debrief after an expert flight: close the gaps, then explain it back.

1. Gaps (local, from the recording and the competence grid): live questions left unanswered,
   behaviour that contradicted a taught rule and was never explained, habits seen several times,
   empty slots of the tasks just flown (limits, abort criteria and exceptions first), rules heard
   only once without their exceptions, and guardrails for situations that never happened.
2. One Haiku call picks DEBRIEF_MIN..MAX_QUESTIONS of them (at least one guardrail) and words them
   for the voice, tied to the moment of the flight they are about.
3. Each answer goes through the knowledge manager into its slot, like a live answer (phase
   "debrief"); a vague answer gets one follow-up.
4. Teach-back: Claude explains the whole process back in under a minute from the Work Map; the
   expert confirms or corrects. Corrections go into their slots and only the corrected points are
   explained again. Done = every gap answered or skipped AND the expert confirmed the teach-back.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from pydantic import BaseModel, Field

from backend.core.config import (
    DEBRIEF_MAX_QUESTIONS,
    DEBRIEF_MIN_QUESTIONS,
    DEBRIEF_MODEL,
    SESSIONS_DIR,
    TEACHBACK_MAX_ROUNDS,
)
from backend.llm import work_map
from backend.llm.attention import TYPE_PRIORITY
from backend.llm.client import LLMClient
from backend.llm.knowledge_manager import KnowledgeManager
from backend.storage import competence_store, debrief_store, episode_store
from backend.storage.session_episodes import SessionFlight, load_flight
from backend.storage.session_recorder import append_transcript, load_jsonl

logger = logging.getLogger("robot-apprentice.debrief")

GUARDRAIL_TYPES = work_map.GUARDRAIL_TYPES
MAX_CANDIDATES = 9
# judgment calls first: a novice gets these wrong; the take-off routine can wait
TASK_BOOST = {"road_crossing": 1.5, "defect_assessment": 1.5, "emergency": 1.5, "insulator_inspection": 0.5,
              "preflight_takeoff": -1.0}
DONE_WHEN = "every gap question answered or skipped, and the expert confirms the teach-back"


# ---- 1. gaps --------------------------------------------------------------------------

def _tasks_flown(flight: SessionFlight) -> list[str]:
    tasks = list(dict.fromkeys(ep.task for ep in flight.episodes if not flight.is_off_record(ep.start, ep.end)))
    if flight.events_of("defect_spotted"):
        tasks.append("defect_assessment")
    if flight.events_of("compass_interference"):
        tasks.append("interference_wind")
    return tasks


def _moment_of(flight: SessionFlight, task: str) -> float | None:
    """The screen moment a question about a task points at: middle of its longest episode."""
    if task == "defect_assessment":
        ev = flight.events_of("defect_spotted")
        return float(ev[0]["t"]) if ev else None
    if task == "interference_wind":
        ev = flight.events_of("compass_interference")
        return float(ev[0]["t"]) if ev else None
    eps = [ep for ep in flight.episodes_of(task) if not flight.is_off_record(ep.start, ep.end)]
    if not eps:
        return None
    ep = max(eps, key=lambda e: e.duration)
    return round(ep.start + ep.duration / 2, 1)


def _live_questions(flight: SessionFlight) -> list[dict[str, Any]]:
    """The apprentice's questions during the flight, each with whether it got an answer."""
    out: list[dict[str, Any]] = []
    for r in flight.transcript:
        if r.get("role") == "apprentice_model":
            out.append({"t": float(r.get("t", 0)), "question": r.get("text", ""), "slot": r.get("slot"),
                        "kind": r.get("kind"), "answered": False})
        elif r.get("role") == "expert_operator" and r.get("kind") == "answer" and out:
            out[-1]["answered"] = True
    return out


def _deviations(session_id: str) -> list[dict[str, Any]]:
    rows = load_jsonl(SESSIONS_DIR / session_id / "observer.jsonl")
    return [r for r in rows if (r.get("rule_check") or {}).get("status") == "deviation"]


def gap_candidates(session_id: str) -> list[dict[str, Any]]:
    """Everything the apprentice is still unsure about after this flight, most valuable first."""
    flight = load_flight(session_id)
    if flight is None:
        return []
    entries = competence_store.load()
    flown = _tasks_flown(flight)
    live = _live_questions(flight)
    asked_slots = {q["slot"] for q in live if q["slot"]}
    cands: list[dict[str, Any]] = []

    def add(priority: float, kind: str, slot: str, why: str, question: str, t: float | None, **extra: Any) -> None:
        if any(c["slot"] == slot for c in cands):
            return
        spec = competence_store.SLOTS[slot]
        priority += TASK_BOOST.get(spec["task"], 0.0) if kind in ("rule", "hypothesis", "unsure") else 0.0
        cands.append({"priority": priority, "kind": kind, "slot": slot, "task": spec["task"], "type": spec["type"],
                      "label": competence_store.slot_name(slot), "learn": spec["learn"], "why": why,
                      "example_question": question, "t": t, **extra})

    # a rule the expert contradicted in this flight, never explained
    explained = {q["slot"] for q in live if q["kind"] == "deviation" and q["answered"]}
    for row in _deviations(session_id):
        rc = row["rule_check"]
        if rc["slot"] in entries and rc["slot"] not in explained:
            add(9, "deviation", rc["slot"], f"flew it differently from the taught rule (expected {rc['expected']}, now {rc['now']})",
                f"You taught me: {entries[rc['slot']]['rule']} In this flight you did it differently. What made the difference?",
                float(row.get("t", 0)), deviation={"slot": rc["slot"], "expected": rc["expected"], "now": rc["now"],
                                                   "rule": competence_store.describe(rc["slot"], entries[rc["slot"]])})

    # live questions the expert could not answer
    for q in live:
        if q["slot"] in competence_store.SLOTS and not q["answered"] and q["slot"] not in entries:
            add(8, "unanswered", q["slot"], "asked during the flight, no answer", q["question"], q["t"])

    # habits seen several times, and the empty slots of the tasks just flown
    for task in flown:
        t = _moment_of(flight, task)
        for s in sorted(competence_store.open_slots(task, entries), key=lambda s: TYPE_PRIORITY.get(s["type"], 9)):
            hyp = competence_store.hypothesis(s["slot"], episode_store.by_task(task))
            if hyp:
                add(7, "hypothesis", s["slot"], hyp["text"], hyp["example_question"], t, hypothesis=hyp)
            elif s["slot"] not in asked_slots:
                weight = 6 if s["type"] in GUARDRAIL_TYPES else 5 - 0.2 * TYPE_PRIORITY.get(s["type"], 5)
                add(weight, "rule", s["slot"], f"flew {competence_store.TASKS[task]['short'].lower()}, the rule is unknown",
                    s["example_question"], t)

    # rules heard once, without their exceptions: the cases not seen yet
    for slot, e in entries.items():
        if competence_store.SLOTS[slot]["task"] in flown and e.get("confidence") == "once" and not e.get("conditions"):
            add(3, "unsure", slot, "heard once, no exception known",
                f"You told me: {e['rule']} When would you do it differently, or not at all?",
                _moment_of(flight, competence_store.SLOTS[slot]["task"]))

    # guardrails for situations that did not happen in this flight
    for slot, spec in competence_store.SLOTS.items():
        if spec["task"] not in flown and spec["type"] in ("abort", "condition") and slot not in entries:
            add(2, "unseen", slot, "never seen in a flight", spec["ask_example"], None)

    cands.sort(key=lambda c: -c["priority"])
    return cands


def _pick(cands: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Deterministic choice: highest priority, at most two per task, at least one guardrail."""
    chosen: list[dict[str, Any]] = []
    per_task: dict[str, int] = {}
    for c in cands:
        if len(chosen) >= n:
            break
        if per_task.get(c["task"], 0) >= 2:
            continue
        chosen.append(c)
        per_task[c["task"]] = per_task.get(c["task"], 0) + 1
    if chosen and not any(c["type"] in GUARDRAIL_TYPES for c in chosen):
        guard = next((c for c in cands if c["type"] in GUARDRAIL_TYPES and c not in chosen), None)
        if guard:
            chosen[-1] = guard
    return chosen


class PlannedQuestion(BaseModel):
    candidate_id: int = Field(description="id of the gap from the candidates list")
    question: str = Field(description="The spoken question, at most 30 words.")


class DebriefPlan(BaseModel):
    questions: list[PlannedQuestion] = Field(description=f"{DEBRIEF_MIN_QUESTIONS} to {DEBRIEF_MAX_QUESTIONS} questions, most valuable first.")


PLAN_SYSTEM = f"""You are the AI apprentice of an expert drone pilot who just finished inspecting a high-voltage power line in a simulator. During the flight you asked only a few short questions; now, in a short spoken debrief, you close the gaps that are left before you write the Work Map a novice will learn from.

You receive the candidate gaps (each with its kind, the competence slot, what the slot must capture, why it is a gap and an example question) and the tasks the expert flew with what telemetry measured. Kinds: deviation = the expert did something differently from a rule they taught; unanswered = a question from the flight that got no answer; hypothesis = a habit seen several times, to confirm as a rule; rule = an empty slot of a task just flown; unsure = a rule heard once, ask for its exceptions; unseen = a guardrail for a situation that never happened.

Pick {DEBRIEF_MIN_QUESTIONS} to {DEBRIEF_MAX_QUESTIONS} questions, most valuable first: what a novice would get dangerously wrong comes first. At least one must be about a guardrail (a limit, an abort criterion, an exception, when to stop and ask). Never two questions about the same thing.

Write each question to be read aloud: one sentence (two short ones for a hypothesis), at most 30 words, plain spoken English, no slot ids or coordinates, rounded numbers. Tie it to the moment it is about ("At the road crossing, you climbed to about 25 metres: ..."). Ask for the rule, its reason and its limit, not for numbers telemetry already measured. For a hypothesis, state the habit and ask whether it is the rule and what decides it. For unseen, say you did not see that situation."""


class TeachBack(BaseModel):
    speech: str = Field(description="The whole process explained back, spoken, under 140 words, ending with a question asking whether that is right.")
    slots: list[str] = Field(description="The slot ids of the rules used in the explanation.")


TEACHBACK_SYSTEM = """You are the AI apprentice of an expert drone pilot. You just debriefed a power line inspection flight with them. Now explain the whole process back to them in your own words, so they can confirm or correct it: this proves you understood.

You receive the Work Map: the steps of the flight in order, with what the expert did and the rules they taught (with their conditions and reasons). Write speech:
- spoken aloud, under 140 words (less than a minute), plain English, addressed to the expert ("So, here is how I understood it: you ...");
- follow the steps in order, but only the ones with a decision worth teaching: for each, what you do, why, and the guardrails (the limits, the exceptions, when to stop or abort);
- the rules are what matters: quote a measured number only when it is part of a rule, never recite speeds or distances that are just telemetry;
- use only what the rules say: never invent a number or a reason; for a step with no rule, say briefly that you would still ask about it;
- end with a question such as "Is that how you do it?".
slots: the slot ids of the rules you used."""


class Correction(BaseModel):
    slot: str = Field(description='The slot id the correction is about (from the teach-back slots), or "" if unclear.')
    correction: str = Field(description="What the expert said is actually right, in their words.")


class TeachBackVerdict(BaseModel):
    confirmed: bool = Field(description="True if the expert agrees the explanation is right (even with a tiny remark that changes nothing).")
    corrections: list[Correction] = Field(description="Each point the expert corrected or added; empty if confirmed.")


VERDICT_SYSTEM = """An AI apprentice explained a drone inspection process back to the expert pilot who taught it. You receive that explanation, the rules it used (slot id: rule) and the expert's spoken reply (a transcript, possibly with filler words). Decide whether the expert confirmed it, and list every correction or addition the expert made, each tied to the slot it changes, in the expert's own words."""

YES = re.compile(r"\b(yes|yeah|yep|correct|right|exactly|that's it|that is it|perfect|good|ok|okay|oui|exact|ja|genau|richtig)\b", re.I)
NO = re.compile(r"\b(no|not|but|except|wrong|actually|instead|never|non|nein|aber)\b", re.I)


class DebriefAgent:
    def __init__(self, client: LLMClient | None = None, knowledge: KnowledgeManager | None = None) -> None:
        self.client = client or LLMClient(model=DEBRIEF_MODEL, timeout=30.0, purpose="debrief")
        self._knowledge = knowledge

    @property
    def knowledge(self) -> KnowledgeManager:
        if self._knowledge is None:
            self._knowledge = KnowledgeManager()
        return self._knowledge

    # ---- 2. plan ---------------------------------------------------------------------

    def start(self, session_id: str) -> dict[str, Any]:
        cands = gap_candidates(session_id)[:MAX_CANDIDATES]
        chosen: list[dict[str, Any]] = []
        if cands and self.client.is_available:
            flight = load_flight(session_id)
            story = [{"task": ep.task, "from_s": round(ep.start), "for_s": round(ep.duration, 1), "measured": ep.signature}
                     for ep in (flight.episodes if flight else [])][-12:]
            prompt = json.dumps({
                "candidates": [{"id": i, **{k: c[k] for k in ("kind", "label", "learn", "why", "example_question", "type")},
                                "moment_s": c["t"]} for i, c in enumerate(cands)],
                "tasks_flown": story,
            }, ensure_ascii=False)
            try:
                plan = self.client.parse(prompt=prompt, system=PLAN_SYSTEM, output_format=DebriefPlan, max_tokens=700)
                seen: set[int] = set()
                for q in plan.questions:
                    if 0 <= q.candidate_id < len(cands) and q.candidate_id not in seen and q.question.strip():
                        seen.add(q.candidate_id)
                        chosen.append({**cands[q.candidate_id], "question": q.question.strip()})
                chosen = chosen[:DEBRIEF_MAX_QUESTIONS]
                if chosen and not any(c["type"] in GUARDRAIL_TYPES for c in chosen):
                    guard = next((c for c in cands if c["type"] in GUARDRAIL_TYPES and c["slot"] not in {x["slot"] for x in chosen}), None)
                    if guard:
                        chosen.append({**guard, "question": guard["example_question"]})
            except Exception:
                logger.exception("Debrief planning failed, using the local choice")
                chosen = []
        if len(chosen) < min(DEBRIEF_MIN_QUESTIONS, len(cands)):
            chosen = [{**c, "question": c["example_question"]} for c in _pick(cands, DEBRIEF_MIN_QUESTIONS + 1)]

        items = [{
            "id": i + 1, "slot": c["slot"], "slot_name": c["label"], "kind": c["kind"], "type": c["type"],
            "guardrail": c["type"] in GUARDRAIL_TYPES, "question": c["question"], "why": c["why"], "t": c["t"],
            "task": c["task"], "hypothesis": c.get("hypothesis"), "deviation": c.get("deviation"),
            "status": "pending", "answer": None, "learned": None,
        } for i, c in enumerate(chosen)]
        n = len(items)
        state = {
            "session_id": session_id,
            "started_at": time.time(),
            "phase": "questions" if items else "teachback",
            "intro": (f"Flight ended. Before I write the Work Map, I have {n} question{'s' if n > 1 else ''} "
                      "about things I could not ask during the flight." if items
                      else "Flight ended. I have no open questions, so let me explain back what I understood."),
            "done_when": DONE_WHEN,
            "items": items,
            "teach_back": None,
        }
        debrief_store.save(state)
        return state

    # ---- 3. answers -------------------------------------------------------------------

    def answer(self, session_id: str, item_id: int, text: str) -> dict[str, Any]:
        state = debrief_store.load(session_id)
        if not state:
            raise KeyError("no debrief for this session")
        item = next((it for it in state["items"] if it["id"] == item_id), None)
        if item is None:
            raise KeyError("unknown question")
        flight = load_flight(session_id)
        eps = [ep for ep in flight.episodes_of(item["task"])] if flight else []
        ep = max(eps, key=lambda e: e.duration) if eps else None
        result = self.knowledge.process_operator_response(
            question=item["question"], answer=text, target_slot=item["slot"],
            deviation=item.get("deviation"), hypothesis=item.get("hypothesis"),
            measured=ep.signature if ep else {}, measured_task=ep.task if ep else None,
            session=session_id, t=float(item["t"] or 0.0), phase="debrief",
        )
        item["status"] = "answered"
        item["answer"] = text
        item["learned"] = result.get("insight")
        item["learned_slot"] = result.get("slot_name")
        follow = (result.get("follow_up") or "").strip()
        if follow and item["kind"] != "follow_up":
            pos = state["items"].index(item) + 1
            state["items"].insert(pos, {**item, "id": max(it["id"] for it in state["items"]) + 1, "kind": "follow_up",
                                        "question": follow, "why": "the answer was too vague", "status": "pending",
                                        "answer": None, "learned": None, "hypothesis": None, "deviation": None})
        append_transcript(session_id, {"role": "expert_operator", "kind": "debrief_answer", "text": text, "t": item["t"],
                                       "slot": result.get("slot"), "question": item["question"], "distilled_insight": result.get("insight")})
        self._advance(state)
        debrief_store.save(state)
        return {"result": result, "state": state}

    def skip(self, session_id: str, item_id: int) -> dict[str, Any]:
        state = debrief_store.load(session_id)
        if not state:
            raise KeyError("no debrief for this session")
        for it in state["items"]:
            if it["id"] == item_id and it["status"] == "pending":
                it["status"] = "skipped"
        self._advance(state)
        debrief_store.save(state)
        return state

    @staticmethod
    def _advance(state: dict[str, Any]) -> None:
        if state["phase"] == "questions" and all(it["status"] != "pending" for it in state["items"]):
            state["phase"] = "teachback"

    # ---- 4. teach-back ----------------------------------------------------------------

    def teach_back(self, session_id: str) -> dict[str, Any]:
        state = debrief_store.load(session_id)
        if not state:
            raise KeyError("no debrief for this session")
        wm = work_map.build(session_id) or {"steps": []}
        steps = [{"step": s["title"], "what_the_expert_did": s.get("decision"),
                  "rules": {r["slot"]: {"rule": r["rule"], "conditions": r["conditions"], "reason": r["reason"]} for r in s.get("rules", [])}}
                 for s in wm["steps"] if not s.get("off_record")]
        known = {r["slot"] for s in wm["steps"] for r in s.get("rules", [])}
        tb: TeachBack | None = None
        if self.client.is_available and steps:
            try:
                tb = self.client.parse(prompt=json.dumps({"work_map": steps}, ensure_ascii=False), system=TEACHBACK_SYSTEM,
                                       output_format=TeachBack, max_tokens=600)
            except Exception:
                logger.exception("Teach-back failed, using the local explanation")
        if tb is None or not tb.speech.strip():
            tb = TeachBack(speech=_local_teach_back(wm), slots=sorted(known))
        slots = [s for s in tb.slots if s in known] or sorted(known)
        state["phase"] = "teachback"
        state["teach_back"] = {"speech": tb.speech.strip(), "slots": slots, "round": 1, "confirmed": False,
                               "history": [{"speech": tb.speech.strip(), "reply": None}], "all_slots": slots}
        debrief_store.save(state)
        return state

    def teach_back_reply(self, session_id: str, reply: str) -> dict[str, Any]:
        state = debrief_store.load(session_id)
        if not state or not state.get("teach_back"):
            raise KeyError("no teach-back for this session")
        tb = state["teach_back"]
        tb["history"][-1]["reply"] = reply
        entries = competence_store.load()
        verdict: TeachBackVerdict | None = None
        if self.client.is_available:
            try:
                verdict = self.client.parse(
                    prompt=json.dumps({"explanation": tb["speech"], "rules": {k: entries[k]["rule"] for k in tb["slots"] if k in entries},
                                       "expert_reply": reply}, ensure_ascii=False),
                    system=VERDICT_SYSTEM, output_format=TeachBackVerdict, max_tokens=500)
            except Exception:
                logger.exception("Teach-back verdict failed, using the local check")
        if verdict is None:
            confirmed = bool(YES.search(reply)) and not NO.search(reply)
            verdict = TeachBackVerdict(confirmed=confirmed, corrections=[] if confirmed else [Correction(slot="", correction=reply)])

        corrected: list[dict[str, Any]] = []
        for c in verdict.corrections:
            if not c.correction.strip():
                continue
            slot = c.slot if c.slot in competence_store.SLOTS else None
            res = self.knowledge.process_operator_response(
                question=f"Teach-back: {tb['speech']}", answer=c.correction, target_slot=slot,
                session=session_id, t=float((entries.get(slot or "", {}).get("learned_in") or {}).get("t", 0.0)), phase="teach-back")
            corrected.append({"slot": res.get("slot") or slot, "slot_name": res.get("slot_name"), "said": c.correction,
                              "learned": res.get("insight")})
        append_transcript(session_id, {"role": "expert_operator", "kind": "teachback_reply", "text": reply,
                                       "confirmed": verdict.confirmed, "corrections": corrected})
        tb.setdefault("corrections", []).extend(corrected)

        if verdict.confirmed and not corrected:
            tb["confirmed"] = True
            state["phase"] = "done"
            competence_store.mark_teachback(tb["all_slots"], session_id)
        elif tb["round"] >= TEACHBACK_MAX_ROUNDS:
            state["phase"] = "done"  # corrections are saved; the expert was asked enough
        else:
            fixed = [c for c in corrected if c["learned"]]
            entries = competence_store.load()
            if fixed:
                lines = " ".join(f"{competence_store.SLOTS[c['slot']]['label']}: {entries[c['slot']]['rule']}"
                                 for c in fixed if c["slot"] in entries)
                speech = f"Thank you, I corrected it. {lines} Is that right now?"
                tb["slots"] = [c["slot"] for c in fixed if c["slot"]]
                tb["all_slots"] = sorted(set(tb["all_slots"]) | set(tb["slots"]))
            else:
                speech = "I did not get the correction. Can you tell me which part is wrong and what you do instead?"
            tb["round"] += 1
            tb["speech"] = speech
            tb["history"].append({"speech": speech, "reply": None})
        debrief_store.save(state)
        return {"state": state, "verdict": verdict.model_dump(), "corrected": corrected}


def _local_teach_back(wm: dict[str, Any]) -> str:
    parts = ["So, here is how I understood it."]
    seen: set[str] = set()
    for s in wm.get("steps", []):
        for r in s.get("rules", []):
            if r["slot"] in seen:
                continue
            seen.add(r["slot"])
            parts.append(r["rule"].rstrip(".") + ".")
            if len(seen) >= 6:
                break
    if len(parts) == 1:
        parts.append("You have not taught me a rule for these steps yet, so I would ask before each one.")
    parts.append("Is that how you do it?")
    return " ".join(parts)
