from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.core.config import CABLE_DANGER_M, ROAD_MIN_CROSSING_ALT_M, TUTOR_MODEL
from backend.llm.client import LLMClient
from backend.llm.teach import HAZARD_TOPIC, caught_explanation, expert_moment
from backend.sim.predictor import ACTIONS
from backend.storage import competence_store
from backend.storage.knowledge_store import get_knowledge

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "advisor.txt"
logger = logging.getLogger("robot-apprentice.advisor")

# Cable proximity alerts are spoken without the LLM: they must be instant and never skipped
DANGER_CABLE_DIST_M = CABLE_DANGER_M  # the HUD turns red at the same distance
CLOSING_CABLE_DIST_M = 4.0
CLOSING_SPEED_MS = 1.5  # approaching a cable faster than this, inside CLOSING_CABLE_DIST_M, is a warning


def _away_command(telemetry: dict[str, Any]) -> str:
    """The stick move that takes the drone straight away from the nearest cable, in the pilot's frame."""
    pos, cable = telemetry["pos"], telemetry["nearest_cable_point"]
    away = [pos[i] - cable[i] for i in range(3)]
    yaw = telemetry["yaw"]
    fwd = away[0] * math.cos(yaw) + away[1] * math.sin(yaw)
    left = -away[0] * math.sin(yaw) + away[1] * math.cos(yaw)
    options = [
        (away[2], "climb"), (-away[2], "descend"),
        (fwd, "move forward"), (-fwd, "back up"),
        (left, "move left"), (-left, "move right"),
    ]
    return max(options)[1]


def safety_alert(telemetry: dict[str, Any]) -> dict[str, Any] | None:
    """Spoken alert when the drone is dangerously close to a cable or closing in on one fast."""
    dist = telemetry["cable_dist"]
    if dist >= CLOSING_CABLE_DIST_M:
        return None
    pos, cable, vel = telemetry["pos"], telemetry["nearest_cable_point"], telemetry["vel"]
    closing = -sum((pos[i] - cable[i]) * vel[i] for i in range(3)) / max(dist, 0.1)
    move = _away_command(telemetry)
    if dist < DANGER_CABLE_DIST_M:
        return {
            "speech": f"Danger! Cable {dist:.1f} metres away. {move.capitalize()} now.",
            "category": "safety_alert",
            "urgency": "high",
            "knowledge_reference": f"Cable standoff: never closer than {DANGER_CABLE_DIST_M:.0f} m",
            "caught": "cable",  # the expert's moment is attached when the alert is spoken (server._speak_alert)
        }
    if closing > CLOSING_SPEED_MS:
        return {
            "speech": f"Slow down, you are closing on the cable fast, {dist:.0f} metres. Release the sticks or {move}.",
            "category": "safety_alert",
            "urgency": "medium",
            "knowledge_reference": "Maximum approach speed near conductors",
        }
    return None


class Advice(BaseModel):
    speech: str = Field(description="At most 2 short sentences, 30 words, the instruction first. Spoken aloud.")
    category: Literal["safety_alert", "technique_tip", "qa_response"]
    urgency: Literal["low", "medium", "high"]
    knowledge_reference: str = Field(description="The expert rule or knowledge.md section applied.")


def expert_rule(slot: str) -> str | None:
    """The short form of a rule the expert taught, for a spoken sentence (None if not learned yet)."""
    entry = competence_store.load().get(slot)
    if not entry:
        return None
    rule = entry["rule"].split(";")[0].split(" because ")[0].strip().rstrip(".,")
    words = rule.split()
    return " ".join(words[:22]) + ("" if len(words) <= 22 else "...")


_HAZARD = {"cable": "a cable", "tower": "the tower", "tree": "a tree", "ground": "the ground"}


def caught_topic(event: dict[str, Any]) -> str | None:
    """The situation a predicted mistake belongs to, for the expert's rule and moment."""
    kind = event.get("type")
    if kind == "road_ahead":
        return "road" if event.get("low") else None
    if kind == "cannot_stop":
        return "speed"
    if kind in ("conflict_predicted", "guardian_engaged"):
        return HAZARD_TOPIC.get(str(event.get("hazard") or ""))
    if kind == "very_close_cable":
        return "cable"
    return None


def predictive_alert(event: dict[str, Any]) -> dict[str, Any] | None:
    """Instant spoken alert for a predicted danger (no LLM). A predicted mistake (low over the road)
    also says what the expert does there and why; "caught" names the situation, so the expert's
    moment can be replayed with it."""
    alert = _predictive_alert(event)
    topic = caught_topic(event)
    if alert and topic:
        alert["caught"] = topic
    return alert


def attach_expert_moment(alert: dict[str, Any]) -> dict[str, Any]:
    """Add the expert's rule, words and screen moment for the situation the alert caught."""
    topic = alert.get("caught")
    moment = expert_moment(topic) if topic else None
    if moment:
        alert = {**alert, "replay": moment, "knowledge_reference": f"Expert rule: {moment['slot_name']}"}
    return alert


def _predictive_alert(event: dict[str, Any]) -> dict[str, Any] | None:
    kind = event.get("type")
    if kind == "road_ahead":
        rule = expert_rule("road_crossing.crossing_rule")
        if event.get("low"):
            act = ACTIONS.get(event.get("action") or "brake_climb", ACTIONS["brake_climb"])["say"]
            speech = f"Road ahead in {event['in_s']:.0f} seconds and you are only {event['alt']:.0f} metres up: {act}." \
                + caught_explanation("road")
        else:
            speech = f"Road ahead in {event['in_s']:.0f} seconds: cross straight and quickly, never hover over traffic."
        return {"speech": speech, "category": "safety_alert", "urgency": "medium" if event.get("low") else "low",
                "knowledge_reference": f"Expert rule: {rule}" if rule else f"Road crossing: at least {ROAD_MIN_CROSSING_ALT_M:.0f} m, straight, no hovering"}
    if kind == "conflict_predicted":
        what = _HAZARD.get(event.get("hazard") or "", "an obstacle")
        act = ACTIONS.get(event.get("action") or "brake", ACTIONS["brake"])["say"]
        return {"speech": f"You will hit {what} in {event.get('in_s') or 1:.0f} seconds: {act}!",
                "category": "safety_alert", "urgency": "high",
                "knowledge_reference": "Predictive safety: 3-second trajectory forecast"}
    if kind == "cannot_stop":
        what = _HAZARD.get(event.get("hazard") or "", "the obstacle")
        return {"speech": f"Too fast: at {event['speed']:.0f} metres per second you cannot stop before {what}. Release the sticks now.",
                "category": "safety_alert", "urgency": "high",
                "knowledge_reference": f"Stopping distance {event['stop_dist']:.0f} m at this speed"}
    if kind == "guardian_engaged":
        act = ACTIONS.get(event.get("action") or "brake", ACTIONS["brake"])["label"].lower()
        return {"speech": f"Guardian: I am taking over to {act}.", "category": "safety_alert", "urgency": "high",
                "knowledge_reference": "AI Guardian: collision avoidance"}
    return None


def coaching_hint(mission: dict[str, Any] | None) -> dict[str, Any]:
    """What to do next, with the expert's rule for it when one was learned (no LLM)."""
    learned = expert_rule("insulator_inspection.inspection_standoff")
    return {"speech": _next_step(mission), "category": "technique_tip", "urgency": "low",
            "knowledge_reference": "Expert rule: inspection distance" if learned else "Mission plan"}


def _next_step(mission: dict[str, Any] | None) -> str:
    todo = (mission or {}).get("remaining_insulators") or []
    if not todo:
        return "All insulators are inspected. Fly back to the take-off point and land."
    n = todo[0]
    above = float(n.get("height_above_drone_m", 0.0))
    level = f", {above:.0f} metres above you" if above > 3 else (f", {-above:.0f} metres below you" if above < -3 else "")
    how = expert_rule("insulator_inspection.inspection_standoff") or "Approach slowly and hover about 3 metres from it"
    return f"Next, insulator {n['id']}, about {n['distance_m']:.0f} metres {n['direction']}{level}. {how}."


def _fallback_advice(
    telemetry: dict[str, Any],
    event: dict[str, Any] | None = None,
    novice_query: str | None = None,
    mission: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cable_dist = telemetry.get("cable_dist", 99.0)
    speed = telemetry.get("speed", 0.0)
    alt = telemetry.get("altitude", 0.0)

    if novice_query:
        query_l = novice_query.lower()
        if "road" in query_l:
            return {
                "speech": "Maintain at least 20 meters altitude over the road to clear vehicular traffic and avoid electro-magnetic interference.",
                "category": "qa_response",
                "urgency": "medium",
                "knowledge_reference": "Section 1: Altitude Range",
            }
        if "cable" in query_l or "distance" in query_l:
            return {
                "speech": "Keep a minimum standoff distance of 2.5 meters while traveling and never approach closer than 1.5 meters during inspection.",
                "category": "qa_response",
                "urgency": "medium",
                "knowledge_reference": "Section 1: Cable Standoff Distance",
            }
        if "insulator" in query_l:
            return {
                "speech": "Approach the insulator horizontally at eye-level, hold steady for 3 seconds, and look for flashover burn marks.",
                "category": "qa_response",
                "urgency": "medium",
                "knowledge_reference": "Section 2: Insulator Strings",
            }
        return {
            "speech": _next_step(mission),
            "category": "qa_response",
            "urgency": "low",
            "knowledge_reference": "General Maintenance Protocol",
        }

    # Proactive advice based on live telemetry & events
    alert = safety_alert(telemetry)
    if alert:
        return alert
    if cable_dist < DANGER_CABLE_DIST_M:
        return {
            "speech": f"Caution! Cable clearance is critically low at {cable_dist:.1f} meters. Increase standoff immediately.",
            "category": "safety_alert",
            "urgency": "high",
            "knowledge_reference": "Section 1: Cable Standoff Distance (1.5m hard limit)",
        }
    if cable_dist < 2.5 and speed > 1.2:
        return {
            "speech": f"Reduce speed to under 1 meter per second when flying within 4 meters of conductors.",
            "category": "technique_tip",
            "urgency": "medium",
            "knowledge_reference": "Section 1: Maximum Approach Speed",
        }
    if event and event.get("type") == "over_road" and alt < 20.0:
        return {
            "speech": "You are low over the roadway. Climb above 20 meters as mandated by the corridor crossing protocol.",
            "category": "safety_alert",
            "urgency": "high",
            "knowledge_reference": "Section 1: Road Clearance",
        }
    if event and event.get("type") == "hover_start" and cable_dist < 3.5:
        return {
            "speech": "Stabilized in inspection stance. Maintain position for 3 seconds to ensure sharp imagery of the insulator clamp.",
            "category": "technique_tip",
            "urgency": "low",
            "knowledge_reference": "Section 2: Insulator Inspection Dwell",
        }

    return {
        "speech": _next_step(mission),
        "category": "technique_tip",
        "urgency": "low",
        "knowledge_reference": "Standard Operating Limits",
    }


class Advisor:
    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient(model=TUTOR_MODEL, timeout=15.0, purpose="tutor")
        self.system_prompt = (
            PROMPT_PATH.read_text(encoding="utf-8")
            if PROMPT_PATH.exists()
            else "You are an AI Flight Instructor tutoring a novice drone operator."
        )

    def advise(
        self,
        telemetry: dict[str, Any],
        image_b64: str | None = None,
        event: dict[str, Any] | None = None,
        novice_query: str | None = None,
        mission: dict[str, Any] | None = None,
        prediction: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        knowledge = get_knowledge()
        tel = {k: v for k, v in telemetry.items() if k not in ("quat", "rpy", "acc", "wind", "nearest_cable_point")}
        forecast = {k: v for k, v in (prediction or {}).items() if k not in ("path", "stop_point")}

        words = []
        for slot, entry in competence_store.load().items():
            said = competence_store.quote(entry)
            if said:
                words.append({"rule": competence_store.slot_name(slot), "expert_said": said["text"][:300]})
        prompt = (
            f"Knowledge Base (knowledge.md):\n{knowledge}\n\n"
            f"The expert's own words, per rule (quote them when you explain why):\n"
            f"{json.dumps(words[-14:], ensure_ascii=False, separators=(',', ':'))}\n\n"
            f"Current Telemetry:\n{json.dumps(tel, separators=(',', ':'))}\n\n"
            f"3-second forecast (risk, predicted conflict, recommended manoeuvre, road ahead, stopping distance):\n"
            f"{json.dumps(forecast or None, separators=(',', ':'))}\n\n"
            f"Mission Progress:\n{json.dumps(mission or {}, separators=(',', ':'))}\n\n"
            f"Event Context:\n{json.dumps(event or {}, separators=(',', ':'))}\n\n"
            f"Novice Pilot Query (if any):\n{novice_query or 'None (provide proactive coaching)'}"
        )

        try:
            if self.client.is_available:
                advice = self.client.parse(prompt=prompt, system=self.system_prompt, output_format=Advice,
                                           max_tokens=300, image_b64=image_b64)
                if advice.speech.strip():
                    return advice.model_dump()
                logger.warning("Tutor reply has no speech, using the fallback")
        except Exception:
            logger.exception("Tutor call failed, using the fallback advice")

        return _fallback_advice(telemetry, event, novice_query, mission)
