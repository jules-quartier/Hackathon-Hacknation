"""Flight debrief: deterministic metrics from the recording, plus a short narrative written by Claude.

Every number comes from the telemetry, events and transcript; the LLM only writes the prose, through
a structured output, so a debrief can never contradict what was measured.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from backend.core.config import CABLE_DANGER_M, DEBRIEF_MODEL, ROAD_MIN_CROSSING_ALT_M, SESSIONS_DIR
from backend.llm.client import LLMClient
from backend.llm.teach import mastery
from backend.sim.scene import Scene
from backend.storage import competence_store
from backend.storage.session_recorder import load_jsonl

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "summarizer.txt"
logger = logging.getLogger("robot-apprentice.debrief")


class DebriefNarrative(BaseModel):
    key_maneuvers: list[str] = Field(description="3 to 5 short sentences: the significant actions of the flight.")
    operational_summary: str = Field(description="One paragraph on performance and findings, defects found or missed included.")
    coaching_points: list[str] = Field(description="2 or 3 concrete things to do better next flight.")


def _hover_stability(telemetry: list[dict[str, Any]]) -> float:
    """1-10 from the drift while the drone holds position (GPS hold): wind is fine, wandering is not."""
    holding = (lambda r: r.get("position_hold")) if any("position_hold" in r for r in telemetry[:50]) \
        else (lambda r: float(r.get("speed", 9.0)) < 0.5)  # older recordings: no GPS-hold flag
    drift = [float(r.get("horizontal_speed", r.get("speed", 0.0))) for r in telemetry
             if holding(r) and float(r.get("altitude", 0.0)) > 1.0]
    if len(drift) < 10:
        return 0.0
    mean = sum(drift) / len(drift)
    return round(max(1.0, min(10.0, 10.0 - 12.0 * mean)), 1)


def compute_metrics(session_id: str) -> dict[str, Any]:
    root = SESSIONS_DIR / session_id
    telemetry = load_jsonl(root / "telemetry.jsonl")
    events = load_jsonl(root / "events.jsonl")
    transcript = load_jsonl(root / "transcript.jsonl")

    meta: dict[str, Any] = {}
    if (root / "meta.json").exists():
        try:
            meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        except Exception:
            pass

    start_t = telemetry[0].get("t", 0.0) if telemetry else 0.0
    end_t = telemetry[-1].get("t", 0.0) if telemetry else 0.0

    by_type: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        by_type.setdefault(str(ev.get("type")), []).append(ev)
    defects_present = (by_type.get("defects_placed") or [{}])[-1].get("defects", [])
    found_ids = {str(ev.get("defect_id")) for ev in by_type.get("defect_spotted", [])}
    inspected = sorted({str(ev.get("insulator_id")) for ev in by_type.get("insulator_inspected", [])})

    violations = [f"Cable closer than {CABLE_DANGER_M:.0f} m at t={ev.get('t', 0.0):.1f}s" for ev in by_type.get("very_close_cable", [])]
    violations += [f"Collision with {ev.get('with')} at t={ev.get('t', 0.0):.1f}s" for ev in by_type.get("collision", [])]
    near_structures = [float(r.get("speed", 0.0)) for r in telemetry
                       if min(float(r.get("cable_dist", 99)), float(r.get("pylon_dist", 99))) < 10.0]
    crossings = [{k: ev.get(k) for k in ("min_altitude", "mean_speed", "checked_before_s", "hovered_over_road_s", "low")}
                 for ev in by_type.get("road_crossed", [])]
    learned = sorted({competence_store.slot_name(str(r["slot"])) for r in transcript
                      if r.get("role") == "expert_operator" and r.get("slot") in competence_store.SLOTS})

    total_insulators = len(Scene().insulators)
    return {
        "session_id": session_id,
        "operator_type": meta.get("mode", "expert"),
        "flight_duration_sec": round(max(0.0, end_t - start_t), 1),
        "insulators_inspected": inspected,
        "inspection_coverage_percent": round(100.0 * len(inspected) / max(1, total_insulators), 1),
        "min_cable_distance": round(min((float(r.get("cable_dist", 99.0)) for r in telemetry), default=99.0), 2),
        "safety_violations_count": len(violations),
        "safety_violations": violations[:10],
        "max_speed_recorded": round(max((float(r.get("speed", 0.0)) for r in telemetry), default=0.0), 2),
        "max_speed_near_structures": round(max(near_structures, default=0.0), 2),
        "time_inside_cable_danger_s": round(sum(1 for r in telemetry if float(r.get("cable_dist", 99)) < CABLE_DANGER_M) / 10.0, 1),
        "hover_stability_score": _hover_stability(telemetry),
        "defects_found": [f"{d['label']} ({d['target']})" for d in defects_present if d["id"] in found_ids],
        "defects_missed": [f"{d['label']} ({d['target']})" for d in defects_present if d["id"] not in found_ids],
        "road_crossings": crossings,
        "low_road_crossings": sum(1 for c in crossings if c.get("low")),
        "road_min_crossing_alt_m": ROAD_MIN_CROSSING_ALT_M,
        "predicted_conflicts": len(by_type.get("conflict_predicted", [])),
        "too_fast_to_stop": len(by_type.get("cannot_stop", [])),
        "guardian_interventions": len(by_type.get("guardian_engaged", [])),
        "questions_asked": sum(1 for r in transcript if r.get("role") == "apprentice_model"),
        "answers_given": sum(1 for r in transcript if r.get("role") == "expert_operator" and r.get("kind") != "note"),
        "notes_recorded": sum(1 for r in transcript if r.get("role") == "expert_operator" and r.get("kind") == "note"),
        "knowledge_learned": learned,
        "tutor_messages": sum(1 for r in transcript if r.get("role") == "tutor_model"),
        # novice: the expert rules applied this flight and the ones to practise next (Module 3)
        **({"mastery": mastery(session_id)} if meta.get("mode") in ("novice", "tutor") else {}),
        "events_count": len(events),
        "telemetry_samples": len(telemetry),
    }


class FlightSummarizer:
    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient(model=DEBRIEF_MODEL, purpose="debrief")
        self.system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    def generate_summary(self, session_id: str, ai_usage: dict[str, Any] | None = None) -> dict[str, Any]:
        metrics = compute_metrics(session_id)
        if ai_usage:
            metrics["ai_usage"] = ai_usage
        root = SESSIONS_DIR / session_id
        transcript = load_jsonl(root / "transcript.jsonl")
        dialogue = [{k: r.get(k) for k in ("role", "text", "slot") if r.get(k)} for r in transcript][-16:]

        narrative: DebriefNarrative | None = None
        try:
            if self.client.is_available:
                narrative = self.client.parse(
                    prompt=f"Flight metrics:\n{json.dumps(metrics, ensure_ascii=False)}\n\n"
                           f"Dialogue during the flight (last messages):\n{json.dumps(dialogue, ensure_ascii=False)}",
                    system=self.system_prompt,
                    output_format=DebriefNarrative,
                    max_tokens=700,
                )
        except Exception:
            logger.exception("Debrief narrative failed; using the deterministic summary")

        if narrative is None:
            role = metrics["operator_type"].capitalize()
            narrative = DebriefNarrative(
                key_maneuvers=[
                    f"{role} pilot flew for {metrics['flight_duration_sec']} s.",
                    f"Inspected {len(metrics['insulators_inspected'])} insulators ({metrics['inspection_coverage_percent']}% coverage).",
                    f"Closest cable approach: {metrics['min_cable_distance']} m.",
                ],
                operational_summary=(
                    f"{role} flight of {metrics['flight_duration_sec']} s with {metrics['inspection_coverage_percent']}% insulator "
                    f"coverage, {len(metrics['defects_found'])} defects found and {len(metrics['defects_missed'])} missed, "
                    f"{metrics['safety_violations_count']} safety violations."
                ),
                coaching_points=["Keep the cable standoff above the danger margin.", "Hold steady at each insulator until it is inspected."],
            )

        summary = {**metrics, **narrative.model_dump()}
        (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary
