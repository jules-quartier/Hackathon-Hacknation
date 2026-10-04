"""Module 2 (debrief, teach-back, Work Map), Module 3 (predictions, expert moments, mastery), trust."""

import numpy as np
from fastapi.testclient import TestClient

from backend.core.privacy import redact
from backend.core.state import runtime
from backend.llm import teach, work_map
from backend.llm.advisor import attach_expert_moment, predictive_alert
from backend.llm.debrief import DebriefAgent, gap_candidates
from backend.server import app
from backend.sim.detector import EventDetector
from backend.sim.drone import DroneSim
from backend.sim.scene import Scene
from backend.storage import competence_store, debrief_store, episode_store
from backend.storage.session_recorder import session_meta, start_session

DT = 1.0 / 60.0
client = TestClient(app)


def _record_flight(mode: str, waypoints: list[tuple[float, float, float, float]]) -> str:
    """Fly a kinematic path and record it like the sim loop does: telemetry, events and frames."""
    scene = Scene()
    scene.defects = []
    drone, detector = DroneSim(scene), EventDetector(scene)
    drone.reset()
    rec = start_session(mode)
    rec.record_event({"type": "defects_placed", "t": 0.0, "defects": []})
    for dur, x, y, z in waypoints:
        a, b = drone.pos.copy(), np.array([x, y, z], dtype=float)
        n = max(1, int(round(dur / DT)))
        for i in range(n):
            p = a + (b - a) * (i + 1) / n
            drone.vel = (p - drone.pos) / DT
            drone.pos = p
            drone.elapsed_time += DT
            for e in detector.update(drone, DT):
                rec.record_event(e)
            rec.record_state(drone.elapsed_time, drone.snapshot())
            if i % 120 == 0:
                rec.save_frame(drone.elapsed_time, "aGVsbG8=")  # a frame every 2 s
        drone.vel = np.zeros(3)
    rec.stop()
    return rec.session_id


# take off, inspect i1 from 5 m, fly along the line, cross the road at 25 m (expert) or 12 m (novice)
def _path(road_alt: float):
    return [(3, -10, -10, 25), (4, -4, -6, 25), (6, -3.5, -6.5, 25.5), (8, 40, -9, road_alt), (6, 75, -9, road_alt),
            (3, 95, -9, road_alt), (2, 100, -9, road_alt)]


def _reset_knowledge():
    competence_store.reset()
    episode_store.reset()


def test_redaction():
    assert redact("Call me at +49 151 2345 6789 or jon@x.de") == "Call me at [NUMBER] or [EMAIL]"
    assert redact("My name is Sabine Keller and I keep 5 to 6 m") == "My name is [NAME] and I keep 5 to 6 m"
    assert redact("cross at 25 m, 12 m/s, 2.5 to 3.0 metres") == "cross at 25 m, 12 m/s, 2.5 to 3.0 metres"


def test_work_map_links_steps_and_guardrails_to_moments_and_words():
    _reset_knowledge()
    sid = _record_flight("expert", _path(25.0))
    competence_store.fill("road_crossing.crossing_rule", "Cross roads at 25 m, straight and fast, never hover over traffic.",
                          [], "a falling drone on a car", {"altitude_median": 25.0}, "What is your rule for roads?",
                          "Always 25 metres, straight across, I never hover over cars.", sid, 30.0, False, phase="live")
    competence_store.fill("insulator_inspection.inspection_standoff", "Hold about 5 m from the string.", [],
                          "the compass drifts closer in", {"insulator_dist_median": 5.0}, "How close?",
                          "Five metres, closer the compass goes wild.", sid, 10.0, False)
    wm = work_map.build(sid)
    tasks = [s["task"] for s in wm["steps"]]
    assert "road_crossing" in tasks and "insulator_inspection" in tasks
    road = next(s for s in wm["steps"] if s["task"] == "road_crossing")
    assert road["moment"]["frame"] and "Crossed the road at 25 m" in road["decision"]
    assert road["reason"]["words"]["text"].startswith("Always 25 metres")
    assert wm["counts"]["guardrails"] >= 1  # inspection distance is a threshold
    g = wm["guardrails"][0]
    assert g["words"] and g["moment"]["session"] == sid


def test_debrief_gaps_answers_and_teach_back():
    _reset_knowledge()
    sid = _record_flight("expert", _path(25.0))
    cands = gap_candidates(sid)
    assert len(cands) >= 3 and any(c["type"] in ("threshold", "abort", "condition") for c in cands)
    agent = DebriefAgent()  # no API key in the tests: local choice and local teach-back
    state = agent.start(sid)
    assert state["phase"] == "questions" and len(state["items"]) >= 3
    assert any(it["guardrail"] for it in state["items"])
    for it in list(state["items"]):
        state = agent.answer(sid, it["id"], "I keep five metres")["state"] if it["id"] == 1 else agent.skip(sid, it["id"])
    assert state["phase"] == "teachback"
    competence_store.fill("road_crossing.crossing_rule", "Cross roads at 25 m.", [], "", {}, "q", "25 m", sid, 30.0, False)
    tb = agent.teach_back(sid)["teach_back"]
    assert "Cross roads at 25 m" in tb["speech"] and "road_crossing.crossing_rule" in tb["slots"]
    again = agent.teach_back_reply(sid, "No, actually 30 metres")["state"]
    assert again["phase"] == "teachback" and again["teach_back"]["round"] == 2
    done = agent.teach_back_reply(sid, "Yes, exactly")["state"]
    assert done["phase"] == "done" and done["teach_back"]["confirmed"]
    assert competence_store.load()["road_crossing.crossing_rule"]["teachback"]["confirmed"]
    assert debrief_store.load(sid)["phase"] == "done"


def test_tutor_prediction_and_expert_moment():
    _reset_knowledge()
    assert teach.prediction_question("road") is None  # nothing to judge the answer against yet
    sid = _record_flight("expert", _path(25.0))
    competence_store.fill("road_crossing.crossing_rule", "Cross roads at 25 m, straight, never hover over traffic.", [],
                          "a falling drone on a car", {"altitude_median": 25.0}, "Your rule for roads?",
                          "Always 25 metres and straight across.", sid, 4.0, False)
    q = teach.prediction_question("road")
    assert q and q["slot"] == "road_crossing.crossing_rule" and "what would the expert do" in q["question"]
    assert teach.PredictionCoach().judge(q, "climb to 25 metres and cross fast")["verdict"] == "right"
    alert = predictive_alert({"type": "road_ahead", "t": 5.0, "in_s": 3.0, "alt": 12.0, "low": True, "action": "brake_climb"})
    assert alert["caught"] == "road" and "The expert crosses at about 25 metres" in alert["speech"]
    alert = attach_expert_moment(alert)
    assert alert["replay"]["quote"].startswith("Always 25") and alert["replay"]["frame"]


def test_novice_mastery_measured_against_the_expert():
    _reset_knowledge()
    competence_store.fill("road_crossing.crossing_rule", "Cross roads at 25 m.", [], "", {"altitude_median": 25.0},
                          "q", "25 metres", "expert-x", 1.0, False)
    competence_store.fill("insulator_inspection.inspection_standoff", "Hold about 5 m.", [], "", {"insulator_dist_median": 5.0},
                          "q", "five metres", "expert-x", 1.0, False)
    sid = _record_flight("novice", _path(12.0))
    m = teach.mastery(sid)
    practice = {p["slot"]: p for p in m["practice"]}
    assert "road_crossing.crossing_rule" in practice
    assert any("crossed the road at 12 m" in w or "12 metres" in w for w in practice["road_crossing.crossing_rule"]["why"])
    assert practice["road_crossing.crossing_rule"]["expert_words"] == "25 metres"


def test_debrief_api_off_record_and_forget():
    _reset_knowledge()
    client.post("/session/start", json={"mode": "expert"})
    on = client.post("/dialogue/off-record?active=true").json()
    assert on["active"] and runtime.off_record
    assert client.post("/dialogue/trigger-question").status_code in (409, 503)
    answer = client.post("/dialogue/answer", json={"answer": "secret technique"}).json()
    assert answer["off_record"] and answer["insight"] is None
    assert not client.post("/dialogue/off-record?active=false").json()["active"]
    sid = client.post("/session/stop").json()["session_id"]
    assert session_meta(sid)["off_record"] and len(session_meta(sid)["off_record"]) == 1
    assert client.get(f"/workmap/{sid}").status_code == 200
    state = client.post(f"/debrief/{sid}/start").json()
    assert state["items"] and state["phase"] == "questions"
    assert client.get(f"/session/{sid}/frame?t=1").status_code == 404
    assert client.get("/session/..%2F..%2Fetc/frame?t=1").status_code == 404
    competence_store.fill("vegetation.tree_clearance_flight", "Keep 5 m from trees.", [], "", {}, "q", "a", sid, 1.0, False)
    grid = client.delete("/knowledge/competence/vegetation.tree_clearance_flight").json()
    assert grid["coverage"]["filled"] == 0


def test_questions_wait_while_the_pilot_talks():
    from backend.api import observer

    runtime.pilot_speaking = True
    assert observer._pilot_talking(100.0)
    runtime.pilot_speaking = False
    runtime.pilot_speech_end = 99.0
    assert observer._pilot_talking(100.0)  # stopped 1 s ago: still too soon
    assert not observer._pilot_talking(105.0)
    runtime.pilot_speech_end = -999.0
