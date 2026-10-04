from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import debrief_router, dialogue_router, knowledge_router, session_router, teach_router, ws_router, broadcast
from backend.api.observer import observer_loop
from backend.core.config import (
    BROADCAST_HZ,
    PREDICT_HZ,
    PREDICT_MAX_PER_FLIGHT,
    PREDICT_MIN_GAP_S,
    QUIET_AFTER_SPEECH_S,
    SIM_HZ,
)
from backend.core.state import runtime
from backend.llm import usage
from backend.llm.advisor import attach_expert_moment, coaching_hint, predictive_alert, safety_alert
from backend.llm.teach import prediction_question
from backend.sim.flight_log import COLUMNS
from backend.sim.predictor import ACTIONS
from backend.sim.scene import scene_json
from backend.storage import competence_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("robot-apprentice.server")


@asynccontextmanager
async def lifespan(_: FastAPI):
    tasks = [asyncio.create_task(sim_loop()), asyncio.create_task(observer_loop())]
    yield
    for task in tasks:
        task.cancel()


app = FastAPI(title="Robot Apprentice - Electric Cable Inspection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(session_router)
app.include_router(dialogue_router)
app.include_router(knowledge_router)
app.include_router(debrief_router)
app.include_router(teach_router)
app.include_router(ws_router)


# Events after which the novice tutor (LLM) gives a tip; cable proximity and predicted dangers are
# spoken instantly without the LLM (_check_novice_safety, PREDICTIVE_EVENTS)
ADVICE_TRIGGER_EVENTS = {
    "takeoff",
    "near_cable",
    "over_road",
    "hover_start",
    "insulator_inspected",
    "defect_spotted",
    "mission_complete",
    "collision",
}
PREDICTIVE_EVENTS = {"road_ahead", "conflict_predicted", "cannot_stop", "guardian_engaged"}
ADVICE_MIN_GAP_S = 6.0
SAFETY_REPEAT_S = 4.0  # a proximity alert is repeated while the danger lasts
COACHING_IDLE_S = 15.0  # silent this long: say what to do next (local, no LLM)


async def _send_advice(advice: dict[str, Any], t: float) -> None:
    runtime.latest_advice = advice
    await broadcast({"type": "advice", **advice})
    if runtime.recorder:
        runtime.recorder.record_transcript({
            "role": "tutor_model",
            "text": advice.get("speech", ""),
            "t": t,
            "advice": advice,
        })


def _tutor_must_wait(t: float) -> bool:
    """A tip would talk over the novice, or over a question they are answering."""
    return runtime.open_prediction is not None or runtime.pilot_quiet_for(t) < QUIET_AFTER_SPEECH_S


async def _handle_novice_event(ev: dict[str, Any], telemetry: dict[str, Any]) -> None:
    now = time.time()
    if ev.get("type") == "hover_start" and telemetry.get("insulator_dist", 99.0) > 8.0:
        return  # a pause away from the work: nothing to coach
    if _tutor_must_wait(telemetry["t"]):
        return
    if ev.get("type") in ADVICE_TRIGGER_EVENTS and (now - runtime.last_advice_time > ADVICE_MIN_GAP_S):
        runtime.last_advice_time = now
        epoch = runtime.session_epoch
        try:
            frame = runtime.camera.get_latest_frame()
            advice_res = await asyncio.to_thread(
                runtime.advisor.advise,
                telemetry=telemetry,
                image_b64=frame,
                event=ev,
                mission=runtime.mission_context(),
                prediction=runtime.flight_log.prediction,
            )
            if runtime.session_epoch != epoch or not runtime.session_active:
                return  # flight ended while the tutor was thinking
            if runtime.last_safety_time > now:
                return  # a proximity alert was spoken meanwhile: this tip is stale
            await _send_advice(advice_res, telemetry["t"])
        except Exception:
            logger.exception("Error generating novice advice")


async def _speak_alert(alert: dict[str, Any], t: float, always: bool = False) -> None:
    """Instant alert (no LLM): repeated at most every SAFETY_REPEAT_S unless it escalates."""
    now = time.time()
    escalation = alert["urgency"] == "high" and runtime.last_safety_urgency != "high"
    if now - runtime.last_safety_time < SAFETY_REPEAT_S and not (escalation or always):
        return
    runtime.last_safety_time = now
    runtime.last_safety_urgency = alert["urgency"]
    runtime.last_advice_time = now  # the next tip waits instead of talking over the alert
    alert = attach_expert_moment(alert)  # the expert's rule, words and screen moment for this situation
    topic = alert.get("caught")
    if topic and alert.get("replay") and f"why:{topic}" not in runtime.predict_asked:
        runtime.pending_why = {"topic": topic, "t": t}  # once it is calm: "why would the expert not...?"
    if runtime.open_prediction is not None:
        runtime.open_prediction = None  # safety first: the question being answered is dropped
    await _send_advice(alert, t)


PREDICT_CHECK_S = 0.5
WHY_DELAY_S = 4.0
PREDICT_ANSWER_TIMEOUT_S = 30.0
_last_predict_check = -1e9


def _closing(column: str, over_s: float = 1.0) -> float:
    """How much the distance in `column` shrank over the last second (m, positive = getting closer)."""
    rows = runtime.flight_log.rows
    n = int(over_s * runtime.flight_log.record_hz)
    if len(rows) <= n:
        return 0.0
    i = COLUMNS.index(column)
    return float(rows[-n - 1][i] - rows[-1][i])


def _decision_point(state: dict[str, Any]) -> tuple[str, str, bool] | None:
    """(topic, target, why) of the decision the novice is about to make, or None."""
    t = state["t"]
    why = runtime.pending_why
    if why and t - why["t"] >= WHY_DELAY_S:
        runtime.pending_why = None
        return why["topic"], "", True
    if why:
        return None  # a caught mistake is waiting for its "why" question
    defect = next((e for e in reversed(runtime.flight_log.events[-20:]) if e.get("type") == "defect_spotted"), None)
    if defect and t - float(defect["t"]) < 8.0 and state["speed"] < 1.5:
        return "defect", str(defect.get("label", "a defect")).lower(), False
    if 10.0 < state["road_dist"] < 30.0 and (_closing("road_dist") > 0.5 or state["speed"] < 0.6):
        return "road", "", False
    ins = state.get("nearest_insulator")
    if ins and ins not in runtime.detector.inspected and 6.0 < state["insulator_dist"] < 14.0 and _closing("insulator_dist") > 0.3:
        return "insulator", ins, False
    if 8.0 < state["pylon_dist"] < 15.0 and _closing("pylon_dist") > 0.5:
        return "tower", "", False
    if 6.0 < state["tree_dist"] < 12.0 and _closing("tree_dist") > 0.5:
        return "tree", "", False
    return None


async def _check_predictions(state: dict[str, Any]) -> None:
    """Novice: before a decision point, ask what the expert would do; after a caught mistake, why."""
    global _last_predict_check
    t = state["t"]
    if t - _last_predict_check < PREDICT_CHECK_S and t >= _last_predict_check:
        return
    _last_predict_check = t
    q = runtime.open_prediction
    if q and t - q["t"] > PREDICT_ANSWER_TIMEOUT_S:
        runtime.open_prediction = q = None  # never answered
    risky = (runtime.flight_log.prediction or {}).get("risk") in ("medium", "high")
    if (q or state["altitude"] < 1.0 or risky
            or len([k for k in runtime.predict_asked if not k.startswith("none:")]) >= PREDICT_MAX_PER_FLIGHT
            or t - runtime.last_predict_time < PREDICT_MIN_GAP_S and not runtime.pending_why
            or time.time() - runtime.last_safety_time < 5.0
            or runtime.pilot_quiet_for(t) < QUIET_AFTER_SPEECH_S):
        return
    point = _decision_point(state)
    if point is None:
        return
    topic, target, why = point
    key = f"{'why' if why else 'predict'}:{topic}"
    if key in runtime.predict_asked or f"none:{key}" in runtime.predict_asked:
        return
    question = prediction_question(topic, target, why=why)
    if question is None:
        runtime.predict_asked.add(f"none:{key}")  # no expert rule to judge the answer against
        return
    runtime.predict_asked.add(key)
    runtime.last_predict_time = t
    runtime.last_advice_time = time.time()  # tips wait while the novice thinks
    runtime.open_prediction = {**question, "id": uuid.uuid4().hex[:8], "t": t}
    await broadcast({"type": "predict", **runtime.open_prediction})
    if runtime.recorder:
        runtime.recorder.record_transcript({"role": "tutor_model", "kind": question["kind"], "text": question["question"],
                                            "slot": question["slot"], "t": t})


async def _check_novice_safety(state: dict[str, Any]) -> None:
    """Instant spoken alert when a cable is dangerously close: no LLM, no tip cooldown."""
    if state["altitude"] < 0.4 or state["collided"]:
        return
    alert = safety_alert(state)
    if alert is not None:
        await _speak_alert(alert, state["t"])


def _guardian_view(active: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "enabled": runtime.guardian_enabled,
        "armed": runtime.guardian_active,
        "engaged": active is not None,
        "action": ACTIONS[active["action"]]["label"] if active else None,
        "interventions": runtime.guardian_interventions,
    }


async def _sim_step(dt: float, tick: int, predict_every: int) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """One physics step and everything that follows from it; returns the state and the Guardian's action."""
    runtime.drone.set_keys(runtime.keys_down)
    # Predict from the pilot's own command, then let the Guardian override it if a crash is imminent
    prediction = runtime.predictor.predict(runtime.drone) if tick % predict_every == 0 else None
    guardian = runtime.predictor.apply_guardian(runtime.drone) if runtime.guardian_active else None
    runtime.drone.step(dt, wind_enabled=False)

    events = runtime.detector.update(runtime.drone, dt)
    state = runtime.drone.snapshot()
    state["mode"] = runtime.mode
    state["inspected_count"] = len(runtime.detector.inspected)
    state["defects_spotted"] = list(runtime.detector.defects_spotted)
    runtime.flight_log.record(state)
    state["task"] = runtime.flight_log.tasks.current  # competence-grid task the pilot is doing

    if prediction is not None:
        runtime.flight_log.prediction = prediction.to_json()
        for ev in runtime.monitor.update(prediction, state):
            if ev["type"] == "guardian_engaged":
                if not runtime.guardian_active:
                    continue
                runtime.guardian_interventions += 1
            events.append(ev)

    novice = runtime.session_active and runtime.mode in {"novice", "tutor"}
    for ev in events:
        await broadcast({"type": "event", "event": ev})
        runtime.flight_log.record_event(ev)
        if runtime.recorder:
            runtime.recorder.record_event(ev)
        # Expert questions come from the attention loop (backend/api/observer.py)
        if novice:
            if ev["type"] in PREDICTIVE_EVENTS:
                alert = predictive_alert(ev)
                if alert:  # the pilot must always hear that the Guardian took the sticks
                    await _speak_alert(alert, state["t"], always=ev["type"] == "guardian_engaged")
            else:
                asyncio.create_task(_handle_novice_event(ev, state))

    if novice:
        await _check_novice_safety(state)
        await _check_predictions(state)
        if (state["altitude"] > 0.4 and time.time() - runtime.last_advice_time > COACHING_IDLE_S
                and not _tutor_must_wait(state["t"])):
            runtime.last_advice_time = time.time()
            await _send_advice(coaching_hint(runtime.mission_context()), state["t"])

    if runtime.recorder:
        runtime.recorder.record_state(state["t"], state)
    return state, guardian


async def sim_loop() -> None:
    """Fixed-step simulation paced on the wall clock: if a step runs late, the next ones catch up
    (up to MAX_CATCH_UP), so simulated time stays real time however busy the server is."""
    dt = 1.0 / SIM_HZ
    broadcast_interval = 1.0 / BROADCAST_HZ
    predict_every = max(1, int(round(SIM_HZ / PREDICT_HZ)))
    max_catch_up = 4
    next_step = time.perf_counter()
    last_broadcast = 0.0
    tick = 0

    while True:
        try:
            steps = 0
            state, guardian = None, None
            while time.perf_counter() >= next_step and steps < max_catch_up:
                state, guardian = await _sim_step(dt, tick, predict_every)
                tick += 1
                steps += 1
                next_step += dt
            if steps == max_catch_up and time.perf_counter() > next_step:
                next_step = time.perf_counter()  # far behind (debugger, sleep): drop the backlog

            now = time.perf_counter()
            if state is not None and now - last_broadcast >= broadcast_interval:
                task = state["task"]
                await broadcast({
                    "type": "state",
                    **state,
                    "task_name": competence_store.TASKS[task]["short"] if task else None,
                    "knowledge": competence_store.coverage(),
                    "prediction": runtime.flight_log.prediction,
                    "guardian": _guardian_view(guardian),
                    "ai_usage": usage.snapshot()["flight"],
                    "session_active": runtime.session_active,
                })
                last_broadcast = now

            await asyncio.sleep(max(0.0, next_step - time.perf_counter()))
        except Exception:
            logger.exception("Sim loop iteration error")
            await asyncio.sleep(0.05)
            next_step = time.perf_counter()


@app.get("/scene")
async def get_scene() -> dict[str, Any]:
    return scene_json(runtime.scene)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "mode": runtime.mode,
        "clients": len(runtime.clients),
        "drone_time": round(runtime.drone.elapsed_time, 2),
        "ai_usage": usage.snapshot(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=False)
