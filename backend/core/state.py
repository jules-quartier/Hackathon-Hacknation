from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import WebSocket

from backend.llm import usage
from backend.llm.advisor import Advisor
from backend.llm.comparator import FlightComparator
from backend.llm.debrief import DebriefAgent
from backend.llm.knowledge_manager import KnowledgeManager
from backend.llm.observer import FlightObserver
from backend.llm.summarizer import FlightSummarizer
from backend.llm.teach import PredictionCoach
from backend.sim.camera import CameraManager
from backend.sim.defects import generate_defects
from backend.sim.detector import EventDetector
from backend.sim.drone import DroneSim
from backend.sim.flight_log import FlightLog
from backend.sim.predictor import PredictiveMonitor, TrajectoryPredictor
from backend.sim.scene import Scene
from backend.storage.session_recorder import SessionRecorder


@dataclass
class SimRuntime:
    scene: Scene = field(default_factory=Scene)
    drone: DroneSim = field(init=False)
    detector: EventDetector = field(init=False)
    predictor: TrajectoryPredictor = field(init=False)  # model-predictive safety (sim/predictor.py)
    monitor: PredictiveMonitor = field(default_factory=PredictiveMonitor)
    camera: CameraManager = field(default_factory=CameraManager)
    flight_log: FlightLog = field(default_factory=FlightLog)
    mode: str = "expert"  # "expert" | "novice"
    keys_down: set[str] = field(default_factory=set)
    clients: set[WebSocket] = field(default_factory=set)
    recorder: SessionRecorder | None = None

    # LLM modules
    observer: FlightObserver = field(default_factory=FlightObserver)
    knowledge_manager: KnowledgeManager = field(default_factory=KnowledgeManager)
    advisor: Advisor = field(default_factory=Advisor)
    summarizer: FlightSummarizer = field(default_factory=FlightSummarizer)
    comparator: FlightComparator = field(default_factory=FlightComparator)
    debrief: DebriefAgent = field(init=False)  # spoken debrief after an expert flight (Module 2)
    coach: PredictionCoach = field(default_factory=PredictionCoach)  # novice predictions (Module 3)

    # Active dialogue states
    latest_question: dict[str, Any] | None = None
    latest_advice: dict[str, Any] | None = None
    last_question_time: float = -999.0
    last_advice_time: float = -999.0
    last_safety_time: float = -999.0  # instant cable-proximity alert (novice mode)
    last_safety_urgency: str = ""
    # observer questions: [{"t", "question", "answer" (None until answered), "slot", "kind", "deviation"}]
    qa_history: list[dict[str, Any]] = field(default_factory=list)
    observer_busy: bool = False
    # follow-up for a vague answer, asked at the next calm moment: {"t", "question", "slot"}
    pending_follow_up: dict[str, Any] | None = None
    # pilot just did a task differently from a learned rule: {"t", "slot", "expected", "now", "rule"}
    pending_deviation: dict[str, Any] | None = None
    # sim time the pilot started recording their own note; the apprentice doesn't ask meanwhile
    pilot_note_since: float | None = None
    # attention model (llm/attention.py): moments already asked or declined, last summary for the UI
    attention_consumed: set[str] = field(default_factory=set)
    attention_attempts: dict[str, int] = field(default_factory=dict)  # LLM calls per moment
    attention_last_try: dict[str, float] = field(default_factory=dict)
    attention: dict[str, Any] | None = None
    last_observer_call: float = -999.0
    # the pilot is talking (browser voice-activity detection): the AI waits until they are silent
    pilot_speaking: bool = False
    pilot_speech_end: float = -999.0  # sim time the pilot last stopped talking
    # off the record (expert): no questions, no frames, no learning until switched back on
    off_record: bool = False
    off_record_since: float | None = None
    off_record_windows: list[list[float]] = field(default_factory=list)
    # novice: "what would the expert do here?" questions, {"id", "slot", "question", "t", "trigger"}
    predict_asked: set[str] = field(default_factory=set)
    last_predict_time: float = -999.0
    open_prediction: dict[str, Any] | None = None
    pending_why: dict[str, Any] | None = None  # after a caught mistake: "why would the expert not..." {"topic", "t"}
    # novice-mode Guardian: takes the sticks for a moment when the predictor sees an imminent crash
    guardian_enabled: bool = True
    guardian_interventions: int = 0
    # bumped on every session start/stop; LLM results from an older epoch are discarded
    session_epoch: int = 0

    def __post_init__(self) -> None:
        self.debrief = DebriefAgent(knowledge=self.knowledge_manager)
        self.scene.defects = generate_defects(self.scene)
        self.drone = DroneSim(self.scene)
        self.detector = EventDetector(self.scene)
        self.predictor = TrajectoryPredictor(self.scene)

    @property
    def session_active(self) -> bool:
        return self.recorder is not None

    @property
    def guardian_active(self) -> bool:
        """The Guardian only flies for novices, during a flight, when it is switched on."""
        return self.guardian_enabled and self.session_active and self.mode in {"novice", "tutor"}

    def end_session(self) -> SessionRecorder | None:
        """Stop all AI activity immediately; in-flight LLM results will be discarded."""
        recorder, self.recorder = self.recorder, None
        self.session_epoch += 1
        self.latest_question = None
        self.latest_advice = None
        self.pending_follow_up = None
        self.pending_deviation = None
        self.pilot_note_since = None
        self.attention = None
        self.close_off_record()
        if recorder and self.off_record_windows:
            recorder.update_meta(off_record=self.off_record_windows)
        self.open_prediction = None
        self.pending_why = None
        self.pilot_speaking = False
        return recorder

    def open_off_record(self) -> None:
        if not self.off_record:
            self.off_record = True
            self.off_record_since = self.drone.elapsed_time

    def close_off_record(self) -> None:
        if self.off_record and self.off_record_since is not None:
            self.off_record_windows.append([round(self.off_record_since, 1), round(self.drone.elapsed_time, 1)])
            if self.recorder:
                self.recorder.update_meta(off_record=self.off_record_windows)
        self.off_record = False
        self.off_record_since = None

    def pilot_quiet_for(self, t: float) -> float:
        """Seconds since the pilot last spoke (0 while they are talking)."""
        return 0.0 if self.pilot_speaking else max(0.0, t - self.pilot_speech_end)

    def reset(self, mode: str = "expert") -> None:
        self.session_epoch += 1
        self.mode = mode
        self.scene.defects = generate_defects(self.scene)  # new damage to find on every flight
        self.keys_down.clear()
        self.drone.reset()
        self.detector.reset()
        self.predictor.reset()
        self.monitor.reset()
        self.camera.reset()
        self.flight_log.reset()
        self.attention_consumed.clear()
        self.attention_attempts.clear()
        self.attention_last_try.clear()
        self.attention = None
        self.last_observer_call = -999.0
        self.guardian_interventions = 0
        self.pilot_speaking = False
        self.pilot_speech_end = -999.0
        self.off_record = False
        self.off_record_since = None
        self.off_record_windows = []
        self.predict_asked.clear()
        self.last_predict_time = -999.0
        self.open_prediction = None
        self.pending_why = None
        usage.reset_flight()
        self.qa_history.clear()
        self.observer_busy = False
        self.pending_follow_up = None
        self.pending_deviation = None
        self.pilot_note_since = None
        self.latest_question = None
        self.latest_advice = None
        self.last_question_time = -999.0
        self.last_advice_time = -999.0
        self.last_safety_time = -999.0
        self.last_safety_urgency = ""

    def mission_context(self) -> dict[str, Any]:
        """What is left to do, for the novice tutor: insulators not yet inspected, nearest first."""
        pos, yaw = self.drone.pos, self.drone.yaw
        remaining = []
        for ins in self.scene.insulators:
            if ins["id"] in self.detector.inspected:
                continue
            d = np.asarray(ins["pos"], dtype=float) - pos
            bearing = (math.degrees(math.atan2(d[1], d[0]) - yaw) + 180.0) % 360.0 - 180.0  # >0: to the left
            remaining.append({
                "id": ins["id"],
                "name": ins.get("name", ins["id"]),
                "distance_m": round(float(np.linalg.norm(d)), 1),
                "direction": _direction_words(bearing),
                "height_above_drone_m": round(float(d[2]), 1),
            })
        remaining.sort(key=lambda r: r["distance_m"])
        return {
            "inspected": sorted(self.detector.inspected),
            "remaining_insulators": remaining,
            "defects_spotted": len(self.detector.defects_spotted),
        }


def _direction_words(bearing_deg: float) -> str:
    """Relative bearing (degrees, positive = left) in words a pilot can act on."""
    side = "left" if bearing_deg > 0 else "right"
    b = abs(bearing_deg)
    if b < 20:
        return "straight ahead"
    if b < 70:
        return f"ahead to your {side}"
    if b < 110:
        return f"to your {side}"
    if b < 160:
        return f"behind you to the {side}"
    return "behind you"


runtime = SimRuntime()
