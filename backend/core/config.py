from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root (variables already set in the environment win, e.g. in tests)
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

# API Keys & IDs
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_STT_MODEL = os.getenv("ELEVENLABS_STT_MODEL", "scribe_v2")  # transcribes spoken answers
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel default

# Storage Paths (ROBOT_APPRENTICE_DATA_DIR lets the tests work on a throw-away copy)
DATA_DIR = Path(os.getenv("ROBOT_APPRENTICE_DATA_DIR", str(ROOT_DIR / "data")))
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
KNOWLEDGE_FILE = KNOWLEDGE_DIR / "knowledge.md"
SESSIONS_DIR = DATA_DIR / "sessions"
COMPARISONS_DIR = DATA_DIR / "comparisons"

KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)

# Simulation & Telemetry Config
SIM_HZ = 60.0
BROADCAST_HZ = 30.0
RECORD_HZ = 10.0  # flight log and session telemetry
PREDICT_HZ = 10.0  # predictive safety (backend/sim/predictor.py)
PREDICT_HORIZON_S = 3.0

# Safety margins, shared by the event detector, the predictor, the tutor, the HUD and the debrief
CABLE_DANGER_M = 2.0  # red: "very close" alarm, counted as a safety violation
CABLE_CAUTION_M = 4.0  # amber: "near the cable"
STRUCTURE_DANGER_M = 1.5  # surface of a tower or a tree
STRUCTURE_CAUTION_M = 3.0
ROAD_MIN_CROSSING_ALT_M = 20.0  # baseline rule until the expert teaches their own
MAX_APPROACH_SPEED = 2.0  # m/s closing speed towards a cable inside CABLE_CAUTION_M
MIN_CABLE_SAFE_DISTANCE = CABLE_DANGER_M  # older name, kept for older modules

# LLM flight observer (expert mode). A local attention model scores every OBSERVER_TICK_S whether
# the moment is worth a question; Claude is only called for salient moments, at most once every
# OBSERVER_INTERVAL_S, with the last OBSERVER_WINDOW_S of telemetry and the flight story.
OBSERVER_MODEL = os.getenv("OBSERVER_MODEL", "claude-haiku-4-5")
# Live novice tutor and debriefs: short structured answers, so the fast model
TUTOR_MODEL = os.getenv("TUTOR_MODEL", "claude-haiku-4-5")
DEBRIEF_MODEL = os.getenv("DEBRIEF_MODEL", "claude-haiku-4-5")
OBSERVER_TICK_S = float(os.getenv("OBSERVER_TICK_S", "1.0"))
OBSERVER_INTERVAL_S = float(os.getenv("OBSERVER_INTERVAL_S", "4.0"))
OBSERVER_WINDOW_S = float(os.getenv("OBSERVER_WINDOW_S", "5.0"))
QUESTION_COOLDOWN_S = float(os.getenv("QUESTION_COOLDOWN_S", "15.0"))
# The browser sends a camera frame every ~2 s; older frames are not shown to the observer.
OBSERVER_FRAME_MAX_AGE_S = float(os.getenv("OBSERVER_FRAME_MAX_AGE_S", "5.0"))
UNANSWERED_QUESTION_TIMEOUT_S = float(os.getenv("UNANSWERED_QUESTION_TIMEOUT_S", "30.0"))

# The pilot talking (voice-activity detection in the browser): no question until they have been
# silent this long, so the apprentice never talks over them.
QUIET_AFTER_SPEECH_S = float(os.getenv("QUIET_AFTER_SPEECH_S", "2.5"))

# Spoken debrief after an expert flight (Module 2): gap questions, then a teach-back
DEBRIEF_MIN_QUESTIONS = 3
DEBRIEF_MAX_QUESTIONS = 5
TEACHBACK_MAX_ROUNDS = 3  # explain back, correct, explain again... then stop asking

# Novice tutor (Module 3): "what would the expert do here?" before a decision point
PREDICT_MIN_GAP_S = 25.0
PREDICT_MAX_PER_FLIGHT = 4
