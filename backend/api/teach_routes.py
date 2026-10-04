"""Module 3 routes: the novice answers the tutor's "what would the expert do here?" questions."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.api.dialogue_routes import _transcribe_request
from backend.core.privacy import redact
from backend.core.state import runtime
from backend.llm.teach import expert_moment

router = APIRouter(prefix="/teach", tags=["Tutor"])


class PredictAnswer(BaseModel):
    id: str
    answer: str


class PredictRef(BaseModel):
    id: str


def _open(qid: str) -> dict[str, Any]:
    q = runtime.open_prediction
    if not q or q["id"] != qid:
        raise HTTPException(status_code=409, detail="This question is no longer open")
    return q


async def _judge(qid: str, answer: str) -> dict[str, Any]:
    q = _open(qid)
    runtime.open_prediction = None
    runtime.last_advice_time = time.time()  # the next automatic tip waits for the feedback
    verdict = await asyncio.to_thread(runtime.coach.judge, q, answer)
    if runtime.recorder:
        runtime.recorder.record_transcript({"role": "novice_operator", "kind": "prediction", "text": answer, "t": q["t"]})
        runtime.recorder.record_transcript({"role": "tutor_quiz", "slot": q["slot"], "kind": q["kind"], "topic": q["topic"],
                                            "question": q["question"], "answer": answer, "verdict": verdict["verdict"],
                                            "text": verdict["speech"], "t": runtime.drone.elapsed_time})
    replay = expert_moment(q["topic"]) if verdict["verdict"] != "right" else None
    return {**verdict, "slot": q["slot"], "slot_name": q["slot_name"], "replay": replay}


@router.post("/predict-answer")
async def predict_answer(payload: PredictAnswer) -> dict[str, Any]:
    if not payload.answer.strip():
        raise HTTPException(status_code=400, detail="Answer cannot be empty")
    return await _judge(payload.id, redact(payload.answer))


@router.post("/voice-predict-answer")
async def voice_predict_answer(id: str, request: Request) -> dict[str, Any]:
    _open(id)
    transcript = await _transcribe_request(request)
    if not transcript:
        runtime.open_prediction = None
        return {"transcript": ""}
    return {"transcript": transcript, **await _judge(id, transcript)}


@router.post("/predict-skip")
async def predict_skip(payload: PredictRef) -> dict[str, Any]:
    if runtime.open_prediction and runtime.open_prediction["id"] == payload.id:
        runtime.open_prediction = None
    return {"ok": True}
