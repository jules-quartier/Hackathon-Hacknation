"""Module 2 routes: the spoken debrief after an expert flight, the Work Map and its screen moments."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.api.dialogue_routes import _transcribe_request
from backend.core.privacy import redact
from backend.core.state import runtime
from backend.llm import work_map
from backend.storage import debrief_store
from backend.storage.session_recorder import frame_path, session_meta

logger = logging.getLogger("robot-apprentice.debrief-api")
router = APIRouter(tags=["Debrief & Work Map"])
_locks: dict[str, asyncio.Lock] = {}


class DebriefAnswer(BaseModel):
    item_id: int
    answer: str


class ItemRef(BaseModel):
    item_id: int


class TeachBackReply(BaseModel):
    answer: str


def _lock(session_id: str) -> asyncio.Lock:
    return _locks.setdefault(session_id, asyncio.Lock())


def _expert_session(session_id: str) -> None:
    meta = session_meta(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found")
    if meta.get("mode") != "expert":
        raise HTTPException(status_code=400, detail="The debrief is for expert flights")
    if runtime.recorder and runtime.recorder.session_id == session_id:
        raise HTTPException(status_code=409, detail="End the flight first")


@router.post("/debrief/{session_id}/start")
async def debrief_start(session_id: str, restart: bool = False) -> dict[str, Any]:
    """Plan the gap questions (or return the debrief already in progress)."""
    _expert_session(session_id)
    async with _lock(session_id):
        state = debrief_store.load(session_id)
        if state and not restart:
            return state
        return await asyncio.to_thread(runtime.debrief.start, session_id)


@router.get("/debrief/{session_id}")
async def debrief_get(session_id: str) -> dict[str, Any]:
    state = debrief_store.load(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="No debrief for this session")
    return state


async def _answer(session_id: str, item_id: int, text: str) -> dict[str, Any]:
    async with _lock(session_id):
        try:
            return await asyncio.to_thread(runtime.debrief.answer, session_id, item_id, redact(text))
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))


@router.post("/debrief/{session_id}/answer")
async def debrief_answer(session_id: str, payload: DebriefAnswer) -> dict[str, Any]:
    if not payload.answer.strip():
        raise HTTPException(status_code=400, detail="Answer cannot be empty")
    return await _answer(session_id, payload.item_id, payload.answer)


@router.post("/debrief/{session_id}/voice-answer")
async def debrief_voice_answer(session_id: str, item_id: int, request: Request) -> dict[str, Any]:
    transcript = redact(await _transcribe_request(request))
    if not transcript:
        return {"transcript": ""}
    return {"transcript": transcript, **await _answer(session_id, item_id, transcript)}


@router.post("/debrief/{session_id}/skip")
async def debrief_skip(session_id: str, payload: ItemRef) -> dict[str, Any]:
    async with _lock(session_id):
        try:
            return await asyncio.to_thread(runtime.debrief.skip, session_id, payload.item_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))


@router.post("/debrief/{session_id}/teachback")
async def debrief_teachback(session_id: str) -> dict[str, Any]:
    """The apprentice explains the whole process back (after the gap questions)."""
    async with _lock(session_id):
        try:
            return await asyncio.to_thread(runtime.debrief.teach_back, session_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))


async def _reply(session_id: str, text: str) -> dict[str, Any]:
    async with _lock(session_id):
        try:
            return await asyncio.to_thread(runtime.debrief.teach_back_reply, session_id, redact(text))
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))


@router.post("/debrief/{session_id}/teachback/reply")
async def debrief_teachback_reply(session_id: str, payload: TeachBackReply) -> dict[str, Any]:
    if not payload.answer.strip():
        raise HTTPException(status_code=400, detail="Reply cannot be empty")
    return await _reply(session_id, payload.answer)


@router.post("/debrief/{session_id}/teachback/voice-reply")
async def debrief_teachback_voice_reply(session_id: str, request: Request) -> dict[str, Any]:
    transcript = redact(await _transcribe_request(request))
    if not transcript:
        return {"transcript": ""}
    return {"transcript": transcript, **await _reply(session_id, transcript)}


@router.get("/workmap/{session_id}")
async def get_work_map(session_id: str) -> dict[str, Any]:
    wm = await asyncio.to_thread(work_map.build, session_id)
    if wm is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return wm


@router.get("/session/{session_id}/frame")
async def session_frame(session_id: str, t: float) -> FileResponse:
    """The camera frame recorded closest to sim time t: the screen moment of a step or a rule."""
    path = frame_path(session_id, t)
    if path is None:
        raise HTTPException(status_code=404, detail="No frame recorded near this moment")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})
