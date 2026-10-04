from __future__ import annotations

import logging
from typing import Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.core.state import runtime

logger = logging.getLogger("robot-apprentice.ws")
router = APIRouter(tags=["WebSocket"])


async def broadcast(payload: dict[str, Any]) -> None:
    stale = []
    for ws in list(runtime.clients):
        try:
            await ws.send_json(payload)
        except Exception:
            stale.append(ws)
    for ws in stale:
        runtime.clients.discard(ws)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    runtime.clients.add(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "keys":
                runtime.keys_down = set(data.get("down", []))
            elif msg_type == "frame":
                b64 = data.get("frame_b64")
                sim_t = float(data.get("t", runtime.drone.elapsed_time))
                if b64 and not runtime.off_record:  # off the record: the AI sees nothing
                    runtime.camera.set_frame(b64, sim_t)
                    if runtime.recorder:  # kept for the Work Map's screen moments and the tutor's replays
                        runtime.recorder.save_frame(sim_t, b64)
            elif msg_type == "voice":
                # browser voice-activity detection: the pilot started or stopped talking
                speaking = bool(data.get("speaking"))
                if runtime.pilot_speaking and not speaking:
                    runtime.pilot_speech_end = runtime.drone.elapsed_time
                runtime.pilot_speaking = speaking
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        runtime.clients.discard(websocket)
    except Exception:
        logger.exception("WebSocket connection error")
        runtime.clients.discard(websocket)
