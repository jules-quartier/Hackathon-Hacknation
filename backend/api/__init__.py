from backend.api.session_routes import router as session_router
from backend.api.dialogue_routes import router as dialogue_router
from backend.api.knowledge_routes import router as knowledge_router
from backend.api.debrief_routes import router as debrief_router
from backend.api.teach_routes import router as teach_router
from backend.api.ws import router as ws_router, broadcast

__all__ = [
    "session_router",
    "dialogue_router",
    "knowledge_router",
    "debrief_router",
    "teach_router",
    "ws_router",
    "broadcast",
]
