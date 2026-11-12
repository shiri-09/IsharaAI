import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .db import Base, engine, migrate
from .llm.router import llm
from .routers import agent, auth, avatar, chat, data, live, prefs, recognize, tts

logging.basicConfig(level=logging.INFO)
Base.metadata.create_all(engine)
migrate()

app = FastAPI(title="Ishara AI", version="0.1.0", description="Continuous ISL recognition + multilingual dialogue")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in (auth.router, chat.router, tts.router, data.router, recognize.router, prefs.router, avatar.router, live.router, agent.router):
    app.include_router(r)


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "llm_order": [p.name for p in llm.order],
        "llm_configured": {p.name: p.available() for p in llm.order},
    }


# Serve the built frontend (npm run build) as a single-page app, if present.
dist = settings.frontend_dist
if dist.exists():
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = dist / path
        return FileResponse(f if path and f.is_file() else dist / "index.html")
