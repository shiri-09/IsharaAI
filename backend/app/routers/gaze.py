"""Eye-tracking calibration logs: the raw per-dot features of each calibration, kept on this machine so the gaze model can be
evaluated and tuned on real faces (scripts/gaze_eval.py). Nothing here leaves the computer."""

import json
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from ..auth import current_user
from ..config import settings
from ..models import User

router = APIRouter(prefix="/api/gaze", tags=["eye tracking"])
LOG_DIR = settings.data_dir / "gaze_logs"


@router.post("/log")
async def log(request: Request, user: User = Depends(current_user)):
    body = await request.body()
    if len(body) > 8_000_000:
        raise HTTPException(413, "Too large")
    data = json.loads(body)
    folder = LOG_DIR / str(user.id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{int(time.time() * 1000)}.json").write_text(json.dumps(data), encoding="utf-8")
    return {"ok": True}
