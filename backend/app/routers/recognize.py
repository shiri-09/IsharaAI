"""Live recognition over WebSocket.

Client -> server (JSON):
  {"type": "frame", "kp": [FRAME_DIM floats]}   one MediaPipe frame
  {"type": "flush"}                              force a prediction on the buffered frames
  {"type": "reset"}                              drop the buffer
Server -> client:
  {"type": "state", "signing": bool, "frames": int}
  {"type": "prediction", "text", "id", "confidence", "top": [...], "accepted": bool}
  {"type": "error", "message": str}

Segmentation: a sentence starts when hands appear and ends after END_GAP frames with no hands
(the signer drops their hands), or on an explicit flush.
"""

import asyncio

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from ..auth import AuthError, current_user, user_id_from_token
from ..config import settings
from ..ml.features import FRAME_DIM, hands_present
from ..ml.infer import recognizer

router = APIRouter(tags=["recognition"])

MIN_FRAMES = 15
END_GAP = 10
MAX_FRAMES = 300


@router.websocket("/ws/recognize")
async def recognize(ws: WebSocket):
    token = ws.query_params.get("token", "")
    try:
        await asyncio.to_thread(user_id_from_token, token)
    except AuthError:
        await ws.close(code=4401)
        return
    await ws.accept()

    buf: list[np.ndarray] = []
    gap = 0
    signing = False

    async def predict_and_send():
        nonlocal buf, signing, gap
        frames, buf, signing, gap = np.stack(buf), [], False, 0
        await ws.send_json({"type": "state", "signing": False, "frames": 0})
        try:
            res = await asyncio.to_thread(recognizer.predict, frames)
        except RuntimeError as e:
            await ws.send_json({"type": "error", "message": str(e)})
            return
        await ws.send_json({"type": "prediction", **res, "accepted": res["confidence"] >= settings.min_confidence})

    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")
            if kind == "reset":
                buf, signing, gap = [], False, 0
                await ws.send_json({"type": "state", "signing": False, "frames": 0})
            elif kind == "flush":
                if len(buf) >= MIN_FRAMES:
                    await predict_and_send()
            elif kind == "frame":
                kp = msg.get("kp")
                if not isinstance(kp, list) or len(kp) != FRAME_DIM:
                    await ws.send_json({"type": "error", "message": f"kp must have {FRAME_DIM} values"})
                    continue
                frame = np.asarray(kp, dtype=np.float32)
                has_hands = hands_present(frame)
                if has_hands:
                    if not signing:
                        signing = True
                        await ws.send_json({"type": "state", "signing": True, "frames": 0})
                    gap = 0
                    buf.append(frame)
                elif signing:
                    gap += 1
                    buf.append(frame)
                    if gap >= END_GAP:
                        buf = buf[:-gap]  # trim trailing hands-down frames
                        if len(buf) >= MIN_FRAMES:
                            await predict_and_send()
                        else:
                            buf, signing, gap = [], False, 0
                            await ws.send_json({"type": "state", "signing": False, "frames": 0})
                if len(buf) >= MAX_FRAMES:
                    await predict_and_send()
    except WebSocketDisconnect:
        return


class ClipIn(BaseModel):
    frames: list[list[float]] = Field(min_length=1, max_length=3000)
    split: bool = False  # True: one sentence per hands-up ... hands-down stretch


def _segments(arr: np.ndarray, split: bool) -> list[tuple[int, int]]:
    has = np.array([hands_present(f) for f in arr])
    if not has.any():
        return []
    if not split:
        idx = np.flatnonzero(has)
        return [(int(idx[0]), int(idx[-1]) + 1)]
    segs, start, gap = [], None, 0
    for i, h in enumerate(has):
        if h:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= END_GAP:
                segs.append((start, i - gap + 1))
                start, gap = None, 0
    if start is not None:
        segs.append((start, len(has) - gap))
    return [(a, b) for a, b in segs if b - a >= MIN_FRAMES]


@router.post("/api/recognize")
async def recognize_clip(body: ClipIn, _user=Depends(current_user)):
    """Recognise an uploaded video's keypoints (extracted in the browser). Same model as the live reader."""
    if any(len(f) != FRAME_DIM for f in body.frames):
        raise HTTPException(422, f"every frame must have {FRAME_DIM} values")
    arr = np.asarray(body.frames, dtype=np.float32)
    segs = _segments(arr, body.split)
    if not segs:
        raise HTTPException(422, "No signing hands were found in this video. Make sure both hands and upper body are visible.")
    out = []
    for a, b in segs:
        try:
            res = await asyncio.to_thread(recognizer.predict, arr[a:b])
        except RuntimeError as e:
            raise HTTPException(503, str(e)) from e
        out.append({"start": a, "end": b, **res, "accepted": res["confidence"] >= settings.min_confidence})
    return {"frames": len(arr), "segments": out}
