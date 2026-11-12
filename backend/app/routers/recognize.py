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

import jwt
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..auth import decode_token
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
        decode_token(token)
    except (jwt.PyJWTError, ValueError, KeyError):
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
