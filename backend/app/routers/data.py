import asyncio
import subprocess
import sys

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth import current_user
from ..config import ROOT
from ..ml import dataset
from ..ml.features import FRAME_DIM
from ..ml.infer import recognizer
from ..models import User

router = APIRouter(prefix="/api", tags=["dataset & model"])

_train_state: dict = {"running": False, "log": "", "returncode": None}


class SentenceIn(BaseModel):
    text: str = Field(min_length=1, max_length=200)


class SampleIn(BaseModel):
    sentence_id: str
    frames: list[list[float]] = Field(min_length=8)


@router.get("/sentences")
def sentences():
    counts = dataset.sample_counts()
    return [{**s, "samples": counts.get(s["id"], 0)} for s in dataset.load_sentences()]


@router.post("/sentences")
def add_sentence(body: SentenceIn, _: User = Depends(current_user)):
    return dataset.add_sentence(body.text)


@router.post("/samples")
def add_sample(body: SampleIn, _: User = Depends(current_user)):
    if body.sentence_id not in {s["id"] for s in dataset.load_sentences()}:
        raise HTTPException(404, "Unknown sentence id")
    try:
        dataset.save_sample(body.sentence_id, body.frames)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "samples": dataset.sample_counts().get(body.sentence_id, 0), "frame_dim": FRAME_DIM}


@router.get("/model")
def model_status():
    if not recognizer.ready:
        recognizer.load()
    return {
        "ready": recognizer.ready,
        "labels": recognizer.labels,
        **recognizer.meta,
        "training": {k: _train_state[k] for k in ("running", "returncode")},
    }


async def _run_training(epochs: int) -> None:
    _train_state.update(running=True, log="", returncode=None)
    proc = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-m", "app.ml.train", "--epochs", str(epochs)],
        cwd=str(ROOT / "backend"),
        capture_output=True,
        text=True,
    )
    _train_state.update(running=False, log=(proc.stdout + proc.stderr)[-8000:], returncode=proc.returncode)
    if proc.returncode == 0:
        recognizer.load()


@router.post("/model/train")
async def train(epochs: int = 150, _: User = Depends(current_user)):
    if _train_state["running"]:
        raise HTTPException(409, "Training already running")
    asyncio.create_task(_run_training(epochs))
    return {"started": True}


@router.get("/model/train")
def train_status(_: User = Depends(current_user)):
    return _train_state
