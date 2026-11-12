import asyncio
import time
import json
import re
import subprocess
import sys
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..auth import current_user
from ..config import ROOT
from ..ml import dataset

from ..ml.features import FRAME_DIM
from ..ml.infer import recognizer
from ..models import User

MIN_NEW_TAKES = 8  # keep in step with ml/finetune.py
router = APIRouter(prefix="/api", tags=["dataset & model"])

_train_state: dict = {"running": False, "log": "", "returncode": None, "mode": "", "result": None}


class SentenceIn(BaseModel):
    text: str = Field(min_length=1, max_length=200)


class SampleIn(BaseModel):
    sentence_id: str
    frames: list[list[float]] = Field(min_length=8)


VIDEO_ROOT = ROOT / "datasets" / "ISL_CSLRT_Corpus" / "Videos_Sentence_Level"
REF_VIDEOS = ROOT / "data" / "reference_videos"  # browser-playable H.264 copies (scripts/make_reference_videos.py)


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


@lru_cache(maxsize=1)
def _video_folders() -> dict:
    if not VIDEO_ROOT.exists():
        return {}
    return {_norm(d.name): d for d in VIDEO_ROOT.iterdir() if d.is_dir()}


def _reference_video(text: str):
    folder = _video_folders().get(_norm(text))
    if folder is None:
        return None
    vids = sorted(f for f in folder.iterdir() if f.suffix.lower() in {".mp4", ".mov", ".avi"})
    # prefer a clip named after the whole sentence (the clearest full signing)
    full = [v for v in vids if _norm(v.stem).startswith(_norm(text)[:12])]
    return (full or vids or [None])[0]


def _reliability() -> dict:
    """Per-sentence accuracy on videos the model never saw. Prefer the cross-validation report (every video tested
    once, ~7 per sentence) over the single held-out split (1 test video per sentence)."""
    try:
        cv = json.loads((ROOT / "models" / "cv_report.json").read_text(encoding="utf-8"))
        return {_norm(k): v["correct"] / v["total"] for k, v in cv.get("per_class", {}).items() if v.get("total")}
    except (OSError, ValueError):
        pass
    try:
        rep = json.loads((ROOT / "models" / "eval_report.json").read_text(encoding="utf-8"))
        return {_norm(k): v["correct"] / v["total"] for k, v in rep.get("per_class", {}).items() if v.get("total")}
    except (OSError, ValueError):
        return {}


@router.get("/sentences")
def sentences():
    """Every sentence, with whether the trained model knows it and how reliably it was recognised on held-out
    test videos, so users know what they can sign."""
    counts = dataset.sample_counts()
    if not recognizer.ready:
        recognizer.load()
    known = {_norm(l["text"]) for l in recognizer.labels}
    rel = _reliability()
    only = dataset.active_ids()
    out = []
    for s in dataset.load_sentences():
        n = _norm(s["text"])
        out.append(
            {
                **s,
                "samples": counts.get(s["id"], 0),
                "in_model": n in known,
                "active": only is None or s["id"] in only,
                "reliability": rel.get(n),
                "has_video": (REF_VIDEOS / f"{s['id']}.mp4").exists() or _reference_video(s["text"]) is not None,
                "has_avatar": any((dataset.EXTERNAL_DIR / s["id"]).glob("*.npy")),
            }
        )
    return out


@router.get("/sentences/{sid}/video")
def sentence_video(sid: str):
    """A real signer performing this sentence (ISL-CSLTR, CC BY 4.0), for learning how to sign it."""
    s = next((x for x in dataset.load_sentences() if x["id"] == sid), None)
    if s and (REF_VIDEOS / f"{sid}.mp4").exists():
        return FileResponse(REF_VIDEOS / f"{sid}.mp4", media_type="video/mp4")
    v = _reference_video(s["text"]) if s else None
    if v is None:
        raise HTTPException(404, "No reference video for this sentence")
    return FileResponse(v, media_type="video/mp4")


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
    ids = {l["id"] for l in recognizer.labels}
    nr = dataset.new_recordings(ids) if recognizer.ready else {"old_classes": {}, "new_classes": {}}
    pending = {k: v for k, v in nr["new_classes"].items() if v < MIN_NEW_TAKES}  # new sentences that need more takes
    usable_new = sum(nr["old_classes"].values()) + sum(v for v in nr["new_classes"].values() if v >= MIN_NEW_TAKES)
    return {
        "ready": recognizer.ready,
        "labels": recognizer.labels,
        **recognizer.meta,
        "training": {k: _train_state[k] for k in ("running", "returncode")},
        "new_takes": {
            "total": usable_new,
            "old_sentences": nr["old_classes"],
            "new_sentences": {k: v for k, v in nr["new_classes"].items() if v >= MIN_NEW_TAKES},
            "pending": pending,
        },
        "last_finetune": _last_finetune(),
    }


def _last_finetune():
    try:
        return json.loads((ROOT / "models" / "last_finetune.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


async def _run_training(epochs: int, mode: str = "full") -> None:
    _train_state.update(running=True, log="", returncode=None, mode=mode, result=None)
    if mode == "finetune":
        # keeps the trained model and teaches it only what is new (about 30 seconds)
        cmd = [sys.executable, "-m", "app.ml.finetune", "--epochs", str(epochs)]
    else:
        # GPU pipeline with the settings chosen by cross-validation (scripts/exp_cv.py): writes cv_report.json,
        # then trains the deployed ensemble on all recordings
        cmd = [sys.executable, "-m", "app.ml.train_fast", "--epochs", str(epochs)]
    proc = await asyncio.to_thread(
        subprocess.run,
        cmd,
        cwd=str(ROOT / "backend"),
        capture_output=True,
        text=True,
    )
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            try:
                result = json.loads(line[7:])
            except ValueError:
                pass
    if mode == "finetune" and result and result.get("status") == "ok":
        (ROOT / "models" / "last_finetune.json").write_text(json.dumps({**result, "at": time.time()}), encoding="utf-8")
    _train_state.update(running=False, log=(proc.stdout + proc.stderr)[-8000:], returncode=proc.returncode, result=result)
    if proc.returncode == 0:
        recognizer.load()


@router.post("/model/train")
async def train(epochs: int | None = None, mode: str = "finetune", _: User = Depends(current_user)):
    if _train_state["running"]:
        raise HTTPException(409, "Training already running")
    if mode not in ("finetune", "full"):
        raise HTTPException(422, "mode must be finetune or full")
    asyncio.create_task(_run_training(epochs or (40 if mode == "finetune" else 300), mode))
    return {"started": True}


@router.get("/model/train")
def train_status(_: User = Depends(current_user)):
    return _train_state


@router.get("/sentences/{sid}/avatar")
def sentence_avatar(sid: str, _: User = Depends(current_user)):
    """The same sentence performed by the Ishara avatar: a real signer's keypoints (the ones the model learned
    from), replayed as motion."""
    from ..services.avatar import avatar

    s = next((x for x in dataset.load_sentences() if x["id"] == sid), None)
    frames = avatar.clip_for(sid) if s else None
    if not frames:
        raise HTTPException(404, "No keypoint recording for this sentence")
    return {"segments": [{"type": "clip", "label": s["text"], "frames": frames}]}
