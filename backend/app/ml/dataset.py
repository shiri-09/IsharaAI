"""Sentence list + recorded samples on disk (data/sentences.json, data/raw/<sentence_id>/*.npy)."""

import json
import time
from pathlib import Path

import numpy as np

from ..config import settings
from .features import FRAME_DIM

DEFAULT_SENTENCES = [
    "Hello, how are you?",
    "My name is ...",
    "I am fine, thank you.",
    "What is your name?",
    "I need help.",
    "Where is the hospital?",
    "I am hungry.",
    "I want water.",
    "Please call my family.",
    "Thank you very much.",
    "I do not understand.",
    "Please repeat again.",
    "What time is it?",
    "I am going home.",
    "Nice to meet you.",
]

SENTENCES_FILE: Path = settings.data_dir / "sentences.json"
RAW_DIR: Path = settings.data_dir / "raw"
EXTERNAL_DIR: Path = settings.data_dir / "external"


ACTIVE_FILE: Path = settings.data_dir / "active_sentences.json"


def active_ids() -> list[str] | None:
    """Sentence ids the recogniser is trained on (data/active_sentences.json); None = every sentence."""
    if not ACTIVE_FILE.exists():
        return None
    return json.loads(ACTIVE_FILE.read_text(encoding="utf-8")).get("ids")


def load_sentences() -> list[dict]:
    if not SENTENCES_FILE.exists():
        save_sentences([{"id": f"s{i:02d}", "text": t} for i, t in enumerate(DEFAULT_SENTENCES, 1)])
    return json.loads(SENTENCES_FILE.read_text(encoding="utf-8"))


def save_sentences(items: list[dict]) -> None:
    SENTENCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SENTENCES_FILE.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


def add_sentence(text: str) -> dict:
    items = load_sentences()
    nums = [int(s["id"][1:]) for s in items if s["id"].startswith("s") and s["id"][1:].isdigit()]
    item = {"id": f"s{(max(nums) + 1) if nums else 1:02d}", "text": text.strip()}
    items.append(item)
    save_sentences(items)
    return item


def save_sample(sentence_id: str, frames: list[list[float]]) -> Path:
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != FRAME_DIM:
        raise ValueError(f"Expected frames of shape (T, {FRAME_DIM}), got {arr.shape}")
    folder = RAW_DIR / sentence_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{int(time.time() * 1000)}.npy"
    np.save(path, arr)
    return path


def _files(sentence_id: str) -> list[Path]:
    """Own recordings (data/raw, in git) + keypoints extracted from public datasets (data/external, git-ignored)."""
    return sorted((RAW_DIR / sentence_id).glob("*.npy")) + sorted((EXTERNAL_DIR / sentence_id).glob("*.npy"))


MANIFEST_FILE: Path = settings.model_path.parent / "trained_files.json"


def all_paths() -> list[str]:
    """Relative path (to data/) of every recording, in exactly the order load_all() returns them."""
    out = []
    for s in load_sentences():
        for f in _files(s["id"]):
            out.append(f.relative_to(settings.data_dir).as_posix())
    return out


def load_manifest() -> dict:
    """{"files": [...], "history": [...]}: the recordings the deployed model was trained on, and its fine-tune history."""
    try:
        return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"files": [], "history": []}


def save_manifest(files: list[str], history: list[dict]) -> None:
    """Record what the model has learned from. `trained_at` is when: a recording only counts as NEW if it was added after
    this moment (corpus sentences that were deliberately left out of the model are not new)."""
    import time

    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps({"files": sorted(set(files)), "history": history, "trained_at": time.time()}, indent=1), encoding="utf-8")


MIN_FRAMES_BYTES = 8 * 285 * 2  # smaller than this cannot hold 8 frames


def new_recordings(model_ids: set[str]) -> dict:
    """Recordings added since the model was last trained.
    -> {"old_classes": {id: n}, "new_classes": {id: n}} counting only new files of sentences the model knows / does not know."""
    m = load_manifest()
    seen = set(m.get("files", []))
    t0 = float(m.get("trained_at", 0) or 0)
    old: dict[str, int] = {}
    new: dict[str, int] = {}
    if not seen:
        return {"old_classes": old, "new_classes": new}
    for s in load_sentences():
        for f in _files(s["id"]):
            rel = f.relative_to(settings.data_dir).as_posix()
            if rel in seen:
                continue
            st = f.stat()
            if st.st_mtime <= t0 or st.st_size < MIN_FRAMES_BYTES:
                continue
            (old if s["id"] in model_ids else new)[s["id"]] = (old if s["id"] in model_ids else new).get(s["id"], 0) + 1
    return {"old_classes": old, "new_classes": new}


def sample_counts() -> dict[str, int]:
    return {s["id"]: len(_files(s["id"])) for s in load_sentences()}


def load_all() -> tuple[list[np.ndarray], list[str], list[str]]:
    """Returns (sequences, sentence ids, group ids). Group = source video/recording, used to keep splits clean."""
    seqs, labels, groups = [], [], []
    for s in load_sentences():
        for f in _files(s["id"]):
            seqs.append(np.load(f).astype(np.float32))
            labels.append(s["id"])
            groups.append(f.stem.split("__")[0])
    return seqs, labels, groups
