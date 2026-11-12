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
