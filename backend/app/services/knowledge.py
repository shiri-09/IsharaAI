"""Retrieval over the institution's knowledge files (data/knowledge/*.md) for grounded, cited answers.

Each '## ' section is one passage. TF-IDF with word n-grams is enough for a small, curated knowledge base and
needs no model download; swap in embeddings later without changing the interface.
"""

import re
import threading
from dataclasses import dataclass
from pathlib import Path

from ..config import settings

KB_DIR: Path = settings.data_dir / "knowledge"


@dataclass
class Passage:
    source: str  # file name
    title: str
    text: str


class KnowledgeBase:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mtime = 0.0
        self.passages: list[Passage] = []
        self._vec = None
        self._mat = None

    def _load(self) -> None:
        files = sorted(KB_DIR.glob("*.md")) if KB_DIR.exists() else []
        mtime = max((f.stat().st_mtime for f in files), default=0.0)
        if mtime == self._mtime and self._vec is not None:
            return
        passages = []
        for f in files:
            for block in re.split(r"(?m)^## ", f.read_text(encoding="utf-8"))[1:]:
                title, _, body = block.partition("\n")
                if body.strip():
                    passages.append(Passage(f.name, title.strip(), " ".join(body.split())))
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        mat = vec.fit_transform([f"{p.title} {p.title} {p.text}" for p in passages]) if passages else None
        self.passages, self._vec, self._mat, self._mtime = passages, vec, mat, mtime

    def search(self, query: str, k: int = 3, min_score: float = 0.08) -> list[tuple[Passage, float]]:
        with self._lock:
            self._load()
            if not self.passages or self._mat is None:
                return []
            scores = (self._mat @ self._vec.transform([query]).T).toarray().ravel()
        order = scores.argsort()[::-1][:k]
        return [(self.passages[i], float(scores[i])) for i in order if scores[i] >= min_score]


kb = KnowledgeBase()
