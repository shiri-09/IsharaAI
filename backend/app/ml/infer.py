import threading

import numpy as np

from ..config import settings
from .features import to_features


class Recognizer:
    """Lazy-loads the trained checkpoint; thread-safe reload after retraining."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.model = None
        self.labels: list[dict] = []
        self.meta: dict = {}

    def load(self) -> bool:
        if not settings.model_path.exists():
            return False
        import torch  # imported lazily so the API starts fast even without torch

        from .model import SignTransformer

        ckpt = torch.load(settings.model_path, map_location="cpu", weights_only=False)
        model = SignTransformer(ckpt["config"]["in_dim"], len(ckpt["labels"]), seq_len=ckpt["config"]["seq_len"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        with self._lock:
            self.model, self.labels = model, ckpt["labels"]
            self.meta = {k: ckpt.get(k) for k in ("val_acc", "n_train", "n_val")}
        return True

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict(self, frames: np.ndarray, top_k: int = 3) -> dict:
        import torch

        if self.model is None and not self.load():
            raise RuntimeError("No trained model yet. Record samples and run training.")
        x = torch.from_numpy(to_features(frames))[None]
        with self._lock, torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0].numpy()
        order = np.argsort(-probs)[:top_k]
        top = [{"id": self.labels[i]["id"], "text": self.labels[i]["text"], "p": float(probs[i])} for i in order]
        return {"text": top[0]["text"], "id": top[0]["id"], "confidence": top[0]["p"], "top": top}


recognizer = Recognizer()
