import logging
import threading

import numpy as np

from ..config import settings
from .features import to_features

log = logging.getLogger("ishara.ml")


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

        from .features import FEAT_DIM

        try:
            ckpt = torch.load(settings.model_path, map_location="cpu", weights_only=False)
            cfg = ckpt["config"]
            if cfg["in_dim"] != FEAT_DIM:
                raise ValueError(f"checkpoint expects {cfg['in_dim']} features, code produces {FEAT_DIM}")
            models = []
            for sd in ckpt.get("state_dicts") or [ckpt["state_dict"]]:
                m = SignTransformer(
                    cfg["in_dim"],
                    len(ckpt["labels"]),
                    seq_len=cfg["seq_len"],
                    d_model=cfg.get("d_model", 128),
                    n_layers=cfg.get("n_layers", 4),
                )
                m.load_state_dict(sd)
                m.eval()
                models.append(m)
        except Exception as e:  # noqa: BLE001 - an outdated checkpoint must not take the API down
            log.warning("Ignoring incompatible model checkpoint %s: %s. Retrain to fix.", settings.model_path, e)
            return False
        from . import features as feats_mod

        # the checkpoint says how its training data was normalised; live features must match exactly
        feats_mod.NORM_MODE = cfg.get("norm", "frame")
        with self._lock:
            self.model, self.labels = models, ckpt["labels"]
            self.meta = {k: ckpt.get(k) for k in ("val_acc", "test_acc", "test_top3", "n_train", "n_val", "n_test")}
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
            probs = torch.stack([torch.softmax(m(x), dim=1)[0] for m in self.model]).mean(0).numpy()
        order = np.argsort(-probs)[:top_k]
        top = [{"id": self.labels[i]["id"], "text": self.labels[i]["text"], "p": float(probs[i])} for i in order]
        return {"text": top[0]["text"], "id": top[0]["id"], "confidence": top[0]["p"], "top": top}


recognizer = Recognizer()
