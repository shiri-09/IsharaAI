"""Build the avatar's sign dictionary from the ISL-CSLTR word-level frames.

    python scripts/import_word_signs.py

Each word folder holds a short frame sequence of one signer performing that word. We run MediaPipe Holistic on
the frames in order (VIDEO mode, so tracking is smooth) and save one keypoint clip per word in the same 285-value
layout the recogniser uses: data/signs/<GLOSS>.npy, plus data/signs/index.json mapping every gloss and alias
(e.g. "I_ME_MINE_MY" -> I, ME, MINE, MY) to its clip.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.config import settings  # noqa: E402
from app.ml.features import FACE_IDX, FRAME_DIM, N_HAND, N_POSE  # noqa: E402

CORPUS = BACKEND.parent / "datasets" / "ISL_CSLRT_Corpus" / "Frames_Word_Level"
MODEL = BACKEND.parent / "datasets" / "holistic_landmarker.task"
OUT = settings.data_dir / "signs"


def _pts(lms, n, pick=None):
    if not lms:
        return [0.0] * (n * 3)
    out = []
    for i in pick if pick is not None else range(n):
        p = lms[i]
        out += [p.x, p.y, p.z]
    return out


def _first(v):
    if not v:
        return None
    return v[0] if isinstance(v[0], (list, tuple)) else v


def frame_key(p: Path):
    nums = re.findall(r"\d+", p.stem)
    return (int(nums[-1]) if nums else 0, p.name)


def main() -> None:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    OUT.mkdir(parents=True, exist_ok=True)
    index = {}
    folders = sorted(p for p in CORPUS.iterdir() if p.is_dir())
    for folder in folders:
        gloss = folder.name.strip().upper()
        frames = sorted([f for f in folder.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}], key=frame_key)
        if not frames:
            continue
        lm = vision.HolisticLandmarker.create_from_options(
            vision.HolisticLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=vision.RunningMode.VIDEO
            )
        )
        rows = []
        for i, f in enumerate(frames):
            bgr = cv2.imread(str(f))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            if w > 640:
                bgr = cv2.resize(bgr, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            r = lm.detect_for_video(img, i * 100)
            rows.append(
                _pts(_first(r.pose_landmarks), N_POSE)
                + _pts(_first(r.left_hand_landmarks), N_HAND)
                + _pts(_first(r.right_hand_landmarks), N_HAND)
                + _pts(_first(r.face_landmarks), len(FACE_IDX), FACE_IDX)
            )
        lm.close()
        arr = np.asarray(rows, dtype=np.float32)
        hands = (np.abs(arr[:, N_POSE * 3 : (N_POSE + 2 * N_HAND) * 3]).sum(1) > 0).sum()
        if len(arr) == 0 or hands == 0:
            print(f"skip {gloss}: no hands found")
            continue
        assert arr.shape[1] == FRAME_DIM
        safe = re.sub(r"[^A-Z0-9]+", "_", gloss).strip("_")
        np.save(OUT / f"{safe}.npy", arr.astype(np.float16))
        aliases = {a.strip() for a in re.split(r"[_/]", gloss) if a.strip()} | {gloss.replace("_", " ")}
        for a in aliases:
            index.setdefault(a, f"{safe}.npy")
        print(f"{gloss:22s} {len(arr):3d} frames, hands in {hands}")
    (OUT / "index.json").write_text(json.dumps(dict(sorted(index.items())), indent=1))
    print(f"{len(set(index.values()))} signs, {len(index)} words/aliases -> {OUT}")


if __name__ == "__main__":
    main()
