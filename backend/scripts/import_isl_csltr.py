"""Extract MediaPipe Holistic keypoints from ISL-CSLTR sentence videos into Ishara AI's training layout.

    python scripts/import_isl_csltr.py --corpus ../datasets/ISL_CSLRT_Corpus --workers 12

- Uses the same holistic_landmarker.task model and the same 285-value frame layout as the browser
  (app/ml/features.py), so training and live inference see identical features.
- Frames are resized to 640 px wide (webcam resolution) before detection.
- Leading/trailing frames with no hands are trimmed, matching the live segmenter.
- Adds each corpus sentence to data/sentences.json (id cNNN) and writes keypoints to data/external/<id>/.
- Safe to re-run: videos already extracted are skipped (handy while the download is still running).
"""

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.ml.dataset import EXTERNAL_DIR, load_sentences, save_sentences  # noqa: E402
from app.ml.features import FACE_IDX, FRAME_DIM, N_HAND, N_POSE, hands_present  # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}


def _norm_text(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def _pts(lms, n: int, pick=None) -> list[float]:
    if not lms:
        return [0.0] * (n * 3)
    out = []
    for i in pick if pick is not None else range(n):
        p = lms[i]
        out += [p.x, p.y, p.z]
    return out


def _first(v):
    """Holistic results may be a flat landmark list or a list of lists depending on the version."""
    if not v:
        return None
    return v[0] if isinstance(v[0], (list, tuple)) else v


_model_path = ""


def _init(model_path: str) -> None:
    global _model_path
    import mediapipe as mp

    _model_path = model_path
    globals()["mp"] = mp


def _new_landmarker():
    """Fresh tracker per video: VIDEO mode needs increasing timestamps and keeps tracking state."""
    from mediapipe.tasks.python import BaseOptions, vision

    return vision.HolisticLandmarker.create_from_options(
        vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_model_path),
            running_mode=vision.RunningMode.VIDEO,
        )
    )


def extract(video: str, out: str) -> tuple[str, int, str]:
    import cv2

    if Path(out).exists():
        return video, -1, "skip"
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    landmarker = _new_landmarker()
    frames, i = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        h, w = bgr.shape[:2]
        if w > 640:
            bgr = cv2.resize(bgr, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        r = landmarker.detect_for_video(img, int(i * 1000 / fps))
        f = (
            _pts(_first(r.pose_landmarks), N_POSE)
            + _pts(_first(r.left_hand_landmarks), N_HAND)
            + _pts(_first(r.right_hand_landmarks), N_HAND)
            + _pts(_first(r.face_landmarks), len(FACE_IDX), FACE_IDX)
        )
        frames.append(f)
        i += 1
    cap.release()
    landmarker.close()
    if not frames:
        return video, 0, "no frames"
    arr = np.asarray(frames, dtype=np.float32)
    assert arr.shape[1] == FRAME_DIM
    has = [hands_present(fr) for fr in arr]
    if not any(has):
        return video, 0, "no hands detected"
    first, last = has.index(True), len(has) - 1 - has[::-1].index(True)
    arr = arr[first : last + 1]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr.astype(np.float16))
    return video, len(arr), "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(BACKEND.parent / "datasets" / "ISL_CSLRT_Corpus"))
    ap.add_argument("--model", default=str(BACKEND.parent / "datasets" / "holistic_landmarker.task"))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    root = Path(args.corpus) / "Videos_Sentence_Level"
    sentences = load_sentences()
    by_text = {_norm_text(s["text"]): s for s in sentences}
    nums = [int(s["id"][1:]) for s in sentences if s["id"].startswith("c") and s["id"][1:].isdigit()]
    next_num = max(nums) + 1 if nums else 1

    jobs = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        text = folder.name.strip()
        key = _norm_text(text)
        if key not in by_text:
            s = {"id": f"c{next_num:03d}", "text": text[0].upper() + text[1:], "source": "ISL-CSLTR"}
            next_num += 1
            sentences.append(s)
            by_text[key] = s
        sid = by_text[key]["id"]
        for v in sorted(folder.iterdir()):
            if v.suffix.lower() in VIDEO_EXT and not v.name.endswith(".part"):
                tag = hashlib.sha1(str(v.relative_to(root)).encode()).hexdigest()[:10]
                jobs.append((str(v), str(EXTERNAL_DIR / sid / f"isl_{tag}.npy")))
    save_sentences(sentences)
    print(f"{len(jobs)} videos found under {root}", flush=True)

    done = 0
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.model,)) as pool:
        futs = [pool.submit(extract, v, o) for v, o in jobs]
        for f in as_completed(futs):
            done += 1
            try:
                video, n, status = f.result()
            except Exception as e:  # noqa: BLE001
                video, n, status = "?", 0, f"FAILED {e!r}"
            if status != "skip":
                print(f"[{done}/{len(jobs)}] {status} {n:4d} frames  {Path(video).parent.name}/{Path(video).name}", flush=True)
    print(json.dumps({"videos": len(jobs)}), flush=True)


if __name__ == "__main__":
    main()
