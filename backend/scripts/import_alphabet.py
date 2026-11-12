"""Build the avatar's built-in ISL fingerspelling alphabet (A-Z) from real photos.

Dataset: Hambir S, Khandagale A, Santosh J, Gaikwad V, et al., "A Multimodal Indian Sign Language Dataset for
English and Marathi Alphabets Featuring Pose and Skin Segmentation Representations", Mendeley Data V1,
doi:10.17632/2tj6c7pzfk.1, CC BY 4.0. We use only the English RGB photos (A-Z, 4 signers).

    python scripts/fetch_isl_csltr.py --out ../datasets/ISL_Alphabet --match "English/RGB/" \
        --url https://data.mendeley.com/public-files/datasets/2tj6c7pzfk/files/6b88e649-3fae-41e1-b0eb-b78788a00f03/file_downloaded
    python scripts/import_alphabet.py

For every photo, MediaPipe Hands finds both hands. Photos are taken facing the signer and are not mirrored, so the
signer's right hand is on the viewer's left, exactly like the avatar. Both hands are scaled to the avatar's hand
size and placed in front of its chest, keeping their positions relative to each other. The elbows are solved
with two-link inverse kinematics so the arms stay attached. Per letter we keep the medoid photo: the one most like
all the others, so one odd photo cannot decide the shape.

Output: data/alphabet/alphabet.json  {"A": [[x, y] | null] * 95, ...} in the avatar's body-normalised space.
"""

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from app.ml.features import N_HAND, N_POINTS, N_POSE  # noqa: E402

LH0, RH0 = N_POSE, N_POSE + N_HAND
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# Avatar geometry (frontend/src/lib/avatar.ts REST pose): shoulders at x=+-0.5, y=0; arm segment lengths.
SHOULDER = {"L": np.array([0.5, 0.0]), "R": np.array([-0.5, 0.0])}
UPPER, FORE = 0.93, 0.86
HAND_LEN = 0.38  # wrist -> middle-finger knuckle, in shoulder widths (real signers in data/signs: ~0.36)
CENTRE_TWO = np.array([0.0, 0.62])  # where two-handed letters are made: in front of the lower chest
CENTRE_ONE = np.array([-0.32, 0.5])  # one-handed letters: right hand beside the chest


def detect(landmarker, path: Path):
    import mediapipe as mp

    img = mp.Image.create_from_file(str(path))
    r = landmarker.detect(img)
    w, h = img.width, img.height
    hands = [{"pts": np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)} for lm in r.hand_landmarks]
    # Side by image position, not MediaPipe's handedness label (unreliable on non-selfie photos): the photos face
    # the signer, so the signer's right hand is on the viewer's left, as on the avatar. One hand = the right hand
    # (every one-handed photo in this dataset is signed with the right hand).
    hands.sort(key=lambda d: d["pts"][0, 0])
    for hd, side in zip(hands, ["R", "L"]):
        hd["side"] = side
    return hands


def hand_size(pts: np.ndarray) -> float:
    """Wrist -> middle knuckle length, estimated from several bones so a foreshortened (side-on) hand is not
    mistaken for a small one. Ratios are typical adult hand proportions relative to wrist -> middle knuckle."""
    bones = [((0, 9), 1.0), ((0, 5), 0.95), ((0, 17), 0.85), ((5, 17), 0.75), ((1, 5), 0.6)]
    return max(float(np.linalg.norm(pts[a] - pts[b])) / k for (a, b), k in bones)


def place(hands: list[dict]) -> dict[str, np.ndarray]:
    """Image-space hands -> avatar-space hands (21x2 each), keyed by side."""
    scale = HAND_LEN / max(max(hand_size(hd["pts"]) for hd in hands), 1e-6)
    centre_img = np.mean([hd["pts"].mean(0) for hd in hands], axis=0)
    centre = CENTRE_TWO if len(hands) == 2 else (CENTRE_ONE if hands[0]["side"] == "R" else -CENTRE_ONE * [1, -1])
    return {hd["side"]: (hd["pts"] - centre_img) * scale + centre for hd in hands}


def elbow(shoulder: np.ndarray, wrist: np.ndarray, outward: float) -> np.ndarray:
    """Two-link IK in 2D (outward = +1 for the left arm, -1 for the right)."""
    d = wrist - shoulder
    dist = float(np.linalg.norm(d))
    if dist >= UPPER + FORE - 1e-6:
        return shoulder + d / dist * UPPER
    dist = max(dist, abs(UPPER - FORE) + 1e-6)
    a = (UPPER**2 - FORE**2 + dist**2) / (2 * dist)
    hgt = np.sqrt(max(UPPER**2 - a**2, 0.0))
    base = shoulder + a * d / dist
    perp = np.array([-d[1], d[0]]) / dist
    e1, e2 = base + hgt * perp, base - hgt * perp
    # elbows hang down: take the lower solution; if both are about as low, the one away from the body
    if abs(e1[1] - e2[1]) > 0.05:
        return e1 if e1[1] > e2[1] else e2
    return e1 if (e1[0] - e2[0]) * outward > 0 else e2


def to_pose(hands: dict[str, np.ndarray]) -> list:
    pose: list = [None] * N_POINTS
    for side, pts in hands.items():
        h0, (s, e, w, pinky, index, thumb) = (LH0, (11, 13, 15, 17, 19, 21)) if side == "L" else (RH0, (12, 14, 16, 18, 20, 22))
        out = 1.0 if side == "L" else -1.0
        for k in range(N_HAND):
            pose[h0 + k] = [round(float(pts[k, 0]), 3), round(float(pts[k, 1]), 3)]
        # Fingerspelling is done with the forearms pointing forward, elbows resting by the sides. In 2D the
        # forearm then looks short (it points at the viewer), so keep the elbow at the side when the hand is in
        # reach from there, and solve for it only when it is not.
        el = SHOULDER[side] + np.array([0.15 * out, 0.88])
        if np.linalg.norm(pts[0] - el) > FORE:
            el = elbow(SHOULDER[side], pts[0], out)
        pose[s] = SHOULDER[side].round(3).tolist()
        pose[e] = [round(float(el[0]), 3), round(float(el[1]), 3)]
        pose[w] = pose[h0]
        pose[pinky], pose[index], pose[thumb] = pose[h0 + 17], pose[h0 + 5], pose[h0 + 2]
    return pose


def medoid(cands: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if len(cands) == 1:
        return cands[0]
    sides = sorted(cands[0])
    flat = np.stack([np.concatenate([c[s].ravel() for s in sides]) for c in cands])
    d = ((flat[:, None] - flat[None]) ** 2).sum(-1)
    return cands[int(d.sum(1).argmin())]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "datasets" / "ISL_Alphabet" / "Dataset" / "English" / "RGB"))
    ap.add_argument("--manifest", default="", help="optional manifest.csv, to skip padded duplicate photos")
    ap.add_argument("--model", default=str(ROOT / "datasets" / "hand_landmarker.task"))
    ap.add_argument("--out", default=str(ROOT / "data" / "alphabet" / "alphabet.json"))
    args = ap.parse_args()

    from mediapipe.tasks.python import BaseOptions, vision

    model = Path(args.model)
    if not model.exists():
        urllib.request.urlretrieve(MODEL_URL, model)
    dupes: set[tuple[str, str]] = set()
    if args.manifest and Path(args.manifest).exists():
        with open(args.manifest, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["language"] == "English" and row["is_padded_duplicate"] == "True":
                    dupes.add((row["class"], row["index"]))

    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)), num_hands=2, min_hand_detection_confidence=0.3
    )
    src = Path(args.src)
    result, report, picked = {}, {}, {}
    with vision.HandLandmarker.create_from_options(opts) as lm:
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            files = sorted((src / letter).glob("*.jpg"), key=lambda p: int(p.stem))
            files = [f for f in files if (letter, f.stem) not in dupes]
            found = []
            for f in files:
                h = detect(lm, f)
                if h:
                    h[0]["file"] = f.name
                    found.append(h)
            if not found:
                report[letter] = "no hands found"
                continue
            # how many hands this letter uses: the most common count across photos
            # two-handed if a fair share of photos show two hands: MediaPipe often misses a hand that is partly
            # hidden behind the other, but it does not invent one
            two = sum(len(h) == 2 for h in found)
            n = 2 if two >= 0.25 * len(found) else 1
            same = [h for h in found if len(h) == n]
            placed = [place(h) for h in same]
            best = medoid(placed)
            result[letter] = to_pose(best)
            picked[letter] = same[next(i for i, c in enumerate(placed) if c is best)][0]["file"]
            report[letter] = f"{len(files)} photos, {two} with two hands, {len(found) - two} with one -> {n}-handed"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, separators=(",", ":")))
    (out.parent / "SOURCE.md").write_text(
        "Built by backend/scripts/import_alphabet.py from: Hambir S, Khandagale A, Santosh J, Gaikwad V, et al., "
        "A Multimodal Indian Sign Language Dataset for English and Marathi Alphabets Featuring Pose and Skin "
        "Segmentation Representations, Mendeley Data V1, doi:10.17632/2tj6c7pzfk.1, CC BY 4.0 "
        "(https://creativecommons.org/licenses/by/4.0/). Changes: hand keypoints extracted with MediaPipe Hands, "
        "rescaled and placed on the avatar; one representative photo per letter.\n"
    )
    (out.parent / "picked.json").write_text(json.dumps(picked))
    for k, v in report.items():
        print(k, v, "| photo", picked.get(k))
    print(f"{len(result)} letters -> {out}")


if __name__ == "__main__":
    main()
