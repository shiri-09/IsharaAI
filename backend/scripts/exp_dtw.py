"""Nearest-neighbour recognition with dynamic time warping (DTW), evaluated with the same folds as exp_cv.py.

Few examples per sentence (about 6 to train on) is the classic setting where template matching does well, so this
is a strong baseline to compare against (and to combine with) the Transformer.

    python scripts/exp_dtw.py --name dtw --folds 0,1
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.ml.dataset import load_all, load_sentences  # noqa: E402
from app.ml.features import featurize, normalize, resample  # noqa: E402
from exp_cv import OUT, folds  # noqa: E402


def dtw_batch(a: torch.Tensor, b: torch.Tensor, band: int) -> torch.Tensor:
    """a: (Na, T, D), b: (Nb, T, D) -> (Na, Nb) DTW distances (Sakoe-Chiba band), all pairs on the GPU."""
    T = a.shape[1]
    cost = torch.cdist(a.reshape(-1, a.shape[2]), b.reshape(-1, b.shape[2]))  # (Na*T, Nb*T)
    cost = cost.reshape(a.shape[0], T, b.shape[0], T).permute(0, 2, 1, 3)  # (Na, Nb, T, T)
    inf = torch.tensor(float("inf"), device=a.device)
    D = torch.full((a.shape[0], b.shape[0], T + 1, T + 1), float("inf"), device=a.device)
    D[:, :, 0, 0] = 0
    for i in range(1, T + 1):
        lo, hi = max(1, i - band), min(T, i + band)
        for j in range(lo, hi + 1):
            best = torch.minimum(torch.minimum(D[:, :, i - 1, j], D[:, :, i, j - 1]), D[:, :, i - 1, j - 1])
            D[:, :, i, j] = cost[:, :, i - 1, j - 1] + best
    out = D[:, :, T, T]
    return torch.where(torch.isfinite(out), out, inf) / (2 * T)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dtw")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", default="")
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--band", type=int, default=6)
    ap.add_argument("--pca", type=int, default=48)
    ap.add_argument("--knn", type=int, default=1)
    ap.add_argument("--min-samples", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()
    seqs, labels, _ = load_all()
    counts = defaultdict(int)
    for lab in labels:
        counts[lab] += 1
    classes = [s["id"] for s in load_sentences() if counts[s["id"]] >= args.min_samples]
    cidx = {c: i for i, c in enumerate(classes)}
    keep = [i for i, lab in enumerate(labels) if lab in cidx and len(seqs[i]) >= 8]
    labels = [labels[i] for i in keep]
    y = np.array([cidx[lab] for lab in labels])
    feats = np.stack([resample(featurize(normalize(seqs[i])), args.T) for i in keep])  # (N, T, F)
    fold = folds(labels, args.k, args.seed)
    run = [int(f) for f in args.folds.split(",")] if args.folds else range(args.k)
    probs = np.zeros((len(y), len(classes)), dtype=np.float32)
    tested = np.zeros(len(y), dtype=bool)
    dev = "cuda"
    for f in run:
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        flat = feats[tr].reshape(-1, feats.shape[2])
        mu, sd = flat.mean(0), flat.std(0) + 1e-4
        Z = (feats - mu) / sd
        # PCA fitted on the training fold only
        U, S, Vt = np.linalg.svd(((Z[tr].reshape(-1, Z.shape[2]))[:: 4]), full_matrices=False)
        P = Vt[: args.pca].T
        Zp = torch.from_numpy((Z @ P).astype(np.float32)).to(dev)
        dist = torch.cat([dtw_batch(Zp[te[i : i + 16]], Zp[tr], args.band) for i in range(0, len(te), 16)]).cpu().numpy()
        # class score = negative distance to the nearest k training examples of that class
        for row, i in enumerate(te):
            sc = np.full(len(classes), -1e9, dtype=np.float32)
            for c in range(len(classes)):
                d = np.sort(dist[row, y[tr] == c])[: args.knn]
                if len(d):
                    sc[c] = -d.mean()
            e = np.exp((sc - sc.max()) * 4)
            probs[i] = e / e.sum()
        tested[te] = True
        print(f"fold {f}: {(probs[te].argmax(1) == y[te]).mean():.3f}  [{time.time() - t0:.0f}s]", flush=True)
    pr, yt = probs[tested], y[tested]
    top1 = float((pr.argmax(1) == yt).mean())
    top3 = float((np.argsort(-pr, 1)[:, :3] == yt[:, None]).any(1).mean())
    print(f"== {args.name}: CV top-1 {top1:.3f}  top-3 {top3:.3f}  n={len(yt)}  {time.time() - t0:.0f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{args.name}.npz", probs=probs, y=y, tested=tested, classes=np.array(classes))
    with open(OUT / "results.jsonl", "a") as fh:
        fh.write(json.dumps({"name": args.name, "top1": top1, "top3": top3, "n": int(len(yt)), "args": vars(args)}) + "\n")


if __name__ == "__main__":
    main()
