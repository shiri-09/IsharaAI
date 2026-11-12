"""K-fold cross-validation of recogniser variants (experiments; the GPU pipeline makes each run take minutes).

Every recording is tested exactly once across the folds (stratified by sentence; a recording never appears in
both train and test of a fold), so accuracy is measured on all ~690 videos instead of ~100.

    python scripts/exp_cv.py --name base
    python scripts/exp_cv.py --name small --d-model 96 --layers 2
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.ml.dataset import load_all, load_sentences  # noqa: E402
from app.ml.features import FEAT_DIM, canonical_hands, dominant_left, featurize, normalize  # noqa: E402
from app.ml.gpu_aug import GpuAugment, prep  # noqa: E402
from app.ml.model import SignTransformer  # noqa: E402

OUT = BACKEND.parent / "models" / "experiments"


def folds(labels: list[str], k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(labels), dtype=int)
    by = defaultdict(list)
    for i, lab in enumerate(labels):
        by[lab].append(i)
    for idx in by.values():
        idx = rng.permutation(idx)
        off = rng.integers(k)
        for j, i in enumerate(idx):
            fold[i] = (j + off) % k
    return fold


def make_model(args, n_classes):
    return SignTransformer(FEAT_DIM, n_classes, d_model=args.d_model, n_layers=args.layers, n_heads=4,
                           dropout=args.dropout)


def feature_mask(spec: str, device):
    """1/0 mask over FEAT_DIM columns. Layout: xy of KEEP points (85x2) | velocities (85x2) | hand shapes (2x21x3)."""
    m = torch.ones(FEAT_DIM, device=device)
    n_keep = 85
    face = list(range(65, 85))  # face subset points within KEEP
    head = list(range(0, 11))  # pose head points (nose, eyes, ears, mouth)
    for g in filter(None, spec.split(",")):
        if g in ("face", "headpose"):
            for k in (face if g == "face" else head):
                for off in (0, n_keep * 2):
                    m[off + 2 * k : off + 2 * k + 2] = 0
        elif g == "vel":
            m[n_keep * 2 : n_keep * 4] = 0
        elif g == "shape":
            m[n_keep * 4 :] = 0
        elif g == "pos":
            m[: n_keep * 2] = 0
    return m


def train_one(args, xtr, ytr, n_classes, device, seed, mask=None):
    torch.manual_seed(seed)
    model = make_model(args, n_classes).to(device)
    aug = GpuAugment(device, mirror_p=args.mirror, rot_deg=args.rot, scale=args.scale, shift=args.shift,
                     noise=args.noise, warp=args.warp, crop=(args.crop, 1.0))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per = (len(xtr) + args.batch - 1) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per, pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.smooth)
    n = len(xtr)
    for _ in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, args.batch):
            bi = perm[i : i + args.batch]
            xb, yb = aug(xtr[bi]), ytr[bi]
            if mask is not None:
                xb = xb * mask
            if args.mixup > 0:
                lam = float(np.random.beta(args.mixup, args.mixup))
                j = torch.randperm(len(bi), device=device)
                out = model(lam * xb + (1 - lam) * xb[j])
                loss = lam * loss_fn(out, yb) + (1 - lam) * loss_fn(out, yb[j])
            else:
                loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--mirror", type=float, default=0.3)
    ap.add_argument("--rot", type=float, default=15)
    ap.add_argument("--scale", type=float, default=0.15)
    ap.add_argument("--shift", type=float, default=0.15)
    ap.add_argument("--noise", type=float, default=0.015)
    ap.add_argument("--warp", type=float, default=0.3)
    ap.add_argument("--crop", type=float, default=0.8)
    ap.add_argument("--ensemble", type=int, default=1)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--folds", type=str, default="", help="comma list of folds to run (quick checks)")
    ap.add_argument("--norm", default="frame", choices=["frame", "clip"])
    ap.add_argument("--tta", action="store_true", help="average predictions over the full clip and two 90% crops")
    ap.add_argument("--mask", default="", help="feature groups to zero: face,headpose,vel,shape,pos")
    ap.add_argument("--canon", action="store_true", help="mirror left-handed signing so the dominant hand is the right")
    args = ap.parse_args()

    t0 = time.time()
    import app.ml.features as feats_mod

    feats_mod.NORM_MODE = args.norm
    seqs, labels, _ = load_all()
    counts = defaultdict(int)
    for lab in labels:
        counts[lab] += 1
    classes = [s["id"] for s in load_sentences() if counts[s["id"]] >= args.min_samples]
    cidx = {c: i for i, c in enumerate(classes)}
    keep = [i for i, lab in enumerate(labels) if lab in cidx and len(seqs[i]) >= 8]
    seqs = [seqs[i] for i in keep]
    labels = [labels[i] for i in keep]
    y = np.array([cidx[lab] for lab in labels])
    device = "cuda"
    canon = canonical_hands if args.canon else (lambda n: n)
    if args.canon:
        print("left-dominant clips:", sum(dominant_left(normalize(s)) for s in seqs), "of", len(seqs), flush=True)
    X = torch.from_numpy(np.stack([prep(s, canon) for s in seqs])).to(device)
    F = torch.from_numpy(np.stack([featurize(canon(normalize(s))) for s in seqs])).to(device)  # exact inference path
    Y = torch.from_numpy(y).to(device)
    views = [F]
    if args.tta:
        def crop(s, a, b):
            n = len(s)
            return s[int(n * a) : max(int(n * a) + 8, int(n * b))]
        for a, b in ((0.0, 0.9), (0.1, 1.0)):
            views.append(torch.from_numpy(np.stack([featurize(canon(normalize(crop(s, a, b)))) for s in seqs])).to(device))
    fold = folds(labels, args.k, args.seed)
    mask = feature_mask(args.mask, device) if args.mask else None
    if mask is not None:
        F = F * mask
    run = [int(f) for f in args.folds.split(",")] if args.folds else range(args.k)
    probs = np.zeros((len(y), len(classes)), dtype=np.float32)
    tested = np.zeros(len(y), dtype=bool)
    for f in run:
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        p = 0
        for e in range(args.ensemble):
            m = train_one(args, X[tr], Y[tr], len(classes), device, seed=args.seed * 100 + f * 10 + e, mask=mask)
            with torch.no_grad():
                for V in views:
                    Vm = V * mask if mask is not None else V
                    p = p + torch.softmax(m(Vm[te]), 1).cpu().numpy() / len(views)
        probs[te] = p / args.ensemble
        tested[te] = True
        acc = (probs[te].argmax(1) == y[te]).mean()
        print(f"fold {f}: {acc:.3f} (n={len(te)})  [{time.time() - t0:.0f}s]", flush=True)
    pr, yt = probs[tested], y[tested]
    top1 = float((pr.argmax(1) == yt).mean())
    top3 = float((np.argsort(-pr, 1)[:, :3] == yt[:, None]).any(1).mean())
    print(f"== {args.name}: CV top-1 {top1:.3f}  top-3 {top3:.3f}  n={len(yt)} classes={len(classes)}  {time.time() - t0:.0f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{args.name}.npz", probs=probs, y=y, tested=tested, classes=np.array(classes))
    with open(OUT / "results.jsonl", "a") as fh:
        fh.write(json.dumps({"name": args.name, "top1": top1, "top3": top3, "n": int(len(yt)), "args": vars(args)}) + "\n")


if __name__ == "__main__":
    main()
