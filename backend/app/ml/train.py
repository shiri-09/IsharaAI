"""Train the ISL sentence recogniser on recorded samples.

    python -m app.ml.train --epochs 150
"""

import argparse
import random
from collections import Counter

import numpy as np
import torch
from torch import nn

from ..config import settings
from .dataset import load_all, load_sentences
from .features import FEAT_DIM, N_POINTS, SEQ_LEN, normalize, resample
from .model import SignTransformer


def _finish(x: np.ndarray) -> np.ndarray:
    x = resample(x)
    vel = np.diff(x, axis=0, prepend=x[:1])
    return np.concatenate([x, vel], axis=1).astype(np.float32)


def augment(raw: np.ndarray, rng: random.Random) -> np.ndarray:
    """Spatial + temporal augmentation on a raw (T, FRAME_DIM) sample -> model features."""
    x = normalize(raw)
    t = x.shape[0]
    # temporal crop (keep 80-100%) and random frame drop
    keep = max(4, int(t * rng.uniform(0.8, 1.0)))
    start = rng.randint(0, t - keep)
    x = x[start : start + keep]
    if x.shape[0] > 8:
        mask = np.array([rng.random() > 0.1 for _ in range(x.shape[0])])
        mask[0] = mask[-1] = True
        x = x[mask]
    pts = x.reshape(x.shape[0], N_POINTS, 3)
    missing = np.all(pts == 0, axis=2)
    # rotation in image plane, scale, shift, jitter
    a = np.deg2rad(rng.uniform(-15, 15))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.float32)
    pts[..., :2] = pts[..., :2] @ rot.T
    pts *= rng.uniform(0.9, 1.1)
    pts[..., :2] += np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)], dtype=np.float32)
    pts += np.random.normal(0, 0.01, pts.shape).astype(np.float32)
    pts[missing] = 0.0
    return _finish(pts.reshape(x.shape[0], -1))


def clean(raw: np.ndarray) -> np.ndarray:
    return _finish(normalize(raw))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    seqs, labels = load_all()
    counts = Counter(labels)
    class_ids = [s["id"] for s in load_sentences() if counts.get(s["id"], 0) > 0]
    if len(class_ids) < 2:
        raise SystemExit("Need recorded samples for at least 2 sentences. Use the Collect page first.")
    text_of = {s["id"]: s["text"] for s in load_sentences()}
    idx = {c: i for i, c in enumerate(class_ids)}
    print(f"{len(seqs)} samples across {len(class_ids)} sentences: {dict(counts)}")

    # stratified split: ~20% of each class to validation when the class has >= 3 samples
    train_items, val_items = [], []
    for c in class_ids:
        items = [s for s, lab in zip(seqs, labels) if lab == c]
        rng.shuffle(items)
        n_val = len(items) // 5 if len(items) >= 3 else 0
        val_items += [(s, idx[c]) for s in items[:n_val]]
        train_items += [(s, idx[c]) for s in items[n_val:]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SignTransformer(FEAT_DIM, len(class_ids), seq_len=SEQ_LEN).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    xv = torch.tensor(np.stack([clean(s) for s, _ in val_items])) if val_items else None
    yv = torch.tensor([y for _, y in val_items]) if val_items else None
    best_acc, best_state = -1.0, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_items)
        total = 0.0
        for i in range(0, len(train_items), args.batch):
            batch = train_items[i : i + args.batch]
            xb = torch.tensor(np.stack([augment(s, rng) for s, _ in batch])).to(device)
            yb = torch.tensor([y for _, y in batch]).to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(batch)
        sched.step()

        if xv is not None:
            model.eval()
            with torch.no_grad():
                acc = (model(xv.to(device)).argmax(1).cpu() == yv).float().mean().item()
        else:
            acc = 0.0
        if acc >= best_acc:
            best_acc, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  loss {total / len(train_items):.4f}  val_acc {acc:.3f}")

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "labels": [{"id": c, "text": text_of[c]} for c in class_ids],
            "config": {"in_dim": FEAT_DIM, "seq_len": SEQ_LEN},
            "val_acc": best_acc if val_items else None,
            "n_train": len(train_items),
            "n_val": len(val_items),
        },
        settings.model_path,
    )
    print(f"Saved {settings.model_path}  (best val_acc={best_acc:.3f}, val samples={len(val_items)})")


if __name__ == "__main__":
    main()
