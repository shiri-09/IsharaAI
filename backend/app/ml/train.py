"""Train the ISL sentence recogniser.

    python -m app.ml.train --epochs 200            # train/val/test split, report test accuracy
    python -m app.ml.train --epochs 200 --final    # same, then retrain on ALL data for the deployed model

Split: per sentence, whole recordings (videos) go to exactly one of train / val / test, so the test set only
contains recordings the model never saw. Writes models/isl_transformer.pt and models/eval_report.json.
"""

import argparse
import json
import random
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from torch import nn

from ..config import settings
from .dataset import load_all, load_sentences
from .features import FACE_IDX, FEAT_DIM, N_HAND, N_POINTS, N_POSE, SEQ_LEN, featurize, normalize
from .model import SignTransformer

# ---- mirror augmentation: flip x and swap left/right body parts ----
_POSE_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
               (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
_FACE_PAIRS_IDX = [(61, 291), (78, 308), (70, 300), (63, 293), (105, 334), (66, 296), (107, 336)]
_PERM = np.arange(N_POINTS)
for a, b in _POSE_PAIRS:
    _PERM[a], _PERM[b] = b, a
for j in range(N_HAND):
    _PERM[N_POSE + j], _PERM[N_POSE + N_HAND + j] = N_POSE + N_HAND + j, N_POSE + j
_FOFF = N_POSE + 2 * N_HAND
for a, b in _FACE_PAIRS_IDX:
    ia, ib = FACE_IDX.index(a), FACE_IDX.index(b)
    _PERM[_FOFF + ia], _PERM[_FOFF + ib] = _FOFF + ib, _FOFF + ia


MIN_FRAMES = 8  # same order as the live segmenter's minimum


def _finish(x: np.ndarray) -> np.ndarray:
    return featurize(x)


def augment(norm: np.ndarray, rng: random.Random) -> np.ndarray:
    """Spatial + temporal augmentation on a normalised (T, FRAME_DIM) sample -> model features."""
    x = norm
    t = x.shape[0]
    # temporal: crop 80-100%, speed change via random frame drop
    keep = min(t, max(4, int(t * rng.uniform(0.8, 1.0))))
    start = rng.randint(0, t - keep)
    x = x[start : start + keep]
    if x.shape[0] > 8:
        mask = np.array([rng.random() > 0.15 for _ in range(x.shape[0])])
        mask[0] = mask[-1] = True
        x = x[mask]
    pts = x.reshape(x.shape[0], N_POINTS, 3).copy()
    missing = np.all(pts == 0, axis=2)
    if rng.random() < 0.3:  # mirror (left-handed signer)
        pts = pts[:, _PERM]
        missing = missing[:, _PERM]
        pts[..., 0] *= -1
    a = np.deg2rad(rng.uniform(-15, 15))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.float32)
    pts[..., :2] = pts[..., :2] @ rot.T
    pts[..., 0] *= rng.uniform(0.85, 1.15)
    pts[..., 1] *= rng.uniform(0.85, 1.15)
    pts[..., :2] += np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15)], dtype=np.float32)
    pts += np.random.normal(0, 0.015, pts.shape).astype(np.float32)
    # occasionally drop a hand for a few frames (tracking loss)
    if rng.random() < 0.2:
        s = rng.randint(0, pts.shape[0] - 1)
        e = min(pts.shape[0], s + rng.randint(1, 6))
        hand = slice(N_POSE, N_POSE + N_HAND) if rng.random() < 0.5 else slice(N_POSE + N_HAND, N_POSE + 2 * N_HAND)
        missing[s:e, hand] = True
    pts[missing] = 0.0
    return _finish(pts.reshape(x.shape[0], -1))


def split_groups(labels, groups, rng: random.Random, val_frac: float, test_frac: float):
    """Assign whole recordings (groups) of each class to train/val/test."""
    by_class: dict[str, list[str]] = defaultdict(list)
    for lab, g in zip(labels, groups):
        if g not in by_class[lab]:
            by_class[lab].append(g)
    assign: dict[tuple[str, str], str] = {}
    for lab, gs in by_class.items():
        rng.shuffle(gs)
        n = len(gs)
        n_test = round(n * test_frac) if n >= 4 else 0
        n_val = round(n * val_frac) if n >= 5 else 0
        for i, g in enumerate(gs):
            assign[(lab, g)] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return [assign[(lab, g)] for lab, g in zip(labels, groups)]


def evaluate(model, xs: torch.Tensor, ys: torch.Tensor, device: str, batch: int = 256):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(xs), batch):
            preds.append(model(xs[i : i + batch].to(device)).float().cpu())
    logits = torch.cat(preds)
    top1 = (logits.argmax(1) == ys).float().mean().item()
    top3 = (logits.topk(min(3, logits.size(1)), 1).indices == ys[:, None]).any(1).float().mean().item()
    return top1, top3, logits.argmax(1)


def evaluate_ensemble(models, xs: torch.Tensor, ys: torch.Tensor, device: str, batch: int = 256):
    probs = 0
    for m in models:
        m.eval()
        with torch.no_grad():
            probs = probs + torch.cat(
                [torch.softmax(m(xs[i : i + batch].to(device)).float(), 1).cpu() for i in range(0, len(xs), batch)]
            )
    top1 = (probs.argmax(1) == ys).float().mean().item()
    top3 = (probs.topk(min(3, probs.size(1)), 1).indices == ys[:, None]).any(1).float().mean().item()
    return top1, top3, probs.argmax(1)


def fit(train_items, n_classes, args, rng, device, val=None, log_every=20):
    model = SignTransformer(FEAT_DIM, n_classes, seq_len=SEQ_LEN, d_model=args.d_model, n_layers=args.layers,
                            n_heads=4, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * max(1, (len(train_items) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    best = (-1.0, None, 0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_items)
        total = 0.0
        for i in range(0, len(train_items), args.batch):
            batch = train_items[i : i + args.batch]
            xb = torch.from_numpy(np.stack([augment(s, rng) for s, _ in batch])).to(device)
            yb = torch.tensor([y for _, y in batch], device=device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(batch)
        msg = f"epoch {epoch:4d}  loss {total / len(train_items):.4f}"
        if val is not None:
            acc, top3, _ = evaluate(model, *val, device)
            msg += f"  val_acc {acc:.3f}  val_top3 {top3:.3f}"
            if acc >= best[0]:
                best = (acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch)
        if epoch % log_every == 0 or epoch == 1:
            print(msg, flush=True)
    if val is not None and best[1] is not None:
        model.load_state_dict(best[1])
    return model, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--min-samples", type=int, default=2, help="skip sentences with fewer recordings")
    ap.add_argument("--final", action="store_true", help="after evaluation, retrain on all data for deployment")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ensemble", type=int, default=1, help="number of models (different seeds) averaged together")
    args = ap.parse_args()

    t0 = time.time()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    seqs, labels, groups = load_all()
    counts = Counter(labels)
    sentences = load_sentences()
    text_of = {s["id"]: s["text"] for s in sentences}
    class_ids = [s["id"] for s in sentences if counts.get(s["id"], 0) >= args.min_samples]
    if len(class_ids) < 2:
        raise SystemExit("Need recorded samples for at least 2 sentences. Use the Collect page first.")
    idx = {c: i for i, c in enumerate(class_ids)}
    keep = [i for i, lab in enumerate(labels) if lab in idx and len(seqs[i]) >= MIN_FRAMES]
    if len(keep) < len(labels):
        print(f"skipping {sum(1 for lab in labels if lab in idx) - len(keep)} clips shorter than {MIN_FRAMES} frames", flush=True)
    seqs = [normalize(seqs[i]) for i in keep]
    labels = [labels[i] for i in keep]
    groups = [groups[i] for i in keep]
    print(f"{len(seqs)} samples across {len(class_ids)} sentences", flush=True)

    part = split_groups(labels, groups, rng, args.val_frac, args.test_frac)
    items = {k: [(s, idx[lab]) for s, lab, p in zip(seqs, labels, part) if p == k] for k in ("train", "val", "test")}
    print({k: len(v) for k, v in items.items()}, flush=True)

    def tensors(its):
        if not its:
            return None
        return torch.from_numpy(np.stack([_finish(s) for s, _ in its])), torch.tensor([y for _, y in its])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    val, test = tensors(items["val"]), tensors(items["test"])
    models, val_accs, best_epochs = [], [], []
    for k in range(args.ensemble):
        torch.manual_seed(args.seed + k)
        m_rng = random.Random(args.seed + k)
        print(f"--- model {k + 1}/{args.ensemble}", flush=True)
        m, (va, _, be) = fit(list(items["train"]), len(class_ids), args, m_rng, device, val=val)
        models.append(m)
        val_accs.append(va)
        best_epochs.append(be)
    val_acc = float(np.mean(val_accs)) if val is not None else None
    best_epoch = int(np.median(best_epochs))

    report = {
        "dataset": {"samples": len(seqs), "classes": len(class_ids), **{k: len(v) for k, v in items.items()}},
        "split": "per-sentence, whole recordings held out (no recording appears in two splits)",
        "model": {"type": "Conv + Transformer encoder", "d_model": args.d_model, "layers": args.layers,
                  "seq_len": SEQ_LEN, "ensemble": args.ensemble},
        "best_epoch": best_epoch,
        "val_acc_mean": val_acc,
    }
    if test is not None:
        singles = [evaluate(m, *test, device)[0] for m in models]
        acc, top3, pred = evaluate_ensemble(models, *test, device)
        per_class = defaultdict(lambda: [0, 0])
        confusions = Counter()
        for y, p in zip(test[1].tolist(), pred.tolist()):
            per_class[class_ids[y]][1] += 1
            per_class[class_ids[y]][0] += int(y == p)
            if y != p:
                confusions[(text_of[class_ids[y]], text_of[class_ids[p]])] += 1
        report.update(
            test_acc=acc,
            test_top3=top3,
            test_acc_single_models=singles,
            per_class={text_of[c]: {"correct": a, "total": n} for c, (a, n) in per_class.items()},
            top_confusions=[{"true": t, "pred": p, "count": n} for (t, p), n in confusions.most_common(15)],
        )
        print(f"single-model test acc: {[round(x, 3) for x in singles]}", flush=True)
        print(f"TEST accuracy {acc:.3f}  top-3 {top3:.3f}  (ensemble of {len(models)}, n={len(items['test'])})", flush=True)

    if args.final:
        args.epochs = max(best_epoch, args.epochs // 2)
        print(f"Retraining {args.ensemble} model(s) on all {len(seqs)} samples for {args.epochs} epochs...", flush=True)
        everything = [(s, idx[lab]) for s, lab in zip(seqs, labels)]
        models = []
        for k in range(args.ensemble):
            torch.manual_seed(args.seed + 100 + k)
            models.append(fit(list(everything), len(class_ids), args, random.Random(args.seed + 100 + k), device)[0])
        report["deployed_model"] = "retrained on all data after evaluation (test numbers above are from the held-out run)"

    report["train_seconds"] = round(time.time() - t0, 1)
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dicts": [{k: v.detach().cpu() for k, v in m.state_dict().items()} for m in models],
            "labels": [{"id": c, "text": text_of[c]} for c in class_ids],
            "config": {"in_dim": FEAT_DIM, "seq_len": SEQ_LEN, "d_model": args.d_model, "n_layers": args.layers},
            "val_acc": val_acc,
            "test_acc": report.get("test_acc"),
            "test_top3": report.get("test_top3"),
            "n_train": len(items["train"]),
            "n_val": len(items["val"]),
            "n_test": len(items["test"]),
        },
        settings.model_path,
    )
    (settings.model_path.parent / "eval_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved {settings.model_path} and eval_report.json in {report['train_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
