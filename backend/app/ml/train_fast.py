"""Train the ISL sentence recogniser on the GPU pipeline, with an honest cross-validated report.

    python -m app.ml.train_fast                 # 5-fold CV report + final ensemble on all data
    python -m app.ml.train_fast --no-cv         # only the final model (faster)

1. Cross-validation: every recording is tested exactly once by a model that never saw it (stratified by sentence).
   Writes models/cv_report.json: overall top-1 / top-3, per-sentence accuracy, top confusions.
2. Final: an ensemble trained on ALL recordings, saved to models/isl_transformer.pt (what the app uses).
   The checkpoint records the normalisation mode so live inference computes exactly the same features.
"""

import argparse
import json
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from torch import nn

from ..config import settings
from . import features as F
from .dataset import load_all, load_sentences
from .gpu_aug import GpuAugment, prep
from .model import SignTransformer


def folds(labels: list[str], k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(len(labels), dtype=int)
    by = defaultdict(list)
    for i, lab in enumerate(labels):
        by[lab].append(i)
    for idx in by.values():
        idx = rng.permutation(idx)
        off = rng.integers(k)
        for j, i in enumerate(idx):
            out[i] = (j + off) % k
    return out


def train_one(args, xtr, ytr, n_classes, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SignTransformer(F.FEAT_DIM, n_classes, d_model=args.d_model, n_layers=args.layers, n_heads=4, dropout=args.dropout).to(device)
    aug = GpuAugment(device, mirror_p=args.mirror)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per = (len(xtr) + args.batch - 1) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per, pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    for _ in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xtr), device=device)
        for i in range(0, len(xtr), args.batch):
            bi = perm[i : i + args.batch]
            loss = loss_fn(model(aug(xtr[bi])), ytr[bi])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.2)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--mirror", type=float, default=0.0)
    ap.add_argument("--norm", default="clip", choices=["frame", "clip"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--cv-ensemble", type=int, default=1, help="models per fold in the CV report")
    ap.add_argument("--ensemble", type=int, default=5, help="models in the deployed ensemble")
    ap.add_argument("--min-samples", type=int, default=2)
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    F.NORM_MODE = args.norm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seqs, labels, _ = load_all()
    counts = Counter(labels)
    sentences = load_sentences()
    text_of = {s["id"]: s["text"] for s in sentences}
    classes = [s["id"] for s in sentences if counts.get(s["id"], 0) >= args.min_samples]
    if len(classes) < 2:
        raise SystemExit("Need recorded samples for at least 2 sentences. Use the Collect page first.")
    cidx = {c: i for i, c in enumerate(classes)}
    keep = [i for i, lab in enumerate(labels) if lab in cidx and len(seqs[i]) >= 8]
    seqs = [seqs[i] for i in keep]
    labels = [labels[i] for i in keep]
    y = np.array([cidx[lab] for lab in labels])
    print(f"{len(seqs)} recordings, {len(classes)} sentences, device {device}, norm {args.norm}", flush=True)
    X = torch.from_numpy(np.stack([prep(s) for s in seqs])).to(device)
    Feats = torch.from_numpy(np.stack([F.featurize(F.normalize(s)) for s in seqs])).to(device)  # exact live path
    Y = torch.from_numpy(y).to(device)

    report: dict = {"config": vars(args), "recordings": len(seqs), "sentences": len(classes)}
    if not args.no_cv:
        fold = folds(labels, args.k, args.seed)
        probs = np.zeros((len(y), len(classes)), dtype=np.float32)
        for f in range(args.k):
            tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
            p = 0
            for e in range(args.cv_ensemble):
                m = train_one(args, X[tr], Y[tr], len(classes), device, seed=args.seed * 1000 + f * 10 + e)
                with torch.no_grad():
                    p = p + torch.softmax(m(Feats[te]), 1).cpu().numpy()
            probs[te] = p / args.cv_ensemble
            print(f"fold {f + 1}/{args.k}: {(probs[te].argmax(1) == y[te]).mean():.3f}  [{time.time() - t0:.0f}s]", flush=True)
        pred = probs.argmax(1)
        top3 = (np.argsort(-probs, 1)[:, :3] == y[:, None]).any(1)
        per = defaultdict(lambda: [0, 0])
        conf = Counter()
        for yy, pp in zip(y, pred):
            per[classes[yy]][1] += 1
            per[classes[yy]][0] += int(yy == pp)
            if yy != pp:
                conf[(text_of[classes[yy]], text_of[classes[pp]])] += 1
        report.update(
            method=f"{args.k}-fold cross-validation, stratified by sentence: every recording is tested once by a model that never saw it",
            top1=float((pred == y).mean()),
            top3=float(top3.mean()),
            n=int(len(y)),
            per_class={text_of[c]: {"correct": a, "total": n} for c, (a, n) in per.items()},
            top_confusions=[{"true": a, "pred": b, "count": n} for (a, b), n in conf.most_common(20)],
        )
        print(f"CV top-1 {report['top1']:.3f}  top-3 {report['top3']:.3f}  (n={len(y)})", flush=True)
        (settings.model_path.parent / "cv_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    models = []
    for e in range(args.ensemble):
        models.append(train_one(args, X, Y, len(classes), device, seed=args.seed * 1000 + 900 + e))
        print(f"final model {e + 1}/{args.ensemble} trained  [{time.time() - t0:.0f}s]", flush=True)
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dicts": [{k: v.detach().cpu() for k, v in m.state_dict().items()} for m in models],
            "labels": [{"id": c, "text": text_of[c]} for c in classes],
            "config": {"in_dim": F.FEAT_DIM, "seq_len": F.SEQ_LEN, "d_model": args.d_model, "n_layers": args.layers, "norm": args.norm},
            "val_acc": None,
            "test_acc": report.get("top1"),
            "test_top3": report.get("top3"),
            "n_train": len(seqs),
            "n_val": 0,
            "n_test": report.get("n"),
            "eval": "cross-validation" if "top1" in report else "none",
        },
        settings.model_path,
    )
    print(f"Saved {settings.model_path} ({args.ensemble} models) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
