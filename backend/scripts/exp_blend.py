"""Accuracy experiment: classical models on pooled motion features, blended with the neural ensemble.

Uses exactly the same held-out split as app.ml.train (seed 42), so numbers are comparable.

    python scripts/exp_blend.py --nn-probs <npz saved by train.py --save-probs>
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.dataset import load_all, load_sentences  # noqa: E402
from app.ml.features import featurize, normalize  # noqa: E402
from app.ml.train import MIN_FRAMES, augment, split_groups  # noqa: E402


def pooled(x: np.ndarray, segs: int = 8) -> np.ndarray:
    """(64, F) -> temporal pyramid: mean over `segs` segments, plus global std."""
    parts = np.array_split(x, segs, axis=0)
    return np.concatenate([p.mean(0) for p in parts] + [x.std(0)])


def top3(p, y):
    return float((np.argsort(-p, 1)[:, :3] == y[:, None]).any(1).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn-probs", default="")
    ap.add_argument("--aug", type=int, default=12)
    args = ap.parse_args()

    seed = 42
    rng = random.Random(seed)
    np.random.seed(seed)
    seqs, labels, groups = load_all()
    from collections import Counter

    counts = Counter(labels)
    sentences = load_sentences()
    class_ids = [s["id"] for s in sentences if counts.get(s["id"], 0) >= 2]
    idx = {c: i for i, c in enumerate(class_ids)}
    keep = [i for i, lab in enumerate(labels) if lab in idx and len(seqs[i]) >= MIN_FRAMES]
    seqs = [normalize(seqs[i]) for i in keep]
    labels = [labels[i] for i in keep]
    groups = [groups[i] for i in keep]
    part = split_groups(labels, groups, rng, 0.15, 0.15)
    tr = [(s, idx[l]) for s, l, p in zip(seqs, labels, part) if p == "train"]
    va = [(s, idx[l]) for s, l, p in zip(seqs, labels, part) if p == "val"]
    te = [(s, idx[l]) for s, l, p in zip(seqs, labels, part) if p == "test"]
    print(len(tr), len(va), len(te), "classes", len(class_ids), flush=True)

    r2 = random.Random(7)
    Xtr, ytr = [], []
    for s, y in tr:
        Xtr.append(pooled(featurize(s)))
        ytr.append(y)
        for _ in range(args.aug):
            Xtr.append(pooled(augment(s, r2)))
            ytr.append(y)
    Xtr, ytr = np.stack(Xtr), np.array(ytr)
    Xte = np.stack([pooled(featurize(s)) for s, _ in te])
    yte = np.array([y for _, y in te])
    Xva = np.stack([pooled(featurize(s)) for s, _ in va])
    yva = np.array([y for _, y in va])
    print("features", Xtr.shape, flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s, Xva_s = sc.transform(Xtr), sc.transform(Xte), sc.transform(Xva)

    results = {}
    t = time.time()
    for C in (3.0,):
        m = SVC(C=C, kernel="rbf", gamma="scale", probability=True, random_state=0).fit(Xtr_s, ytr)
        p = m.predict_proba(Xte_s)
        full = np.zeros((len(yte), len(class_ids)))
        full[:, m.classes_] = p
        va_acc = float((m.predict(Xva_s) == yva).mean())
        results[f"svm_C{C}"] = (full, va_acc)
        print(f"SVM C={C}: val {va_acc:.3f}  TEST top1 {float((full.argmax(1) == yte).mean()):.3f}  top3 {top3(full, yte):.3f}  ({time.time() - t:.0f}s)", flush=True)

    t = time.time()
    lr = LogisticRegression(C=0.05, max_iter=400).fit(Xtr_s, ytr)
    p = lr.predict_proba(Xte_s)
    full = np.zeros((len(yte), len(class_ids)))
    full[:, lr.classes_] = p
    results["logreg"] = (full, float((lr.predict(Xva_s) == yva).mean()))
    print(f"LogReg: val {results['logreg'][1]:.3f}  TEST top1 {float((full.argmax(1) == yte).mean()):.3f}  top3 {top3(full, yte):.3f}  ({time.time() - t:.0f}s)", flush=True)

    if args.nn_probs:
        z = np.load(args.nn_probs)
        nn = z["probs"]
        assert (z["y"] == yte).all(), "split mismatch between NN run and this script"
        print(f"NN ensemble: TEST top1 {float((nn.argmax(1) == yte).mean()):.3f}  top3 {top3(nn, yte):.3f}", flush=True)
        best_name = max(results, key=lambda k: results[k][1])  # pick by VALIDATION accuracy, not test
        sk = results[best_name][0]
        print("blend with", best_name, "(chosen on validation)", flush=True)
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            b = w * sk + (1 - w) * nn
            print(f"  w_classical={w:.2f}: top1 {float((b.argmax(1) == yte).mean()):.3f}  top3 {top3(b, yte):.3f}", flush=True)


if __name__ == "__main__":
    main()
