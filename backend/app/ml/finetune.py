"""Fast fine-tuning: keep the trained model and teach it only what is new.

    python -m app.ml.finetune            # about 30 seconds

What it does
- Finds recordings the deployed model has not seen (compared with models/trained_files.json).
- Loads the existing 5-model ensemble and continues training from its weights, at a low learning rate, for a few epochs.
- A new sentence is added to the model only when it has at least MIN_NEW_TAKES new recordings. Its output row is added to the
  classifier head; the old rows are kept.
- To avoid "forgetting" the old sentences, each pass also replays the old recordings (they are few, so this is still fast).
  Training on the new takes alone would make the model drift toward them and get worse on everything else.
- If there are at least 10 new takes, about a fifth are held out first. Their accuracy is measured before and after, so you
  get an honest "did it help on takes it never trained on" number.

A full retrain from scratch (`python -m app.ml.train_fast`) is still the way to get a fresh cross-validated accuracy.
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
from . import features as F
from .dataset import ACTIVE_FILE, all_paths, load_all, load_manifest, load_sentences, new_recordings, save_manifest
from .gpu_aug import GpuAugment, prep
from .model import SignTransformer

MIN_NEW_TAKES = 8  # a brand new sentence needs this many recordings before it is added
NEW_OVERSAMPLE = 6  # new (the person's own) recordings are shown this many times per epoch, old ones once


def _ensemble_probs(models, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return sum(torch.softmax(m.eval()(x), 1) for m in models).cpu().numpy() / len(models)


def _expand_head(model: SignTransformer, n_old: int, n_new: int, device: str) -> SignTransformer:
    """Add output rows for new sentences, keeping every trained row."""
    if n_new == n_old:
        return model
    d = model.head.in_features
    head = nn.Linear(d, n_new).to(device)
    with torch.no_grad():
        head.weight.normal_(0, 0.02)
        head.bias.zero_()
        head.weight[:n_old] = model.head.weight
        head.bias[:n_old] = model.head.bias
    model.head = head
    return model


def finetune(epochs: int = 40, lr: float = 3e-4, seed: int = 0) -> dict:
    t0 = time.time()
    rnd = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(settings.model_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    F.NORM_MODE = cfg.get("norm", "frame")
    old_labels = ck["labels"]
    old_ids = [l["id"] for l in old_labels]
    manifest = load_manifest()
    if not manifest["files"]:
        return {"status": "no_manifest", "message": "Run one full training first, so I know what the model has already learned."}
    seen = set(manifest["files"])

    sentences = load_sentences()
    text_of = {s["id"]: s["text"] for s in sentences}
    seqs, labels, _ = load_all()
    paths = all_paths()
    usable = [i for i in range(len(seqs)) if len(seqs[i]) >= 8]
    # new = added after the model was last trained (old corpus files that were left out on purpose are NOT new)
    fresh = new_recordings(set(old_ids))
    fresh_paths = set()
    for s in sentences:
        for f in sorted((settings.data_dir / "raw" / s["id"]).glob("*.npy")) + sorted((settings.data_dir / "external" / s["id"]).glob("*.npy")):
            rel = f.relative_to(settings.data_dir).as_posix()
            if rel not in seen and f.stat().st_mtime > float(manifest.get("trained_at", 0) or 0):
                fresh_paths.add(rel)
    new_idx = [i for i in usable if paths[i] in fresh_paths]
    if not new_idx:
        return {"status": "nothing_new", "message": "There are no new recordings since the model was last trained."}

    by_label = defaultdict(list)
    for i in new_idx:
        by_label[labels[i]].append(i)
    added = [l for l, ix in by_label.items() if l not in old_ids and len(ix) >= MIN_NEW_TAKES]
    skipped = {l: len(ix) for l, ix in by_label.items() if l not in old_ids and l not in added}
    classes = old_ids + added
    cidx = {c: i for i, c in enumerate(classes)}
    new_use = [i for i in new_idx if labels[i] in cidx]
    if not new_use:
        need = ", ".join(f"{text_of.get(l, l)} ({n} of {MIN_NEW_TAKES})" for l, n in skipped.items())
        return {"status": "too_few", "message": f"New sentences need at least {MIN_NEW_TAKES} takes each before they are added. So far: {need}."}

    # hold out about a fifth of the new takes (only if there are enough) to measure the effect honestly
    held: list[int] = []
    if len(new_use) >= 10:
        for l, ix in defaultdict(list, {l: [i for i in new_use if labels[i] == l] for l in {labels[i] for i in new_use}}).items():
            ix = ix[:]
            rnd.shuffle(ix)
            if len(ix) >= 3:
                held += ix[: max(1, len(ix) // 5)]
    train_new = [i for i in new_use if i not in set(held)]
    replay = [i for i in usable if paths[i] in seen and labels[i] in cidx]

    def X(ix):
        return torch.from_numpy(np.stack([prep(seqs[i]) for i in ix])).to(device) if ix else None

    def Fe(ix):
        return torch.from_numpy(np.stack([F.featurize(F.normalize(seqs[i])) for i in ix])).to(device) if ix else None

    # models: the existing ensemble, rebuilt from its weights
    models = []
    for sd in ck["state_dicts"]:
        m = SignTransformer(F.FEAT_DIM, len(old_ids), d_model=cfg.get("d_model", 128), n_layers=cfg.get("n_layers", 3), n_heads=4, dropout=0.3).to(device)
        m.load_state_dict(sd)
        models.append(m.eval())

    held_y = torch.tensor([cidx[labels[i]] for i in held], device=device) if held else None
    held_f = Fe(held)
    before = None
    if held:
        keep = [k for k, i in enumerate(held) if labels[i] in old_ids]  # old model cannot know brand-new sentences
        if keep:
            p = _ensemble_probs(models, held_f[keep])
            before = float((p.argmax(1) == held_y[keep].cpu().numpy()).mean())

    models = [_expand_head(m, len(old_ids), len(classes), device) for m in models]
    Xtr_new, Xtr_old = X(train_new), X(replay)
    y_new = torch.tensor([cidx[labels[i]] for i in train_new], device=device)
    y_old = torch.tensor([cidx[labels[i]] for i in replay], device=device)
    xs = [Xtr_new] * NEW_OVERSAMPLE + ([Xtr_old] if Xtr_old is not None else [])
    ys = [y_new] * NEW_OVERSAMPLE + ([y_old] if Xtr_old is not None else [])
    XA, YA = torch.cat([x for x in xs if x is not None]), torch.cat(ys)

    aug = GpuAugment(device, mirror_p=0.0)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    steps_per = (len(XA) + 63) // 64

    def train(model, X_, Y_, ep, lr_):
        opt = torch.optim.AdamW(model.parameters(), lr=lr_, weight_decay=0.2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr_, total_steps=ep * ((len(X_) + 63) // 64), pct_start=0.15)
        for _ in range(ep):
            model.train()
            perm = torch.randperm(len(X_), device=device)
            for i in range(0, len(X_), 64):
                bi = perm[i : i + 64]
                loss = loss_fn(model(aug(X_[bi])), Y_[bi])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
        return model.eval()

    for k, m in enumerate(models):
        torch.manual_seed(seed + k)
        train(m, XA, YA, epochs, lr)

    after = None
    if held:
        p = _ensemble_probs(models, held_f)
        after = float((p.argmax(1) == held_y.cpu().numpy()).mean())
        # now let the model learn from the held-out takes as well (a short pass)
        XH = X(held)
        X2 = torch.cat([XA, XH, XH])
        Y2 = torch.cat([YA, held_y, held_y])
        for m in models:
            train(m, X2, Y2, max(6, epochs // 6), lr * 0.5)

    # save
    labels_out = old_labels + [{"id": c, "text": text_of.get(c, c)} for c in added]
    ck["state_dicts"] = [{k: v.detach().cpu() for k, v in m.state_dict().items()} for m in models]
    ck["labels"] = labels_out
    ck["n_train"] = int(ck.get("n_train", 0)) + len(new_use)
    torch.save(ck, settings.model_path)
    entry = {
        "at": time.strftime("%Y-%m-%d %H:%M"),
        "new_takes": len(new_use),
        "added_sentences": [text_of.get(c, c) for c in added],
        "skipped_too_few": {text_of.get(l, l): n for l, n in skipped.items()},
        "held_out": len(held),
        "held_out_accuracy_before": before,
        "held_out_accuracy_after": after,
        "epochs": epochs,
        "seconds": round(time.time() - t0, 1),
    }
    save_manifest(list(seen) + [paths[i] for i in new_use], manifest.get("history", []) + [entry])
    # new sentences become part of the model's active set (and so of the Sign library)
    if added and ACTIVE_FILE.exists():
        act = json.loads(ACTIVE_FILE.read_text(encoding="utf-8"))
        act["ids"] = list(dict.fromkeys(list(act.get("ids", [])) + added))
        ACTIVE_FILE.write_text(json.dumps(act, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"status": "ok", **entry}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    res = finetune(args.epochs, args.lr)
    print("RESULT " + json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
