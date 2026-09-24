"""Train an image-only click-position head on RADIO patches; test on games never trained on.

Training games contribute 90% of their recordings to training and 10% (by episode hash)
to validation, which selects the checkpoint. Held-out games are reported only. Baselines
are the window center and the mean training press position.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from laya_vision_stitch.pointer_head import PointerHead


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def summarize(pred, xy, games):
    d = np.sqrt(((pred - xy) ** 2).sum(-1))
    per = defaultdict(list)
    for g, v in zip(games, d, strict=True):
        per[g].append(v)
    out = {
        g: {
            "presses": len(v),
            "median_distance": float(np.median(v)),
            "hit@0.05": float(np.mean(np.array(v) <= 0.05)),
            "hit@0.1": float(np.mean(np.array(v) <= 0.1)),
        }
        for g, v in per.items()
    }
    out["macro"] = {
        k: float(np.mean([out[g][k] for g in per]))
        for k in ["median_distance", "hit@0.05", "hit@0.1"]
    }
    return out


def predict(head, features, index):
    out = []
    for s in range(0, len(index), 256):
        p = head.predict(mx.array(np.asarray(features[np.sort(index[s : s + 256])])))
        mx.eval(p)
        out.append(np.asarray(p))
    return np.concatenate(out) if out else np.zeros((0, 2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260923)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    rows = read(args.cache / "rows.jsonl")
    features = np.load(args.cache / "features.npy", mmap_mode="r")
    xy = np.load(args.cache / "xy.npy")
    games = np.array([r["game"] for r in rows])

    def is_validation(r):
        return int(hashlib.sha256(r["episode"].encode()).hexdigest(), 16) % 10 == 0

    train = np.array(
        [i for i, r in enumerate(rows) if r["split"] == "train" and not is_validation(r)]
    )
    val = np.array([i for i, r in enumerate(rows) if r["split"] == "train" and is_validation(r)])
    held = np.array([i for i, r in enumerate(rows) if r["split"] == "heldout"])
    by_game = defaultdict(list)
    for i in train:
        by_game[games[i]].append(i)
    groups = list(by_game.values())
    head = PointerHead(width=args.width, context_width=0, grid=(14, 24), features=768)
    mx.eval(head.parameters())
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.01)
    value_grad = mx.value_and_grad(lambda m, f, y: m.loss(f, None, y))
    best, best_step, history = float("inf"), 0, []
    for step in range(1, args.steps + 1):
        idx = np.sort(
            [
                g[rng.integers(len(g))]
                for g in (groups[k] for k in rng.integers(0, len(groups), args.batch))
            ]
        )
        loss, grads = value_grad(head, mx.array(np.asarray(features[idx])), mx.array(xy[idx]))
        optimizer.update(head, grads)
        mx.eval(head.parameters(), optimizer.state, loss)
        if step % 500 == 0 or step == args.steps:
            v = summarize(predict(head, features, val), xy[np.sort(val)], games[np.sort(val)])[
                "macro"
            ]["median_distance"]
            history.append(
                {"step": step, "train_loss": float(loss), "validation_median_distance": v}
            )
            print(json.dumps(history[-1]), flush=True)
            if v < best:
                best, best_step = v, step
                head.save_weights(str(args.output / "pointer_head.safetensors"))
    head.load_weights(str(args.output / "pointer_head.safetensors"))
    mean = xy[train].mean(0)
    report = {
        "parameters": sum(v.size for _, v in tree_flatten(head.parameters())),
        "train_presses": {g: len(v) for g, v in by_game.items()},
        "selected_step": best_step,
        "history": history,
        "splits": {},
    }
    for name, index in [("validation_sessions", val), ("heldout_games", held)]:
        if not len(index):
            continue
        order = np.sort(index)
        report["splits"][name] = {
            "pointer_head": summarize(predict(head, features, index), xy[order], games[order]),
            "center": summarize(np.full((len(order), 2), 0.5), xy[order], games[order]),
            "training_mean": summarize(
                np.broadcast_to(mean, (len(order), 2)), xy[order], games[order]
            ),
        }
        print(
            json.dumps({name: {k: v["macro"] for k, v in report["splits"][name].items()}}),
            flush=True,
        )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "config.json").write_text(
        json.dumps(
            {
                "pointer_head": {
                    "width": args.width,
                    "context_width": 0,
                    "grid": [14, 24],
                    "features": 768,
                },
                "encoder": "C-RADIOv3-B 384x224",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
