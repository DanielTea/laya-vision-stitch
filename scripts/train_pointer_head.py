"""Train the absolute click-position head on D2E button presses and test on unseen games.

Baselines: window center, the mean training press position, and each seen game's mean
press position (an oracle unavailable for new games). Metrics use normalized coordinates;
`hit@r` counts presses within radius r of the recorded position.
"""

import argparse
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


def gather(items):
    """Press frames from cache:split pairs -> spatial, contexts, targets, games."""
    spatial, contexts, xy, games = [], [], [], []
    for item in items:
        cache, split = item.rsplit(":", 1)
        cache = Path(cache)
        a = np.load(cache / f"{split}.npz")
        if "press_rows" not in a.files:
            raise ValueError(f"{item} has no press grids; rebuild with --spatial-press")
        rows = read(cache / f"{split}.jsonl")
        index = a["press_rows"]
        spatial.append(a["press_spatial"])
        contexts.append(a["contexts"][index])
        xy.append(a["press_xy"][index])
        games.extend(rows[i]["game"] for i in index)
    return np.concatenate(spatial), np.concatenate(contexts), np.concatenate(xy), games


def distance(a, b):
    return np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).sum(-1))


def summarize(pred, xy, games):
    d = distance(pred, xy)
    per_game = defaultdict(list)
    for g, v in zip(games, d, strict=True):
        per_game[g].append(v)
    report = {
        g: {
            "presses": len(v),
            "median_distance": float(np.median(v)),
            "hit@0.05": float(np.mean(np.asarray(v) <= 0.05)),
            "hit@0.1": float(np.mean(np.asarray(v) <= 0.1)),
        }
        for g, v in per_game.items()
    }
    report["macro"] = {
        k: float(np.mean([report[g][k] for g in per_game]))
        for k in ["median_distance", "hit@0.05", "hit@0.1"]
    }
    return report


def predict(head, spatial, contexts):
    out = []
    for s in range(0, len(spatial), 512):
        p = head.predict(mx.array(spatial[s : s + 512]), mx.array(contexts[s : s + 512]))
        mx.eval(p)
        out.append(np.asarray(p))
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--validation", nargs="+", required=True)
    p.add_argument(
        "--evaluate", nargs="+", required=True, help="cache:split pairs (e.g. held-out games)"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=20260923)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "args.json").write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2
        )
    )
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    ts, tc, txy, tg = gather(args.train)
    vs, vc, vxy, vg = gather(args.validation)
    game_mean = {g: txy[[x == g for x in tg]].mean(0) for g in set(tg)}
    head = PointerHead()
    mx.eval(head.parameters())
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.01)
    value_grad = mx.value_and_grad(lambda m, s, c, y: m.loss(s, c, y))
    # Balance games so frequent clickers do not dominate.
    by_game = defaultdict(list)
    for i, g in enumerate(tg):
        by_game[g].append(i)
    groups = list(by_game.values())
    best, best_step, history = float("inf"), 0, []
    for step in range(1, args.steps + 1):
        idx = np.array(
            [
                g[rng.integers(len(g))]
                for g in (groups[k] for k in rng.integers(0, len(groups), args.batch))
            ]
        )
        loss, grads = value_grad(head, mx.array(ts[idx]), mx.array(tc[idx]), mx.array(txy[idx]))
        optimizer.update(head, grads)
        mx.eval(head.parameters(), optimizer.state, loss)
        if step % 250 == 0 or step == args.steps:
            v = summarize(predict(head, vs, vc), vxy, vg)["macro"]["median_distance"]
            history.append(
                {"step": step, "train_loss": float(loss), "validation_median_distance": v}
            )
            print(json.dumps(history[-1]), flush=True)
            if v < best:
                best, best_step = v, step
                head.save_weights(str(args.output / "pointer_head.safetensors"))
    head.load_weights(str(args.output / "pointer_head.safetensors"))
    report = {
        "parameters": sum(v.size for _, v in tree_flatten(head.parameters())),
        "train_presses": len(tg),
        "train_games": {g: len(v) for g, v in by_game.items()},
        "selected_step": best_step,
        "history": history,
        "splits": {},
    }
    center, global_mean = np.array([0.5, 0.5]), txy.mean(0)
    for item in [*args.validation, *args.evaluate]:
        s, c, xy, games = gather([item])
        result = {
            "pointer_head": summarize(predict(head, s, c), xy, games),
            "center": summarize(np.broadcast_to(center, xy.shape), xy, games),
            "training_mean": summarize(np.broadcast_to(global_mean, xy.shape), xy, games),
        }
        if all(g in game_mean for g in games):
            result["seen_game_mean_oracle"] = summarize(
                np.stack([game_mean[g] for g in games]), xy, games
            )
        report["splits"][item] = result
        print(json.dumps({item: {k: v["macro"] for k, v in result.items()}}), flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
