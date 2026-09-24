"""Train action-chunk heads on cached frozen P2P contexts and compare execution modes.

`flow` is a pi0-style flow-matching head; `deterministic` is the matched ablation.
Validation loss selects the checkpoint; test splits never select weights.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from laya_vision_stitch.flow_action_head import (
    ChunkHead,
    chunk_targets,
    deterministic_loss,
    deterministic_predict,
    flow_loss,
    sample,
    tokens_to_vectors,
    vectors_to_actions,
)
from laya_vision_stitch.sequence_metrics import evaluate_actions, token_buttons, token_mouse


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load(cache, split, horizon):
    a = dict(np.load(cache / f"{split}.npz"))
    rows = read(cache / f"{split}.jsonl")
    vectors = tokens_to_vectors(a["tokens"])
    targets, mask = chunk_targets(vectors, a["sequence_index"], a["steps"], horizon)
    mask = mask * a["label_complete"][:, None]
    return rows, a, targets, mask


def predict_chunks(head, contexts, kind, key, unconditional=None, scale=1.0):
    out = []
    for start in range(0, len(contexts), 512):
        c = mx.array(contexts[start : start + 512])
        if kind == "flow":
            key, sub = mx.random.split(key)
            u = None if unconditional is None else mx.array(unconditional[start : start + 512])
            x = sample(head, c, key=sub, unconditional=u, scale=scale)
        else:
            x = deterministic_predict(head, c)
        mx.eval(x)
        out.append(np.asarray(x))
    return np.concatenate(out)


def execution_report(rows, a, chunks, stride):
    """Execute each chunk open-loop for `stride` steps, re-planning every `stride` frames."""
    n, horizon = len(rows), chunks.shape[1]
    if stride > horizon:
        raise ValueError("Stride exceeds chunk horizon")
    chosen = np.zeros((n, chunks.shape[2]), np.float32)
    covered = np.zeros(n, bool)
    for i in range(n):
        step = int(a["steps"][i])
        origin = i - (step % stride)
        offset = i - origin
        chosen[i], covered[i] = chunks[origin, offset], True
    buttons, mouse = vectors_to_actions(chosen)
    truth = token_buttons(a["tokens"])
    report = evaluate_actions(rows, buttons, truth, mouse, token_mouse(a["tokens"]))
    true_rate = float(np.mean([len(t) for t in truth]))
    report["calibration"] = {
        "predicted_controls_per_frame": float(np.mean([len(b) for b in buttons])),
        "recorded_controls_per_frame": true_rate,
    }
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--kind", choices=["flow", "deterministic"], required=True)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--goal-dropout", type=float, default=0.15)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--eval-splits", nargs="+", default=["validation", "test", "fresh_test"])
    p.add_argument("--train-splits", nargs="+", default=["train"])
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--train", nargs="*", default=[], help="Extra cache:split training pairs")
    p.add_argument("--validation", help="cache:split selecting the checkpoint (default: --cache)")
    p.add_argument("--evaluate", nargs="*", default=[], help="Extra cache:split pairs to report")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "args.json").write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2
        )
    )
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    def pair(item):
        cache, split = item.rsplit(":", 1)
        return Path(cache), split

    parts = [load(args.cache, s, args.horizon) for s in args.train_splits]
    parts += [load(*pair(item), args.horizon) for item in args.train]
    ctx = np.concatenate([x[1]["contexts"] for x in parts])
    unc = np.concatenate([x[1]["unconditional_contexts"] for x in parts])
    tgt = np.concatenate([x[2] for x in parts])
    msk = np.concatenate([x[3] for x in parts])
    _, va, vt, vm = load(
        *(pair(args.validation) if args.validation else (args.cache, "validation")), args.horizon
    )
    head = ChunkHead(horizon=args.horizon, flow=args.kind == "flow")
    mx.eval(head.parameters())
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.01)

    def loss_fn(model, c, t, m, key):
        if args.kind == "flow":
            return flow_loss(model, c, t, m, key)
        return deterministic_loss(model, c, t, m)

    value_grad = mx.value_and_grad(loss_fn)
    val_key = mx.random.key(7)

    def validation_loss():
        total, count = 0.0, 0
        key = val_key
        for start in range(0, len(vt), 512):
            key, sub = mx.random.split(key)
            sl = slice(start, start + 512)
            loss = loss_fn(
                head, mx.array(va["contexts"][sl]), mx.array(vt[sl]), mx.array(vm[sl]), sub
            )
            weight = float(vm[sl].sum())
            total, count = total + float(loss) * weight, count + weight
        return total / max(count, 1)

    best, best_step, history = float("inf"), 0, []
    key = mx.random.key(args.seed)
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(ctx), args.batch)
        drop = rng.random(args.batch) < args.goal_dropout
        c = np.where(drop[:, None], unc[idx], ctx[idx])
        key, sub = mx.random.split(key)
        loss, grads = value_grad(head, mx.array(c), mx.array(tgt[idx]), mx.array(msk[idx]), sub)
        optimizer.update(head, grads)
        mx.eval(head.parameters(), optimizer.state, loss)
        if step % 250 == 0 or step == args.steps:
            v = validation_loss()
            history.append({"step": step, "train_loss": float(loss), "validation_loss": v})
            print(json.dumps(history[-1]), flush=True)
            if v < best:
                best, best_step = v, step
                head.save_weights(str(args.output / "head.safetensors"))
    head.load_weights(str(args.output / "head.safetensors"))
    report = {
        "kind": args.kind,
        "selected_step": best_step,
        "validation_loss": best,
        "history": history,
        "parameters": sum(v.size for _, v in tree_flatten(head.parameters())),
        "splits": {},
    }
    scales = [1.0, 2.0] if args.kind == "flow" and args.goal_dropout > 0 else [1.0]
    targets = [(args.cache, s) for s in args.eval_splits] + [pair(x) for x in args.evaluate]
    for cache, split in targets:
        if not (cache / f"{split}.npz").exists():
            continue
        rows, a, _, _ = load(cache, split, args.horizon)
        split = split if cache == args.cache else f"{cache.name}:{split}"
        result = {}
        repeats = args.samples if args.kind == "flow" else 1
        for scale in scales:
            for r in range(repeats):
                chunks = predict_chunks(
                    head,
                    a["contexts"],
                    args.kind,
                    mx.random.key(100 + r),
                    a["unconditional_contexts"],
                    scale,
                ).reshape(len(rows), args.horizon, -1)
                for stride in sorted({1, args.horizon // 2, args.horizon}):
                    name = f"scale_{scale:g}_stride_{stride}"
                    result.setdefault(name, []).append(execution_report(rows, a, chunks, stride))
        report["splits"][split] = result
        summary = {
            name: {
                k: float(np.mean([x["macro"][k] for x in runs]))
                for k in ["button_f1", "onset_f1", "idle_false_positive_rate"]
            }
            for name, runs in result.items()
        }
        print(json.dumps({split: summary}), flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
