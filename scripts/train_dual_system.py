"""Train a slow planner that conditions the frozen fast P2P controller with stale intent.

Planner outputs from frame t-s condition the fast policy at frame t, with s drawn from
a periodic schedule (period P, latency L) so training matches asynchronous execution.
`--features efficientnet` is the matched control using the fast path's own grid.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from laya_vision_stitch.laya_p2p import LayaP2PRuntime
from laya_vision_stitch.sequence_metrics import evaluate_actions, token_buttons, token_mouse
from laya_vision_stitch.sequence_policy import guided_decode, sequence_contexts
from laya_vision_stitch.slow_planner import SlowPlanner


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load(cache, split, features):
    a = dict(np.load(cache / f"{split}.npz"))
    rows = read(cache / f"{split}.jsonl")
    if features == "radio":
        f = np.load(cache / f"{split}-radio.npz")["patches"]
    else:
        f = a["spatial"].reshape(len(rows), 144, 112)
    groups = defaultdict(list)
    for i, s in enumerate(a["sequence_index"]):
        groups[int(s)].append(i)
    groups = [np.array(sorted(v, key=lambda i: a["steps"][i])) for v in groups.values()]
    return rows, a, f, groups


def plan_sources(frames, period, latency):
    """Index of the latest completed plan for each fast step (clipped at sequence start)."""
    k = np.arange(len(frames))
    source = ((k - latency) // period) * period
    return frames[np.clip(source, 0, None)]


def token_nll(policy, contexts, tokens):
    logits = policy.teacher_logits(contexts[:, None], tokens)
    return sum(
        nn.losses.cross_entropy(logit.astype(mx.float32), tokens[:, j], reduction="none")
        for j, logit in enumerate(logits)
    )


def contexts(policy, planner, a, feats, frames, sources):
    images = mx.array(a["images"][frames])[None]
    goals = mx.array(a["goals"][frames])[None]
    tokens = mx.array(a["tokens"][frames])[None]
    if planner is None:
        return sequence_contexts(policy, images, goals, tokens)[0]
    language, image = planner(mx.array(feats[sources]), mx.array(a["goals"][sources]))
    return sequence_contexts(
        policy, images, goals, tokens, residual=image[None], language=language[None]
    )[0]


def evaluate(policy, planner, cache, split, features, schedule, shuffle=False):
    rows, a, feats, groups = load(cache, split, features)
    period, latency = schedule
    rng = np.random.default_rng(5)
    losses, predicted, used_rows, used_tokens = [], [], [], []
    for n, g in enumerate(groups):
        sources = plan_sources(g, period, latency)
        if shuffle:
            other = groups[(n + 1 + rng.integers(len(groups) - 1)) % len(groups)]
            sources = other[np.clip(sources - g[0], 0, len(other) - 1)]
        c = contexts(policy, planner, a, feats, g, sources)
        t = mx.array(a["tokens"][g])
        loss = token_nll(policy, c, t)
        tokens, _ = guided_decode(policy, c)
        mx.eval(loss, tokens)
        losses.extend(np.asarray(loss).tolist())
        predicted.extend(np.asarray(tokens).tolist())
        used_rows.extend(rows[i] for i in g)
        used_tokens.extend(a["tokens"][g].tolist())
    report = evaluate_actions(
        used_rows,
        token_buttons(predicted),
        token_buttons(used_tokens),
        token_mouse(predicted),
        token_mouse(used_tokens),
    )
    return {"mean_nll": float(np.mean(losses)), "macro": report["macro"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--features", choices=["radio", "efficientnet"], default="radio")
    p.add_argument("--max-period", type=int, default=4)
    p.add_argument("--max-latency", type=int, default=2)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--window", type=int, default=20)
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
    model = LayaP2PRuntime.load(args.bundle).model
    model.freeze()
    policy = model.policy
    _, a, feats, groups = load(args.cache, "train", args.features)
    planner = SlowPlanner(feature_width=feats.shape[-1])
    mx.eval(planner.parameters())
    parameters = sum(v.size for _, v in tree_flatten(planner.parameters()))

    def loss_fn(pl, images, goals, tokens, features, plan_goals):
        b, t = tokens.shape[:2]
        language, image = pl(
            features.reshape(b * t, *features.shape[2:]), plan_goals.reshape(b * t, -1)
        )
        c = sequence_contexts(
            policy,
            images,
            goals,
            tokens,
            residual=image.reshape(b, t, 1024),
            language=language.reshape(b, t, 1024),
        )
        return token_nll(policy, c.reshape(b * t, 1024), tokens.reshape(b * t, 8)).mean()

    value_grad = nn.value_and_grad(planner, loss_fn)
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.01)
    schedule_val = (args.max_period, 1)
    history, best, best_step = [], float("inf"), 0
    for step in range(1, args.steps + 1):
        frames, sources = [], []
        for g in (groups[i] for i in rng.integers(0, len(groups), args.batch)):
            start = rng.integers(0, len(g) - args.window + 1)
            f = g[start : start + args.window]
            period = int(rng.integers(1, args.max_period + 1))
            latency = int(rng.integers(0, args.max_latency + 1))
            s = plan_sources(g, period, latency)[start : start + args.window]
            frames.append(f)
            sources.append(s)
        frames, sources = np.stack(frames), np.stack(sources)
        loss, grads = value_grad(
            planner,
            mx.array(a["images"][frames]),
            mx.array(a["goals"][frames]),
            mx.array(a["tokens"][frames]),
            mx.array(feats[sources]),
            mx.array(a["goals"][sources]),
        )
        optimizer.update(planner, grads)
        mx.eval(planner.parameters(), optimizer.state, loss)
        if step % 250 == 0 or step == args.steps:
            v = evaluate(policy, planner, args.cache, "validation", args.features, schedule_val)
            history.append(
                {"step": step, "train_loss": float(loss), "validation_nll": v["mean_nll"]}
            )
            print(json.dumps(history[-1]), flush=True)
            if v["mean_nll"] < best:
                best, best_step = v["mean_nll"], step
                planner.save_weights(str(args.output / "planner.safetensors"))
    planner.load_weights(str(args.output / "planner.safetensors"))
    report = {
        "parameters": parameters,
        "selected_step": best_step,
        "history": history,
        "splits": {},
    }
    for split in ["validation", "test", "fresh_test"]:
        res = {"no_planner": evaluate(policy, None, args.cache, split, args.features, (1, 0))}
        for name, schedule in {
            "fresh_plan": (1, 0),
            "period4_latency1": (4, 1),
            "period8_latency2": (8, 2),
        }.items():
            res[name] = evaluate(policy, planner, args.cache, split, args.features, schedule)
        res["shuffled_plan_features"] = evaluate(
            policy, planner, args.cache, split, args.features, (1, 0), shuffle=True
        )
        report["splits"][split] = res
        print(
            json.dumps(
                {
                    split: {
                        k: (
                            round(v["mean_nll"], 4),
                            round(v["macro"]["button_f1"], 4),
                            round(v["macro"]["onset_f1"], 4),
                        )
                        for k, v in res.items()
                    }
                }
            ),
            flush=True,
        )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
