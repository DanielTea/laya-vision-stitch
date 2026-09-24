"""Train a fovea fusion (central high-resolution crop) through the frozen P2P policy.

A zero-initialized residual adds the crop's frozen image token to the full-frame token.
`--source duplicate` is the capacity control (the full-frame token again, no new pixels);
`--source none` trains only decoder LoRA through the same loop.
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
from laya_vision_stitch.p2p_adaptation import install_control_adapter
from laya_vision_stitch.sequence_metrics import (
    baselines,
    evaluate_actions,
    token_buttons,
    token_mouse,
)
from laya_vision_stitch.sequence_policy import guided_decode, sequence_contexts


class FoveaFusion(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.norm = nn.LayerNorm(1024, affine=False)
        self.hidden = nn.Linear(1024, hidden)
        self.output = nn.Linear(hidden, 1024)
        self.output.weight = mx.zeros_like(self.output.weight)
        self.output.bias = mx.zeros_like(self.output.bias)

    def __call__(self, crop):
        return self.output(nn.silu(self.hidden(self.norm(crop))))


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load(cache, split):
    a = dict(np.load(cache / f"{split}.npz"))
    groups = defaultdict(list)
    for i, s in enumerate(a["sequence_index"]):
        groups[int(s)].append(i)
    groups = [np.array(sorted(v, key=lambda i: a["steps"][i])) for v in groups.values()]
    return read(cache / f"{split}.jsonl"), a, groups


def extra_input(a, frames, source):
    if source == "fovea":
        return a["fovea_images"][frames]
    return a["images"][frames]


def token_nll(policy, contexts, tokens):
    logits = policy.teacher_logits(contexts[:, None], tokens)
    return sum(
        nn.losses.cross_entropy(logit.astype(mx.float32), tokens[:, j], reduction="none")
        for j, logit in enumerate(logits)
    )


def forward(model, fusion, a, frames, source):
    policy = model.policy
    images = mx.array(a["images"][frames])
    residual = None if source == "none" else fusion(mx.array(extra_input(a, frames, source)))
    return sequence_contexts(
        policy,
        images,
        mx.array(a["goals"][frames]),
        mx.array(a["tokens"][frames]),
        residual=residual,
    )


def evaluate(model, fusion, cache, split, source, samples=2):
    rows, a, groups = load(cache, split)
    losses, greedy, sampled = [], [], [[] for _ in range(samples)]
    order = []
    key = mx.random.key(9)
    for g in groups:
        c = forward(model, fusion, a, g[None], source)[0]
        t = mx.array(a["tokens"][g])
        loss = token_nll(model.policy, c, t)
        tokens, _ = guided_decode(model.policy, c)
        drawn = []
        for s in range(samples):
            key, sub = mx.random.split(key)
            drawn.append(guided_decode(model.policy, c, temperature=1.0, key=sub)[0])
        mx.eval(loss, tokens, *drawn)
        losses.extend(np.asarray(loss).tolist())
        greedy.extend(np.asarray(tokens).tolist())
        for s in range(samples):
            sampled[s].extend(np.asarray(drawn[s]).tolist())
        order.extend(g.tolist())
    rows = [rows[i] for i in order]
    truth, mouse = token_buttons(a["tokens"][order]), token_mouse(a["tokens"][order])

    def score(tokens):
        return evaluate_actions(rows, token_buttons(tokens), truth, token_mouse(tokens), mouse)

    return {
        "mean_nll": float(np.mean(losses)),
        "greedy": score(greedy),
        "sampled": [score(s) for s in sampled],
        **baselines(rows, truth, mouse),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", choices=["fovea", "duplicate", "none"], required=True)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
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
    install_control_adapter(model, rank=args.rank)
    fusion = FoveaFusion()
    mx.eval(fusion.parameters())
    _, a, groups = load(args.cache, "train")
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    trainable += (
        0 if args.source == "none" else sum(v.size for _, v in tree_flatten(fusion.parameters()))
    )

    class Joint(nn.Module):
        def __init__(self):
            super().__init__()
            self.model, self.fusion = model, fusion

    joint = Joint()

    def loss_fn(j, frames):
        c = forward(j.model, j.fusion, a, frames, args.source)
        b, t = frames.shape
        return token_nll(
            j.model.policy, c.reshape(b * t, 1024), mx.array(a["tokens"][frames]).reshape(b * t, 8)
        ).mean()

    value_grad = nn.value_and_grad(joint, loss_fn)
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.01)
    val = evaluate(model, fusion, args.cache, "validation", args.source, samples=0)["mean_nll"]
    best, best_step, history = val, 0, [{"step": 0, "validation_nll": val}]
    joint.save_weights(str(args.output / "adapter.safetensors"))
    for step in range(1, args.steps + 1):
        frames = []
        for g in (groups[i] for i in rng.integers(0, len(groups), args.batch)):
            start = rng.integers(0, len(g) - args.window + 1)
            frames.append(g[start : start + args.window])
        loss, grads = value_grad(joint, np.stack(frames))
        optimizer.update(joint, grads)
        mx.eval(joint.parameters(), optimizer.state, loss)
        if step % 250 == 0 or step == args.steps:
            v = evaluate(model, fusion, args.cache, "validation", args.source, samples=0)[
                "mean_nll"
            ]
            history.append({"step": step, "train_loss": float(loss), "validation_nll": v})
            print(json.dumps(history[-1]), flush=True)
            if v < best:
                best, best_step = v, step
                joint.save_weights(str(args.output / "adapter.safetensors"))
    joint.load_weights(str(args.output / "adapter.safetensors"))
    report = {
        "source": args.source,
        "trainable_parameters": trainable,
        "selected_step": best_step,
        "history": history,
        "splits": {
            s: evaluate(model, fusion, args.cache, s, args.source) for s in ["validation", "test"]
        },
    }
    for split, r in report["splits"].items():
        print(
            json.dumps(
                {
                    split: {
                        "nll": round(r["mean_nll"], 4),
                        "greedy_f1": round(r["greedy"]["macro"]["button_f1"], 4),
                        "greedy_onset": round(r["greedy"]["macro"]["onset_f1"], 4),
                        "sampled_onset": round(
                            float(np.mean([x["macro"]["onset_f1"] for x in r["sampled"]])), 4
                        ),
                        "repeat_f1": round(r["repeat_previous"]["macro"]["button_f1"], 4),
                    }
                }
            ),
            flush=True,
        )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
