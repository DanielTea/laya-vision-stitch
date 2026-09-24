"""Train an elapsed-time input for the frozen P2P memory and compare gap handling.

Frames are dropped from recorded 20 FPS sequences to simulate slow live screenshots.
Conditions: reset memory at >100 ms gaps (current runtime), keep memory without timing,
and keep memory with the learned time-gap residual. Offline teacher-forced evaluation.
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
from laya_vision_stitch.time_gap_adapter import install_time_gap_adapter

STEP_SECONDS = 0.05


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load(cache, split):
    a = dict(np.load(cache / f"{split}.npz"))
    rows = read(cache / f"{split}.jsonl")
    groups = defaultdict(list)
    for i, s in enumerate(a["sequence_index"]):
        groups[int(s)].append(i)
    return rows, a, [np.array(sorted(v, key=lambda i: a["steps"][i])) for v in groups.values()]


def keep_pattern(length, rng, keep, gap_probability):
    """Indices kept from one sequence: a contiguous run, or runs separated by 1-6 frame gaps."""
    if rng.random() >= gap_probability:
        start = rng.integers(0, length - keep + 1)
        return np.arange(start, start + keep)
    kept, position = [], 0
    while len(kept) < keep and position < length:
        run = int(rng.integers(2, 7))
        kept.extend(range(position, min(length, position + run)))
        position += run + int(rng.integers(1, 7))
    kept = np.array(kept[:keep])
    return kept if len(kept) == keep else np.arange(length - keep, length)


def gaps(steps):
    dt = np.diff(steps, prepend=steps[0] - 1) * STEP_SECONDS
    return dt.astype(np.float32)


def contexts_for(policy, a, frames, dt, mode):
    """Contexts for kept frames of one sequence under a gap-handling mode."""
    images, goals, tokens = a["images"][frames], a["goals"][frames], a["tokens"][frames]
    if mode == "reset":
        out, start = [], 0
        cuts = [k for k in range(1, len(frames)) if dt[k] > 0.1] + [len(frames)]
        for end in cuts:
            sl = slice(start, end)
            out.append(
                sequence_contexts(
                    policy,
                    mx.array(images[sl])[None],
                    mx.array(goals[sl])[None],
                    mx.array(tokens[sl])[None],
                )[0]
            )
            start = end
        return mx.concatenate(out)
    residual = None
    if mode == "time_gap":
        residual = policy.time_gap_adapter(mx.array(dt))[None]
    return sequence_contexts(
        policy,
        mx.array(images)[None],
        mx.array(goals)[None],
        mx.array(tokens)[None],
        residual=residual,
    )[0]


def token_nll(policy, contexts, tokens):
    logits = policy.teacher_logits(contexts[:, None], tokens)
    return sum(
        nn.losses.cross_entropy(logit.astype(mx.float32), tokens[:, j], reduction="none")
        for j, logit in enumerate(logits)
    )


def evaluate(policy, cache, split, keep, seed):
    rows, a, groups = load(cache, split)
    rng = np.random.default_rng(seed)
    patterns = [(g, g[keep_pattern(len(g), rng, keep, 1.0)]) for g in groups]
    result = {}
    for mode in ["full_rate", "reset", "keep_memory", "time_gap"]:
        losses, predicted, kept_rows, kept_tokens, post_gap = [], [], [], [], []
        for group, frames in patterns:
            if mode == "full_rate":
                frames = group
            dt = gaps(a["steps"][frames])
            c = contexts_for(
                policy,
                a,
                frames,
                dt,
                "time_gap" if mode == "time_gap" else mode if mode == "reset" else "keep",
            )
            t = mx.array(a["tokens"][frames])
            loss = token_nll(policy, c, t)
            tokens, _ = guided_decode(policy, c)
            mx.eval(loss, tokens)
            losses.extend(np.asarray(loss).tolist())
            predicted.extend(np.asarray(tokens).tolist())
            kept_rows.extend(rows[i] for i in frames)
            kept_tokens.extend(a["tokens"][frames].tolist())
            since = np.zeros(len(frames), int)
            last = -99
            for k in range(len(frames)):
                if dt[k] > 0.1:
                    last = k
                since[k] = k - last
            post_gap.extend((since <= 2).tolist())
        truth = token_buttons(kept_tokens)
        report = evaluate_actions(
            kept_rows,
            token_buttons(predicted),
            truth,
            token_mouse(predicted),
            token_mouse(kept_tokens),
        )
        losses, post_gap = np.array(losses), np.array(post_gap)
        result[mode] = {
            "frames": len(losses),
            "mean_nll": float(losses.mean()),
            "post_gap_frames": int(post_gap.sum()),
            "post_gap_mean_nll": float(losses[post_gap].mean()) if post_gap.any() else None,
            "macro": report["macro"],
        }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--keep", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=20260923)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    runtime = LayaP2PRuntime.load(args.bundle)
    model = runtime.model
    install_time_gap_adapter(model)
    model.freeze()
    policy = model.policy
    policy.time_gap_adapter.unfreeze()
    mx.eval(policy.time_gap_adapter.parameters())
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    _, a, groups = load(args.cache, "train")
    groups = [g for g in groups if len(g) >= args.keep]

    def loss_fn(m, images, goals, tokens, dt):
        pol = m.policy
        c = sequence_contexts(pol, images, goals, tokens, residual=pol.time_gap_adapter(dt))
        b, t = tokens.shape[:2]
        return token_nll(pol, c.reshape(b * t, 1024), tokens.reshape(b * t, 8)).mean()

    value_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.01)
    history, best, best_step = [], float("inf"), 0
    for step in range(1, args.steps + 1):
        chosen = [groups[i] for i in rng.integers(0, len(groups), args.batch)]
        frames = [g[keep_pattern(len(g), rng, args.keep, 0.7)] for g in chosen]
        batch = {
            k: mx.array(np.stack([a[k][f] for f in frames])) for k in ["images", "goals", "tokens"]
        }
        dt = mx.array(np.stack([gaps(a["steps"][f]) for f in frames]))
        loss, grads = value_grad(model, batch["images"], batch["goals"], batch["tokens"], dt)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        if step % 250 == 0 or step == args.steps:
            v = evaluate(policy, args.cache, "validation", args.keep, 11)["time_gap"]["mean_nll"]
            history.append({"step": step, "train_loss": float(loss), "validation_gap_nll": v})
            print(json.dumps(history[-1]), flush=True)
            if v < best:
                best, best_step = v, step
                policy.time_gap_adapter.save_weights(
                    str(args.output / "time_gap_adapter.safetensors")
                )
    policy.time_gap_adapter.load_weights(str(args.output / "time_gap_adapter.safetensors"))
    report = {
        "trainable_parameters": trainable,
        "selected_step": best_step,
        "history": history,
        "splits": {
            s: evaluate(policy, args.cache, s, args.keep, 29)
            for s in ["validation", "test", "fresh_test"]
        },
    }
    for split, res in report["splits"].items():
        print(
            json.dumps(
                {
                    split: {
                        m: (
                            r["mean_nll"],
                            r["post_gap_mean_nll"],
                            r["macro"]["button_f1"],
                            r["macro"]["onset_f1"],
                        )
                        for m, r in res.items()
                    }
                }
            ),
            flush=True,
        )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
