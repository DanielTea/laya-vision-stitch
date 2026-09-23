"""Capacity diagnostic on 32 training images; never a gameplay qualification."""

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
from laya_vision_stitch.p2p_adaptation import install_control_adapter, install_visual_adapter
from scripts.train_p2p_control_adapter import frozen_hash
from scripts.train_p2p_visual_adapter import context, digest, evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=800)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    identity = json.loads((args.cache / "identity.json").read_text())
    if identity["weights"] != digest(args.bundle / "model.safetensors"):
        raise ValueError("Feature cache and frozen model differ")
    mx.random.seed(20260924)
    rng = np.random.default_rng(20260924)
    all_rows = [json.loads(s) for s in (args.cache / "train.jsonl").read_text().splitlines()]
    groups = defaultdict(list)
    for i, r in enumerate(all_rows):
        if r["game"] == "Hordes.io":
            groups[tuple(r["action"]["buttons"])].append(i)
    indices = [int(i) for group in groups.values() for i in rng.choice(group, 4, replace=False)]
    rows = [all_rows[i] for i in indices]
    source = np.load(args.cache / "train.npz")
    arrays = {k: source[k][indices] for k in source.files}
    split = rows, arrays
    x, g, y = [mx.array(arrays[k]) for k in ("images", "goals", "tokens")]
    runtime = LayaP2PRuntime.load(args.bundle)
    install_control_adapter(runtime.model, rank=16)
    install_visual_adapter(runtime.model, bottleneck=64)
    model = runtime.model
    fingerprint = frozen_hash(model)
    optimizer = optim.AdamW(learning_rate=0.001, weight_decay=0.001)

    def loss_fn(m, idx):
        logits = m.policy.teacher_logits(context(m.policy, x[idx], g[idx]), y[idx])
        weights = [2.0, 0.25, 0.25, 0.25, 1.0, 0.25, 0.5, 0.5]
        return sum(
            w * nn.losses.cross_entropy(v, y[idx, i], reduction="mean")
            for i, (w, v) in enumerate(zip(weights, logits, strict=True))
        ) / sum(weights)

    value_grad = nn.value_and_grad(model, loss_fn)
    history = []
    for step in range(1, args.steps + 1):
        idx = mx.array(rng.choice(len(rows), 8, replace=False))
        loss, grad = value_grad(model, idx)
        grad, norm = optim.clip_grad_norm(grad, 1.0)
        optimizer.update(model, grad)
        mx.eval(loss, norm, model.trainable_parameters(), optimizer.state)
        if not np.isfinite(float(loss)):
            raise FloatingPointError("Nonfinite diagnostic loss")
        if step % 100 == 0 or step == args.steps:
            m = evaluate(model.policy, split)["Hordes.io"]
            history.append({"step": step, "loss": float(loss), "fit": m})
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(loss),
                        "f1": m["button_f1"],
                        "exact": m["exact_button_set_accuracy"],
                    }
                ),
                flush=True,
            )
            if m["button_f1"] >= 0.95 and m["exact_button_set_accuracy"] >= 0.9:
                break
    fitted = evaluate(model.policy, split)["Hordes.io"]
    perm = rng.permutation(len(rows))
    shuffled = evaluate(model.policy, split, permutation=perm)["Hordes.io"]
    unchanged = frozen_hash(model) == fingerprint
    if not unchanged:
        raise RuntimeError("Frozen parent changed")
    mx.save_safetensors(
        str(args.output / "diagnostic-adapters.safetensors"),
        dict(tree_flatten(model.trainable_parameters())),
    )
    report = {
        "examples": len(rows),
        "row_ids": [r["id"] for r in rows],
        "history": history,
        "fit": fitted,
        "shuffled_images": shuffled,
        "parents_unchanged": unchanged,
        "deployment_eligible": False,
        "scope": "Training-set memorization/capacity diagnostic only. No held-out performance or gameplay claim. No inputs sent.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "fit": fitted["button_f1"],
                "shuffle": shuffled["button_f1"],
                "parents_unchanged": unchanged,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
