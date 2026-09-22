"""Diagnostic only: can frozen visual features discriminate recorded controls?"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from laya_vision_stitch.gameplay_buttons import button_metrics
from laya_vision_stitch.policy_data import read_manifest
from laya_vision_stitch.policy_training import cache_examples
from laya_vision_stitch.trainable_model import TrainableRuntime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = TrainableRuntime.load(args.bundle)
    config = runtime.module.policy_config
    rows = {
        s: [r for r in read_manifest(args.data / f"{s}.jsonl", config) if r["game"] == "Hordes.io"]
        for s in ("train", "validation")
    }
    features = {}
    for split, items in rows.items():
        examples = cache_examples(runtime, items, "artifacts/radio-feature-cache")
        vectors = []
        for _, inputs in examples:
            patches, coords = inputs[:2]
            xy = np.asarray(coords[:, :2])
            index = np.clip(((xy + 1) * 4).astype(int), 0, 7)
            cells = index[:, 1] * 8 + index[:, 0]
            matrix = np.array([cells == i for i in range(64)], dtype=np.float32)
            matrix /= np.maximum(matrix.sum(-1, keepdims=True), 1)
            normalized = nn.LayerNorm(patches.shape[-1], affine=False)(patches)
            vectors.append((mx.array(matrix) @ normalized).reshape(-1))
        features[split] = mx.stack(vectors)
        mx.eval(features[split])
    keys = sorted({b for r in rows["train"] for b in r["action"]["buttons"]})
    truth = mx.array(
        [[b in r["action"]["buttons"] for b in keys] for r in rows["train"]], mx.float32
    )
    groups = {}
    for i, r in enumerate(rows["train"]):
        groups.setdefault(tuple(r["action"]["buttons"]), []).append(i)
    groups = list(groups.values())
    model = nn.Sequential(
        nn.Linear(features["train"].shape[-1], 128), nn.GELU(), nn.Linear(128, len(keys))
    )
    rng = np.random.default_rng(71)
    optimizer = optim.AdamW(learning_rate=3e-4, weight_decay=0.1)

    def loss(m, indices):
        logits = m(features["train"][indices])
        y = truth[indices]
        return (y * 6 * mx.logaddexp(0, -logits) + (1 - y) * mx.logaddexp(0, logits)).mean()

    gradient = nn.value_and_grad(model, loss)
    report = {"scope": "Unstitched visual diagnostic, not a deployable policy", "checks": []}
    for step in range(1500):
        indices = mx.array(
            [int(rng.choice(groups[int(rng.integers(len(groups)))])) for _ in range(16)]
        )
        value, grads = gradient(model, indices)
        grads, _ = optim.clip_grad_norm(grads, 1)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        if (step + 1) % 250 == 0:
            result = {}
            for split in ("train", "validation"):
                pred = np.asarray(model(features[split])) >= 0
                result[split] = button_metrics(
                    rows[split],
                    [[key for key, on in zip(keys, row, strict=True) if on] for row in pred],
                    keys,
                )
                j = keys.index("1")
                target = np.array(["1" in r["action"]["buttons"] for r in rows[split]])
                result[split]["attack_precision"] = float(
                    (pred[:, j] & target).sum() / max(1, pred[:, j].sum())
                )
                result[split]["attack_recall"] = float((pred[:, j] & target).sum() / target.sum())
            report["checks"].append({"step": step + 1, **result})
            print(json.dumps(report["checks"][-1]), flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
