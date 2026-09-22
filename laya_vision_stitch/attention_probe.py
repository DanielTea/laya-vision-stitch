"""Measure action-attention saturation before and after input normalization."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .policy_data import read_manifest
from .trainable_model import TrainableRuntime


def run(bundle, manifest, output, samples):
    runtime = TrainableRuntime.load(bundle)
    rows = read_manifest(manifest, runtime.module.policy_config)
    unique = {}
    for row in rows:
        unique.setdefault(tuple(f["sha256"] for f in row["frames"]), row)
    selected = list(unique.values())
    selected = [
        selected[i]
        for i in np.linspace(0, len(selected) - 1, min(samples, len(selected)), dtype=int)
    ]
    records = []
    for row in selected:
        h, _ = runtime.module.action_context(*runtime.features(row), *runtime.prepare(row))
        a = runtime.module.actions
        record = {
            "id": row["id"],
            "hidden_abs_max": float(mx.max(mx.abs(h))),
            "hidden_rms": float(mx.sqrt(mx.mean(h.astype(mx.float32) ** 2))),
        }
        for name, hh in [
            ("raw", h.astype(mx.float32)),
            ("normalized", mx.fast.layer_norm(h.astype(mx.float32), None, None, 1e-5)),
        ]:
            x = a.context(hh)
            q = a.action_queries[None] + a.context(hh[:, 0])[:, None]
            q = a.attention.query_proj(q)
            k = a.attention.key_proj(x)
            heads = runtime.module.policy_config.heads
            d = q.shape[-1] // heads
            logits = (
                q.reshape(1, -1, heads, d).transpose(0, 2, 1, 3)
                @ k.reshape(1, -1, heads, d).transpose(0, 2, 3, 1)
            ) / d**0.5
            p = mx.softmax(logits, axis=-1)
            record[name] = {
                "logit_abs_max": float(mx.max(mx.abs(logits))),
                "mean_max_probability": float(mx.mean(mx.max(p, axis=-1))),
                "entropy": float(mx.mean(-mx.sum(p * mx.log(mx.maximum(p, 1e-20)), axis=-1))),
            }
        records.append(record)
    result = {
        "source": str(bundle),
        "manifest": str(manifest),
        "samples": records,
        "scope": "One action query block on fixed inputs; diagnostic, not gameplay performance",
    }
    with output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(records, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=Path("artifacts/decoder-probe-004/bundle"))
    p.add_argument(
        "--manifest", type=Path, default=Path("artifacts/goal-curriculum-001/train.jsonl")
    )
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.bundle, a.manifest, a.output, a.samples)


if __name__ == "__main__":
    main()
