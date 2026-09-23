"""Audit a BF16 policy with FP32 vision, then time fresh stitched predictions."""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from PIL import Image

from laya_vision_stitch.laya_p2p import LayaP2PRuntime
from laya_vision_stitch.p2p_adaptation import encode_action
from laya_vision_stitch.p2p_pretrained_vision import preprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--trial", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference = LayaP2PRuntime.load(args.bundle)
    candidate = LayaP2PRuntime.load(args.bundle)
    candidate.model.policy.load_weights(
        [
            (k, v.astype(mx.bfloat16))
            for k, v in tree_flatten(candidate.model.policy.parameters())
            if not k.startswith("vision.")
        ],
        strict=False,
    )
    mx.eval(candidate.model.parameters())
    rows = [json.loads(s) for s in (args.trial / "events.jsonl").read_text().splitlines()]
    if len(rows) < 240 or not all(row["applied"] for row in rows[:240]):
        raise ValueError("Precision audit requires 240 frames with applied-action feedback")
    goal = json.loads((args.trial / "config.json").read_text())["goal"]
    prepared = reference.prepare_goal(goal)
    ref_goal = reference.model.bridge(reference.model.goal_features(*prepared))
    cand_goal = candidate.model.bridge(candidate.model.goal_features(*prepared))
    states = [None, None]
    divergences, disagreements = [], 0
    for i, row in enumerate(rows[:64]):
        pixels = mx.array(preprocess(Image.open(args.trial / row["image"])))
        actual = mx.array([encode_action(row["bounded_action"])], mx.int32)
        logits = []
        for j, (runtime, text) in enumerate(((reference, ref_goal), (candidate, cand_goal))):
            out = runtime.model.policy.step(
                pixels, text=text, caches=states[j], position=i * 12, forced=actual
            )
            mx.eval(out)
            states[j] = out[2]
            logits.append(out[1])
        for a, b in zip(*logits, strict=True):
            pa = np.asarray(mx.softmax(a.astype(mx.float32)))
            pb = np.asarray(mx.softmax(b.astype(mx.float32)))
            if not np.isfinite(pb).all():
                raise FloatingPointError("Nonfinite BF16 logits")
            divergences.append(float((pa * (np.log(pa + 1e-30) - np.log(pb + 1e-30))).sum()))
            disagreements += int(pa.argmax() != pb.argmax())
    report = {
        "parity_frames": 64,
        "logit_distributions": len(divergences),
        "mean_kl": float(np.mean(divergences)),
        "max_kl": max(divergences),
        "argmax_disagreements": disagreements,
        "scope": "Controlled precision comparison on recorded frames and applied-action history. Offline timing excludes capture and dispatch. No gameplay validation.",
    }
    if report["mean_kl"] > 0.01 or disagreements / len(divergences) > 0.05:
        raise RuntimeError(f"Precision gate failed: {report}")
    report["timing"] = {}
    for name, runtime in (("fp32", reference), ("bf16_policy_fp32_vision", candidate)):
        times = []
        state = None
        for i, row in enumerate(rows[:240]):
            image = Image.open(args.trial / row["image"]).convert("RGB")
            start = time.perf_counter()
            out = runtime.model.step(
                mx.array(preprocess(image)),
                *runtime.prepare_goal(goal),
                state,
                i * 12,
                temperature=1.0,
            )
            mx.eval(out)
            state = out[2]
            times.append((time.perf_counter() - start) * 1000)
        report["timing"][name] = {
            "full_cache_samples": 40,
            "p50_ms": float(np.median(times[200:])),
            "p95_ms": float(np.percentile(times[200:], 95)),
        }
        print(json.dumps({name: report["timing"][name]}), flush=True)
    candidate.metadata.update(
        policy_compute_dtype="bfloat16",
        vision_compute_dtype="float32",
        precision_audit=report,
        deployment_eligible=False,
    )
    candidate.save(args.output / "bundle")
    loaded = LayaP2PRuntime.load(args.output / "bundle")
    first = mx.array(preprocess(Image.open(args.trial / rows[0]["image"])))
    a = candidate.model.step(first, *prepared)[0]
    b = loaded.model.step(first, *loaded.prepare_goal(goal))[0]
    if not np.array_equal(np.asarray(a), np.asarray(b)):
        raise RuntimeError("Reload action differs")
    report["reload_greedy_tokens_match"] = True
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
