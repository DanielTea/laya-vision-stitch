"""Fresh-frame stitched inference and goal/sampling diagnostics; no live inputs."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from laya_vision_stitch.laya_p2p import LayaP2PRuntime
from laya_vision_stitch.p2p_pretrained_policy import KEY_NAMES, physical_action
from laya_vision_stitch.p2p_pretrained_vision import preprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--alignment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=240)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frames = sorted(args.frames.glob("*.jpg"))[: args.steps]
    if len(frames) != args.steps or args.steps <= 205:
        raise ValueError("Need more than 205 distinct frames to time a full temporal cache")
    images = [Image.open(f).convert("RGB") for f in frames]
    runtime = LayaP2PRuntime.load(args.bundle)
    goals = [json.loads(s) for s in (args.alignment / "goals.jsonl").read_text().splitlines()]
    teacher_idx = next(i for i, r in enumerate(goals) if r["goal"] == "Attack the opponent")
    teacher = mx.array(np.load(args.alignment / "features.npz")["gemma"][teacher_idx][None])
    conditions = [
        ("no_goal_greedy", None, 0),
        ("teacher_attack_greedy", "teacher", 0),
        ("laya_attack_greedy", "Attack the opponent", 0),
        ("laya_wait_greedy", "Stand still and do nothing.", 0),
        ("no_goal_sampled", None, 1),
        ("laya_attack_sampled", "Attack the opponent", 1),
    ]
    report = {
        "bundle": str(args.bundle),
        "preprocessing": "fast_image_resize 5.1.4 Hamming interpolation, byte-parity checked",
        "scope": "Offline fresh screenshots, preprocessing, fresh Laya goal encoding and autoregressive control. No capture, input posting or gameplay-success measurement. Teacher condition is diagnostic only.",
        "input_events_sent": 0,
        "conditions": {},
    }
    for name, goal, temperature in conditions:
        mx.random.seed(20260922)
        caches, times, records = None, [], []
        counts, active_mouse = Counter(), 0
        for i, (image, frame) in enumerate(zip(images, frames, strict=True)):
            start = time.perf_counter()
            pixels = mx.array(preprocess(image))
            if goal not in (None, "teacher"):
                result = runtime.model.step(
                    pixels, *runtime.prepare_goal(goal), caches, i * 12, temperature
                )
            else:
                result = runtime.model.policy.step(
                    pixels,
                    text=teacher if goal else None,
                    caches=caches,
                    position=i * 12,
                    temperature=temperature,
                )
            tokens, logits, caches, context = result
            mx.eval(tokens, logits, caches, context)
            if not all(np.isfinite(np.asarray(x)).all() for x in logits):
                raise FloatingPointError("Nonfinite policy logits")
            action = physical_action(
                np.asarray(tokens)[0].tolist(),
                key_names=runtime.metadata.get("key_names", KEY_NAMES),
            )
            times.append(1000 * (time.perf_counter() - start))
            counts.update(action["buttons"])
            active_mouse += int(any(action["mouse_delta"]))
            records.append(
                {
                    "frame": frame.name,
                    "tokens": tokens.tolist()[0],
                    **action,
                    "latency_ms": times[-1],
                    "first_key_probabilities": mx.softmax(logits[0]).tolist()[0],
                }
            )
        result = {
            "goal": goal,
            "temperature": temperature,
            "button_counts": dict(counts),
            "nonzero_mouse_steps": active_mouse,
            "active_steps": sum(bool(r["buttons"] or any(r["mouse_delta"])) for r in records),
        }
        for label, values in (("growing_cache", times[5:200]), ("full_cache", times[200:])):
            result[label] = {
                "samples": len(values),
                "p50_ms": float(np.median(values)),
                "p95_ms": float(np.percentile(values, 95)),
            }
        report["conditions"][name] = result
        (args.output / (name + ".jsonl")).write_text(
            "".join(json.dumps(r, allow_nan=False) + "\n" for r in records)
        )
        print(json.dumps({name: result}), flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
