"""Offline latency diagnostic; never captures a screen or sends game inputs.

Run from the repository with PYTHONPATH=. .venv/bin/python scripts/profile_latency.py.
Cached-history timings assume the older frame was encoded when first received.
They still freshly encode the current frame on every prediction. Stage fences
add synchronization overhead, so compare end-to-end variants independently.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from laya_vision_stitch.policy_data import read_manifest
from laya_vision_stitch.trainable_model import TrainableRuntime


def stats(values):
    return {
        "n": len(values),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path, default=Path("artifacts/gameplay-buttons-002/bundle")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/learning-gate-002/validation.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/latency-profile-002"))
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("Sample count must be positive")
    bundle = args.bundle
    runtime = TrainableRuntime.load(bundle)
    rows = read_manifest(args.manifest, runtime.module.policy_config)[: args.samples]
    if any(len(r["frames"]) != 2 for r in rows):
        raise ValueError("This profile requires exactly two frames per example")
    model = runtime.module
    if model.policy_config.action_context_source != "encoder":
        raise ValueError("Skipping choice head requires encoder action context")
    stages, variants, errors = defaultdict(list), defaultdict(list), defaultdict(list)
    button_disagreements = defaultdict(int)

    @mx.compile
    def compiled_actions(state, batch, start):
        output = model.from_state(state, batch, start)
        return {key: value for key, value in output.items() if key != "choices"}

    def timed(name, fn):
        start = time.perf_counter()
        result = fn()
        mx.eval(result)
        stages[name].append((time.perf_counter() - start) * 1000)
        return result

    def action_only(patches, coordinates, prepared):
        # MLX is lazy: omitting choices from the evaluated output discards the
        # unused choice-head graph while preserving the action path exactly.
        output = model.from_features(patches, coordinates, *prepared)
        return {key: value for key, value in output.items() if key != "choices"}

    for index, row in enumerate([rows[0]] * 3 + rows):
        measure = index >= 3
        for label, in_memory in (("original_disk", False), ("two_frames_memory", True)):
            if in_memory:
                memory_row = dict(row)
                memory_row["frames"] = []
                for frame in row["frames"]:
                    with Image.open(frame["image"]) as image:
                        memory_row["frames"].append({**frame, "image": image.convert("RGB")})
                source = memory_row
            else:
                source = row
            start = time.perf_counter()
            reference = model(runtime.frames(source), *runtime.prepare(source))
            mx.eval(reference)
            elapsed = (time.perf_counter() - start) * 1000
            if measure:
                variants[label].append(elapsed)

        prepared = timed("tokenize", lambda: runtime.prepare(memory_row))
        frames = timed("preprocess_two_frames", lambda: runtime.frames(memory_row))
        old = timed("vision_old_frame", lambda: model.encode_frame(*frames[0]))
        new = timed("vision_current_frame", lambda: model.encode_frame(*frames[1]))
        patches, coordinates = mx.concatenate([old[0], new[0]]), mx.concatenate([old[1], new[1]])
        mx.eval(patches, coordinates)
        batch, goal_ids, start_token = prepared
        state = timed(
            "connector",
            lambda: model.connector(
                patches, coordinates, model.laya.encoder.embeddings.tok_embeddings(goal_ids)
            ),
        )
        timed("laya_and_all_heads", lambda: model.from_state(state, batch, start_token))
        timed(
            "laya_actions_without_choice_head",
            lambda: {
                k: v
                for k, v in model.from_state(state, batch, start_token).items()
                if k != "choices"
            },
        )

        for label in (
            "cached_history",
            "cached_history_action_only",
            "cached_history_compiled_laya",
        ):
            start = time.perf_counter()
            # Preprocess only the fresh frame; restore its original temporal
            # index. Historical coordinates already match this request's ages.
            fresh_row = {**memory_row, "frames": [memory_row["frames"][-1]]}
            pixels, grid, age, _ = runtime.frames(fresh_row)[0]
            fresh = model.encode_frame(pixels, grid, age, 1.0)
            features = mx.concatenate([old[0], fresh[0]])
            coords = mx.concatenate([old[1], fresh[1]])
            if label == "cached_history_compiled_laya":
                state = model.connector(
                    features, coords, model.laya.encoder.embeddings.tok_embeddings(goal_ids)
                )
                output = compiled_actions(state, batch, start_token)
            elif label == "cached_history_action_only":
                output = action_only(features, coords, prepared)
            else:
                output = model.from_features(features, coords, *prepared)
            mx.eval(output)
            elapsed = (time.perf_counter() - start) * 1000
            if measure:
                variants[label].append(elapsed)
                errors[label].append(
                    max(
                        float(mx.max(mx.abs(output[key] - reference[key])).item()) for key in output
                    )
                )
                button_disagreements[label] += int(
                    mx.sum((output["buttons"] >= 0) != (reference["buttons"] >= 0)).item()
                )
        if index % 8 == 0:
            print(f"Processed {index + 1}/{len(rows) + 3}", flush=True)
    report = {
        "bundle": str(bundle),
        "samples": len(rows),
        "warmup": 3,
        "variants": {name: stats(values) for name, values in variants.items()},
        "stages_with_sync_fences": {name: stats(values[3:]) for name, values in stages.items()},
        "output_max_abs_error": {name: max(values) for name, values in errors.items()},
        "button_threshold_disagreements": dict(button_disagreements),
        "input_events_sent": 0,
        "scope": "Offline recorded frames; excludes capture, HTTP, decoding and input posting. "
        "Not a game-running contention test. Cache variants reuse prompt tokens and "
        "one previously encoded historical frame; current frame is always encoded.",
    }
    target = args.output
    target.mkdir(exist_ok=True)
    (target / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
