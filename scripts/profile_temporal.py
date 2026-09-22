"""Offline fresh-frame and per-stage timing for the streaming adapter."""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

from laya_vision_stitch.policy_data import read_manifest
from laya_vision_stitch.temporal_adapter import spatial_pool
from laya_vision_stitch.temporal_runtime import TemporalRuntime
from laya_vision_stitch.visual_action_adapter import encode_action_history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--clips", type=int, default=4)
    args = p.parse_args()
    if args.clips < 1:
        p.error("Clip count must be positive")
    stream = TemporalRuntime.load(args.bundle)
    runtime, stages = stream.runtime, defaultdict(list)
    groups = {}
    for row in read_manifest(args.manifest, runtime.module.policy_config):
        groups.setdefault(row["sequence"], []).append(row)
    sequences = list(groups.values())
    selected = np.linspace(0, len(sequences) - 1, min(args.clips, len(sequences)), dtype=int)
    rows = [row for i in selected for row in sequences[i]]

    def timed(name, fn, measure):
        started = time.perf_counter()
        result = fn()
        mx.eval(result)
        if measure:
            stages[name].append((time.perf_counter() - started) * 1000)
        return result

    state, last_sequence = None, None
    for i, row in enumerate([rows[0]] * 3 + rows):
        measure = i >= 3
        prediction = stream.predict(
            row,
            session_id=row["sequence"],
            timestamp_seconds=row["timestamp_seconds"],
            reset=i <= 3,
        )
        if last_sequence != row["sequence"] or i <= 3:
            state = None
        last_sequence = row["sequence"]
        if measure:
            stages["complete_fresh_image_path"].append(prediction["image_to_outputs_ms"])
        frames = timed("image_load_preprocess", lambda: runtime.frames(row), measure)
        patches, coords = timed("vision", lambda: runtime.module.encode_frame(*frames[0]), measure)
        prepared = timed(
            "prompt", lambda: runtime.prepare({**row, "previous_actions": []}), measure
        )
        context = timed(
            "connector_laya",
            lambda: runtime.module.action_context(patches, coords, *prepared)[0],
            measure,
        )
        visual = timed("spatial_pool", lambda: spatial_pool(patches, coords), measure)
        start = prepared[2]
        language = mx.stack(
            [
                context[:, 0],
                context[:, start : start + runtime.module.policy_config.visual_slots].mean(1),
            ],
            1,
        )[0]
        history = encode_action_history(row, runtime.module.policy_config)
        _, state = timed(
            "temporal_adapter",
            lambda: runtime.module.temporal_actions(
                visual[None, None], language[None, None], history[:, None], state
            ),
            measure,
        )
        if (i + 1) % 8 == 0:
            print(f"Profiled {i + 1}/{len(rows) + 3}", flush=True)
    report = {
        "bundle": args.bundle,
        "samples": len(rows),
        "sequences": [sequences[i][0]["sequence"] for i in selected],
        "games": sorted({r["game"] for r in rows}),
        "warmups": 3,
        "scope": "Offline. Complete path includes fresh disk image, current vision, Laya, memory and decode; excludes capture/input/game. Per-stage fences add overhead.",
        "stages": {
            k: {"p50_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95))}
            for k, v in stages.items()
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
