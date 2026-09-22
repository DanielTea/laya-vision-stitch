"""Full public prediction API timing on distinct, freshly encoded screenshots."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from laya_vision_stitch.policy_data import read_manifest
from laya_vision_stitch.trainable_model import TrainableRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=80)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("Positive sample count required")
    runtime = TrainableRuntime.load(args.bundle)
    rows = read_manifest(args.manifest, runtime.module.policy_config)
    indices = np.random.default_rng(20260922).choice(
        len(rows), min(args.samples, len(rows)), replace=False
    )
    samples = []
    for index in indices:
        row = dict(rows[int(index)])
        frames = []
        for frame in row["frames"]:
            with Image.open(frame["image"]) as source:
                frames.append({**frame, "image": source.convert("RGB")})
        row["frames"] = frames
        samples.append(row)
    for _ in range(5):
        runtime.predict(samples[0])
    elapsed, model, outputs = [], [], []
    for row in samples:
        start = time.perf_counter()
        output = runtime.predict(row)
        elapsed.append(1000 * (time.perf_counter() - start))
        model.append(output["image_to_outputs_ms"])
        outputs.append({"id": row["id"], "buttons": output["buttons"], "latency_ms": elapsed[-1]})
    report = {
        "bundle": str(args.bundle),
        "manifest": str(args.manifest),
        "samples": len(samples),
        "fresh_prediction_p50_ms": float(np.median(elapsed)),
        "fresh_prediction_p95_ms": float(np.percentile(elapsed, 95)),
        "internal_model_p50_ms": float(np.median(model)),
        "input_events_sent": 0,
        "scope": "Offline in-memory screenshots; includes preprocessing, fresh vision, prompt tokenization, Laya, actions and Python decoding. Excludes capture, disk image loading, input dispatch and active-game contention. Not screenshot-to-keypress latency.",
        "outputs": outputs,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "outputs"}, indent=2))


if __name__ == "__main__":
    main()
