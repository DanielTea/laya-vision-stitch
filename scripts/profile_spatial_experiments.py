"""Fresh-image full-cache latency; excludes capture, dispatch and game acknowledgement."""

import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from laya_vision_stitch.laya_p2p import LayaP2PRuntime
from laya_vision_stitch.p2p_pretrained_vision import preprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundles", type=Path, nargs="+", required=True)
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    paths = sorted(a.frames.glob("*.jpg"))[:240]
    if len(paths) != 240:
        raise ValueError("Need 240 distinct recorded frames")
    images = [Image.open(p).convert("RGB") for p in paths]
    goal = "Defeat nearby monsters. Avoid attacking players. Retreat when health is low."
    results = []
    for bundle in a.bundles:
        runtime = LayaP2PRuntime.load(bundle)
        state = None
        times = []
        for i, image in enumerate(images):
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
        result = {
            "bundle": str(bundle),
            "frames": 240,
            "full_cache_samples": 40,
            "p50_ms": float(np.median(times[200:])),
            "p95_ms": float(np.percentile(times[200:], 95)),
            "scope": "Fresh preprocessing, frozen vision and Laya, trained adapter, temporal policy and autoregressive controls; excludes screenshot capture, input posting and game response.",
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        del runtime, state, out
        gc.collect()
        mx.clear_cache()
    (a.output / "report.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
