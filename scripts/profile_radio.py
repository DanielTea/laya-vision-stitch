"""Reference parity and fresh-image latency for a proposed vision replacement."""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from laya_vision_stitch.radio_vision import RadioProcessor, RadioVision


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--images", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from transformers import AutoModel

    reference = AutoModel.from_pretrained(
        str(args.source), trust_remote_code=True, local_files_only=True
    ).eval()
    torch.set_num_threads(4)
    processor = RadioProcessor()
    images = [Image.open(path).convert("RGB") for path in args.images]
    report = {"source": str(args.source), "images": [str(p) for p in args.images], "cases": []}
    for dtype in (mx.float32, mx.float16):
        model = RadioVision.from_source(args.source, dtype)
        for width in (224, 320, 384, 448):
            parity, times = [], []
            for image in images:
                im = image.resize((width, round(image.height * width / image.width)))
                pixels = processor(images=[im])["pixel_values"]
                with torch.inference_mode():
                    expected = (
                        reference(torch.from_numpy(pixels).permute(0, 3, 1, 2))
                        .features.numpy()
                        .reshape(-1, 768)
                    )
                actual, _ = model(mx.array(pixels).astype(dtype))
                mx.eval(actual)
                actual = np.asarray(actual).astype(np.float32)
                cosine = (actual * expected).sum(-1) / (
                    np.linalg.norm(actual, axis=-1) * np.linalg.norm(expected, axis=-1)
                )
                parity.append(
                    {
                        "max_abs_error": float(np.max(np.abs(actual - expected))),
                        "mean_abs_error": float(np.mean(np.abs(actual - expected))),
                        "mean_cosine": float(cosine.mean()),
                        "min_cosine": float(cosine.min()),
                    }
                )
            for i in range(43):
                image = images[i % len(images)]
                started = time.perf_counter()
                im = image.resize((width, round(image.height * width / image.width)))
                pixels = processor(images=[im])["pixel_values"]
                out, _ = model(mx.array(pixels).astype(dtype))
                mx.eval(out)
                if i >= 3:
                    times.append((time.perf_counter() - started) * 1000)
            case = {
                "dtype": str(dtype),
                "width": width,
                "parity": parity,
                "fresh_image_to_features_p50_ms": float(np.median(times)),
                "fresh_image_to_features_p95_ms": float(np.percentile(times, 95)),
            }
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
