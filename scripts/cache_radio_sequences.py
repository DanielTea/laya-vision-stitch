"""Cache frozen C-RADIOv3-B patch features aligned with a sequence cache (training only)."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from laya_vision_stitch.radio_vision import RadioProcessor, RadioVision


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="Pinned RADIO weights directory")
    p.add_argument("--data", type=Path, required=True, help="Original sequence manifests")
    p.add_argument("--cache", type=Path, required=True, help="Existing sequence cache to align")
    p.add_argument("--splits", nargs="+", default=["train", "validation", "test", "fresh_test"])
    args = p.parse_args()
    model = RadioVision.from_source(args.source, mx.float16)
    processor = RadioProcessor()
    for split in args.splits:
        target = args.cache / f"{split}-radio.npz"
        if target.exists():
            raise FileExistsError(target)
        paths = {r["id"]: r["frames"][0]["image"] for r in read(args.data / f"{split}.jsonl")}
        rows = read(args.cache / f"{split}.jsonl")
        patches, summaries, grid = None, None, None
        for start in range(0, len(rows), 16):
            batch = rows[start : start + 16]
            pixels = []
            for r in batch:
                path = Path(paths[r["id"]])
                with Image.open(path if path.is_absolute() else args.data / path) as im:
                    pixels.append(processor([im])["pixel_values"][0])
            x, summary = model(mx.array(np.stack(pixels)).astype(mx.float16))
            mx.eval(x, summary)
            if patches is None:
                h, w = pixels[0].shape[0] // 16, pixels[0].shape[1] // 16
                grid = (h, w)
                patches = np.zeros((len(rows), h * w, 768), np.float16)
                summaries = np.zeros((len(rows), summary.shape[-1]), np.float16)
            patches[start : start + len(batch)] = np.asarray(x).reshape(len(batch), -1, 768)
            summaries[start : start + len(batch)] = np.asarray(summary)
        np.savez(target, patches=patches, summaries=summaries, grid=np.array(grid))
        print(json.dumps({split: {"frames": len(rows), "grid": grid}}), flush=True)


if __name__ == "__main__":
    main()
