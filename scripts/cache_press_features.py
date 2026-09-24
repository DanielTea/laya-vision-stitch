"""Cache frozen C-RADIOv3-B patch grids for extracted click frames (training only).

Frames are resized to 384x224 (24x14 patches of 768 features) and written incrementally to
memory-mapped FP16 arrays. Labels, games and splits are stored alongside. Rows whose
image already appears in an earlier input file are skipped.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from numpy.lib.format import open_memmap
from PIL import Image

from laya_vision_stitch.radio_vision import RadioVision

SIZE = (384, 224)


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("artifacts/radio-source"))
    p.add_argument("--presses", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, seen, missing = [], set(), 0
    for path in args.presses:
        for r in read(path):
            if not Path(r["image"]).exists():
                missing += 1  # superseded extraction; the same game was re-extracted elsewhere
                continue
            if r["image"] not in seen:
                seen.add(r["image"])
                rows.append(r)
    model = RadioVision.from_source(args.source, mx.float16)
    features = open_memmap(
        args.output / "features.npy", mode="w+", dtype=np.float16, shape=(len(rows), 14, 24, 768)
    )
    for start in range(0, len(rows), 32):
        batch = rows[start : start + 32]
        pixels = np.stack(
            [
                np.asarray(
                    Image.open(r["image"]).convert("RGB").resize(SIZE, Image.BICUBIC), np.float32
                )
                / 255
                for r in batch
            ]
        )
        patches, _ = model(mx.array(pixels).astype(mx.float16))
        mx.eval(patches)
        features[start : start + len(batch)] = np.asarray(patches).reshape(len(batch), 14, 24, 768)
    features.flush()
    np.save(args.output / "xy.npy", np.array([r["xy"] for r in rows], np.float32))
    (args.output / "rows.jsonl").write_text(
        "".join(
            json.dumps({k: r[k] for k in ["id", "game", "split", "episode", "button", "image"]})
            + "\n"
            for r in rows
        )
    )
    summary = {
        "presses": len(rows),
        "skipped_missing_images": missing,
        "games": sorted({r["game"] for r in rows}),
        "grid": [14, 24],
        "size": SIZE,
    }
    (args.output / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
