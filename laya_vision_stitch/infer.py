"""Evaluate a saved connector on one screenshot, without sending game input."""

import argparse
import json
import time
from pathlib import Path

from .alignment import RidgeConnector
from .backends import CHOICES, QWEN_ID, QWEN_REVISION, LayaEmbeddings, QwenVision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--repeat", type=int, default=1, help="Repeated offline measurements")
    args = parser.parse_args()
    if args.repeat < 1 or not args.image.is_file():
        parser.error("Need an existing image and a positive repeat count")
    manifest = json.loads((args.artifact / "manifest.json").read_text())
    if manifest["vision"]["model"] != QWEN_ID or manifest["vision"]["revision"] != QWEN_REVISION:
        raise ValueError("Connector was fitted with a different visual checkpoint")
    started = time.perf_counter()
    connector = RidgeConnector.load(args.artifact / "connector.npz")
    laya = LayaEmbeddings()
    if manifest["laya"] != laya.metadata:
        raise ValueError("Connector was fitted with a different Laya interface")
    vision = QwenVision(manifest["vision"]["max_image_width"])
    initialization_ms = (time.perf_counter() - started) * 1000
    for i in range(args.repeat):
        started = time.perf_counter()
        features, visual_ms = vision.encode(args.image)
        latent = connector.predict(features[None])[0]
        logits, laya_ms = laya.predict(latent)
        elapsed_ms = (time.perf_counter() - started) * 1000
        print(
            json.dumps(
                {
                    "iteration": i + 1,
                    "first_call": i == 0,
                    "initialization_ms": initialization_ms if i == 0 else None,
                    "image_to_scores_ms": elapsed_ms,
                    "vision_ms": visual_ms,
                    "laya_ms": laya_ms,
                    "choice": list(CHOICES)[int(logits.argmax())],
                    "raw_scores": dict(zip(CHOICES, logits.tolist(), strict=True)),
                    "input_events_sent": 0,
                    "note": "Experimental scores, not action correctness or probabilities.",
                },
                allow_nan=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
