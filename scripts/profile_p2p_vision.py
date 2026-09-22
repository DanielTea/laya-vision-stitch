"""Validate pretrained gameplay vision conversion; never invokes a control policy."""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from laya_vision_stitch.p2p_pretrained_vision import P2P_ID, P2P_REVISION, OpenP2PVision, preprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from torchvision.models import efficientnet_b0

    torch.set_num_threads(4)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)[
        "state_dict"
    ]
    reference = efficientnet_b0(weights=None).features[:6].eval()
    reference.load_state_dict(
        {
            k.removeprefix("image_tokenizer.efficientnet_preprocess."): v
            for k, v in state.items()
            if k.startswith("image_tokenizer.efficientnet_preprocess.")
        },
        strict=True,
    )
    projection = torch.nn.Sequential(
        torch.nn.Linear(112 * 12 * 12, 1024), torch.nn.LayerNorm(1024)
    ).eval()
    projection.load_state_dict(
        {
            k.removeprefix("image_tokenizer.mlp."): v
            for k, v in state.items()
            if k.startswith("image_tokenizer.mlp.")
        },
        strict=True,
    )
    numpy_state = {
        k: v.float().numpy() for k, v in state.items() if k.startswith("image_tokenizer.")
    }
    images = [Image.open(path).convert("RGB") for path in args.images]
    report = {
        "model": P2P_ID,
        "revision": P2P_REVISION,
        "scope": "Image tokenizer only, no Laya or action model. Model parity uses identical tensors; upstream Rust resize parity is not established.",
        "cases": [],
    }
    for dtype in (mx.float32, mx.bfloat16):
        model = OpenP2PVision.from_state(numpy_state, dtype)
        parity = []
        for image in images:
            pixels = preprocess(image)
            with torch.inference_mode():
                spatial = reference(torch.from_numpy(pixels).permute(0, 3, 1, 2))
                expected = projection(spatial.flatten(1)).numpy()
            features, actual = model(mx.array(pixels))
            mx.eval(features, actual)
            actual = np.asarray(actual.astype(mx.float32))
            cosine = float(
                (actual * expected).sum() / (np.linalg.norm(actual) * np.linalg.norm(expected))
            )
            error = float(np.abs(actual - expected).max())
            parity.append(
                {
                    "token_cosine": cosine,
                    "token_max_abs_error": error,
                    "spatial_max_abs_error": float(
                        np.abs(
                            np.asarray(features.astype(mx.float32))
                            - spatial.numpy().transpose(0, 2, 3, 1)
                        ).max()
                    ),
                }
            )
            if not np.isfinite([cosine, error]).all() or cosine < (
                0.99999 if dtype == mx.float32 else 0.995
            ):
                raise RuntimeError(f"Pretrained vision conversion failed parity: {parity[-1]}")
        times = []
        for i in range(85):
            start = time.perf_counter()
            result = model(mx.array(preprocess(images[i % len(images)])))
            mx.eval(result)
            if i >= 5:
                times.append(1000 * (time.perf_counter() - start))
        case = {
            "dtype": str(dtype),
            "parity": parity,
            "fresh_encoder_p50_ms": float(np.median(times)),
            "fresh_encoder_p95_ms": float(np.percentile(times, 95)),
            "samples": len(times),
        }
        print(json.dumps(case), flush=True)
        report["cases"].append(case)
        model.save_weights(
            str(
                args.output
                / ("vision-fp32.safetensors" if dtype == mx.float32 else "vision-bf16.safetensors")
            )
        )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
