"""Balanced synthetic visual probes and negative controls; no fitting or gameplay."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from PIL import Image, ImageDraw

from .lexical_stitch import StitchRuntime


def fixtures(directory):
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for color in ("red", "blue"):
        for side in ("left", "right"):
            for shape in ("square", "circle"):
                path = directory / f"{color}-{side}-{shape}.png"
                image = Image.new("RGB", (384, 256), "white")
                draw = ImageDraw.Draw(image)
                x = 80 if side == "left" else 304
                box = (x - 40, 88, x + 40, 168)
                (draw.rectangle if shape == "square" else draw.ellipse)(box, fill=color)
                image.save(path)
                text = f"There is a {color} {shape} on the {side} side of a white image."
                for task, question, labels, expected in (
                    ("color", "What color is the object?", ("red", "blue"), color),
                    ("position", "Which side contains the object?", ("left", "right"), side),
                    ("shape", "What shape is the object?", ("square", "circle"), shape),
                ):
                    for reverse in (False, True):
                        order = labels[::-1] if reverse else labels
                        rows.append(
                            dict(
                                image=path,
                                task=task,
                                question=question,
                                choices={k: k for k in order},
                                expected=expected,
                                oracle_text=text,
                                reversed=reverse,
                            )
                        )
    return rows


def text_oracle(runtime, text, question, choices):
    from laya_mlx.agent import collate_items

    item = runtime.agent.prepare(
        text, {"decision": {"type": "choice", "instructions": question, "criteria": choices}}
    )[0][0]
    batch = {
        k: mx.array(v) for k, v in collate_items([item], runtime.agent.tok.pad_token_id).items()
    }
    original, _ = runtime.module.laya(**batch)
    injected = runtime.module.score_embeddings(
        runtime.module.laya.encoder.embeddings.tok_embeddings(batch["input_ids"]),
        **{k: v for k, v in batch.items() if k != "input_ids"},
    )
    mx.eval(original, injected)
    error = float(mx.max(mx.abs(original - injected)).item())
    return list(choices)[int(mx.argmax(original[0]).item())], error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.json").exists():
        parser.error("Use a new output directory to preserve prior evidence")
    started = time.perf_counter()
    runtime = StitchRuntime.load(args.bundle)
    load_seconds = time.perf_counter() - started
    rows = fixtures(args.output / "images")
    blank = args.output / "images" / "blank.png"
    Image.new("RGB", (384, 256), "white").save(blank)
    first = rows[0]
    warmup = runtime.decide(first["image"], first["question"], first["choices"])
    results, blind = [], {}
    original_values = runtime.module.bridge.anchor_values
    for i, row in enumerate(rows):
        actual = runtime.decide(row["image"], row["question"], row["choices"])
        key = (row["task"], row["reversed"])
        if key not in blind:
            blind[key] = runtime.decide(blank, row["question"], row["choices"])["choice"]
        oracle, parity = text_oracle(runtime, row["oracle_text"], row["question"], row["choices"])
        # Fixed reversal destroys correspondence while preserving the set of word embeddings.
        runtime.module.bridge.anchor_values = original_values[::-1]
        shuffled = runtime.decide(row["image"], row["question"], row["choices"])["choice"]
        runtime.module.bridge.anchor_values = original_values
        results.append(
            {
                **row,
                "image": row["image"].name,
                "stitched": actual,
                "blank_image": blind[key],
                "text_oracle": oracle,
                "reversed_word_pairs": shuffled,
                "parity_max_error": parity,
            }
        )
        if i % 8 == 0:
            print(f"Probe {i + 1}/{len(rows)}; {actual['image_to_scores_ms']:.1f} ms", flush=True)

    methods = ("stitched", "blank_image", "text_oracle", "reversed_word_pairs")
    metrics = {}
    for method in methods:
        scores = {}
        for task in ("color", "position", "shape", "all"):
            group = [r for r in results if task == "all" or r["task"] == task]
            predictions = [
                r[method]["choice"] if method == "stitched" else r[method] for r in group
            ]
            scores[task] = {
                "accuracy": float(
                    np.mean([p == r["expected"] for p, r in zip(predictions, group, strict=True)])
                ),
                "n": len(group),
                "predictions": dict(Counter(predictions)),
            }
        metrics[method] = scores
    times = [r["stitched"]["image_to_scores_ms"] for r in results]
    summary = {
        "claim": "Synthetic visual probes only; no game-playing or cross-game capability established.",
        "new_training": False,
        "connector_fitting": False,
        "qwen_decoder_layers_executed": 0,
        "generated_tokens": 0,
        "trainable_tensor_count": len(tree_flatten(runtime.module.trainable_parameters())),
        "word_count": len(runtime.words),
        "load_seconds": load_seconds,
        "first_call_ms": warmup["image_to_scores_ms"],
        "warm_p50_ms": float(np.median(times)),
        "warm_p95_ms": float(np.percentile(times, 95)),
        "parity_max_error": max(r["parity_max_error"] for r in results),
        "metrics": metrics,
        "limitations": [
            "Eight synthetic images; questions and reversed choices are correlated trials.",
            "No hyperparameters selected against these labels.",
            "Fixed temperature and top-k are design choices, not learned weights.",
            "Only a single screenshot is supported; no temporal memory.",
            "Raw token embeddings need not align with pre-decoder visual features.",
        ],
    }
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
