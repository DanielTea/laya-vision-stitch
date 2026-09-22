"""Run an offline frozen-model stitching pilot; never sends game input."""

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .alignment import RidgeConnector, agreement
from .backends import CHOICES, QWEN_ID, QWEN_REVISION, LayaEmbeddings, QwenVision, write_json
from .data import collect, validate_splits


def score(laya, vectors):
    return np.array([int(np.argmax(laya.predict(v)[0])) for v in vectors])


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = collect(args.runs, args.max_runs, args.per_run)
    validate_splits(rows)
    settings = {"width": args.width, "rows": rows, "protocol": 1}
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    print(
        f"Dataset: {len(rows)} frames; splits {dict(Counter(r['split'] for r in rows))}", flush=True
    )
    laya = LayaEmbeddings()
    parity = max(laya.parity(text) for text in list(dict.fromkeys(r["text"] for r in rows))[:5])
    print(f"Embedding injection parity max logit error: {parity:.6g}", flush=True)
    cache = out / "features.npz"
    if cache.exists():
        if not args.reuse:
            raise ValueError("Output already contains a run. Use a new directory or --reuse.")
        manifest = json.loads((out / "manifest.json").read_text())
        if (
            manifest["fingerprint"] != fingerprint
            or manifest["laya"] != laya.metadata
            or manifest["vision"]["model"] != QWEN_ID
            or manifest["vision"]["revision"] != QWEN_REVISION
        ):
            raise ValueError("Cached dataset/config does not match this run")
        with np.load(cache, allow_pickle=False) as saved:
            x, y, vision_ms = saved["x"], saved["y"], saved["vision_ms"]
        vision = None
    else:
        vision = QwenVision(args.width)
        # Exclude one warm-up on a fitting image from the reported inference timings.
        vision.encode(rows[0]["image"])
        laya.predict(laya.encode(rows[0]["text"]))
        xs, ys, times = [], [], []
        for i, row in enumerate(rows):
            feature, elapsed = vision.encode(row["image"])
            xs.append(feature)
            ys.append(laya.encode(row["text"]).reshape(-1))
            times.append(elapsed)
            if i % 16 == 0:
                print(f"Encoded {i + 1}/{len(rows)}; vision {elapsed:.1f} ms", flush=True)
        x, y, vision_ms = np.array(xs), np.array(ys), np.array(times)
        manifest = {
            "fingerprint": fingerprint,
            "laya": laya.metadata,
            "vision": vision.metadata,
            "settings": settings,
            "machine": platform.platform(),
            "architecture": platform.machine(),
            "parity_max_logit_error": parity,
        }
        write_json(out / "manifest.json", manifest)
        np.savez_compressed(cache, x=x, y=y, vision_ms=vision_ms)
    masks = {
        name: np.array([r["split"] == name for r in rows])
        for name in ("train", "validation", "test")
    }
    train, val, test = (masks[k] for k in ("train", "validation", "test"))
    teacher = score(laya, y)
    ordinary = np.array([laya.ordinary_choice(r["text"]) for r in rows])
    candidates = []
    for alpha in (0.01, 0.1, 1.0, 10.0):
        connector = RidgeConnector().fit(x[train], y[train], alpha)
        prediction = score(laya, connector.predict(x[val]))
        result = agreement(teacher[val], prediction)
        candidates.append((result["balanced_agreement"], result["agreement"], alpha))
        print(f"Validation alpha={alpha}: {result}", flush=True)
    # Select once on validation. Never refit or choose against the test partition.
    best = max(candidates)
    connector = RidgeConnector().fit(x[train], y[train], best[2])
    connector.save(out / "connector.npz")
    stitched, total_ms, head_ms, bridge_ms = [], [], [], []
    for index in np.flatnonzero(test):
        started = time.perf_counter()
        latent = connector.predict(x[index : index + 1])[0]
        projected_ms = (time.perf_counter() - started) * 1000
        logits, elapsed = laya.predict(latent)
        stitched.append(int(logits.argmax()))
        head_ms.append(elapsed)
        bridge_ms.append(projected_ms)
        total_ms.append(vision_ms[index] + projected_ms + elapsed)
    mean_latent = y[train].mean(0)
    constant = int(np.argmax(laya.predict(mean_latent)[0]))
    majority = Counter(teacher[train].tolist()).most_common(1)[0][0]
    normalized_train = (x[train] - connector.mean) / connector.scale
    normalized_test = (x[test] - connector.mean) / connector.scale
    distances = (
        np.square(normalized_test).sum(1)[:, None]
        + np.square(normalized_train).sum(1)[None, :]
        - 2 * normalized_test @ normalized_train.T
    )
    nearest = teacher[train][distances.argmin(1)]
    shuffled_y = np.random.default_rng(42).permutation(y[train])
    shuffled = RidgeConnector().fit(x[train], shuffled_y, best[2])
    control = score(laya, shuffled.predict(x[test]))
    methods = {
        "stitched": stitched,
        "nearest_training_image": nearest,
        "mean_state_embedding": [constant] * int(test.sum()),
        "training_teacher_majority": [majority] * int(test.sum()),
        "shuffled_pair_connector": control,
    }
    qwen_ms = []
    if args.qwen_baseline:
        vision = vision or QwenVision(args.width)
        vision.choose(rows[0]["image"])
        predicted = []
        for n, index in enumerate(np.flatnonzero(test)):
            choice, elapsed = vision.choose(rows[index]["image"])
            predicted.append(choice)
            qwen_ms.append(elapsed)
            if n % 8 == 0:
                print(f"Qwen baseline {n + 1}/{int(test.sum())}; {elapsed:.1f} ms", flush=True)
        methods["qwen_single_prefill"] = predicted
    labels = list(CHOICES)
    test_labels = np.array(labels)[teacher[test]]
    results = {
        name: agreement(test_labels, np.array(labels)[pred]) for name, pred in methods.items()
    }
    for i, index in enumerate(np.flatnonzero(test)):
        rows[index]["teacher"] = labels[teacher[index]]
        rows[index]["predictions"] = {name: labels[pred[i]] for name, pred in methods.items()}
    write_json(out / "test_predictions.json", [r for r in rows if r["split"] == "test"])

    def timing(values):
        return (
            {"p50_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}
            if len(values)
            else None
        )

    summary = {
        "claim": "Offline weak-supervision pilot; agreement with frozen text teacher, not gameplay accuracy.",
        "backbone_weights_updated": False,
        "connector_fitted_from_data": True,
        "input_events_sent": 0,
        "human_labeled_accuracy": None,
        "split_counts": dict(Counter(r["split"] for r in rows)),
        "split_runs": {s: sorted({r["run"] for r in rows if r["split"] == s}) for s in masks},
        "state_counts": dict(Counter(r["text"] for r in rows)),
        "selected_alpha": best[2],
        "validation_candidates": candidates,
        "teacher_protocol_vs_ordinary_laya": agreement(ordinary, teacher),
        "test": results,
        "timing": {
            "vision": timing(vision_ms[test]),
            "connector": timing(bridge_ms),
            "laya_graph": timing(head_ms),
            "stitched_sum": timing(total_ms),
            "qwen_single_prefill": timing(qwen_ms),
        },
        "limitations": [
            "State descriptions come from historical OCR/heuristics and can be wrong.",
            "Splits hold out recording runs, not independent players, maps or days.",
            "Exact duplicate image files are removed; similar scenes can remain.",
            "State descriptions must have exactly 19 tokens; no state-dependent masks or padding.",
            "No action labels, allowed-action masks or policy phases enter the connector.",
            "Timing sums separate offline feature extraction and head evaluation; not live latency.",
            "No live control, equipment-pickup proof, or general multimodal capability is established.",
        ],
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results: {out.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", required=True, type=Path, help="Read-only ScreenQuest runs directory"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-runs", type=int, default=8)
    parser.add_argument("--per-run", type=int, default=32)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--reuse", action="store_true", help="Reuse matching extracted features")
    parser.add_argument("--qwen-baseline", action="store_true")
    args = parser.parse_args()
    if args.max_runs < 6 or args.per_run < 2 or not 128 <= args.width <= 1024:
        parser.error("Need >=6 runs, >=2 frames per run, and width in 128..1024")
    run(args)


if __name__ == "__main__":
    main()
