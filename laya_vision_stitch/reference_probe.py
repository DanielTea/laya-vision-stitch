"""Evaluate paired-reference transfer. Labels are used only for scoring."""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from PIL import Image

from .reference_fixtures import QUESTIONS, create, validate_separation
from .reference_stitch import ReferenceRuntime, load_laya


def text_choice(agent, text, question, choices):
    from laya_mlx.agent import collate_items

    item = agent.prepare(
        text, {"decision": {"type": "choice", "instructions": question, "criteria": choices}}
    )[0][0]
    batch = {k: mx.array(v) for k, v in collate_items([item], agent.tok.pad_token_id).items()}
    logits, _ = agent.model(**batch)
    mx.eval(logits)
    return list(choices)[int(mx.argmax(logits[0]).item())]


def metrics(rows, methods):
    result = {}
    for method in methods:
        result[method] = {}
        groups = {
            "all": rows,
            "familiar_composition": [r for r in rows if not r["novel_composition"]],
            "novel_composition": [r for r in rows if r["novel_composition"]],
        }
        groups.update({task: [r for r in rows if r["task"] == task] for task in QUESTIONS})
        for group, records in groups.items():
            result[method][group] = {
                "accuracy": float(np.mean([r[method] == r["expected"] for r in records])),
                "n": len(records),
            }
    return result


def uncertainty(rows):
    """Exploratory paired bootstrap; correlated questions stay with their image."""
    ids = sorted({r["id"] for r in rows})
    methods = ("mixture", "nearest_laya", "blank_image", "shuffled_correspondence")
    values = {
        m: np.array([np.mean([r[m] == r["expected"] for r in rows if r["id"] == i]) for i in ids])
        for m in methods
    }
    samples = np.random.default_rng(97).integers(0, len(ids), size=(5000, len(ids)))
    return {
        "bootstrap_unit": "image; groups all six correlated questions/options together",
        "replicates": 5000,
        "accuracy_intervals_95": {
            m: np.percentile(values[m][samples].mean(1), [2.5, 97.5]).tolist() for m in methods
        },
        "mixture_minus_control_intervals_95": {
            m: np.percentile((values["mixture"] - values[m])[samples].mean(1), [2.5, 97.5]).tolist()
            for m in methods
            if m != "mixture"
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    references, tests = create(out / "data")
    validate_separation(references, tests)
    print(f"References: {len(references)}; evaluation images: {len(tests)}", flush=True)
    started = time.perf_counter()
    runtime = ReferenceRuntime.build(out / "data/references.json", laya_kind="english")
    build_seconds = time.perf_counter() - started
    runtime.save(out / "bundle")
    other = load_laya("multilingual")
    other.model.freeze()
    blank = out / "blank.png"
    Image.new("RGB", (320, 320), "white").save(blank)
    # One warm-up; no inference results feed back into the memory or model.
    first = tests[0]
    warm = runtime.decide(
        out / "data" / first["image"],
        QUESTIONS["color"][0],
        dict.fromkeys(QUESTIONS["color"][1], None),
    )
    token_ids, token_mask = runtime.module.reference_ids, runtime.module.reference_mask
    permutation = mx.array(np.random.default_rng(431).permutation(len(references)))
    records, blind_cache = [], {}
    for number, row in enumerate(tests):
        for task, (question, labels) in QUESTIONS.items():
            for reversed_order in (False, True):
                choices = {k: k for k in (labels[::-1] if reversed_order else labels)}
                path = out / "data" / row["image"]
                prediction = runtime.decide(path, question, choices)
                nearest = references[prediction["reference_indices"][0]][task]
                oracle = text_choice(runtime.agent, row["caption"], question, choices)
                multilingual = text_choice(other, row["caption"], question, choices)
                # A fixed, predeclared alternative question style is measured, not selected.
                alternate_question = f"Read the description and select the object's {task}."
                oracle_alternate = text_choice(
                    runtime.agent, row["caption"], alternate_question, choices
                )
                cache_key = (task, reversed_order)
                if cache_key not in blind_cache:
                    blind_cache[cache_key] = runtime.decide(blank, question, choices)["choice"]
                # Scramble only correspondence; keep the set of captions and all backbone weights.
                runtime.module.reference_ids = token_ids[permutation]
                runtime.module.reference_mask = token_mask[permutation]
                shuffled = runtime.decide(path, question, choices)["choice"]
                runtime.module.reference_ids, runtime.module.reference_mask = token_ids, token_mask
                text_key = runtime.decide(path, question, choices, key_source="text")["choice"]
                records.append(
                    {
                        "id": row["id"],
                        "task": task,
                        "expected": row[task],
                        "novel_composition": row["novel_composition"],
                        "style": row["style"],
                        "reversed_order": reversed_order,
                        "mixture": prediction["choice"],
                        "nearest_laya": prediction["nearest_laya_choice"],
                        "nearest_reference_fact": nearest,
                        "text_key_mixture": text_key,
                        "blank_image": blind_cache[cache_key],
                        "shuffled_correspondence": shuffled,
                        "english_oracle": oracle,
                        "multilingual_oracle": multilingual,
                        "english_oracle_alternate": oracle_alternate,
                        "latency_ms": prediction["image_to_scores_ms"],
                        "reference_ids": prediction["reference_ids"],
                        "weights": prediction["reference_weights"],
                    }
                )
        if number % 6 == 0:
            print(
                f"Evaluated {number + 1}/{len(tests)} images; {len(records)} decisions", flush=True
            )
    methods = [
        "mixture",
        "nearest_laya",
        "nearest_reference_fact",
        "text_key_mixture",
        "blank_image",
        "shuffled_correspondence",
        "english_oracle",
        "multilingual_oracle",
        "english_oracle_alternate",
    ]
    times = [r["latency_ms"] for r in records]
    summary = {
        "claim": "Synthetic grounding experiment, not cross-game gameplay validation.",
        "reference_images": len(references),
        "evaluation_images": len(tests),
        "evaluation_decisions": len(records),
        "new_training": False,
        "connector_fitting": False,
        "generated_tokens": 0,
        "input_events_sent": 0,
        "trainable_tensors": len(tree_flatten(runtime.module.trainable_parameters())),
        "build_seconds": build_seconds,
        "first_call_ms": warm["image_to_scores_ms"],
        "warm_p50_ms": float(np.median(times)),
        "warm_p95_ms": float(np.percentile(times, 95)),
        "metrics": metrics(records, methods),
        "uncertainty": uncertainty(records),
        "limitations": [
            "Procedurally rendered geometric objects; no natural photos or game footage.",
            "Questions and reversed options share images and are correlated trials.",
            "Reference memory covers only its supplied descriptions; no new captions are generated.",
            "Novel compositions exclude three color-shape pairs, not every attribute.",
            "Reference similarity and mixture scores are not calibrated action confidence.",
            "No parameters or reference-temperature settings selected using evaluation labels.",
        ],
    }
    (out / "predictions.json").write_text(json.dumps(records, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
