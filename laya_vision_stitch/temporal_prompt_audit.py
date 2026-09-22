"""Evaluate goal interventions on the same images; no training or desktop input."""

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .policy_training import fingerprint
from .temporal_runtime import language_context
from .temporal_training import cache_sequences, grouped_metrics, predict_sequence, read_sequences
from .trainable_model import TrainableRuntime


def altered_contexts(runtime, sequences, output, frozen):
    selected = [
        i for i, rows in enumerate(sequences) if rows[0]["instruction_provenance"]["annotator"]
    ]
    goals = {}
    for i in selected:
        row = sequences[i][0]
        goals.setdefault(row["game"], set()).add(row["goal"])
    selected = [i for i in selected if len(goals[sequences[i][0]["game"]]) > 1]
    cached = {}
    output.mkdir(parents=True, exist_ok=True)
    for i in selected:
        rows = sequences[i]
        row = rows[0]
        alternatives = sorted(goals[row["game"]] - {row["goal"]})
        prompts = {
            "generic_goal": "Continue the current activity.",
            "different_goal": alternatives[0],
        }
        signature = {
            "parent": frozen,
            "policy_config": runtime.metadata["policy_config"],
            "prompts": prompts,
            "images": [r["frames"][0]["sha256"] for r in rows],
            "controls": row["controls"],
            "choices": row.get("choices"),
        }
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        path = output / f"{key}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                values = {k: mx.array(data[k]) for k in data.files}
        else:
            contexts = {v: [] for v in prompts}
            for r in rows:
                patches, coords = runtime.features({"frames": r["frames"]})
                for variant, prompt in prompts.items():
                    context = language_context(runtime, patches, coords, {**r, "goal": prompt})
                    mx.eval(context)
                    contexts[variant].append(context)
            values = {k: mx.stack(v) for k, v in contexts.items()}
            np.savez_compressed(path, **{k: np.asarray(v) for k, v in values.items()})
        cached[i] = values
        print(f"Prepared goal intervention {len(cached)}/{len(selected)}", flush=True)
    return cached


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    parent = TrainableRuntime.load(args.parent)
    frozen = {
        k: fingerprint(getattr(parent.module, k))
        for k in ("vision", "laya", "connector", "actions")
    }
    _, groups = read_sequences(args.data, parent.module.policy_config)
    sequences = groups["validation"]
    data = cache_sequences(parent, {"validation": sequences}, args.cache, frozen)["validation"]
    changed = altered_contexts(parent, sequences, args.cache / "goal-interventions", frozen)
    if not changed:
        raise ValueError("No validation sequences with distinct instructed goals")
    rows = [r for i in changed for r in sequences[i]]
    report = {
        "scope": "Offline goal sensitivity on weak instructed validation clips; not counterfactual action ground truth or goal completion",
        "examples": len(rows),
        "sequences": len(changed),
        "models": {},
    }
    for path in args.bundles:
        runtime = TrainableRuntime.load(path)
        if frozen != {k: fingerprint(getattr(runtime.module, k)) for k in frozen}:
            raise ValueError("Model parent differs from cached context parent")
        config = runtime.module.policy_config
        predictions, results, probabilities = {}, {}, {}
        for variant in ("actual", "generic_goal", "different_goal"):
            prediction, probability = [], []
            for i, contexts in changed.items():
                item = (
                    data[i] if variant == "actual" else {**data[i], "language": contexts[variant]}
                )
                actions, logits = predict_sequence(runtime.module.temporal_actions, item, config)
                prediction.extend(actions)
                probability.append(np.asarray(mx.sigmoid(logits["buttons"][0])))
            predictions[variant] = prediction
            probabilities[variant] = np.concatenate(probability)
            results[variant] = grouped_metrics(rows, prediction, config)
        for variant in ("generic_goal", "different_goal"):
            results[variant]["button_set_change_rate"] = float(
                np.mean(
                    [
                        set(a["buttons"]) != set(b["buttons"])
                        for a, b in zip(predictions["actual"], predictions[variant], strict=True)
                    ]
                )
            )
            results[variant]["action_change_rate"] = float(
                np.mean(
                    [
                        set(a["buttons"]) != set(b["buttons"])
                        or a["mouse_delta"] != b["mouse_delta"]
                        for a, b in zip(predictions["actual"], predictions[variant], strict=True)
                    ]
                )
            )
            difference = np.abs(probabilities["actual"] - probabilities[variant])
            results[variant]["mean_abs_button_probability_change"] = float(difference.mean())
            results[variant]["max_abs_button_probability_change"] = float(difference.max())
        report["models"][str(path)] = results
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, default=Path("artifacts/p2p-pilot-004/bundle"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--bundles", type=Path, nargs="+", required=True)
    p.add_argument("--cache", type=Path, default=Path("artifacts/temporal-feature-cache"))
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
