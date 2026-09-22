"""Offline held-game imitation audit; never equates action agreement with gameplay."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .policy_data import check_separation, read_manifest
from .policy_training import cache_examples
from .trainable_model import TrainableRuntime


def metrics(rows, predictions, buttons):
    target = np.array([[b in r["action"]["buttons"] for b in buttons] for r in rows])
    pred = np.array([[b in p["buttons"] for b in buttons] for p in predictions])
    tp = (target & pred).sum(axis=0)
    fp = (~target & pred).sum(axis=0)
    fn = (target & ~pred).sum(axis=0)
    support = target.sum(axis=0)
    total = 2 * tp + fp + fn
    active = total > 0
    target_mouse = np.array([r["action"]["mouse_delta"] for r in rows])
    pred_mouse = np.array([p["mouse_delta"] for p in predictions])
    moving = abs(target_mouse) >= 0.02
    return {
        "examples": len(rows),
        "button_exact_match": float((target == pred).all(axis=1).mean()),
        "button_micro_f1": float(2 * tp.sum() / max(1, total.sum())),
        "button_macro_f1_active": float((2 * tp[active] / total[active]).mean())
        if active.any()
        else 0.0,
        "button_per_key": {
            b: {
                "support": int(support[i]),
                "f1": float(2 * tp[i] / max(1, total[i])),
                "predicted_positive": int(pred[:, i].sum()),
            }
            for i, b in enumerate(buttons)
            if active[i]
        },
        "mouse_mae": float(abs(target_mouse - pred_mouse).mean()),
        "mouse_moving_axis_count": int(moving.sum()),
        "mouse_direction_accuracy": float(
            (np.sign(target_mouse[moving]) == np.sign(pred_mouse[moving])).mean()
        )
        if moving.any()
        else None,
    }


def infer(runtime, examples, variant):
    output = []
    for index, (row, inputs) in enumerate(examples):
        if variant == "zero_vision":
            inputs = (mx.zeros_like(inputs[0]), *inputs[1:])
        elif variant == "shuffled_vision":
            # Fixed cyclic derangement of both frames together; prompt/history unchanged.
            other = examples[(index + max(1, len(examples) // 2)) % len(examples)][1]
            inputs = (*other[:2], *inputs[2:])
        elif variant == "no_action_history":
            changed = dict(row, previous_actions=[])
            inputs = (*inputs[:2], *runtime.prepare(changed))
        result = runtime.module.from_features(*inputs)
        mx.eval(result)
        probabilities = np.asarray(mx.sigmoid(result["buttons"][0]))
        output.append(
            {
                "buttons": [
                    b
                    for b, p in zip(
                        runtime.module.policy_config.buttons, probabilities, strict=True
                    )
                    if p >= 0.5
                ],
                "mouse_delta": np.asarray(result["mouse"][0]).tolist(),
            }
        )
    return output


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = TrainableRuntime.load(args.bundle)
    rows = read_manifest(args.manifest, runtime.module.policy_config)
    train = read_manifest(args.training_manifest, runtime.module.policy_config)
    check_separation(train, rows, holdout_games=True)
    buttons = runtime.module.policy_config.buttons
    report = {
        "bundle": str(args.bundle),
        "games": sorted({r["game"] for r in rows}),
        "game_holdout_verified": True,
        "closed_loop_gameplay_tested": False,
        "protocol": "Fixed 0.5 button threshold; no test tuning. 100 ms imitation targets.",
        "variants": {},
    }
    examples = cache_examples(runtime, rows, args.feature_cache)
    predictions = {}
    changes = [
        i
        for i, r in enumerate(rows)
        if set(r["action"]["buttons"]) != set(r["previous_actions"][-1]["buttons"])
    ]
    report["button_transition_subset"] = {"count": len(changes), "variants": {}}
    for variant in ("actual", "zero_vision", "shuffled_vision", "no_action_history"):
        predictions[variant] = infer(runtime, examples, variant)
        report["variants"][variant] = metrics(rows, predictions[variant], buttons)
        if changes:
            report["button_transition_subset"]["variants"][variant] = metrics(
                [rows[i] for i in changes], [predictions[variant][i] for i in changes], buttons
            )
        print(
            variant,
            {k: v for k, v in report["variants"][variant].items() if k != "button_per_key"},
            flush=True,
        )
    repeat = [r["previous_actions"][-1] for r in rows]
    no_action = [{"buttons": [], "mouse_delta": [0, 0]} for _ in rows]
    freq = {b: sum(b in r["action"]["buttons"] for r in train if "action" in r) for b in buttons}
    n = sum("action" in r for r in train)
    constant = [
        {"buttons": [b for b in buttons if freq[b] >= n / 2], "mouse_delta": [0, 0]} for _ in rows
    ]
    report["baselines"] = {
        name: metrics(rows, p, buttons)
        for name, p in [
            ("repeat_previous_action", repeat),
            ("no_action", no_action),
            ("train_frequency", constant),
        ]
    }
    if changes:
        report["button_transition_subset"]["repeat_previous_action"] = metrics(
            [rows[i] for i in changes], [repeat[i] for i in changes], buttons
        )
    if args.reference_bundle:
        mx.random.seed(17)
        reference = TrainableRuntime.load(args.reference_bundle)
        reference.expand_buttons(buttons)
        # Reuse identical frozen Qwen features, but rebuild text for the reference runtime.
        ref_examples = [(row, (*inputs[:2], *reference.prepare(row))) for row, inputs in examples]
        report["baselines"]["before_game_training"] = metrics(
            rows, infer(reference, ref_examples, "actual"), buttons
        )
    latencies = [runtime.predict(rows[i % len(rows)])["image_to_outputs_ms"] for i in range(9)]
    report["warm_image_to_outputs_p50_ms"] = float(np.median(latencies[1:]))
    report["warm_image_to_outputs_p95_ms"] = float(np.percentile(latencies[1:], 95))
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "predictions.json").write_text(
        json.dumps(
            [
                {
                    "id": r["id"],
                    "label": r["action"],
                    **{name: p[i] for name, p in predictions.items()},
                }
                for i, r in enumerate(rows)
            ],
            indent=2,
        )
        + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "manifest", "training-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--reference-bundle", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
