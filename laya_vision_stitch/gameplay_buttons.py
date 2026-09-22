"""Isolate first-step recorded gameplay buttons using frozen visual contexts."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .adapter_buttons import train_adapter_buttons
from .decoder_probe import freeze_decoder_only, prepare, stratum_positive_weights, train_decoder
from .policy_data import check_separation, manifest_digest, read_manifest
from .policy_training import cache_examples, fingerprint
from .trainable_model import TrainableRuntime


def button_metrics(rows, predicted, buttons):
    truth = np.array([[b in r["action"]["buttons"] for b in buttons] for r in rows])
    pred = np.array([[b in p for b in buttons] for p in predicted])
    correct = (truth == pred).all(-1)
    tp, fp, fn = (truth & pred).sum(), (~truth & pred).sum(), (truth & ~pred).sum()
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(tuple(sorted(row["action"]["buttons"])), []).append(i)
    return {
        "examples": len(rows),
        "button_exact_match": float(correct.mean()),
        "button_micro_f1": float(2 * tp / max(1, 2 * tp + fp + fn)),
        "action_set_balanced_exact": float(np.mean([correct[g].mean() for g in groups.values()])),
        "empty_prediction_rate": float(np.mean([not p for p in predicted])),
    }


def score(runtime, rows, contexts):
    buttons = runtime.module.policy_config.buttons
    predicted = []
    for h in contexts:
        p = np.asarray(mx.sigmoid(runtime.module.actions(h[:, 0], h)["buttons"][0]))
        predicted.append([b for b, on in zip(buttons, p >= 0.5, strict=True) if on])
    return {
        **button_metrics(rows, predicted, buttons),
        "predictions": [
            {"id": r["id"], "target": r["action"]["buttons"], "buttons": p}
            for r, p in zip(rows, predicted, strict=True)
        ],
    }


def run(args):
    if args.steps < 1 or args.learning_rate <= 0 or not np.isfinite(args.learning_rate):
        raise ValueError("Positive training steps and learning rate required")
    args.output.mkdir(parents=True, exist_ok=False)
    mx.random.seed(41)
    runtime = TrainableRuntime.load(args.bundle)
    if args.train_adapters:
        for key in ("mouse", "chunk_mouse", "pointer", "pointer_active", "duration"):
            if hasattr(runtime.module.actions, key):
                getattr(runtime.module.actions, key).freeze()
    else:
        freeze_decoder_only(runtime)
    rows = {
        s: read_manifest(args.data / f"{s}.jsonl", runtime.module.policy_config)
        for s in ("train", "validation")
    }
    check_separation(rows["train"], rows["validation"])
    for split in rows.values():
        for row in split:
            if row.get("previous_actions"):
                raise ValueError("This visual diagnostic excludes previous-action inputs")
            row["choices"] = {"act": "Take the next action.", "wait": "Wait."}
    weights = (
        mx.array(
            stratum_positive_weights(rows["train"], runtime.module.policy_config.buttons),
            mx.float32,
        )
        if args.balance_positive_keys
        else None
    )
    protocol = {
        "source": str(args.bundle),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "seed": 41,
        "batch_size": 4 if args.train_adapters else 1,
        "train_adapters": args.train_adapters,
        "positive_key_weights": weights.tolist() if weights is not None else None,
        "manifest_digests": {s: manifest_digest(r) for s, r in rows.items()},
        "scope": "Recorded first-step buttons only; no task intent, camera or live gameplay",
        "sampling": (
            "Four random game-goal/button-set strata with replacement per update"
            if args.train_adapters
            else "Round-robin game-goal/button-set strata with replacement"
        ),
        "train_gate": {"button_f1_min": 0.95, "exact_min": 0.90},
        "validation_gate": {
            "button_f1_min": 0.70,
            "shuffle_margin_min": 0.10,
            "must_exceed_previous_action": True,
        },
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    roots = ("vision", "laya") if args.train_adapters else ("vision", "connector", "laya")
    frozen = {
        k: fingerprint(getattr(runtime.module, k), frozen_only=args.train_adapters) for k in roots
    }
    actual = {s: prepare(runtime, r, args.cache)[0] for s, r in rows.items()}
    report = {
        "protocol": protocol,
        "parameters": runtime.parameter_counts(),
        "before": {s: score(runtime, r, actual[s]) for s, r in rows.items()},
    }
    with (args.output / "steps.jsonl").open("w") as log:
        if args.train_adapters:
            train_adapter_buttons(
                runtime,
                list(cache_examples(runtime, rows["train"], args.cache)),
                args.steps,
                args.learning_rate,
                41,
                log,
                batch_size=4,
                positive_weights=weights,
            )
        else:
            train_decoder(
                runtime,
                rows["train"],
                actual["train"],
                args.steps,
                args.learning_rate,
                41,
                log,
                positive_weights=weights,
            )
    if args.train_adapters:
        actual = {s: prepare(runtime, r, args.cache)[0] for s, r in rows.items()}
    report["after"] = {}
    for s, r in rows.items():
        report["after"][s] = {"actual": score(runtime, r, actual[s])}
        for variant in ("shuffled_vision", "zero_vision"):
            report["after"][s][variant] = score(
                runtime, r, prepare(runtime, r, args.cache, variant)[0]
            )
        report["after"][s]["repeat_previous_action"] = button_metrics(
            r,
            [x["recorded_previous_actions"][-1]["buttons"] for x in r],
            runtime.module.policy_config.buttons,
        )
    if frozen != {
        k: fingerprint(getattr(runtime.module, k), frozen_only=args.train_adapters) for k in frozen
    }:
        raise RuntimeError("Frozen upstream weights changed")
    a, v = report["after"]["train"]["actual"], report["after"]["validation"]
    report["train_gate_passed"] = a["button_micro_f1"] >= 0.95 and a["button_exact_match"] >= 0.90
    report["validation_gate_passed"] = (
        v["actual"]["button_micro_f1"] >= 0.70
        and v["actual"]["button_micro_f1"] - v["shuffled_vision"]["button_micro_f1"] >= 0.10
        and v["actual"]["button_micro_f1"] > v["repeat_previous_action"]["button_micro_f1"]
    )
    report["frozen_backbone_weights_unchanged"] = True
    report["entire_upstream_path_frozen"] = not args.train_adapters
    report["live_inputs_sent"] = 0
    runtime.metadata["gameplay_buttons"] = protocol
    runtime.save(args.output / "bundle")
    loaded = TrainableRuntime.load(args.output / "bundle")
    np.testing.assert_allclose(
        list(runtime.predict(rows["validation"][0])["button_probabilities"].values()),
        list(loaded.predict(rows["validation"][0])["button_probabilities"].values()),
        atol=1e-6,
    )
    report["reload_matches"] = True
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                s: {
                    v: {k: x for k, x in m.items() if k != "predictions"}
                    for v, m in variants.items()
                }
                for s, variants in report["after"].items()
            },
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=Path("artifacts/robust-decoder-003/bundle"))
    p.add_argument("--data", type=Path, default=Path("artifacts/learning-gate-002"))
    p.add_argument("--cache", type=Path, default=Path("artifacts/vision-feature-cache"))
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--learning-rate", type=float, default=0.0001)
    p.add_argument("--train-adapters", action="store_true")
    p.add_argument("--balance-positive-keys", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
