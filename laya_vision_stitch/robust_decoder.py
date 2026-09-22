"""Decoder curriculum with unseen wording, option order and visual ablations."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np

from .adapter_buttons import train_adapter_buttons
from .decoder_probe import audit, freeze_decoder_only, prepare, train_decoder
from .policy_data import check_separation, manifest_digest, read_manifest
from .policy_training import cache_examples, fingerprint
from .trainable_model import ActionHeads, TrainableRuntime


def score(runtime, rows, contexts, choices):
    overall = audit(runtime, rows, contexts, choices)
    overall.pop("predictions")
    by_template = {}
    for template in sorted({r["template_id"] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r["template_id"] == template]
        result = audit(
            runtime,
            [rows[i] for i in indices],
            [contexts[i] for i in indices],
            [choices[i] for i in indices],
        )
        result.pop("predictions")
        by_template[template] = result
    return {"overall": overall, "by_template": by_template}


def run(args):
    if args.steps < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Positive steps and learning rate required")
    if args.matched_pairs and not args.train_adapters:
        raise ValueError("Matched-pair sampling currently requires adapter training")
    args.output.mkdir(parents=True, exist_ok=False)
    mx.random.seed(args.seed)
    runtime = TrainableRuntime.load(args.bundle)
    if args.normalize_action_context:
        runtime.normalize_action_inputs()
    if args.encoder_context:
        runtime.module.policy_config.action_context_source = "encoder"
        runtime.metadata["policy_config"] = asdict(runtime.module.policy_config)
    if args.reset_decoder:
        old = runtime.module.actions
        new = ActionHeads(
            runtime.module.laya.encoder.config.hidden_size, runtime.module.policy_config
        )
        for key in ("mouse", "chunk_mouse", "pointer", "pointer_active", "duration"):
            if hasattr(old, key):
                setattr(new, key, getattr(old, key))
        runtime.module.actions = new
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
    frozen_roots = ("vision", "laya") if args.train_adapters else ("vision", "connector", "laya")
    frozen = {
        k: fingerprint(getattr(runtime.module, k), frozen_only=args.train_adapters)
        for k in frozen_roots
    }
    protocol = {
        "source": str(args.bundle),
        "normalize_action_context": runtime.module.policy_config.normalize_action_context,
        "action_context_source": runtime.module.policy_config.action_context_source,
        "reset_decoder": args.reset_decoder,
        "data": str(args.data),
        "steps": args.steps,
        "batch_size": 4 if args.train_adapters else 8,
        "train_adapters": args.train_adapters,
        "matched_pairs": args.matched_pairs,
        "objective": "balanced buttons only",
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "digests": {s: manifest_digest(r) for s, r in rows.items()},
        "curriculum": json.loads((args.data / "curriculum.json").read_text()),
        "train_gate": {"exact_min": 0.95, "balanced_min": 0.95, "both_goals_min": 0.90},
        "validation_gate": {
            "each_template_exact_min": 0.85,
            "each_template_balanced_min": 0.80,
            "each_template_both_goals_min": 0.75,
            "shuffle_balanced_margin_min": 0.15,
        },
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print("Preparing pre-training contexts and audit...", flush=True)
    actual = {s: prepare(runtime, r, args.cache) for s, r in rows.items()}
    report = {
        "protocol": protocol,
        "before": {s: score(runtime, r, *actual[s]) for s, r in rows.items()},
        "parameters": runtime.parameter_counts(),
    }
    with (args.output / "steps.jsonl").open("w") as log:
        if args.train_adapters:
            examples = list(cache_examples(runtime, rows["train"], args.cache))
            train_adapter_buttons(
                runtime,
                examples,
                args.steps,
                args.learning_rate,
                args.seed,
                log,
                matched=args.matched_pairs,
            )
        else:
            train_decoder(
                runtime,
                rows["train"],
                actual["train"][0],
                args.steps,
                args.learning_rate,
                args.seed,
                log,
                batch_size=8,
            )
    if args.train_adapters:
        actual = {}
        for s, r in rows.items():
            print(f"Recomputing {s} contexts after adapter updates...", flush=True)
            actual[s] = prepare(runtime, r, args.cache)
    print("Scoring trained model...", flush=True)
    report["after"] = {s: {"actual": score(runtime, r, *actual[s])} for s, r in rows.items()}
    for variant in ("shuffled_vision", "zero_vision"):
        print(f"Scoring validation with {variant}...", flush=True)
        report["after"]["validation"][variant] = score(
            runtime, rows["validation"], *prepare(runtime, rows["validation"], args.cache, variant)
        )
    a = report["after"]["train"]["actual"]["overall"]
    report["train_gate_passed"] = (
        a["button_exact_match"] >= 0.95
        and a["balanced_button_exact_match"] >= 0.95
        and a["both_goals_correct"] >= 0.90
    )
    variants = report["after"]["validation"]
    report["validation_gate_passed"] = (
        all(
            m["button_exact_match"] >= 0.85
            and m["balanced_button_exact_match"] >= 0.80
            and m["both_goals_correct"] >= 0.75
            for m in variants["actual"]["by_template"].values()
        )
        and variants["actual"]["overall"]["balanced_button_exact_match"]
        - variants["shuffled_vision"]["overall"]["balanced_button_exact_match"]
        >= 0.15
    )
    if frozen != {
        k: fingerprint(getattr(runtime.module, k), frozen_only=args.train_adapters) for k in frozen
    }:
        raise RuntimeError("Frozen upstream weights changed")
    report["frozen_backbone_weights_unchanged"] = True
    report["entire_upstream_path_frozen"] = not args.train_adapters
    if report["train_gate_passed"] and report["validation_gate_passed"]:
        print("Earlier gates passed; evaluating reserved wording...", flush=True)
        sealed = read_manifest(args.data / "sealed.jsonl", runtime.module.policy_config)
        check_separation(rows["train"], sealed)
        report["sealed_wording"] = score(runtime, sealed, *prepare(runtime, sealed, args.cache))
    else:
        report["sealed_wording"] = "Not evaluated: earlier gates failed"
    runtime.metadata["robust_decoder"] = protocol
    print("Saving and verifying checkpoint...", flush=True)
    runtime.save(args.output / "bundle")
    loaded = TrainableRuntime.load(args.output / "bundle")
    original = runtime.predict(rows["validation"][0])
    reloaded = loaded.predict(rows["validation"][0])
    np.testing.assert_allclose(
        list(original["button_probabilities"].values()),
        list(reloaded["button_probabilities"].values()),
        atol=1e-6,
    )
    report["reload_matches"] = True
    report["live_inputs_sent"] = 0
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["after"], indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=Path("artifacts/decoder-probe-004/bundle"))
    p.add_argument("--data", type=Path, default=Path("artifacts/goal-curriculum-001"))
    p.add_argument("--cache", type=Path, default=Path("artifacts/vision-feature-cache"))
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--normalize-action-context", action="store_true")
    p.add_argument("--encoder-context", action="store_true")
    p.add_argument("--reset-decoder", action="store_true")
    p.add_argument("--train-adapters", action="store_true")
    p.add_argument("--matched-pairs", action="store_true")
    p.add_argument("--learning-rate", type=float, default=0.00003)
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
