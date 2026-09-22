"""CLI for trainable visual stitching. All commands are offline; no input events."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np

from .policy_data import check_separation, manifest_digest, read_manifest
from .policy_training import cache_examples, evaluate, fingerprint, train
from .trainable_model import PolicyConfig, TrainableRuntime


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_training(args, config):
    if args.bundle:
        # Load config before model allocation so invalid data fails cheaply.
        saved = json.loads((args.bundle / "config.json").read_text())
        config = PolicyConfig(**saved["policy_config"])
    train_rows = read_manifest(args.train, config)
    validation_rows = read_manifest(args.validation, config)
    check_separation(train_rows, validation_rows, args.holdout_games)
    args.output.mkdir(parents=True, exist_ok=False)
    mx.random.seed(args.seed)
    print(
        "Loading pretrained backbones; training connector, action heads and configured LoRA only.",
        flush=True,
    )
    runtime = TrainableRuntime.load(args.bundle) if args.bundle else TrainableRuntime.build(config)
    print(json.dumps(runtime.parameter_counts()), flush=True)
    frozen_before = {
        k: fingerprint(getattr(runtime.module, k), frozen_only=True) for k in ("vision", "laya")
    }
    connector_before = fingerprint(runtime.module.connector)
    actions_before = fingerprint(runtime.module.actions)
    examples = cache_examples(runtime, train_rows)
    validation = cache_examples(runtime, validation_rows)
    before = {"train": evaluate(runtime, examples), "validation": evaluate(runtime, validation)}
    write_json(args.output / "before.json", before)
    with (args.output / "steps.jsonl").open("w") as handle:

        def log(record):
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if record["step"] == 1 or record["step"] % 5 == 0:
                print(json.dumps(record), flush=True)

        history = train(runtime, examples, args.steps, args.learning_rate, args.seed, log)
    frozen_after = {
        k: fingerprint(getattr(runtime.module, k), frozen_only=True) for k in ("vision", "laya")
    }
    if frozen_before != frozen_after:
        raise RuntimeError("Pretrained weights changed; refusing to export")
    after = {
        "train": evaluate(runtime, examples),
        "validation": evaluate(runtime, validation),
        "validation_zero_visual": evaluate(runtime, validation, zero_visual=True),
    }
    runtime.metadata["training"] = {
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "steps_this_run": args.steps,
        "train_digest": manifest_digest(train_rows),
        "validation_digest": manifest_digest(validation_rows),
        "holdout_games": args.holdout_games,
        "frozen_sha256": frozen_after,
        "teacher_examples": sum("teacher_probs" in r for r in train_rows),
        "action_examples": sum("action" in r for r in train_rows),
        "optimizer_state_saved": False,
    }
    runtime.save(args.output / "bundle")
    # Ensure exported checkpoint runs the entire raw-image graph, including vision.
    prediction = runtime.predict(validation_rows[0])
    loaded = TrainableRuntime.load(args.output / "bundle")
    reloaded = loaded.predict(validation_rows[0])
    numeric_keys = ("mouse_delta_normalized", "pointer_xy_normalized", "pointer_active_probability")
    for key in numeric_keys:
        np.testing.assert_allclose(prediction[key], reloaded[key], atol=1e-6, rtol=1e-6)
    for key in ("button_probabilities", "choice_probabilities"):
        if key in prediction:
            np.testing.assert_allclose(
                list(prediction[key].values()), list(reloaded[key].values()), atol=1e-6, rtol=1e-6
            )
    timings = [loaded.predict(validation_rows[0])["image_to_outputs_ms"] for _ in range(10)]
    report = {
        "parameters": runtime.parameter_counts(),
        "before": before,
        "after": after,
        "frozen_weights_unchanged": frozen_before == frozen_after,
        "connector_changed": connector_before != fingerprint(runtime.module.connector),
        "action_heads_changed": actions_before != fingerprint(runtime.module.actions),
        "nonzero_gradient_steps": sum(h["gradient_norm"] > 0 for h in history),
        "reload_matches": True,
        "warm_image_to_outputs_p50_ms": float(np.median(timings)),
        "warm_image_to_outputs_p95_ms": float(np.percentile(timings, 95)),
        "sample_prediction": prediction,
        "training": runtime.metadata["training"],
        "input_events_sent": 0,
        "cross_game_capability_established": False,
    }
    write_json(args.output / "report.json", report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("before", "after", "sample_prediction", "training")
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"Report: {args.output / 'report.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser(
        "train", help="Train connector/action heads; export one self-contained model"
    )
    fit.add_argument("--train", required=True, type=Path)
    fit.add_argument("--validation", required=True, type=Path)
    fit.add_argument("--output", required=True, type=Path)
    fit.add_argument("--bundle", type=Path, help="Continue from weights; optimizer restarts")
    fit.add_argument("--config", type=Path)
    fit.add_argument("--steps", type=int, default=50)
    fit.add_argument("--learning-rate", type=float, default=1e-4)
    fit.add_argument("--seed", type=int, default=17)
    fit.add_argument("--holdout-games", action="store_true")
    teacher = sub.add_parser("teacher", help="Run full Qwen offline to label named choices")
    teacher.add_argument("--manifest", required=True, type=Path)
    teacher.add_argument("--output", required=True, type=Path)
    teacher.add_argument("--config", type=Path)
    teacher.add_argument("--temperature", type=float, default=2.0)
    teacher.add_argument("--mode", choices=("generate", "logits"), default="generate")
    teacher.add_argument("--max-tokens", type=int, default=256)
    for name in ("predict", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--bundle", required=True, type=Path)
        command.add_argument("--manifest", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
    fixtures = sub.add_parser(
        "fixtures", help="Generate synthetic smoke data, not game demonstrations"
    )
    fixtures.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = (
        PolicyConfig(**json.loads(args.config.read_text()))
        if getattr(args, "config", None)
        else PolicyConfig()
    )
    if args.command == "train":
        run_training(args, config)
    elif args.command == "teacher":
        from .policy_teacher import label_examples

        rows = read_manifest(args.manifest, config, supervised=False)
        label_examples(
            rows, args.output, args.temperature, config.image_width, args.mode, args.max_tokens
        )
    elif args.command == "fixtures":
        from .policy_fixtures import create

        create(args.output)
        write_json(args.output / "config.json", asdict(config))
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        runtime = TrainableRuntime.load(args.bundle)
        rows = read_manifest(
            args.manifest, runtime.module.policy_config, supervised=args.command == "evaluate"
        )
        if args.command == "predict":
            result = [{"id": row["id"], **runtime.predict(row)} for row in rows]
        else:
            result = evaluate(runtime, cache_examples(runtime, rows))
        write_json(args.output, result)


if __name__ == "__main__":
    main()
