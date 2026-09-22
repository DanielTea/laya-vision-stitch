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
    if args.train_eval_limit < 0:
        raise ValueError("Training evaluation limit must be nonnegative")
    if args.bundle:
        # Load config before model allocation so invalid data fails cheaply.
        saved = json.loads((args.bundle / "config.json").read_text())
        config = PolicyConfig(**saved["policy_config"])
    if args.buttons_file:
        config.buttons = tuple(json.loads(args.buttons_file.read_text()))
        config.__post_init__()
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
    if args.buttons_file and args.bundle:
        runtime.expand_buttons(config.buttons)
    if args.add_lora_rank:
        runtime.add_lora(args.add_lora_rank, args.add_lora_layers)
    if not args.bundle and config.connector_type == "aligned":
        descriptions = [r["description"] for r in train_rows if r.get("description")]
        if not descriptions:
            raise ValueError("Aligned initialization requires training descriptions")
        vectors = []
        for description in descriptions:
            ids = runtime.agent.tok(description, add_special_tokens=False)["input_ids"]
            if len(ids) > config.visual_slots:
                raise ValueError("Description exceeds visual slots")
            ids += [runtime.agent.tok.pad_token_id] * (config.visual_slots - len(ids))
            vectors.append(
                runtime.module.laya.encoder.embeddings.tok_embeddings(mx.array(ids)).astype(
                    mx.float32
                )
            )
        runtime.module.connector.anchors = mx.stack(vectors).mean(axis=0)
        runtime.metadata["anchor_initialization"] = "mean training-description token embeddings"
    print(json.dumps(runtime.parameter_counts()), flush=True)
    frozen_before = {
        k: fingerprint(getattr(runtime.module, k), frozen_only=True) for k in ("vision", "laya")
    }
    connector_before = fingerprint(runtime.module.connector)
    actions_before = fingerprint(runtime.module.actions)
    examples = cache_examples(runtime, train_rows, args.feature_cache)
    validation = cache_examples(runtime, validation_rows, args.feature_cache)
    button_weights = None
    if args.balance_buttons:
        action_rows = [r for r in train_rows if "action" in r]
        if not action_rows:
            raise ValueError("Button balancing requires recorded actions")
        positives = np.array(
            [sum(b in r["action"]["buttons"] for r in action_rows) for b in config.buttons]
        )
        button_weights = np.where(
            positives > 0,
            np.clip(np.sqrt((len(action_rows) - positives) / np.maximum(positives, 1)), 1, 10),
            1,
        ).tolist()
    training_eval = examples
    if args.train_eval_limit and len(examples) > args.train_eval_limit:
        from .policy_training import ExampleSubset

        indices = np.random.default_rng(args.seed + 1).choice(
            len(examples), args.train_eval_limit, replace=False
        )
        training_eval = ExampleSubset(examples, indices)
    before = {
        "train": evaluate(runtime, training_eval),
        "validation": evaluate(runtime, validation),
    }
    write_json(args.output / "before.json", before)
    with (args.output / "steps.jsonl").open("w") as handle:

        def log(record):
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if record["step"] == 1 or record["step"] % 100 == 0:
                print(json.dumps(record), flush=True)

        history = train(
            runtime,
            examples,
            args.steps,
            args.learning_rate,
            args.seed,
            log,
            args.shuffle_options,
            button_positive_weights=button_weights,
        )
    frozen_after = {
        k: fingerprint(getattr(runtime.module, k), frozen_only=True) for k in ("vision", "laya")
    }
    if frozen_before != frozen_after:
        raise RuntimeError("Pretrained weights changed; refusing to export")
    after = {
        "train": evaluate(runtime, training_eval),
        "validation": evaluate(runtime, validation),
        "validation_zero_visual": evaluate(runtime, validation, zero_visual=True),
    }
    if "training" in runtime.metadata:
        runtime.metadata.setdefault("training_history", []).append(runtime.metadata["training"])
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
        "shuffle_options": args.shuffle_options,
        "train_examples": len(examples),
        "train_evaluation_examples": len(training_eval),
        "button_positive_weights": button_weights,
        "datasets": list(
            {
                (r["provenance"]["dataset"], r["provenance"].get("revision", "unspecified")): {
                    k: r["provenance"].get(k, "unspecified")
                    for k in ("dataset", "revision", "license")
                }
                for r in train_rows
                if isinstance(r.get("provenance"), dict) and "dataset" in r["provenance"]
            }.values()
        ),
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
    fit.add_argument(
        "--buttons-file", type=Path, help="JSON action vocabulary; preserves old heads"
    )
    fit.add_argument(
        "--balance-buttons", action="store_true", help="Capped train-only positive weights"
    )
    fit.add_argument(
        "--shuffle-options",
        action="store_true",
        help="Randomize named choices during training; targets remain keyed by name",
    )
    fit.add_argument(
        "--add-lora-rank", type=int, default=0, help="Add LoRA to an unadapted checkpoint"
    )
    fit.add_argument("--add-lora-layers", type=int, default=2)
    fit.add_argument(
        "--train-eval-limit",
        type=int,
        default=0,
        help="Bound training-set evaluation; validation remains complete",
    )
    fit.add_argument(
        "--feature-cache", type=Path, help="Persistent frozen features, eight resident histories"
    )
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
