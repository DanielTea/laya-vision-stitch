"""Fit a tiny real-image curriculum; assess vision, session transfer and retention."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .d2e_data import BUTTONS
from .game_eval import infer, metrics
from .policy_data import check_separation, manifest_digest, read_manifest
from .policy_training import cache_examples, evaluate, fingerprint, train
from .trainable_model import TrainableRuntime, decode_action_chunk


def audit(runtime, examples):
    rows = [r for r, _ in examples]
    actual = infer(runtime, examples, "actual")
    zero = infer(runtime, examples, "zero_vision")
    # Shuffle within each game so game identity cannot explain the visual margin.
    indices = {}
    for i, row in enumerate(rows):
        indices.setdefault(row["game"], []).append(i)
    mapping = {}
    for group in indices.values():
        mapping.update(
            {i: group[(j + max(1, len(group) // 2)) % len(group)] for j, i in enumerate(group)}
        )
    shuffled = [
        (r, (*examples[mapping[i]][1][:2], *inputs[2:])) for i, (r, inputs) in enumerate(examples)
    ]
    shuffled_pred = infer(runtime, shuffled, "actual")
    repeat = [r["recorded_previous_actions"][-1] for r in rows]
    output = {
        name: metrics(rows, pred, runtime.module.policy_config.buttons)
        for name, pred in [
            ("actual", actual),
            ("zero_vision", zero),
            ("shuffled_vision", shuffled_pred),
            ("repeat_previous_action", repeat),
        ]
    }
    output["predictions"] = [
        {"id": r["id"], "target": r["action"], "actual": a, "shuffled": s}
        for r, a, s in zip(rows, actual, shuffled_pred, strict=True)
    ]
    if runtime.module.policy_config.action_chunk_size > 1:
        targets, predicted = [], []
        for row, inputs in examples:
            result = runtime.module.from_features(*inputs)
            mx.eval(result)
            chunk = decode_action_chunk(result, runtime.module.policy_config)
            targets.extend({"action": action} for action in row["action_chunk"])
            predicted.extend(chunk)
        output["all_chunk_steps"] = metrics(
            targets, predicted, runtime.module.policy_config.buttons
        )
    return output


def grounding_audit(runtime, subset):
    report = {}
    for name, marker in (("factual", "-menu-question-"), ("conditional", "-menu-instruction-")):
        selected = [(row, inputs) for row, inputs in subset if marker in row["id"]]
        groups = {}
        for i, (row, _) in enumerate(selected):
            groups.setdefault(row["game"], []).append(i)
        mapping = {}
        for group in groups.values():
            # Two opposing prompts belong to each screenshot pair. Move by an
            # even offset to keep each pair's images aligned under shuffling.
            offset = max(2, (len(group) // 4) * 2)
            mapping.update({i: group[(j + offset) % len(group)] for j, i in enumerate(group)})
        shuffled = [
            (row, (*selected[mapping[i]][1][:2], *inputs[2:]))
            for i, (row, inputs) in enumerate(selected)
        ]
        report[name] = {}
        for variant, examples, zero in (
            ("actual", selected, False),
            ("zero_vision", selected, True),
            ("shuffled_vision", shuffled, False),
        ):
            result = evaluate(runtime, examples, zero_visual=zero)
            by_answer = {}
            for (row, _), prediction in zip(examples, result["predictions"], strict=True):
                by_answer.setdefault((row["goal"], row["answer"]), []).append(
                    prediction["choice"] == row["answer"]
                )
            result["answer_balanced_choice_accuracy"] = float(
                np.mean([np.mean(values) for values in by_answer.values()])
            )
            report[name][variant] = {k: v for k, v in result.items() if k != "predictions"}
    return report


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    mx.random.seed(17)
    runtime = TrainableRuntime.load(args.bundle)
    runtime.expand_buttons(BUTTONS)
    if args.temporal:
        runtime.upgrade_temporal()
    config = runtime.module.policy_config
    train_rows = read_manifest(args.data / "train.jsonl", config)
    validation_rows = read_manifest(args.data / "validation.jsonl", config)
    check_separation(train_rows, validation_rows)
    replay_rows = read_manifest(args.replay, config)
    if not 0 <= args.replay_count <= len(replay_rows):
        raise ValueError("Replay count exceeds available examples")
    replay_rows = [
        replay_rows[i]
        for i in np.random.default_rng(41).choice(
            len(replay_rows), args.replay_count, replace=False
        )
    ]
    retention_rows = read_manifest(args.retention, config)
    check_separation(train_rows + replay_rows, retention_rows)
    frozen = {
        k: fingerprint(getattr(runtime.module, k), frozen_only=True) for k in ("vision", "laya")
    }
    cache = args.cache
    real = list(cache_examples(runtime, train_rows, cache))
    validation = list(cache_examples(runtime, validation_rows, cache))
    replay = list(cache_examples(runtime, replay_rows, cache))
    retention = list(cache_examples(runtime, retention_rows, cache))
    # The unchanged source checkpoint is an offline preservation teacher only.
    teacher = TrainableRuntime.load(args.preserve_bundle)
    teacher_buttons = tuple(teacher.module.policy_config.buttons)
    if tuple(config.buttons[: len(teacher_buttons)]) != teacher_buttons:
        raise ValueError("Preservation button vocabulary must match the source prefix")
    for i, (row, inputs) in enumerate(replay):
        original = teacher.module.from_features(*inputs[:2], *teacher.prepare(row))
        mx.eval(original)
        labelled = dict(
            row,
            _preserve_buttons=np.asarray(mx.sigmoid(original["buttons"][0])).tolist(),
            _preserve_choices=np.asarray(mx.softmax(original["choices"][0])).tolist(),
            _preserve_mouse=np.asarray(original["mouse"][0]).tolist(),
            _preserve_state=np.asarray(original["visual_state"]).tolist(),
        )
        replay[i] = labelled, inputs
    del teacher
    counts = np.array(
        [sum(b in r["action"]["buttons"] for r in train_rows) for b in config.buttons]
    )
    weights = np.where(
        counts > 0, np.clip(np.sqrt((len(real) - counts) / np.maximum(counts, 1)), 1, 10), 1
    ).tolist()
    examples = real + replay
    grounding = None
    if args.grounding:
        grounding_rows = read_manifest(args.data / "train-grounding.jsonl", config)
        grounding_validation = read_manifest(args.data / "validation-grounding.jsonl", config)
        check_separation(grounding_rows, grounding_validation)
        grounding_train = list(cache_examples(runtime, grounding_rows, cache))
        grounding = list(cache_examples(runtime, grounding_validation, cache))
        # Explicit task mixture: real imitation receives half the sample weight
        # when replay_count=64; factual/conditional probes receive one third.
        examples = real * 6 + replay * 2 + grounding_train
    logs = []
    with (args.output / "steps.jsonl").open("w") as f:

        def log(row):
            f.write(json.dumps(row) + "\n")
            f.flush()
            if row["step"] == 1 or row["step"] % 250 == 0:
                print(json.dumps(row), flush=True)

        logs = train(
            runtime,
            examples,
            steps=args.steps,
            learning_rate=args.learning_rate,
            seed=17,
            log=log,
            button_positive_weights=weights,
        )
    if frozen != {
        k: fingerprint(getattr(runtime.module, k), frozen_only=True) for k in ("vision", "laya")
    }:
        raise RuntimeError("Frozen backbone changed")
    report = {
        "train": audit(runtime, real),
        "validation": audit(runtime, validation),
        "retention": {k: v for k, v in evaluate(runtime, retention).items() if k != "predictions"},
        "parameters": runtime.parameter_counts(),
        "frozen_weights_unchanged": True,
        "steps": args.steps,
        "variant": config.connector_type,
        "learning_rate": args.learning_rate,
        "replay_count": args.replay_count,
        "source_checkpoint": str(args.bundle),
        "preservation_checkpoint": str(args.preserve_bundle),
        "manifest_digests": {
            "train": manifest_digest(train_rows),
            "validation": manifest_digest(validation_rows),
        },
        "training_slots": {
            "total": len(examples),
            "real": len(real) * (6 if args.grounding else 1),
            "replay": len(replay) * (2 if args.grounding else 1),
            "grounding": len(grounding_train) if args.grounding else 0,
        },
        "previous_actions_in_prompt": False,
        "preservation_teacher_at_inference": False,
        "nonzero_gradient_steps": sum(x["gradient_norm"] > 0 for x in logs),
    }
    if grounding is not None:
        report["grounding"] = {}
        for name, subset in (("train", grounding_train), ("validation", grounding)):
            report["grounding"][name] = grounding_audit(runtime, subset)
        report["grounding_labels"] = (
            "assistant-reviewed menu state and synthetic conditional-control probes; not expert intent"
        )
    criteria = json.loads((args.data / "protocol.json").read_text())
    a = report["train"]["actual"]
    v = report["validation"]["actual"]
    small = criteria["small_fit_gate"]
    general = criteria["generalization_gate"]
    report["small_fit_passed"] = all(
        [
            a["button_micro_f1"] >= small["button_micro_f1_min"],
            a["button_exact_match"] >= small["button_exact_match_min"],
            a["mouse_direction_accuracy"] >= small["mouse_direction_accuracy_min"],
        ]
    )
    report["session_generalization_passed"] = all(
        [
            v["button_micro_f1"] >= general["button_micro_f1_min"],
            v["button_micro_f1"] - report["validation"]["shuffled_vision"]["button_micro_f1"]
            >= general["image_shuffle_f1_margin_min"],
            v["button_micro_f1"]
            > report["validation"]["repeat_previous_action"]["button_micro_f1"],
        ]
    )
    runtime.metadata["learning_gate"] = {
        k: v for k, v in report.items() if k not in ("train", "validation", "retention")
    }
    runtime.metadata["data_provenance"] = {
        "dataset": "open-world-agents/D2E-480p",
        "license": "CC-BY-NC-4.0",
        "manifest": str(args.data),
        "source_checkpoint": str(args.bundle),
    }
    runtime.save(args.output / "bundle")
    loaded = TrainableRuntime.load(args.output / "bundle")
    expected = runtime.predict(train_rows[0])
    actual = loaded.predict(train_rows[0])
    np.testing.assert_allclose(
        list(expected["button_probabilities"].values()),
        list(actual["button_probabilities"].values()),
        atol=1e-6,
    )
    report["reload_matches"] = True
    report["warm_latency_p50_ms"] = float(
        np.median([loaded.predict(validation_rows[i])["image_to_outputs_ms"] for i in range(5)])
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("train", "validation")}, indent=2),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=Path("artifacts/scaled-robust-001/bundle"))
    p.add_argument("--data", type=Path, default=Path("artifacts/learning-gate-001"))
    p.add_argument(
        "--replay",
        type=Path,
        default=Path("artifacts/scaled-recovery-data-001/canonical-train.jsonl"),
    )
    p.add_argument(
        "--retention", type=Path, default=Path("artifacts/robust-final-test-001/validation.jsonl")
    )
    p.add_argument("--cache", type=Path, default=Path("artifacts/vision-feature-cache"))
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--learning-rate", type=float, default=0.0001)
    p.add_argument("--temporal", action="store_true")
    p.add_argument("--replay-count", type=int, default=64)
    p.add_argument("--grounding", action="store_true")
    p.add_argument(
        "--preserve-bundle", type=Path, default=Path("artifacts/scaled-robust-001/bundle")
    )
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
