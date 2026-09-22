"""Isolate action decoding on paired goals, with the entire upstream model frozen."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from .policy_data import check_separation, manifest_digest, read_manifest
from .policy_training import cache_examples, fingerprint
from .trainable_model import TrainableRuntime


def freeze_decoder_only(runtime):
    runtime.module.freeze()
    heads = runtime.module.actions
    heads.unfreeze()
    # This diagnostic supervises buttons only. Preserve the other output heads.
    for name in ("mouse", "chunk_mouse", "pointer", "pointer_active", "duration"):
        if hasattr(heads, name):
            getattr(heads, name).freeze()
    names = [k for k, _ in tree_flatten(runtime.module.trainable_parameters())]
    if not names or any(not k.startswith("actions.") for k in names):
        raise RuntimeError("Only the action decoder may train")


def button_loss(logits, target, varying):
    """Equal weight to demonstrated keys and suppressing unrelated keys.

    Prevent 50 always-off keys from diluting the supervised changing key.
    This uses training labels to weight the loss, never an inference rule.
    """
    loss = nn.losses.binary_cross_entropy(logits, target, with_logits=True, reduction="none")
    active = mx.array(varying, mx.float32)[None]
    other = 1 - active
    return (
        mx.sum(loss * active) / mx.maximum(active.sum(), 1)
        + mx.sum(loss * other) / mx.maximum(other.sum(), 1)
    ) / 2


def groups(rows):
    result = {}
    for i, row in enumerate(rows):
        result.setdefault((row["goal"], tuple(row["action"]["buttons"])), []).append(i)
    return list(result.values())


def prepare(runtime, rows, cache, variant="actual"):
    examples = list(cache_examples(runtime, rows, cache))
    mapping = list(range(len(rows)))
    if variant == "shuffled_vision":
        by_game = {}
        for i, row in enumerate(rows):
            by_game.setdefault(row["game"], []).append(i)
        for group in by_game.values():
            offset = max(2, len(group) // 4 * 2)
            for j, i in enumerate(group):
                mapping[i] = group[(j + offset) % len(group)]
    contexts, answers = [], []
    for i, (row, inputs) in enumerate(examples):
        if variant == "zero_vision":
            inputs = (mx.zeros_like(inputs[0]), *inputs[1:])
        elif variant == "shuffled_vision":
            inputs = (*examples[mapping[i]][1][:2], *inputs[2:])
        h, choice = runtime.module.action_context(*inputs)
        h = mx.stop_gradient(h)
        mx.eval(h, choice)
        contexts.append(h)
        answers.append(list(row["choices"])[int(mx.argmax(choice[0]))])
    return contexts, answers


def audit(runtime, rows, contexts, choices):
    expected, predictions, probabilities = [], [], []
    for row, h in zip(rows, contexts, strict=True):
        output = runtime.module.actions(h[:, 0], h)
        p = np.asarray(mx.sigmoid(output["buttons"][0]))
        probabilities.append(p.tolist())
        predictions.append(p >= 0.5)
        expected.append(
            [b in row["action"]["buttons"] for b in runtime.module.policy_config.buttons]
        )
    truth, pred = np.array(expected), np.array(predictions)
    correct = (truth == pred).all(axis=1)
    tp, fp, fn = (truth & pred).sum(), (~truth & pred).sum(), (truth & ~pred).sum()
    paired = {}
    for i, row in enumerate(rows):
        key = tuple((f["sha256"], f["age_seconds"]) for f in row["frames"])
        paired.setdefault(key, []).append(i)
    pairs = list(paired.values())
    if any(len(p) != 2 or np.array_equal(truth[p[0]], truth[p[1]]) for p in pairs):
        raise ValueError("Each image must have two opposing action goals")
    return {
        "examples": len(rows),
        "button_exact_match": float(correct.mean()),
        "balanced_button_exact_match": float(np.mean([correct[g].mean() for g in groups(rows)])),
        "button_micro_f1": float(2 * tp / max(1, 2 * tp + fp + fn)),
        "both_goals_correct": float(np.mean([correct[g].all() for g in pairs])),
        "opposing_actions_rate": float(
            np.mean([not np.array_equal(pred[g[0]], pred[g[1]]) for g in pairs])
        ),
        "frozen_choice_accuracy": float(
            np.mean([c == r["answer"] for r, c in zip(rows, choices, strict=True)])
        ),
        "predictions": [
            {
                "id": r["id"],
                "expected": r["action"]["buttons"],
                "buttons": [
                    b for b, on in zip(runtime.module.policy_config.buttons, p, strict=True) if on
                ],
                "probabilities": probs,
            }
            for r, p, probs in zip(rows, pred, probabilities, strict=True)
        ],
    }


def train_decoder(runtime, rows, contexts, steps, learning_rate, seed, log, batch_size=1):
    heads = runtime.module.actions
    truth = np.array(
        [[b in r["action"]["buttons"] for b in runtime.module.policy_config.buttons] for r in rows]
    )
    varying = np.any(truth, axis=0) & ~np.all(truth, axis=0)
    if not varying.any():
        raise ValueError("Diagnostic requires changing action targets")
    strata = groups(rows)
    if batch_size < 1 or (batch_size > 1 and batch_size % len(strata)):
        raise ValueError("Batch size must be one or a multiple of the goal/action strata")
    rng = np.random.default_rng(seed)
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0)

    def loss(model, items):
        return sum(
            button_loss(model(h[:, 0], h)["buttons"], target, varying) for h, target in items
        ) / len(items)

    update = nn.value_and_grad(heads, loss)
    for step in range(steps):
        # Balanced over goal wording AND action target; majority menu state
        # cannot dominate the training distribution.
        selected_groups = (
            [strata[step % len(strata)]]
            if batch_size == 1
            else strata * (batch_size // len(strata))
        )
        indices = [int(rng.choice(group)) for group in selected_groups]
        items = [(contexts[i], mx.array(truth[i : i + 1], mx.float32)) for i in indices]
        value, grad = update(heads, items)
        grad, norm = optim.clip_grad_norm(grad, 1.0)
        mx.eval(value, norm)
        if not np.isfinite(float(value)) or not np.isfinite(float(norm)):
            raise FloatingPointError("Nonfinite decoder loss/gradient")
        optimizer.update(heads, grad)
        mx.eval(heads.parameters(), optimizer.state)
        record = {"step": step + 1, "loss": float(value), "gradient_norm": float(norm)}
        log.write(json.dumps(record) + "\n")
        if (step + 1) % 500 == 0:
            log.flush()
            print(json.dumps(record), flush=True)
    runtime.metadata["training_steps"] += steps
    runtime.metadata["action_training_examples"] += steps * batch_size


def run(args):
    if args.steps < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Positive steps and learning rate required")
    args.output.mkdir(parents=True, exist_ok=False)
    mx.random.seed(args.seed)
    runtime = TrainableRuntime.load(args.bundle)
    freeze_decoder_only(runtime)
    config = runtime.module.policy_config
    if config.action_chunk_size < 2:
        raise ValueError("This probe requires the temporal action-decoder checkpoint")
    rows = {
        s: [
            r
            for r in read_manifest(args.data / f"{s}-grounding.jsonl", config)
            if "-menu-instruction-" in r["id"]
        ]
        for s in ("train", "validation")
    }
    check_separation(rows["train"], rows["validation"])
    frozen = {k: fingerprint(getattr(runtime.module, k)) for k in ("vision", "connector", "laya")}
    auxiliaries = {
        k: fingerprint(getattr(runtime.module.actions, k))
        for k in ("mouse", "chunk_mouse", "pointer", "pointer_active", "duration")
    }
    protocol = {
        "source_checkpoint": str(args.bundle),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "training_presentations": args.steps * args.batch_size,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "training_scope": "action decoder, buttons only",
        "sampling": "equal goal/action strata; no preservation, mouse or language losses",
        "manifest_digests": {s: manifest_digest(r) for s, r in rows.items()},
        "train_gate": {"exact_min": 0.95, "balanced_min": 0.95, "both_goals_min": 0.90},
        "validation_gate": {
            "exact_min": 0.85,
            "balanced_min": 0.80,
            "both_goals_min": 0.75,
            "shuffle_balanced_margin_min": 0.15,
        },
        "source_selection": (
            "Decoder-only continuation after improved training fit; development validation is reused"
            if "decoder_probe" in runtime.metadata
            else "grounding-001 retains 88% conditional answer accuracy; selected on previously used development sessions"
        ),
        "source_decoder_probe": runtime.metadata.get("decoder_probe"),
        "dataset": "open-world-agents/D2E-480p",
        "license": "CC-BY-NC-4.0",
        "labels": "assistant-reviewed menu state and synthetic opposing instructions, not player intent",
        "not_measured": [
            "gameplay success",
            "cross-game transfer",
            "camera actions",
            "four-step action quality",
        ],
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    actual = {s: prepare(runtime, r, args.cache) for s, r in rows.items()}
    report = {
        "protocol": protocol,
        "parameters": runtime.parameter_counts(),
        "before": {s: audit(runtime, r, *actual[s]) for s, r in rows.items()},
    }
    with (args.output / "steps.jsonl").open("w") as log:
        train_decoder(
            runtime,
            rows["train"],
            actual["train"][0],
            args.steps,
            args.learning_rate,
            args.seed,
            log,
            batch_size=args.batch_size,
        )
    report["after"] = {s: {"actual": audit(runtime, r, *actual[s])} for s, r in rows.items()}
    for split, r in rows.items():
        for variant in ("zero_vision", "shuffled_vision"):
            report["after"][split][variant] = audit(
                runtime, r, *prepare(runtime, r, args.cache, variant)
            )
    for variant in ("paraphrased_goal", "reversed_choices"):
        changed = []
        for row in rows["validation"]:
            item = dict(row)
            if variant == "reversed_choices":
                item["choices"] = dict(reversed(list(row["choices"].items())))
            else:
                opened = "overlay is open;" in row["goal"]
                state = "open" if opened else "closed"
                item["goal"] = (
                    f"If the large game menu is {state}, keep W pressed. Otherwise let go of every key."
                )
            changed.append(item)
        report["after"]["validation"][variant] = audit(
            runtime, changed, *prepare(runtime, changed, args.cache)
        )
    if frozen != {k: fingerprint(getattr(runtime.module, k)) for k in frozen}:
        raise RuntimeError("Frozen visual/language path changed, including connector or LoRA")
    if auxiliaries != {k: fingerprint(getattr(runtime.module.actions, k)) for k in auxiliaries}:
        raise RuntimeError("Unsupervised action heads changed")
    report["frozen_path_unchanged"] = True
    report["auxiliary_heads_unchanged"] = True
    for split, limits in [
        ("train", protocol["train_gate"]),
        ("validation", protocol["validation_gate"]),
    ]:
        a = report["after"][split]["actual"]
        passed = (
            a["button_exact_match"] >= limits["exact_min"]
            and a["balanced_button_exact_match"] >= limits["balanced_min"]
            and a["both_goals_correct"] >= limits["both_goals_min"]
        )
        if split == "validation":
            passed = (
                passed
                and a["balanced_button_exact_match"]
                - report["after"][split]["shuffled_vision"]["balanced_button_exact_match"]
                >= limits["shuffle_balanced_margin_min"]
            )
        report[split + "_gate_passed"] = bool(passed)
    runtime.metadata["decoder_probe"] = {**protocol, "frozen_path_unchanged": True}
    runtime.save(args.output / "bundle")
    loaded = TrainableRuntime.load(args.output / "bundle")
    before, after = runtime.predict(rows["validation"][0]), loaded.predict(rows["validation"][0])
    np.testing.assert_allclose(
        list(before["button_probabilities"].values()),
        list(after["button_probabilities"].values()),
        atol=1e-6,
    )
    report["reload_matches"] = True
    report["live_inputs_sent"] = 0
    report["joint_training_resumed"] = False
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                s: {
                    v: {k: val for k, val in m.items() if k != "predictions"}
                    for v, m in report["after"][s].items()
                }
                for s in rows
            },
            indent=2,
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=Path("artifacts/fit-gate-grounding-001/bundle"))
    p.add_argument("--data", type=Path, default=Path("artifacts/learning-gate-002"))
    p.add_argument("--cache", type=Path, default=Path("artifacts/vision-feature-cache"))
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=0.0001)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
