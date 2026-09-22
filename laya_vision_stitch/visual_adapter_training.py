"""Train visual residuals without changing either pretrained backbone or parent policy."""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .decoder_probe import button_loss
from .p2p_training import evaluate, metrics
from .policy_data import check_separation, manifest_digest, read_manifest
from .policy_training import cache_examples, fingerprint
from .trainable_model import TrainableRuntime
from .visual_action_adapter import attach, encode_action_history


def evaluate_grouped(runtime, rows, examples, variant="actual"):
    result = evaluate(runtime, rows, examples, variant)
    games = sorted({r["game"] for r in rows})
    result["by_game"] = {}
    for game in games:
        indices = [i for i, r in enumerate(rows) if r["game"] == game]
        result["by_game"][game] = metrics(
            [rows[i] for i in indices],
            [result["predictions"][i] for i in indices],
            runtime.module.policy_config,
        )
    return result


def compare_outputs(reference, candidate):
    errors = {}
    for key in ("buttons", "chunk_mouse"):
        a, b = np.asarray(reference[key]), np.asarray(candidate[key])
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(
            a >= 0 if key == "buttons" else a.argmax(-1),
            b >= 0 if key == "buttons" else b.argmax(-1),
        )
        errors[key] = float(np.max(np.abs(a - b)))
    return errors


def cached_contexts(runtime, rows, examples):
    """Only frozen computations are reused; both history conditions are real forward passes."""
    result = []
    for i, (row, inputs) in enumerate(examples):
        conditions = []
        for history in (True, False):
            prepared = inputs[2:] if history else runtime.prepare({**row, "previous_actions": []})
            h, _ = runtime.module.action_context(*inputs[:2], *prepared)
            base = runtime.module.actions(h[:, 0], h)
            mx.eval(h, base)
            conditions.append(
                (
                    mx.stop_gradient(h),
                    {k: mx.stop_gradient(base[k]) for k in ("chunk_buttons", "chunk_mouse")},
                )
            )
        result.append(conditions)
        if (i + 1) % 100 == 0:
            print(f"Prepared frozen contexts {i + 1}/{len(rows)}", flush=True)
    return result


def train(runtime, rows, examples, contexts, steps, lr, seed, dropout, log):
    config = runtime.module.policy_config
    adapter = runtime.module.visual_actions
    truth = np.array([[b in r["action"]["buttons"] for b in config.buttons] for r in rows])
    varying = truth.any(0) & ~truth.all(0)
    bins = np.array(config.mouse_bins)
    mouse = np.array([r["action"]["mouse_delta"] for r in rows])
    labels = np.abs(mouse[..., None] - bins).argmin(-1)
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault((row["game"], tuple(row["action"]["buttons"])), []).append(i)
    groups = list(groups.values())
    weights = []
    for axis in range(2):
        counts = np.bincount(labels[:, axis], minlength=len(bins))
        active = counts > 0
        weight = np.ones(len(bins))
        weight[active] = np.sqrt(counts.sum() / (active.sum() * counts[active]))
        weights.append(np.clip(weight, 0.25, 4))
    weights = mx.array(np.stack(weights), mx.float32)
    rng = np.random.default_rng(seed)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)
    history = [
        [encode_action_history(row, config), encode_action_history({}, config)]
        if config.numeric_action_history
        else [None, None]
        for row in rows
    ]

    def loss(model, button_items, camera_items):
        button_losses, camera_losses = [], []
        for i, condition in button_items:
            features, coords = examples[i][1][:2]
            context, base = contexts[i][condition]
            logits = (
                base["chunk_buttons"][:, 0]
                + model.buttons(features, coords, context, history[i][condition])[:, 0]
            )
            button_losses.append(
                button_loss(logits, mx.array(truth[i : i + 1], mx.float32), varying)
            )
        for i, condition in camera_items:
            features, coords = examples[i][1][:2]
            context, base = contexts[i][condition]
            delta = model.camera(features, coords, context, history[i][condition]).reshape(
                1, -1, 2, len(bins)
            )[:, 0]
            logits = base["chunk_mouse"][:, 0] + delta
            target = mx.array(labels[i : i + 1])
            ce = nn.losses.cross_entropy(logits, target, reduction="none")
            ce = (ce * weights[mx.arange(2), target]).mean()
            # Penalize confident large camera errors, including false motion.
            distance = mx.abs(mx.array(bins)[None, None] - mx.array(mouse[i : i + 1])[..., None])
            expected_error = (mx.softmax(logits, -1) * distance).sum(-1).mean()
            camera_losses.append(ce + 4 * expected_error)
        return sum(button_losses) / len(button_losses) + sum(camera_losses) / len(camera_losses)

    gradient_fn = nn.value_and_grad(adapter, loss)
    for step in range(steps):
        started = time.perf_counter()
        button_indices = [
            int(rng.integers(len(rows))),
            int(rng.choice(groups[int(rng.integers(len(groups)))])),
        ]
        camera_indices = rng.integers(len(rows), size=2).tolist()
        button_items = [(i, int(rng.random() < dropout)) for i in button_indices]
        camera_items = [(i, int(rng.random() < dropout)) for i in camera_indices]
        value, grads = gradient_fn(adapter, button_items, camera_items)
        # Disjoint gradients; a difficult camera sample cannot clip button learning.
        norms = []
        for branch in ("buttons", "camera"):
            grads[branch], norm = optim.clip_grad_norm(grads[branch], 1.0)
            norms.append(norm)
        mx.eval(value, norms)
        if not np.isfinite([float(value), *[float(n) for n in norms]]).all():
            raise FloatingPointError("Nonfinite adapter update")
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)
        item = {"step": step + 1, "loss": float(value), "seconds": time.perf_counter() - started}
        log.write(json.dumps(item) + "\n")
        if step == 0 or (step + 1) % 100 == 0:
            log.flush()
            print(json.dumps(item), flush=True)
    runtime.metadata["training_steps"] += steps
    runtime.metadata["action_training_examples"] += steps * 4


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    mx.random.seed(53)
    runtime = TrainableRuntime.load(args.bundle)
    splits = ["train", "validation"]
    if (args.data / "test.jsonl").exists():
        splits.append("test")
    rows = {
        s: read_manifest(args.data / f"{s}.jsonl", runtime.module.policy_config) for s in splits
    }
    check_separation(rows["train"], rows["validation"], holdout_games=not args.session_validation)
    if "test" in rows:
        check_separation(rows["train"], rows["test"], holdout_games=True)
        check_separation(rows["validation"], rows["test"], holdout_games=True)
    frozen = {
        k: fingerprint(getattr(runtime.module, k))
        for k in ("vision", "laya", "connector", "actions")
    }
    examples = {s: cache_examples(runtime, r, args.cache) for s, r in rows.items()}
    before = {}
    for split, r in rows.items():
        print(f"Evaluating parent: {split}", flush=True)
        before[split] = evaluate_grouped(runtime, r, examples[split])
    reference = runtime.module.from_features(*examples["validation"][0][1])
    mx.eval(reference)
    attach(runtime, numeric_history=args.numeric_history)
    if args.numeric_history:
        examples = {
            s: [(row, (*inputs[:2], *runtime.prepare(row))) for row, inputs in items]
            for s, items in examples.items()
        }
    converted = runtime.module.from_features(*examples["validation"][0][1])
    conversion_errors = compare_outputs(reference, converted)
    protocol = {
        "source": str(args.bundle),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "history_dropout": args.history_dropout,
        "numeric_action_history": args.numeric_history,
        "validation_split": "recording" if args.session_validation else "game",
        "seed": 53,
        "trainable": "Independent visual button/camera residuals only",
        "sampling": "buttons half natural/half game-action strata; camera natural",
        "manifest_digests": {s: manifest_digest(r) for s, r in rows.items()},
        "exploratory_reused_development_game": True,
        "holdout_used_for_training_or_checkpoint_selection": False,
        "deployment_eligible": False,
        "scope": "First-step buttons and camera, no future chunks/pointer supervision",
        "supervised_action_steps": 1,
        "action_step_seconds": 0.05,
    }
    report = {
        "protocol": protocol,
        "parameters": runtime.parameter_counts(),
        "conversion_actions_match": True,
        "conversion_max_abs_error": conversion_errors,
        "before": before,
    }
    (args.output / "protocol.json").write_text(json.dumps(report, indent=2) + "\n")
    contexts = cached_contexts(runtime, rows["train"], examples["train"])
    with (args.output / "steps.jsonl").open("w") as log:
        train(
            runtime,
            rows["train"],
            examples["train"],
            contexts,
            args.steps,
            args.learning_rate,
            53,
            args.history_dropout,
            log,
        )
    report["after"] = {}
    for split, r in rows.items():
        report["after"][split] = {}
        for v in (
            "actual",
            "shuffled_vision",
            "zero_vision",
            "shuffled_goals",
            "no_previous_actions",
        ):
            print(f"Evaluating adapter: {split}/{v}", flush=True)
            report["after"][split][v] = evaluate_grouped(runtime, r, examples[split], v)
        report["after"][split]["repeat_previous_action"] = metrics(
            r, [x["recorded_previous_actions"][-1] for x in r], runtime.module.policy_config
        )
        report["after"][split]["no_input"] = metrics(
            r, [{"buttons": [], "mouse_delta": [0, 0]} for _ in r], runtime.module.policy_config
        )
        # Persist each completed split, even if a later export fails.
        (args.output / "partial-report.json").write_text(json.dumps(report, indent=2) + "\n")
    if frozen != {k: fingerprint(getattr(runtime.module, k)) for k in frozen}:
        raise RuntimeError("Parent policy changed")
    report["entire_parent_policy_unchanged"] = True
    v = report["after"]["validation"]
    a, baseline = v["actual"], v["repeat_previous_action"]
    report["qualification_passed"] = bool(
        a["button_micro_f1"] >= 0.70
        and a["button_micro_f1"] > baseline["button_micro_f1"]
        and a["button_micro_f1"] - v["shuffled_vision"]["button_micro_f1"] >= 0.05
        and a["moving_mouse_mae_px"]
        < min(baseline["moving_mouse_mae_px"], v["no_input"]["moving_mouse_mae_px"])
    )
    runtime.metadata["visual_adapter_experiment"] = protocol
    runtime.save(args.output / "bundle")
    loaded = TrainableRuntime.load(args.output / "bundle")
    inputs = examples["validation"][0][1]
    a, b = runtime.module.from_features(*inputs), loaded.module.from_features(*inputs)
    report["reload_max_abs_error"] = compare_outputs(a, b)
    report["reload_matches"] = True
    report["live_inputs_sent"] = 0
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                s: {v: {k: x for k, x in m.items() if k != "predictions"} for v, m in vs.items()}
                for s, vs in report["after"].items()
            },
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=Path("artifacts/p2p-pilot-004/bundle"))
    p.add_argument("--data", type=Path, default=Path("artifacts/p2p-pilot-data-002"))
    p.add_argument("--cache", type=Path, default=Path("artifacts/vision-feature-cache"))
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--history-dropout", type=float, default=0.5)
    p.add_argument("--session-validation", action="store_true")
    p.add_argument("--numeric-history", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.steps < 1 or not 0 < args.learning_rate < 0.01 or not 0 <= args.history_dropout <= 1:
        p.error("Invalid steps, learning rate or dropout")
    run(args)


if __name__ == "__main__":
    main()
