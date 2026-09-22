"""Staged first-step button/camera learning; no live input or gameplay claims."""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from .adapter_buttons import train_adapter_buttons
from .decoder_probe import button_loss
from .gameplay_buttons import button_metrics
from .policy_data import check_separation, manifest_digest, read_manifest
from .policy_training import cache_examples, fingerprint
from .trainable_model import TrainableRuntime, decode_action_chunk


def metrics(rows, predictions, config):
    result = button_metrics(rows, [p["buttons"] for p in predictions], config.buttons)
    truth = np.array([r["action"]["mouse_delta"] for r in rows]) * 512
    mouse = np.array([p["mouse_delta"] for p in predictions]) * 512
    moving = np.abs(truth).max(-1) >= 4
    previous = [r["recorded_previous_actions"][-1]["buttons"] for r in rows]
    transitions = [i for i, r in enumerate(rows) if set(r["action"]["buttons"]) != set(previous[i])]
    result.update(
        {
            "mouse_mae_px": float(np.abs(mouse - truth).mean()),
            "moving_examples": int(moving.sum()),
            "moving_mouse_mae_px": (
                float(np.abs(mouse[moving] - truth[moving]).mean()) if moving.any() else None
            ),
            "moving_direction_accuracy": (
                float((np.sign(mouse[moving]) == np.sign(truth[moving])).mean())
                if moving.any()
                else None
            ),
            "button_transition_examples": len(transitions),
            "transition_exact": (
                float(
                    np.mean(
                        [
                            set(predictions[i]["buttons"]) == set(rows[i]["action"]["buttons"])
                            for i in transitions
                        ]
                    )
                )
                if transitions
                else None
            ),
        }
    )
    return result


def evaluate(runtime, rows, examples, variant="actual"):
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["game"], []).append(i)
    permutation = list(range(len(rows)))
    for group in groups.values():
        for j, i in enumerate(group):
            permutation[i] = group[(j + max(1, len(group) // 2)) % len(group)]
    predictions = []
    for i, (row, inputs) in enumerate(examples):
        if variant == "shuffled_vision":
            inputs = (*examples[permutation[i]][1][:2], *inputs[2:])
        elif variant in {"shuffled_goals", "no_previous_actions"}:
            changed = dict(row)
            if variant == "shuffled_goals":
                changed["goal"] = rows[permutation[i]]["goal"]
            else:
                changed["previous_actions"] = []
            inputs = (*inputs[:2], *runtime.prepare(changed))
        elif variant == "zero_vision":
            inputs = (mx.zeros_like(inputs[0]), *inputs[1:])
        out = runtime.module.from_features(*inputs)
        mx.eval(out)
        predictions.append(decode_action_chunk(out, runtime.module.policy_config)[0])
    return {
        **metrics(rows, predictions, runtime.module.policy_config),
        "predictions": [
            {"id": r["id"], "buttons": p["buttons"], "mouse_delta": p["mouse_delta"]}
            for r, p in zip(rows, predictions, strict=True)
        ],
    }


def configure_training(runtime, adapters):
    model = runtime.module
    model.freeze()
    model.actions.unfreeze()
    # No pointer/click-location, duration or future-chunk training in this pilot.
    for name in ("mouse", "pointer", "pointer_active", "duration"):
        getattr(model.actions, name).freeze()
    if adapters:
        model.connector.unfreeze()
        for layer in model.laya.encoder.layers:
            for name in ("Wqkv", "Wo"):
                linear = getattr(layer.attn, name)
                if hasattr(linear, "lora_a"):
                    linear.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
    names = [k for k, _ in tree_flatten(model.trainable_parameters())]
    if not names or any(
        not (
            k.startswith(("actions.", "connector."))
            or (k.startswith("laya.encoder.layers.") and k.endswith((".lora_a", ".lora_b")))
        )
        for k in names
    ):
        raise ValueError("Unexpected trainable backbone weights")


def train_stage(runtime, rows, examples, steps, lr, seed, adapters, log, camera_only=False):
    if camera_only:
        if adapters:
            raise ValueError("Camera-only fitting must preserve the learned button path")
        runtime.module.freeze()
        runtime.module.actions.chunk_mouse.unfreeze()
    else:
        configure_training(runtime, adapters)
    config = runtime.module.policy_config
    truth = np.array([[b in r["action"]["buttons"] for b in config.buttons] for r in rows])
    varying = truth.any(0) & ~truth.all(0)
    bins = np.array(config.mouse_bins)
    mouse_labels = np.abs(
        np.array([r["action"]["mouse_delta"] for r in rows])[..., None] - bins
    ).argmin(-1)
    # Balance classes using training data only; cap rare-bin weights.
    weights = []
    for axis in range(2):
        counts = np.bincount(mouse_labels[:, axis], minlength=len(bins))
        present = counts > 0
        w = np.ones(len(bins))
        w[present] = np.sqrt(counts[present].sum() / (present.sum() * counts[present]))
        weights.append(np.clip(w, 0.25, 4))
    weights = mx.array(np.stack(weights), mx.float32)
    camera_groups = {}
    for i, label in enumerate(mouse_labels):
        camera_groups.setdefault(tuple(label), []).append(i)
    camera_groups = list(camera_groups.values())
    if camera_only:
        # Sampling already balances the target bin pairs.
        weights = mx.ones_like(weights)
    # Decoder warm-up reuses exactly frozen upstream states.
    contexts = []
    if not adapters:
        for _, inputs in examples:
            h, _ = runtime.module.action_context(*inputs)
            mx.eval(h)
            contexts.append(mx.stop_gradient(h))
    rng = np.random.default_rng(seed)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0)

    def loss(model, indices):
        values = []
        for i in indices:
            output = (
                model.from_features(*examples[i][1])
                if adapters
                else model.actions(contexts[i][:, 0], contexts[i])
            )
            buttons = button_loss(
                output["buttons"], mx.array(truth[i : i + 1], mx.float32), varying
            )
            target = mx.array(mouse_labels[i : i + 1])
            camera = nn.losses.cross_entropy(output["chunk_mouse"][:, 0], target, reduction="none")
            camera = (camera * weights[mx.arange(2), target]).mean()
            values.append(camera if camera_only else buttons + 0.25 * camera)
        return sum(values) / len(values)

    gradient_fn = nn.value_and_grad(runtime.module, loss)
    for step in range(steps):
        started = time.perf_counter()
        indices = (
            [
                int(rng.choice(camera_groups[int(g)]))
                for g in rng.integers(len(camera_groups), size=4)
            ]
            if camera_only
            else rng.integers(len(rows), size=4).tolist()
        )
        value, gradient = gradient_fn(runtime.module, indices)
        gradient, norm = optim.clip_grad_norm(gradient, 1)
        mx.eval(value, norm)
        if not np.isfinite([float(value), float(norm)]).all():
            raise FloatingPointError("Nonfinite loss/gradient")
        optimizer.update(runtime.module, gradient)
        mx.eval(runtime.module.trainable_parameters(), optimizer.state)
        item = {
            "stage": "camera_only" if camera_only else "adapters" if adapters else "decoder",
            "step": step + 1,
            "loss": float(value),
            "gradient_norm": float(norm),
            "seconds": time.perf_counter() - started,
        }
        log.write(json.dumps(item) + "\n")
        if step == 0 or (step + 1) % 50 == 0:
            log.flush()
            print(json.dumps(item), flush=True)
    runtime.metadata["training_steps"] += steps
    runtime.metadata["action_training_examples"] += steps * 4


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    mx.random.seed(41)
    runtime = TrainableRuntime.load(args.bundle)
    if runtime.module.policy_config.action_chunk_size < 2:
        raise ValueError("Pilot needs the categorical camera head")
    rows = {
        s: read_manifest(args.data / f"{s}.jsonl", runtime.module.policy_config)
        for s in ("train", "validation")
    }
    check_separation(rows["train"], rows["validation"], holdout_games=True)
    for split in rows.values():
        for row in split:
            if row.get("action_supervision") != ["buttons", "relative_mouse"]:
                raise ValueError("Expected audited P2P human controls")
    protocol = {
        "source_bundle": str(args.bundle),
        "decoder_steps": args.decoder_steps,
        "adapter_steps": args.adapter_steps,
        "learning_rate": args.learning_rate,
        "seed": 41,
        "batch_size": 4,
        "recipe": args.recipe,
        "sampling": (
            "uniform game/button-set strata, then mouse-bin-pair strata"
            if args.recipe == "separate"
            else "uniform training examples"
        ),
        "loss": (
            "button BCE only, then camera CE with entire button path frozen"
            if args.recipe == "separate"
            else "varying-key BCE + 0.25 * training-frequency-weighted camera CE"
        ),
        "manifest_digests": {s: manifest_digest(r) for s, r in rows.items()},
        "scope": "First 50 ms buttons and raw mouse; whole-game Roblox holdout",
        "qualification": {
            "button_f1_min": 0.70,
            "vision_f1_margin_min": 0.05,
            "must_beat_persistence_buttons_and_moving_mouse": True,
        },
        "holdout_used_for_gradients_or_checkpoint_selection": False,
        "exploratory_followup_to_joint_pilot": args.recipe == "separate",
        "deployment_eligible": False,
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    configure_training(runtime, True)
    frozen = {
        k: fingerprint(getattr(runtime.module, k), frozen_only=True) for k in ("vision", "laya")
    }
    if args.recipe == "separate":
        # CachedExamples copies rows, so set sampling metadata before caching.
        for row in rows["train"]:
            row["_sampling_stratum"] = [row["game"], ",".join(row["action"]["buttons"])]
    examples = {s: cache_examples(runtime, r, args.cache) for s, r in rows.items()}
    report = {
        "protocol": protocol,
        "parameters": runtime.parameter_counts(),
        "before": {s: evaluate(runtime, r, examples[s]) for s, r in rows.items()},
    }
    (args.output / "before.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "steps.jsonl").open("w") as log:
        if args.recipe == "separate":
            configure_training(runtime, True)
            runtime.module.actions.chunk_mouse.freeze()
            train_adapter_buttons(
                runtime,
                examples["train"],
                args.adapter_steps,
                args.learning_rate,
                41,
                log,
                batch_size=4,
            )
            report["button_train"] = evaluate(runtime, rows["train"], examples["train"])
            (args.output / "button-stage.json").write_text(json.dumps(report, indent=2) + "\n")
            preserved = {
                k: fingerprint(getattr(runtime.module, k)) for k in ("vision", "laya", "connector")
            }
            button_head = fingerprint(runtime.module.actions.buttons)
            train_stage(
                runtime,
                rows["train"],
                examples["train"],
                args.decoder_steps,
                args.learning_rate,
                42,
                False,
                log,
                camera_only=True,
            )
            if preserved != {
                k: fingerprint(getattr(runtime.module, k)) for k in preserved
            } or button_head != fingerprint(runtime.module.actions.buttons):
                raise RuntimeError("Camera training altered the frozen button path")
            report["camera_training_preserved_button_path"] = True
        else:
            train_stage(
                runtime,
                rows["train"],
                examples["train"],
                args.decoder_steps,
                args.learning_rate,
                41,
                False,
                log,
            )
            report["decoder_train"] = evaluate(runtime, rows["train"], examples["train"])
            train_stage(
                runtime,
                rows["train"],
                examples["train"],
                args.adapter_steps,
                args.learning_rate,
                42,
                True,
                log,
            )
    configure_training(runtime, True)
    report["after"] = {}
    for split, r in rows.items():
        variants = {
            v: evaluate(runtime, r, examples[split], v)
            for v in (
                "actual",
                "shuffled_vision",
                "zero_vision",
                "shuffled_goals",
                "no_previous_actions",
            )
        }
        variants["repeat_previous_action"] = metrics(
            r, [x["recorded_previous_actions"][-1] for x in r], runtime.module.policy_config
        )
        variants["no_input"] = metrics(
            r, [{"buttons": [], "mouse_delta": [0, 0]} for _ in r], runtime.module.policy_config
        )
        report["after"][split] = variants
    if frozen != {k: fingerprint(getattr(runtime.module, k), frozen_only=True) for k in frozen}:
        raise RuntimeError("Pretrained backbone weights changed")
    val = report["after"]["validation"]
    actual, repeat = val["actual"], val["repeat_previous_action"]
    report["qualification_passed"] = bool(
        actual["button_micro_f1"] >= 0.70
        and actual["button_micro_f1"] > repeat["button_micro_f1"]
        and actual["button_micro_f1"] - val["shuffled_vision"]["button_micro_f1"] >= 0.05
        and actual["moving_mouse_mae_px"] is not None
        and actual["moving_mouse_mae_px"]
        < min(repeat["moving_mouse_mae_px"], val["no_input"]["moving_mouse_mae_px"])
    )
    report["frozen_backbone_weights_unchanged"] = True
    report["live_inputs_sent"] = 0
    runtime.metadata["p2p_pilot"] = protocol
    runtime.metadata["cross_game_capability_established"] = False
    runtime.save(args.output / "bundle")
    loaded = TrainableRuntime.load(args.output / "bundle")
    sample = examples["validation"][0][1]
    a, b = runtime.module.from_features(*sample), loaded.module.from_features(*sample)
    for head in ("buttons", "chunk_mouse"):
        np.testing.assert_allclose(np.asarray(a[head]), np.asarray(b[head]), atol=1e-6)
    report["reload_matches"] = True
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
    p.add_argument("--bundle", type=Path, default=Path("artifacts/gameplay-buttons-002/bundle"))
    p.add_argument("--data", type=Path, default=Path("artifacts/p2p-pilot-data-002"))
    p.add_argument("--cache", type=Path, default=Path("artifacts/vision-feature-cache"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--decoder-steps", type=int, default=400)
    p.add_argument("--adapter-steps", type=int, default=800)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--recipe", choices=("joint", "separate"), default="joint")
    args = p.parse_args()
    if min(args.decoder_steps, args.adapter_steps) < 1 or not 0 < args.learning_rate < 0.01:
        p.error("Positive step counts and a learning rate in (0, 0.01) required")
    run(args)


if __name__ == "__main__":
    main()
