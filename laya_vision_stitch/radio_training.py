"""Small-parameter RADIO-to-Laya control learning with run-disjoint evaluation."""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from .decoder_probe import button_loss
from .gameplay_buttons import button_metrics
from .p2p_training import configure_training
from .policy_data import check_separation, manifest_digest, read_manifest
from .policy_training import cache_examples, fingerprint
from .trainable_model import PolicyConfig, TrainableRuntime, decode_action_chunk


def score(runtime, rows, examples, shuffle=False):
    by_game = {}
    for i, row in enumerate(rows):
        by_game.setdefault(row["game"], []).append(i)
    mapping = {
        i: group[(j + max(1, len(group) // 2)) % len(group)]
        for group in by_game.values()
        for j, i in enumerate(group)
    }
    predictions = []
    for i, (_, inputs) in enumerate(examples):
        if shuffle:
            inputs = (*examples[mapping[i]][1][:2], *inputs[2:])
        out = runtime.module.from_features(*inputs)
        mx.eval(out)
        predictions.append(decode_action_chunk(out, runtime.module.policy_config)[0])
    result = {}
    for name, indices in {"all": list(range(len(rows))), **by_game}.items():
        selected_rows, selected_predictions = (
            [rows[i] for i in indices],
            [predictions[i] for i in indices],
        )
        metrics = button_metrics(
            selected_rows,
            [a["buttons"] for a in selected_predictions],
            runtime.module.policy_config.buttons,
        )
        per_button = {}
        for key in ("tab", "1", "w", "mouse_right"):
            true = np.array([key in r["action"]["buttons"] for r in selected_rows])
            pred = np.array([key in r["buttons"] for r in selected_predictions])
            per_button[key] = {
                "precision": float((true & pred).sum() / max(1, pred.sum())),
                "recall": float((true & pred).sum() / max(1, true.sum())),
                "targets": int(true.sum()),
                "predictions": int(pred.sum()),
            }
        truth = np.array([r["action"]["mouse_delta"] for r in selected_rows]) * 512
        predicted = np.array([r["mouse_delta"] for r in selected_predictions]) * 512
        metrics.update(buttons=per_button, mouse_mae_px=float(np.abs(truth - predicted).mean()))
        result[name] = metrics
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    mx.random.seed(71)
    config = PolicyConfig(
        visual_slots=36,
        connector_width=256,
        max_frames=1,
        image_width=384,
        lora_rank=8,
        lora_layers=2,
        connector_type="spatial",
        action_chunk_size=2,
        normalize_action_context=True,
        action_context_source="encoder",
        visual_fusion_layers=args.visual_fusion_layers,
        visual_action_adapter=args.visual_action_adapter,
        buttons=tuple(json.loads(Path("configs/desktop-buttons.json").read_text())),
    )
    runtime = TrainableRuntime.build(config, radio_source=args.source)
    configure_training(runtime, True)
    if args.visual_action_adapter:
        runtime.module.visual_actions.unfreeze()
    frozen = {
        name: fingerprint(getattr(runtime.module, name), frozen_only=True)
        for name in ("vision", "laya")
    }
    rows = {
        s: read_manifest(args.data / f"{s}.jsonl", config) for s in ("train", "validation", "test")
    }
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        check_separation(rows[a], rows[b])
    if args.small_fit_per_action:
        buckets = {}
        for row in rows["train"]:
            if row["game"] == "Hordes.io":
                buckets.setdefault(tuple(row["action"]["buttons"]), []).append(row)
        rng = np.random.default_rng(71)
        rows["train"] = [
            bucket[int(i)]
            for bucket in buckets.values()
            for i in rng.choice(
                len(bucket), min(len(bucket), args.small_fit_per_action), replace=False
            )
        ]
    runtime.metadata["first_step_control_only"] = True
    report = {
        "config": runtime.metadata,
        "parent_fingerprints": frozen,
        "manifest_digests": {s: manifest_digest(r) for s, r in rows.items()},
        "parameters": runtime.parameter_counts(),
        "selection": [],
        "experiment": {
            "small_fit_per_action": args.small_fit_per_action,
            "batch_size": args.batch_size,
        },
    }
    examples = {s: cache_examples(runtime, r, args.cache) for s, r in rows.items() if s != "test"}
    train_rows = rows["train"]
    truth = np.array([[b in r["action"]["buttons"] for b in config.buttons] for r in train_rows])
    varying = truth.any(0) & ~truth.all(0)
    camera_labels = np.abs(
        np.array([r["action"]["mouse_delta"] for r in train_rows])[..., None]
        - np.array(config.mouse_bins)
    ).argmin(-1)
    groups = {}
    for i, row in enumerate(train_rows):
        domain = "hordes" if row["game"] == "Hordes.io" else "replay"
        groups.setdefault(domain, {}).setdefault(
            (row["game"], tuple(row["action"]["buttons"])), []
        ).append(i)
    groups = {d: list(g.values()) for d, g in groups.items()}
    # Expected key frequency under the actual stratified sampler, not the raw
    # dataset frequency. This only weights training supervision.
    probability = np.mean(
        [
            np.mean([truth[indices].mean(0) for indices in buckets], axis=0)
            for buckets in groups.values()
        ],
        axis=0,
    )
    positive_weights = (
        mx.array(np.clip((1 - probability) / np.maximum(probability, 1e-4), 1, 20), mx.float32)
        if args.balance_positive
        else None
    )
    report["positive_weights"] = positive_weights.tolist() if positive_weights is not None else None
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.01)
    rng = np.random.default_rng(71)

    def loss(model, indices):
        terms = []
        for i in indices:
            output = model.from_features(*examples["train"][i][1])
            buttons = button_loss(
                output["buttons"], mx.array(truth[i : i + 1], mx.float32), varying, positive_weights
            )
            camera = nn.losses.cross_entropy(
                output["chunk_mouse"][:, 0], mx.array(camera_labels[i : i + 1]), reduction="mean"
            )
            terms.append(buttons + 0.25 * camera)
        return sum(terms) / len(terms)

    gradient = nn.value_and_grad(runtime.module, loss)
    best, best_weights, selected = -float("inf"), None, None
    with (args.output / "steps.jsonl").open("w") as log:
        for step in range(args.steps):
            domains = list(groups.values())
            indices = [
                int(
                    rng.choice(
                        domains[j % len(domains)][int(rng.integers(len(domains[j % len(domains)])))]
                    )
                )
                for j in range(args.batch_size)
            ]
            started = time.perf_counter()
            value, grads = gradient(runtime.module, indices)
            grads, norm = optim.clip_grad_norm(grads, 1)
            mx.eval(value, norm)
            if not np.isfinite([float(value), float(norm)]).all():
                raise FloatingPointError("Nonfinite optimization step")
            optimizer.update(runtime.module, grads)
            mx.eval(runtime.module.trainable_parameters(), optimizer.state)
            item = {
                "step": step + 1,
                "loss": float(value),
                "gradient_norm": float(norm),
                "seconds": time.perf_counter() - started,
            }
            log.write(json.dumps(item) + "\n")
            if step == 0 or (step + 1) % 50 == 0:
                print(json.dumps(item), flush=True)
                log.flush()
            if (step + 1) % 250 == 0 or step + 1 == args.steps:
                if args.small_fit_per_action:
                    print(
                        json.dumps(
                            {
                                "small_fit_step": step + 1,
                                "train": score(runtime, rows["train"], examples["train"]),
                                "shuffled": score(runtime, rows["train"], examples["train"], True),
                            }
                        ),
                        flush=True,
                    )
                validation = score(runtime, rows["validation"], examples["validation"])
                # Both domains participate; no test observations select checkpoints.
                selection = np.mean(
                    [v["button_micro_f1"] for k, v in validation.items() if k != "all"]
                )
                report["selection"].append(
                    {"step": step + 1, "macro_game_f1": float(selection), "metrics": validation}
                )
                if selection > best:
                    best, selected = selection, step + 1
                    best_weights = [
                        (k, mx.stop_gradient(v))
                        for k, v in tree_flatten(runtime.module.trainable_parameters())
                    ]
                    mx.eval([v for _, v in best_weights])
                print(
                    json.dumps(
                        {
                            "validation_step": step + 1,
                            "macro_game_f1": float(selection),
                            "hordes": validation["Hordes.io"],
                        }
                    ),
                    flush=True,
                )
                (args.output / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
    runtime.module.load_weights(best_weights, strict=False)
    runtime.metadata.update(
        training_steps=args.steps,
        action_training_examples=args.steps * args.batch_size,
        selected_step=selected,
    )
    report["selected_step"] = selected
    if args.small_fit_per_action:
        report["small_fit"] = {
            "actual": score(runtime, rows["train"], examples["train"]),
            "shuffled_images": score(runtime, rows["train"], examples["train"], True),
        }
    report["parent_unchanged"] = frozen == {
        name: fingerprint(getattr(runtime.module, name), frozen_only=True) for name in frozen
    }
    if not report["parent_unchanged"]:
        raise RuntimeError("Frozen parent changed")
    examples["test"] = cache_examples(runtime, rows["test"], args.cache)
    report["results"] = {
        s: {
            "actual": score(runtime, rows[s], examples[s]),
            "shuffled_images": score(runtime, rows[s], examples[s], True),
        }
        for s in ("validation", "test")
    }
    runtime.save(args.output / "bundle")
    reloaded = TrainableRuntime.load(args.output / "bundle")
    probe = rows["validation"][0]
    a, b = runtime.predict(probe), reloaded.predict(probe)
    report["reload_matches"] = a["action_chunk"][0] == b["action_chunk"][0]
    if not report["reload_matches"]:
        raise RuntimeError("Export differs")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_step": selected, "results": report["results"]}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=Path("artifacts/radio-feature-cache"))
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--balance-positive", action="store_true")
    p.add_argument("--visual-fusion-layers", type=int, default=0)
    p.add_argument("--visual-action-adapter", action="store_true")
    p.add_argument("--small-fit-per-action", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    args = p.parse_args()
    if (
        args.steps < 1
        or args.learning_rate <= 0
        or args.batch_size < 1
        or args.small_fit_per_action < 0
    ):
        p.error("Positive steps and learning rate required")
    run(args)


if __name__ == "__main__":
    main()
