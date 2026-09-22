"""Matched offline memory experiments with frozen Qwen/Laya and causal sequence audits."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from itertools import combinations
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from .p2p_training import metrics
from .policy_data import check_separation, manifest_digest, read_manifest
from .policy_training import fingerprint
from .temporal_adapter import TemporalActionAdapter, dynamics_target
from .temporal_runtime import TemporalRuntime, decode, frame_inputs
from .trainable_model import TrainableRuntime
from .visual_action_adapter import encode_action_history


def read_sequences(data, config):
    splits = ["train", "validation", "test"]
    if (data / "fresh_test.jsonl").exists():
        splits.append("fresh_test")
    pairs = list(combinations(splits, 2))
    rows = {s: read_manifest(data / f"{s}.jsonl", config) for s in splits}
    for a, b in pairs:
        check_separation(rows[a], rows[b], holdout_games=b in {"test", "fresh_test"})
    result, hashes = {}, {}
    for split, examples in rows.items():
        groups, hashes[split] = {}, set()
        for row in examples:
            groups.setdefault(row["sequence"], []).append(row)
            future = (data / row["future_image"]).resolve()
            row["future_image"] = str(future)
            hashes[split].add(hashlib.sha256(future.read_bytes()).hexdigest())
            hashes[split].update(f["sha256"] for f in row["frames"])
        for sequence in groups.values():
            if (
                [r["sequence_step"] for r in sequence] != list(range(len(sequence)))
                or len(sequence) != sequence[0]["sequence_length"]
                or len({(r["game"], r["episode"], r["goal"]) for r in sequence}) != 1
            ):
                raise ValueError("Nonconsecutive or mixed sequence")
            times = [r["timestamp_seconds"] for r in sequence]
            if not np.allclose(np.diff(times), 0.05, atol=0.002):
                raise ValueError("Invalid sequence cadence")
            for row in sequence:
                if row["source"]["action_annotation_index"] != row["frame_index"] + 1:
                    raise ValueError("Invalid current/action alignment")
            for a, b in zip(sequence, sequence[1:]):
                if a["action"] != b["recorded_previous_actions"][-1]:
                    raise ValueError("Previous control differs from preceding action label")
                if (
                    a["future_image"] != b["frames"][0]["image"]
                    or a["frame_index"] + 1 != b["frame_index"]
                ):
                    raise ValueError("Future-target alignment differs from next current frame")
        result[split] = list(groups.values())
    if any(hashes[a] & hashes[b] for a, b in pairs):
        raise ValueError("Current/future image leakage across splits")
    return rows, result


def cache_sequences(runtime, groups, directory, parent_hashes):
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for split, sequences in groups.items():
        result[split] = []
        for i, rows in enumerate(sequences):
            signature = {
                "format": 1,
                "parent": parent_hashes,
                "config": runtime.metadata["policy_config"],
                "rows": manifest_digest(rows),
                "future_sha256": hashlib.sha256(
                    Path(rows[-1]["future_image"]).read_bytes()
                ).hexdigest(),
            }
            key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
            path = directory / f"{key}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    values = {k: mx.array(saved[k]) for k in saved.files}
            else:
                visuals, languages = [], []
                for row in rows:
                    visual, language = frame_inputs(runtime, row)
                    mx.eval(visual, language)
                    visuals.append(visual)
                    languages.append(language)
                future_row = {
                    **rows[-1],
                    "frames": [{"image": rows[-1]["future_image"], "age_seconds": 0}],
                }
                # Future pixels are only encoded for a fixed auxiliary target, never the policy context.
                from .temporal_adapter import spatial_pool

                patches, coords = runtime.features(future_row)
                future_visual = spatial_pool(patches, coords)
                visual = mx.stack(visuals)
                next_visual = mx.concatenate([visual[1:], future_visual[None]], 0)
                values = {
                    "visual": visual,
                    "language": mx.stack(languages),
                    "future_delta": dynamics_target(next_visual) - dynamics_target(visual),
                }
                mx.eval(values)
                np.savez_compressed(path, **{k: np.asarray(v) for k, v in values.items()})
            values["history"] = mx.concatenate(
                [encode_action_history(r, runtime.module.policy_config) for r in rows]
            )
            values["buttons"] = mx.array(
                [
                    [b in r["action"]["buttons"] for b in runtime.module.policy_config.buttons]
                    for r in rows
                ],
                mx.float32,
            )
            values["mouse"] = mx.array([r["action"]["mouse_delta"] for r in rows], mx.float32)
            values["transition"] = mx.array(
                [
                    set(r["action"]["buttons"])
                    != set(r["recorded_previous_actions"][-1]["buttons"])
                    for r in rows
                ],
                mx.float32,
            )
            result[split].append(values)
            print(f"Frozen sequence features: {split} {i + 1}/{len(sequences)}", flush=True)
    return result


def training_stats(data, config):
    truth = np.concatenate([np.asarray(s["buttons"]) for s in data])
    labels = np.abs(
        np.concatenate([np.asarray(s["mouse"]) for s in data])[..., None]
        - np.array(config.mouse_bins)
    ).argmin(-1)
    weights = []
    for axis in range(2):
        counts = np.bincount(labels[:, axis], minlength=len(config.mouse_bins))
        active = counts > 0
        w = np.ones(len(counts))
        w[active] = np.sqrt(counts.sum() / (active.sum() * counts[active]))
        weights.append(np.clip(w, 0.25, 4))
    deltas = np.concatenate([np.asarray(s["future_delta"]) for s in data])
    return {
        "varying": mx.array(truth.any(0) & ~truth.all(0), mx.float32),
        "camera_weights": mx.array(np.stack(weights), mx.float32),
        "delta_scale": mx.array(np.maximum(np.sqrt(np.mean(deltas**2, 0)), 0.001)),
    }


def loss_terms(model, batch, stats, config, aux_weight):
    output, _ = model(batch["visual"], batch["language"], batch["history"])
    bce = nn.losses.binary_cross_entropy(
        output["buttons"], batch["buttons"], with_logits=True, reduction="none"
    )
    active = stats["varying"]
    per_step = (
        (bce * active).sum(-1) / mx.maximum(active.sum(), 1)
        + (bce * (1 - active)).sum(-1) / mx.maximum((1 - active).sum(), 1)
    ) / 2
    transition_weights = 1 + 2 * batch["transition"]
    buttons = (per_step * transition_weights).sum() / transition_weights.sum()
    bins = mx.array(config.mouse_bins)
    labels = mx.argmin(mx.abs(batch["mouse"][..., None] - bins), -1)
    ce = nn.losses.cross_entropy(output["camera"], labels, reduction="none")
    camera = (ce * stats["camera_weights"][mx.arange(2), labels]).mean()
    expected_error = (
        (mx.softmax(output["camera"], -1) * mx.abs(bins - batch["mouse"][..., None])).sum(-1).mean()
    )
    control_loss = buttons + camera + 4 * expected_error
    if aux_weight == 0:
        return control_loss
    action = mx.concatenate([batch["buttons"], batch["mouse"]], -1)
    prediction = model.future_delta(output["hidden"], action)
    dynamics = mx.mean(((prediction - batch["future_delta"]) / stats["delta_scale"]) ** 2)
    return control_loss + aux_weight * dynamics


def augmented_batch(data, selected, rng, regularize, augmentation_rng=None):
    batch = {k: mx.stack([data[i][k] for i in selected]) for k in data[0]}
    keep = mx.array((rng.random((len(selected), 1, 1)) >= 0.5).astype(np.float32))
    batch["history"] = batch["history"] * keep
    if regularize:
        augmentation_rng = rng if augmentation_rng is None else augmentation_rng
        length = batch["visual"].shape[1]
        crop = max(2, length * 3 // 4)
        offsets = augmentation_rng.integers(length - crop + 1, size=len(selected))
        batch = {
            k: mx.stack(
                [v[i, int(offset) : int(offset) + crop] for i, offset in enumerate(offsets)]
            )
            for k, v in batch.items()
        }
        # Same missing spatial tokens across time: no invented motion cues.
        mask = mx.array(
            (augmentation_rng.random((len(selected), 1, 16, 1)) >= 0.1).astype(np.float32)
        )
        batch["visual"] = batch["visual"] * mask
    return batch


def selection_loss(adapter, data, stats, config):
    # Select by supervised controls only. Neither test set nor future targets selects weights.
    total, count = 0.0, 0
    for item in data:
        batch = {k: v[None] for k, v in item.items() if k != "future_delta"}
        value = loss_terms(adapter, batch, stats, config, 0.0)
        total += float(value) * item["buttons"].shape[0]
        count += item["buttons"].shape[0]
    return total / count


def train(
    adapter,
    data,
    config,
    steps,
    lr,
    seed,
    stats,
    output,
    aux_weight,
    validation=None,
    validation_every=0,
    regularize=False,
):
    rng = np.random.default_rng(seed)
    augmentation_rng = np.random.default_rng(seed + 1)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)
    gradient = nn.value_and_grad(
        adapter, lambda m, batch: loss_terms(m, batch, stats, config, aux_weight)
    )
    selection = {
        "criterion": "validation control loss (no auxiliary target)",
        "checks": [],
        "selected_step": steps,
    }
    best, best_weights = float("inf"), None
    with (output / "steps.jsonl").open("x") as log:
        for step in range(steps):
            started = time.perf_counter()
            selected = rng.integers(len(data), size=2)
            batch = augmented_batch(data, selected, rng, regularize, augmentation_rng)
            value, grads = gradient(adapter, batch)
            grads, norm = optim.clip_grad_norm(grads, 1.0)
            mx.eval(value, norm)
            if not np.isfinite([float(value), float(norm)]).all():
                raise FloatingPointError("Nonfinite temporal update")
            optimizer.update(adapter, grads)
            mx.eval(adapter.parameters(), optimizer.state)
            item = {
                "step": step + 1,
                "loss": float(value),
                "gradient_norm": float(norm),
                "seconds": time.perf_counter() - started,
            }
            log.write(json.dumps(item) + "\n")
            if step == 0 or (step + 1) % 100 == 0:
                log.flush()
                print(json.dumps(item), flush=True)
            if validation_every and ((step + 1) % validation_every == 0 or step + 1 == steps):
                score = selection_loss(adapter, validation, stats, config)
                if not np.isfinite(score):
                    raise FloatingPointError("Nonfinite validation loss")
                selection["checks"].append({"step": step + 1, "loss": score})
                print(json.dumps({"validation_step": step + 1, "control_loss": score}), flush=True)
                if score < best:
                    best = score
                    best_weights = [
                        (k, mx.stop_gradient(v)) for k, v in tree_flatten(adapter.parameters())
                    ]
                    mx.eval([v for _, v in best_weights])
                    selection["selected_step"] = step + 1
    if best_weights is not None:
        adapter.load_weights(best_weights, strict=True)
    selection["selected_loss"] = best if best_weights is not None else None
    return selection


def predict_sequence(adapter, data, config, variant="actual"):
    visual, language, history = (data[k][None] for k in ("visual", "language", "history"))
    if variant == "no_previous_actions":
        history = mx.zeros_like(history)
    if variant in {"actual", "no_previous_actions", "shuffled_observations"}:
        out, _ = adapter(visual, language, history)
        mx.eval(out)
        return decode(out, config), out
    outputs, state = [], None
    for t in range(visual.shape[1]):
        h = history[:, t : t + 1]
        if variant == "self_fed_controls" and t:
            h = encode_action_history(
                {"previous_actions": [decode(outputs[-1], config)[0]]}, config
            )[:, None]
        if variant == "reset_each_frame":
            state = None
        if variant == "shuffled_history":
            # Current frame stays current; only the strictly earlier prefix is reordered.
            order = np.random.default_rng(900 + t).permutation(t).tolist() + [t]
            out, _ = adapter(visual[:, order], language[:, order], history[:, order])
            out = {k: v[:, -1:] for k, v in out.items()}
        else:
            out, state = adapter(visual[:, t : t + 1], language[:, t : t + 1], h, state)
        mx.eval(out, state)
        outputs.append(out)
    merged = {k: mx.concatenate([o[k] for o in outputs], 1) for k in outputs[0]}
    return decode(merged, config), merged


def grouped_metrics(rows, predictions, config):
    result = metrics(rows, predictions, config)
    result["by_game"] = {}
    for game in sorted({r["game"] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r["game"] == game]
        result["by_game"][game] = metrics(
            [rows[i] for i in indices], [predictions[i] for i in indices], config
        )
    return result


def evaluate(adapter, sequences, data, config, stats, variants):
    report = {}
    rows = [r for seq in sequences for r in seq]
    # Exchange observation embeddings only among frames with the exact same goal and game.
    sources = {}
    for i, seq in enumerate(sequences):
        for j, r in enumerate(seq):
            sources.setdefault((r["game"], r["goal"]), []).append((i, j))
    mapping = {}
    for group in sources.values():
        order = np.random.default_rng(901).permutation(len(group))
        mapping.update({dest: group[int(src)] for dest, src in zip(group, order, strict=True)})
    for variant in variants:
        predictions, dynamic_errors, zero_errors = [], [], []
        for i, item in enumerate(data):
            changed = item
            if variant == "shuffled_observations":
                indices = [mapping[(i, j)] for j in range(len(sequences[i]))]
                changed = {
                    **item,
                    **{
                        k: mx.stack([data[a][k][b] for a, b in indices])
                        for k in ("visual", "language")
                    },
                }
            pred, out = predict_sequence(adapter, changed, config, variant)
            predictions.extend(pred)
            if variant == "actual":
                action = mx.concatenate([item["buttons"], item["mouse"]], -1)[None]
                future = adapter.future_delta(out["hidden"], action)[0]
                dynamic_errors.append(
                    np.asarray(((future - item["future_delta"]) / stats["delta_scale"]) ** 2)
                )
                zero_errors.append(np.asarray((item["future_delta"] / stats["delta_scale"]) ** 2))
        report[variant] = grouped_metrics(rows, predictions, config)
        if variant == "actual":
            report[variant]["future_delta_standardized_mse"] = float(
                np.concatenate(dynamic_errors).mean()
            )
            report[variant]["zero_delta_standardized_mse"] = float(
                np.concatenate(zero_errors).mean()
            )
        print(
            json.dumps(
                {
                    "variant": variant,
                    "f1": report[variant]["button_micro_f1"],
                    "transition_exact": report[variant]["transition_exact"],
                    "moving_mouse_mae_px": report[variant]["moving_mouse_mae_px"],
                }
            ),
            flush=True,
        )
    report["repeat_previous_action"] = grouped_metrics(
        rows, [r["recorded_previous_actions"][-1] for r in rows], config
    )
    report["no_input"] = grouped_metrics(
        rows, [{"buttons": [], "mouse_delta": [0, 0]} for _ in rows], config
    )
    return report


def qualify(result):
    a, p, shuffled = [
        result[k] for k in ("actual", "repeat_previous_action", "shuffled_observations")
    ]
    return bool(
        a["button_micro_f1"] > p["button_micro_f1"]
        and a["button_micro_f1"] - shuffled["button_micro_f1"] >= 0.05
        and a["transition_exact"] is not None
        and a["transition_exact"] > 0.25
        and a["moving_mouse_mae_px"] is not None
        and a["moving_mouse_mae_px"]
        < min(p["moving_mouse_mae_px"], result["no_input"]["moving_mouse_mae_px"])
    )


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = TrainableRuntime.load(args.bundle)
    if (
        runtime.module.policy_config.visual_action_adapter
        or runtime.module.policy_config.temporal_adapter != "none"
    ):
        raise ValueError("Start from the frozen parent without another action adapter")
    frozen = {
        k: fingerprint(getattr(runtime.module, k))
        for k in ("vision", "laya", "connector", "actions")
    }
    rows, groups = read_sequences(args.data, runtime.module.policy_config)
    data = cache_sequences(runtime, groups, args.cache, frozen)
    stats = training_stats(data["train"], runtime.module.policy_config)
    for kind in args.kinds:
        outdir = args.output / kind
        outdir.mkdir(exist_ok=False)
        mx.random.seed(83)
        config = replace(
            runtime.module.policy_config, temporal_adapter=kind, temporal_width=args.width
        )
        runtime.module.policy_config = config
        runtime.metadata["policy_config"] = asdict(config)
        runtime.module.temporal_actions = TemporalActionAdapter(
            runtime.module.vision.config.out_hidden_size,
            runtime.module.laya.encoder.config.hidden_size,
            config,
        )
        runtime.module.freeze()
        runtime.module.temporal_actions.unfreeze()
        adapter = runtime.module.temporal_actions
        protocol = {
            "parent": str(args.bundle),
            "parent_fingerprints": frozen,
            "kind": kind,
            "seed": 83,
            "steps": args.steps,
            "batch_sequences": 2,
            "sequence_length": len(groups["train"][0]),
            "learning_rate": args.learning_rate,
            "auxiliary_weight": args.auxiliary_weight,
            "history_dropout": 0.5,
            "random_crop_and_spatial_masking": args.regularize,
            "validation_every": args.validation_every,
            "test_used_for_selection": False,
            "button_transition_loss_weight": 3,
            "manifest_digests": {s: manifest_digest(r) for s, r in rows.items()},
            "training_only_future_target": "normalized frozen spatial features at t+1 minus t, 64 dimensions",
            "goal_source": "Retrospective weak P2P label or generic continuation, fixed per clip",
            "memory_placement": "After frozen Qwen vision and goal-conditioned Laya contexts; learned moment pooling feeds temporal blocks",
            "inference": "One checkpoint, one current screenshot, goal, previous applied action, explicit causal memory",
            "scope": "50 ms first-step physical buttons and relative mouse; no pointer coordinates",
            "split": "recording validation, held-out experiences test; reused developmental data; not player independent",
            "deployment_eligible": False,
            "input_events_sent": 0,
        }
        runtime.metadata["temporal_experiment"] = protocol
        report = {"protocol": protocol, "parameters": runtime.parameter_counts()}
        (outdir / "protocol.json").write_text(json.dumps(report, indent=2) + "\n")
        report["checkpoint_selection"] = train(
            adapter,
            data["train"],
            config,
            args.steps,
            args.learning_rate,
            83,
            stats,
            outdir,
            args.auxiliary_weight,
            validation=data["validation"],
            validation_every=args.validation_every,
            regularize=args.regularize,
        )
        runtime.metadata["temporal_experiment"]["checkpoint_selection"] = report[
            "checkpoint_selection"
        ]
        report["results"] = {}
        for split in data:
            variants = [
                "actual",
                "no_previous_actions",
                "reset_each_frame",
                "shuffled_observations",
                "shuffled_history",
                "self_fed_controls",
            ]
            if split == "train":
                variants = ["actual", "reset_each_frame"]
            print(f"Evaluating {kind}/{split}", flush=True)
            report["results"][split] = evaluate(
                adapter, groups[split], data[split], config, stats, variants
            )
            (outdir / "partial-report.json").write_text(json.dumps(report, indent=2) + "\n")
        report["qualification_passed"] = qualify(report["results"]["validation"])
        report["frozen_parent_unchanged"] = frozen == {
            k: fingerprint(getattr(runtime.module, k)) for k in frozen
        }
        if not report["frozen_parent_unchanged"]:
            raise RuntimeError("Frozen parent weights changed")
        runtime.metadata["temporal_experiment"]["qualification_passed"] = report[
            "qualification_passed"
        ]
        runtime.save(outdir / "bundle")
        loaded = TrainableRuntime.load(outdir / "bundle")
        for v in ("actual", "self_fed_controls"):
            before, out1 = predict_sequence(adapter, data["validation"][0], config, v)
            after, out2 = predict_sequence(
                loaded.module.temporal_actions, data["validation"][0], config, v
            )
            if before != after:
                raise RuntimeError("Reload changed actions")
            for k in ("buttons", "camera"):
                np.testing.assert_allclose(
                    np.asarray(out1[k]), np.asarray(out2[k]), atol=1e-6, rtol=1e-6
                )
        report["checkpoint_reload_actions_match"] = True
        streaming = TemporalRuntime(loaded)
        latencies, parity, cached_state = [], [], None
        for i, row in enumerate(groups["validation"][0]):
            actual = streaming.predict(
                row, session_id="latency-validation", timestamp_seconds=row["timestamp_seconds"]
            )
            cached = data["validation"][0]
            expected, cached_state = loaded.module.temporal_actions(
                cached["visual"][i : i + 1][None],
                cached["language"][i : i + 1][None],
                cached["history"][i : i + 1][None],
                cached_state,
            )
            mx.eval(expected, cached_state)
            expected = decode(expected, config)[0]
            parity.append(all(actual[k] == expected[k] for k in ("buttons", "mouse_delta")))
            if i >= 3:
                latencies.append(actual["image_to_outputs_ms"])
        if not all(parity):
            raise RuntimeError("Fresh-image streaming differs from cached training graph")
        report["latency"] = {
            "samples": len(latencies),
            "warmup_frames": 3,
            "p50_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "includes": "disk image load, current vision, prompt preparation, Laya, memory, action decode",
            "excludes": "screenshot capture, event posting, game response/contention",
            "fresh_image_streaming_actions_match_cached_graph": True,
        }
        (outdir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "kind": kind,
                    "qualified": report["qualification_passed"],
                    "latency": report["latency"],
                }
            ),
            flush=True,
        )
        del loaded, streaming


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=Path("artifacts/temporal-feature-cache"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--validation-every", type=int, default=0)
    p.add_argument("--regularize", action="store_true")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--auxiliary-weight", type=float, default=0.1)
    p.add_argument(
        "--kinds", nargs="+", choices=("mamba3", "attention"), default=["mamba3", "attention"]
    )
    args = p.parse_args()
    if (
        args.steps < 1
        or args.learning_rate <= 0
        or args.auxiliary_weight < 0
        or args.validation_every < 0
    ):
        p.error("Invalid training settings")
    run(args)


if __name__ == "__main__":
    main()
