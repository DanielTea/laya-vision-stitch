"""Fit small visual/control adapters through a frozen pretrained P2P policy.

Cached vision and Laya features are for training only. Export runs fresh images.
Hordes labels remain weak; validation selects checkpoints, never live success.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from PIL import Image

from laya_vision_stitch.laya_p2p import LayaP2PRuntime
from laya_vision_stitch.p2p_adaptation import (
    KEYS_WITH_TAB,
    encode_action,
    install_control_adapter,
    install_visual_adapter,
)
from laya_vision_stitch.p2p_pretrained_policy import physical_action
from laya_vision_stitch.p2p_pretrained_vision import preprocess
from scripts.train_p2p_control_adapter import frozen_hash


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def cache_features(runtime, data, cache, source):
    identity = {
        "weights": digest(source / "model.safetensors"),
        "splits": {s: digest(data / f"{s}.jsonl") for s in ("train", "validation", "test")},
    }
    if cache.exists():
        if json.loads((cache / "identity.json").read_text()) != identity:
            raise ValueError("Feature cache does not match source model and manifests")
    else:
        cache.mkdir(parents=True)
        goal_features = {}
        for split in identity["splits"]:
            rows, images, goals, labels = [], [], [], []
            excluded = Counter()
            for line in (data / f"{split}.jsonl").read_text().splitlines():
                row = json.loads(line)
                try:
                    tokens = encode_action(row["action"])
                except ValueError as error:
                    excluded[str(error)] += 1
                    continue
                with Image.open(row["frames"][-1]["image"]) as image:
                    _, vision = runtime.model.policy.vision(mx.array(preprocess(image)))
                if row["goal"] not in goal_features:
                    text = runtime.model.bridge(
                        runtime.model.goal_features(*runtime.prepare_goal(row["goal"]))
                    )
                    goal_features[row["goal"]] = np.asarray(text.astype(mx.float32))[0]
                images.append(np.asarray(vision.astype(mx.float32))[0])
                goals.append(goal_features[row["goal"]])
                labels.append(tokens)
                rows.append(row)
            np.savez(
                cache / f"{split}.npz",
                images=np.stack(images),
                goals=np.stack(goals),
                tokens=np.array(labels, np.int32),
            )
            (cache / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            print(
                json.dumps({"cached": split, "examples": len(rows), "excluded": dict(excluded)}),
                flush=True,
            )
        (cache / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    result, seen_episodes, seen_images = {}, {}, {}
    for split in identity["splits"]:
        rows = [json.loads(line) for line in (cache / f"{split}.jsonl").read_text().splitlines()]
        arrays = dict(np.load(cache / f"{split}.npz"))
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise FloatingPointError("Nonfinite cached inputs")
        for row in rows:
            episode = (row["game"], row["episode"])
            image = digest(Path(row["frames"][-1]["image"]))
            if image != row["frames"][-1].get("sha256"):
                raise ValueError("Image contents differ from the dataset manifest")
            for key, seen in ((episode, seen_episodes), (image, seen_images)):
                if key in seen and seen[key] != split:
                    raise ValueError(f"Cross-split leakage: {key}")
                seen[key] = split
        result[split] = rows, arrays
    return result


def context(policy, images, goals):
    return policy.context(policy.prefix(images, goals))[0]


def metrics(rows, predictions, targets):
    grouped = defaultdict(
        lambda: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "exact": 0,
            "examples": 0,
            "idle_examples": 0,
            "idle_false_positives": 0,
            "buttons": {},
        }
    )
    for row, pred, truth in zip(rows, predictions, targets, strict=True):
        got = set(physical_action(pred, key_names=KEYS_WITH_TAB)["buttons"])
        wanted = set(physical_action(truth, key_names=KEYS_WITH_TAB)["buttons"])
        m = grouped[row["game"]]
        for k, v in {
            "tp": len(got & wanted),
            "fp": len(got - wanted),
            "fn": len(wanted - got),
            "exact": int(got == wanted),
            "examples": 1,
            "idle_examples": int(not wanted),
            "idle_false_positives": int(not wanted and bool(got)),
        }.items():
            m[k] += v
        for button in got | wanted:
            b = m["buttons"].setdefault(button, {"tp": 0, "fp": 0, "fn": 0})
            b["tp"] += int(button in got & wanted)
            b["fp"] += int(button in got - wanted)
            b["fn"] += int(button in wanted - got)
    for m in grouped.values():
        m["button_f1"] = 2 * m["tp"] / max(1, 2 * m["tp"] + m["fp"] + m["fn"])
        m["exact_button_set_accuracy"] = m["exact"] / m["examples"]
        for b in m["buttons"].values():
            b["f1"] = 2 * b["tp"] / max(1, 2 * b["tp"] + b["fp"] + b["fn"])
        supported = [b["f1"] for b in m["buttons"].values() if b["tp"] + b["fn"] > 0]
        m["supported_button_macro_f1"] = float(np.mean(supported)) if supported else 0.0
    return dict(grouped)


def evaluate(policy, split, batch=8, permutation=None):
    rows, a = split
    predictions = []
    for start in range(0, len(rows), batch):
        indices = np.arange(start, min(start + batch, len(rows)))
        image_indices = indices if permutation is None else permutation[indices]
        c = context(policy, mx.array(a["images"][image_indices]), mx.array(a["goals"][indices]))
        tokens, _ = policy.decode(c)
        mx.eval(tokens)
        predictions.extend(tokens.tolist())
    return metrics(rows, predictions, a["tokens"])


def within_game_permutation(rows, seed):
    rng = np.random.default_rng(seed)
    result = np.arange(len(rows))
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["game"]].append(i)
    for indices in groups.values():
        result[indices] = rng.permutation(indices)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument(
        "--bottleneck", type=int, default=64, help="Zero gives matched decoder-only control"
    )
    p.add_argument("--learning-rate", type=float, default=0.0003)
    p.add_argument("--replay-kl", type=float, default=0.2)
    p.add_argument("--balanced-idle", action="store_true")
    p.add_argument("--shortcut-gates", action="store_true")
    p.add_argument("--seed", type=int, default=20260923)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "args.json").write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2
        )
        + "\n"
    )
    rng = np.random.default_rng(args.seed)
    mx.random.seed(args.seed)
    runtime = LayaP2PRuntime.load(args.bundle)
    if runtime.metadata.get("visual_adapter") or runtime.metadata.get("control_adapter"):
        raise ValueError("Start from the unadapted stitched checkpoint")
    data = cache_features(runtime, args.data, args.cache, args.bundle)
    install_control_adapter(runtime.model, rank=args.rank)
    if args.bottleneck:
        install_visual_adapter(runtime.model, args.bottleneck)
    model, policy = runtime.model, runtime.model.policy
    fingerprint = frozen_hash(model)
    parameters = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    baseline = evaluate(policy, data["validation"])
    base_public = np.mean([v["button_f1"] for g, v in baseline.items() if g != "Hordes.io"])
    base_idle = baseline["Hordes.io"]["idle_false_positives"] / max(
        1, baseline["Hordes.io"]["idle_examples"]
    )
    val_permutation = within_game_permutation(data["validation"][0], args.seed + 1)

    def score(m, shuffled=None):
        replay = float(np.mean([v["button_f1"] for g, v in m.items() if g != "Hordes.io"]))
        h = m["Hordes.io"]
        value = (h["button_f1"] + h["supported_button_macro_f1"] + replay) / 3
        if args.shortcut_gates and shuffled is not None:
            idle_rate = h["idle_false_positives"] / max(1, h["idle_examples"])
            if idle_rate > base_idle + 0.1:
                return -1.0
            if h["button_f1"] < shuffled["Hordes.io"]["button_f1"] + 0.02:
                return -1.0
        return value if replay >= base_public - 0.05 else -1.0

    train_rows, arrays = data["train"]
    x, g, y = [mx.array(arrays[k]) for k in ("images", "goals", "tokens")]
    # Original teacher distributions anchor public replay; no live teacher is exported.
    teacher = [[] for _ in range(8)]
    for start in range(0, len(train_rows), args.batch_size):
        sl = slice(start, start + args.batch_size)
        logits = policy.teacher_logits(context(policy, x[sl], g[sl]), y[sl])
        mx.eval(logits)
        for collection, logit in zip(teacher, logits, strict=True):
            collection.append(logit)
    teacher = tuple(mx.concatenate(v) for v in teacher)
    groups = defaultdict(list)
    for i, row in enumerate(train_rows):
        groups[(row["game"], tuple(sorted(row["action"]["buttons"])))].append(i)
    hordes = [v for (game, _), v in groups.items() if game == "Hordes.io"]
    hordes_idle = [
        v for (game, buttons), v in groups.items() if game == "Hordes.io" and not buttons
    ]
    hordes_active = [v for (game, buttons), v in groups.items() if game == "Hordes.io" and buttons]
    public = [v for (game, _), v in groups.items() if game != "Hordes.io"]
    replay_mask = mx.array([float(r["game"] != "Hordes.io") for r in train_rows])
    weights = [2.0, 0.25, 0.25, 0.25, 1.0, 0.25, 0.5, 0.5]

    def loss_fn(m, indices):
        logits = m.policy.teacher_logits(context(m.policy, x[indices], g[indices]), y[indices])
        ce, kl = mx.array(0.0), mx.array(0.0)
        for i, logit in enumerate(logits):
            ce = ce + weights[i] * nn.losses.cross_entropy(logit, y[indices, i], reduction="mean")
            ref = teacher[i][indices]
            log_ref = ref - mx.logsumexp(ref, axis=-1, keepdims=True)
            log_pred = logit - mx.logsumexp(logit, axis=-1, keepdims=True)
            divergence = (mx.exp(log_ref) * (log_ref - log_pred)).sum(-1)
            kl = kl + (divergence * replay_mask[indices]).mean()
        return ce / sum(weights) + args.replay_kl * kl / 8

    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.01)
    value_grad = nn.value_and_grad(model, loss_fn)
    best_score, best_step = score(baseline), 0
    best = [(k, mx.array(v)) for k, v in tree_flatten(model.trainable_parameters())]
    history = [{"step": 0, "validation": baseline, "score": best_score}]
    print(
        json.dumps({"parameters": parameters, "baseline": baseline, "score": best_score}),
        flush=True,
    )
    for step in range(1, args.steps + 1):
        choices = []
        for _ in range(args.batch_size):
            if rng.random() >= 0.5:
                choices.append(public)
            elif args.balanced_idle:
                choices.append(hordes_idle if rng.random() < 0.5 else hordes_active)
            else:
                choices.append(hordes)
        indices = mx.array([int(rng.choice(c[rng.integers(len(c))])) for c in choices])
        loss, grads = value_grad(model, indices)
        grads, norm = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        mx.eval(loss, norm, model.trainable_parameters(), optimizer.state)
        if not np.isfinite(float(loss)) or not np.isfinite(float(norm)):
            raise FloatingPointError("Nonfinite training loss/gradient")
        if step % 50 == 0 or step == args.steps:
            result = evaluate(policy, data["validation"])
            shuffled_validation = (
                evaluate(policy, data["validation"], permutation=val_permutation)
                if args.shortcut_gates
                else None
            )
            selection = score(result, shuffled_validation)
            item = {
                "step": step,
                "loss": float(loss),
                "gradient_norm": float(norm),
                "validation": result,
                "shuffled_validation": shuffled_validation,
                "score": selection,
            }
            history.append(item)
            (args.output / "progress.json").write_text(json.dumps(history, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(loss),
                        "score": selection,
                        "hordes_f1": result["Hordes.io"]["button_f1"],
                        "attack": result["Hordes.io"]["buttons"].get("1", {}),
                    }
                ),
                flush=True,
            )
            if selection > best_score:
                best_score, best_step = selection, step
                best = [(k, mx.array(v)) for k, v in tree_flatten(model.trainable_parameters())]
                mx.save_safetensors(str(args.output / "best-adapters.safetensors"), dict(best))
    model.load_weights(best, strict=False)
    unchanged = frozen_hash(model) == fingerprint
    if not unchanged:
        raise RuntimeError("Frozen pretrained parameters changed")
    result = {s: evaluate(policy, data[s]) for s in ("train", "validation", "test")}
    rows = data["test"][0]
    permutation = np.arange(len(rows))
    by_game = defaultdict(list)
    for i, row in enumerate(rows):
        by_game[row["game"]].append(i)
    for indices in by_game.values():
        permutation[indices] = rng.permutation(indices)
    shuffled = evaluate(policy, data["test"], permutation=permutation)
    runtime.metadata.update(
        control_adapter={"rank": args.rank},
        key_names=list(KEYS_WITH_TAB),
        selected_control_step=best_step,
        deployment_eligible=False,
        control_training="Small residual visual adapter and control LoRA; weak Hordes labels plus public human replay; frozen parents",
    )
    if args.bottleneck:
        runtime.metadata["visual_adapter"] = {"bottleneck": args.bottleneck}
    runtime.save(args.output / "bundle")
    loaded = LayaP2PRuntime.load(args.output / "bundle")
    # Fresh preprocessing + vision + Laya + learned adapter + policy after export.
    row = rows[0]
    with Image.open(row["frames"][-1]["image"]) as image:
        pixels = mx.array(preprocess(image))
    out = loaded.model.step(pixels, *loaded.prepare_goal(row["goal"]))
    expected = policy.decode(
        context(
            policy, mx.array(data["test"][1]["images"][:1]), mx.array(data["test"][1]["goals"][:1])
        )
    )[0]
    if out[0].tolist() != expected.tolist():
        raise RuntimeError("Fresh-image export/reload action differs from cached-feature path")
    report = {
        "selected_step": best_step,
        "trainable_parameters": parameters,
        "parents_unchanged": unchanged,
        "frozen_sha256": fingerprint,
        "fresh_image_reload_tokens_match": True,
        "metrics": result,
        "within_game_image_shuffle_test": shuffled,
        "history": history,
        "scope": "Single-frame weak Hordes imitation, not verified successful play. Public games may overlap parent pretraining. No live inputs. No inference-time gameplay rules. Temporal memory and goal following need separate validation.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"selected_step": best_step, "test": result["test"], "parents_unchanged": unchanged}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
