"""Controlled static-image experiments; auxiliary success is not gameplay success."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from PIL import Image

from laya_vision_stitch.laya_p2p import LayaP2PRuntime
from laya_vision_stitch.p2p_adaptation import KEYS_WITH_TAB, install_control_adapter
from laya_vision_stitch.p2p_pretrained_vision import preprocess
from laya_vision_stitch.spatial_goal_adapter import install_spatial_adapter
from scripts.prepare_spatial_experiments import read
from scripts.train_p2p_control_adapter import frozen_hash
from scripts.train_p2p_visual_adapter import digest, metrics


def policy_context(policy, spatial, image, goal):
    return policy.context(policy.prefix(image, goal, spatial))[0]


def load_data(path):
    result = {}
    for split in ["train", "validation", "test"]:
        arrays = dict(np.load(path / f"{split}.npz"))
        rows = read(path / f"{split}.jsonl")
        result[split] = (rows, arrays)
    return result


def permutation(rows):
    rng = np.random.default_rng(901)
    groups = defaultdict(list)
    perm = np.arange(len(rows))
    for i, r in enumerate(rows):
        groups[(r["game"], r["kind"])].append(i)
    for idx in groups.values():
        perm[idx] = rng.permutation(idx)
    return perm


def evaluate(policy, data, teacher=None, shuffle=False, teacher_mean=None):
    rows, a = data
    pred = []
    ground = []
    cosines = []
    perm = permutation(rows) if shuffle else np.arange(len(rows))
    for start in range(0, len(rows), 16):
        idx = np.arange(start, min(start + 16, len(rows)))
        imidx = perm[idx]
        s, i, g = [
            mx.array(a[k][j]) for k, j in [("spatial", imidx), ("images", imidx), ("goals", idx)]
        ]
        c = policy_context(policy, s, i, g)
        tokens, _ = policy.decode(c)
        patches, pool, _ = policy.spatial_adapter.features(s, g)
        labels = policy.spatial_adapter.grounding_head(pool).argmax(-1)
        mx.eval(tokens, labels)
        pred.extend(tokens.tolist())
        ground.extend(labels.tolist())
        if teacher is not None:
            t = mx.array(teacher[idx].astype(np.float32) - teacher_mean).reshape(-1, 144, 768)
            v = policy.spatial_adapter.distillation_head(patches)
            cos = (v * t).sum(-1) / (mx.sqrt((v * v).sum(-1) * (t * t).sum(-1)) + 1e-6)
            cosines.extend(np.asarray(cos.mean(-1)).tolist())
    game_idx = [j for j, r in enumerate(rows) if r["kind"] == "gameplay"]
    gameplay = metrics(
        [rows[j] for j in game_idx], [pred[j] for j in game_idx], a["tokens"][game_idx]
    )
    pairs = defaultdict(list)
    correct = []
    goal_predictions = []
    for j, r in enumerate(rows):
        if r["kind"] == "goal":
            match = (set(pred[j][:4]) - {0}) == (
                set(a["tokens"][j, :4].tolist()) - {0}
            ) and not any(pred[j][4:6])
            correct.append(match)
            pairs[r["pair"]].append(match)
            goal_predictions.append((r["pair"], pred[j][:4]))
    ground_idx = [j for j, r in enumerate(rows) if r["kind"] == "grounding"]
    ground_accuracy = float(np.mean([ground[j] == int(a["labels"][j]) for j in ground_idx]))
    balanced_ground = float(
        np.mean(
            [
                np.mean([ground[j] == label for j in ground_idx if int(a["labels"][j]) == label])
                for label in [0, 1]
            ]
        )
    )
    return {
        "gameplay": gameplay,
        "grounding_accuracy": ground_accuracy,
        "grounding_balanced_accuracy": balanced_ground,
        "goal_accuracy": float(np.mean(correct)),
        "both_goals_correct": float(np.mean([all(x) for x in pairs.values()])),
        "distillation_cosine": float(np.mean(cosines)) if cosines else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--mode", choices=["spatial", "grounded", "distilled", "goals", "combined"], required=True
    )
    p.add_argument("--steps", type=int, default=800)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "args.json").write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2
        )
        + "\n"
    )
    if json.loads((args.data / "metadata.json").read_text())["source_weights_sha256"] != digest(
        args.bundle / "model.safetensors"
    ):
        raise ValueError("Feature cache source differs")
    ground_mode = args.mode in ["grounded", "combined"]
    distill_mode = args.mode in ["distilled", "combined"]
    goal_mode = args.mode in ["goals", "combined"]
    mx.random.seed(20260925)
    rng = np.random.default_rng(20260925)
    runtime = LayaP2PRuntime.load(args.bundle)
    install_control_adapter(runtime.model, rank=4)
    install_spatial_adapter(runtime.model)
    model = runtime.model
    policy = model.policy
    if not ground_mode:
        policy.spatial_adapter.grounding_head.freeze()
    if not distill_mode:
        policy.spatial_adapter.distillation_head.freeze()
    fingerprint = frozen_hash(model)
    data = load_data(args.data)
    teacher = (
        {s: np.load(args.data / f"{s}-siglip.npy", mmap_mode="r") for s in data}
        if distill_mode
        else {}
    )
    teacher_mean = None
    if distill_mode:
        unique = {}
        for j, row in enumerate(data["train"][0]):
            unique.setdefault(row["frames"][0]["sha256"], j)
        teacher_mean = np.zeros((12, 12, 768), np.float64)
        for j in unique.values():
            teacher_mean += teacher["train"][j].astype(np.float64)
        teacher_mean = (teacher_mean / len(unique)).astype(np.float32)
        np.save(args.output / "teacher-training-mean.npy", teacher_mean)
    rows, a = data["train"]
    x, im, g, y = [mx.array(a[k]) for k in ["spatial", "images", "goals", "tokens"]]
    groups = defaultdict(list)
    for j, r in enumerate(rows):
        if r["kind"] == "gameplay":
            groups[(r["game"], bool(any(r["tokens"][:6])))].append(j)
    hordes = [v for (game, _), v in groups.items() if game == "Hordes.io"]
    public = [v for (game, _), v in groups.items() if game != "Hordes.io"]
    goals = [j for j, r in enumerate(rows) if r["kind"] == "goal"]
    grounded = [j for j, r in enumerate(rows) if r["kind"] == "grounding"]
    if ground_mode:
        accepted = {
            r["id"]: r
            for r in read(args.data / "qwen-grounding-reviewed.jsonl")
            if r["split"] == "train" and r["accepted_after_review"]
        }
        grounded = [j for j in grounded if rows[j]["id"] in accepted]
        for j in grounded:
            review = accepted[rows[j]["id"]]
            if review["image_sha256"] != rows[j]["frames"][0]["sha256"] or review[
                "reviewed_label"
            ] != int(a["labels"][j]):
                raise ValueError("Reviewed teacher label does not match this image")
        if len(grounded) < 16:
            raise ValueError("Insufficient reviewed teacher labels")
        if min(sum(a["labels"][j] == label for j in grounded) for label in [0, 1]) < 4:
            raise ValueError("Insufficient reviewed labels in each grounding class")
    ground_groups = [[j for j in grounded if a["labels"][j] == label] for label in [0, 1]]
    goal_groups = [
        [j for j in goals if a["labels"][j] == label and bool(a["tokens"][j, 0]) == active]
        for label in [0, 1]
        for active in [False, True]
    ]
    original = [[] for _ in range(8)]
    for start in range(0, len(rows), 16):
        sl = slice(start, start + 16)
        c = policy_context(policy, x[sl], im[sl], g[sl])
        logits = policy.teacher_logits(c, y[sl])
        mx.eval(logits)
        for target, logit in zip(original, logits, strict=True):
            target.append(logit)
    original = tuple(mx.concatenate(v) for v in original)
    replay = mx.array([float(r["kind"] == "gameplay" and r["game"] != "Hordes.io") for r in rows])
    weights = [2.0, 0.25, 0.25, 0.25, 1.0, 0.25, 0.5, 0.5]

    def ce_loss(p, c, tokens):
        return sum(
            w * nn.losses.cross_entropy(logit, tokens[:, j], reduction="mean")
            for j, (w, logit) in enumerate(zip(weights, p.teacher_logits(c, tokens), strict=True))
        ) / sum(weights)

    def loss_fn(m, indices, aux_indices, goal_indices, teacher_targets):
        p = m.policy
        c = policy_context(p, x[indices], im[indices], g[indices])
        logits = p.teacher_logits(c, y[indices])
        ce = sum(
            w * nn.losses.cross_entropy(logit, y[indices, j], reduction="mean")
            for j, (w, logit) in enumerate(zip(weights, logits, strict=True))
        ) / sum(weights)
        kl = mx.array(0.0)
        for j, logit in enumerate(logits):
            ref = original[j][indices]
            lr = ref - mx.logsumexp(ref, axis=-1, keepdims=True)
            lp = logit - mx.logsumexp(logit, axis=-1, keepdims=True)
            kl = kl + ((mx.exp(lr) * (lr - lp)).sum(-1) * replay[indices]).mean() / 8
        loss = ce + kl
        if ground_mode:
            _, pool, _ = p.spatial_adapter.features(x[aux_indices], g[aux_indices])
            loss = loss + 0.5 * nn.losses.cross_entropy(
                p.spatial_adapter.grounding_head(pool),
                mx.array(a["labels"])[aux_indices],
                reduction="mean",
            )
        if goal_mode:
            cgoal = policy_context(p, x[goal_indices], im[goal_indices], g[goal_indices])
            loss = loss + 0.5 * ce_loss(p, cgoal, y[goal_indices])
        if distill_mode:
            patches, _, _ = p.spatial_adapter.features(x[indices], g[indices])
            value = p.spatial_adapter.distillation_head(patches)
            target = teacher_targets.reshape(-1, 144, 768)
            cosine = (value * target).sum(-1) / (
                mx.sqrt((value * value).sum(-1) * (target * target).sum(-1)) + 1e-6
            )
            loss = loss + 0.5 * (1 - cosine.mean())
        return loss

    value_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=0.0003, weight_decay=0.01)
    baseline = evaluate(
        policy, data["validation"], teacher.get("validation"), teacher_mean=teacher_mean
    )
    replay_base = np.mean(
        [v["button_f1"] for k, v in baseline["gameplay"].items() if k != "Hordes.io"]
    )

    def score(result):
        replay_now = np.mean(
            [v["button_f1"] for k, v in result["gameplay"].items() if k != "Hordes.io"]
        )
        if replay_now < replay_base - 0.05:
            return -1.0
        values = [result["gameplay"]["Hordes.io"]["button_f1"]]
        if ground_mode:
            values.append(result["grounding_balanced_accuracy"])
        if goal_mode:
            values.append(result["both_goals_correct"])
        if distill_mode:
            values.append(result["distillation_cosine"])
        return float(np.mean(values))

    best_score = score(baseline)
    best_step = 0
    best = [(k, mx.array(v)) for k, v in tree_flatten(model.trainable_parameters())]
    history = [{"step": 0, "validation": baseline, "score": best_score}]
    print(
        json.dumps(
            {
                "mode": args.mode,
                "baseline": baseline,
                "trainable": sum(v.size for _, v in best),
                "grounding_examples": len(grounded),
            }
        ),
        flush=True,
    )
    for step in range(1, args.steps + 1):
        choices = [hordes if rng.random() < 0.5 else public for _ in range(8)]
        idx = np.array([int(rng.choice(c[rng.integers(len(c))])) for c in choices])
        aux = mx.array(np.concatenate([rng.choice(group, 4) for group in ground_groups]))
        gi = mx.array(np.concatenate([rng.choice(group, 2) for group in goal_groups]))
        target = (
            mx.array(teacher["train"][idx].astype(np.float32) - teacher_mean)
            if distill_mode
            else mx.zeros((1,))
        )
        loss, grads = value_grad(model, mx.array(idx), aux, gi, target)
        grads, norm = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        mx.eval(loss, norm, model.trainable_parameters(), optimizer.state)
        if not np.isfinite(float(loss)):
            raise FloatingPointError("Nonfinite training")
        if step % 100 == 0 or step == args.steps:
            r = evaluate(
                policy, data["validation"], teacher.get("validation"), teacher_mean=teacher_mean
            )
            s = score(r)
            history.append({"step": step, "loss": float(loss), "score": s, "validation": r})
            (args.output / "progress.json").write_text(json.dumps(history, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "mode": args.mode,
                        "step": step,
                        "score": s,
                        "hordes_f1": r["gameplay"]["Hordes.io"]["button_f1"],
                        "ground": r["grounding_accuracy"],
                        "paired_goals": r["both_goals_correct"],
                        "feature_cosine": r["distillation_cosine"],
                    }
                ),
                flush=True,
            )
            if s > best_score:
                best_score, best_step = s, step
                best = [(k, mx.array(v)) for k, v in tree_flatten(model.trainable_parameters())]
    mx.save_safetensors(
        str(args.output / "last-adapters.safetensors"),
        dict(tree_flatten(model.trainable_parameters())),
    )
    final_fit = evaluate(policy, data["train"], teacher.get("train"), teacher_mean=teacher_mean)
    model.load_weights(best, strict=False)
    if frozen_hash(model) != fingerprint:
        raise RuntimeError("Frozen parents changed")
    results = {
        s: evaluate(policy, d, teacher.get(s), teacher_mean=teacher_mean) for s, d in data.items()
    }
    shuffled = evaluate(
        policy, data["test"], teacher.get("test"), shuffle=True, teacher_mean=teacher_mean
    )
    runtime.metadata.update(
        control_adapter={"rank": 4},
        spatial_adapter={"width": 64, "queries": 4},
        key_names=list(KEYS_WITH_TAB),
        selected_control_step=best_step,
        experiment_mode=args.mode,
        deployment_eligible=False,
        auxiliary_heads_runtime=False,
        training_scope="Static screenshots; weak controls and explicit grounding/conditional probes; offline teachers only",
    )
    runtime.save(args.output / "bundle")
    loaded = LayaP2PRuntime.load(args.output / "bundle")
    row = data["test"][0][0]
    with Image.open(row["frames"][0]["image"]) as image:
        pixels = mx.array(preprocess(image))
    before = model.step(pixels, *runtime.prepare_goal(row["goal"]))
    after = loaded.model.step(pixels, *loaded.prepare_goal(row["goal"]))
    mx.eval(before, after)
    reload_error = max(float(mx.abs(a - b).max()) for a, b in zip(before[1], after[1], strict=True))
    if reload_error > 1e-5:
        raise RuntimeError("Fresh-image reload differs")
    report = {
        "mode": args.mode,
        "selected_step": best_step,
        "metrics": results,
        "shuffled_test": shuffled,
        "final_step_training_metrics": final_fit,
        "history": history,
        "frozen_parents_unchanged": True,
        "reload_logits_max_error": reload_error,
        "deployment_eligible": False,
        "scope": "Static-image research. Auxiliary-head or synthetic-goal accuracy is not combat success. Holdouts have been reused across prior experiments; fresh validation is still required.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "mode": args.mode,
                "selected": best_step,
                "test": results["test"],
                "shuffled_test": shuffled,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
