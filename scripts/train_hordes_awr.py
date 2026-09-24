"""Advantage-weighted imitation (AWR) of weak Hordes controller labels.

Rewards are computed offline from the old controller's logged OCR state and only
reweight training examples. Logged state is never a model input: inference sees
the screenshot and goal only. Returns are 1.5 s observations, not causal credit;
other players, same-named targets, OCR lag and earlier attacks can explain them.
BC and AWR share seeds, sampler, updates and public-replay KL preservation.
Validation selects checkpoints; test never selects weights. Hordes validation
and test are two recordings each and are reused development holdouts. No game
inputs are sent.
"""

import argparse
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
    install_control_adapter,
    install_visual_adapter,
)
from laya_vision_stitch.p2p_pretrained_policy import physical_action
from laya_vision_stitch.p2p_pretrained_vision import preprocess
from scripts.train_p2p_control_adapter import frozen_hash
from scripts.train_p2p_visual_adapter import (
    cache_features,
    context,
    metrics,
    within_game_permutation,
)

HORDES = "Hordes.io"
SPLITS = ("train", "validation", "test")
ATTACK, SPACE = KEYS_WITH_TAB.index("1"), KEYS_WITH_TAB.index("space")
# Per-token loss weights of the existing visual-adapter recipe.
TOKEN_WEIGHTS = [2.0, 0.25, 0.25, 0.25, 1.0, 0.25, 0.5, 0.5]
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
KEY_FIELDS = (
    "hordes_button_f1",
    "idle_false_positive_rate",
    "public_mean_button_f1",
    "shuffled_hordes_button_f1",
    "damage_auc",
    "shuffled_damage_auc",
    "attack_vs_idle_auc",
    "shuffled_attack_vs_idle_auc",
)


def load_log(path):
    """Controller events plus a step index; times must be present and increasing."""
    sequence = [json.loads(line) for line in Path(path).read_text().splitlines()]
    times = [r.get("elapsed_s") for r in sequence]
    if not sequence or any(t is None for t in times):
        raise ValueError(f"Log lacks elapsed times: {path}")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError(f"Log times must increase: {path}")
    steps = {r["step"]: i for i, r in enumerate(sequence)}
    if len(steps) != len(sequence):
        raise ValueError(f"Duplicate controller steps: {path}")
    return sequence, steps


def example_return(sequence, index, horizon=1.5, threshold=0.02, health_penalty=0.5):
    """Observed outcome in (t, t + horizon]; zero with a status when unobservable.

    reward = target_damage - health_penalty * player_health_loss, where damage is
    the drop of the same-named selected target's OCR health fraction and loss the
    drop of the player's; drops at or below `threshold` count as zero (OCR noise).
    """
    if horizon <= 0 or threshold < 0 or health_penalty < 0:
        raise ValueError("Invalid reward settings")
    start = sequence[index]["elapsed_s"]
    end = index + 1
    while end < len(sequence) and sequence[end]["elapsed_s"] - start <= horizon:
        end += 1
    later = [r.get("state", {}) for r in sequence[index + 1 : end]]
    state = sequence[index].get("state", {})
    name, hp = state.get("target"), state.get("target_health")
    damage = 0.0
    if name is not None and hp is not None:
        seen = [s["target_health"] for s in later if s.get("target") == name]
        seen = [v for v in seen if v is not None]
        status = "observed" if seen else "unobserved"
        if seen and hp - min(seen) > threshold:
            damage = float(hp - min(seen))
    else:
        # A visible panel without a trusted name/health is missing OCR, not "no target".
        status = "unidentified" if state.get("target_panel") else "no_target"
    health = state.get("health")
    seen = [s["health"] for s in later if s.get("health") is not None]
    loss = 0.0
    if health is not None and seen and health - min(seen) > threshold:
        loss = float(health - min(seen))
    return {
        "reward": damage - health_penalty * loss,
        "target_damage": damage,
        "player_health_loss": loss,
        "target_status": status,
        "player_health_observed": health is not None and bool(seen),
        "out_of_range": any(bool(s.get("out_of_range")) for s in later),
        "complete_window": sequence[-1]["elapsed_s"] >= start + horizon,
    }


def row_returns(rows, logs=None, **settings):
    """Join Hordes rows to logged records via provenance; public replay gets None."""
    logs = {} if logs is None else logs
    result = []
    for row in rows:
        if row["game"] != HORDES:
            result.append(None)
            continue
        source = row["provenance"]
        if source["log"] not in logs:
            logs[source["log"]] = load_log(source["log"])
        sequence, steps = logs[source["log"]]
        if source["step"] not in steps:
            raise ValueError(f"Step {source['step']} missing from {source['log']}")
        index = steps[source["step"]]
        image = Path(source["log"]).parent / sequence[index].get("image", "missing")
        if image.resolve() != Path(row["frames"][-1]["image"]).resolve():
            raise ValueError(f"Row {row['id']} does not match its logged screenshot")
        result.append(example_return(sequence, index, **settings))
    return result


def action_group(row, outcome):
    buttons = row["action"]["buttons"]
    if buttons == ["1"]:
        return "attack_damage" if outcome["target_damage"] > 0 else "attack_no_damage"
    return "idle" if not buttons else "other_active"


def reward_statistics(rows, outcomes):
    pairs = [(r, o) for r, o in zip(rows, outcomes, strict=True) if o is not None]
    if not pairs:
        raise ValueError("No Hordes rewards")
    rewards = np.array([o["reward"] for _, o in pairs])
    groups = defaultdict(list)
    for r, o in pairs:
        groups[action_group(r, o)].append(o["reward"])
    attacks = [o for r, o in pairs if r["action"]["buttons"] == ["1"]]
    damaged = sum(o["target_damage"] > 0 for o in attacks)
    unknown = sum(o["target_damage"] == 0 and not o["complete_window"] for o in attacks)
    return {
        "examples": len(pairs),
        "reward": {
            "mean": float(rewards.mean()),
            "std": float(rewards.std()),
            "min": float(rewards.min()),
            "max": float(rewards.max()),
            "positive": int((rewards > 0).sum()),
            "negative": int((rewards < 0).sum()),
            "zero": int((rewards == 0).sum()),
        },
        "target_damage_examples": sum(o["target_damage"] > 0 for _, o in pairs),
        "player_health_loss_examples": sum(o["player_health_loss"] > 0 for _, o in pairs),
        "target_status": dict(Counter(o["target_status"] for _, o in pairs)),
        "player_health_unobserved": sum(not o["player_health_observed"] for _, o in pairs),
        "incomplete_windows": sum(not o["complete_window"] for _, o in pairs),
        "out_of_range_windows": sum(o["out_of_range"] for _, o in pairs),
        "by_action": {
            g: {"examples": len(v), "mean_reward": float(np.mean(v))}
            for g, v in sorted(groups.items())
        },
        "attack": {
            "labels": len(attacks),
            "damage_followed": damaged,
            "no_damage_complete_window": len(attacks) - damaged - unknown,
            "no_damage_incomplete_window": unknown,
            "out_of_range_after": sum(o["out_of_range"] for o in attacks),
        },
    }


def ridge_fit(x, y, alpha):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if alpha <= 0 or x.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Ridge needs a positive penalty and matching 2D inputs")
    mean, scale = x.mean(0), x.std(0) + 1e-6
    z = (x - mean) / scale
    weights = np.linalg.solve(z.T @ z + alpha * np.eye(x.shape[1]), z.T @ (y - y.mean()))
    return {"mean": mean, "scale": scale, "weights": weights, "offset": float(y.mean())}


def ridge_predict(fit, x):
    return ((np.asarray(x, np.float64) - fit["mean"]) / fit["scale"]) @ fit["weights"] + fit[
        "offset"
    ]


def cross_fitted_values(x, y, groups, alphas=ALPHAS):
    """Leave-one-run-out V(s); no training advantage uses a fit on its own run."""
    groups = np.asarray(groups)
    unique = sorted(set(groups.tolist()))
    if len(unique) < 2:
        raise ValueError("Need at least two runs for cross-fitting")
    y = np.asarray(y, np.float64)
    predictions, errors = {}, {}
    for alpha in alphas:
        out = np.empty(len(y))
        for group in unique:
            held = groups == group
            out[held] = ridge_predict(ridge_fit(x[~held], y[~held], alpha), x[held])
        predictions[alpha], errors[alpha] = out, float(np.mean((out - y) ** 2))
    best = min(alphas, key=errors.get)
    return predictions[best], best, errors


def awr_weights(returns, values, beta, max_weight=20.0):
    """w = min(exp(A / beta), max_weight), A = R - V scaled to unit std; mean one."""
    if beta <= 0 or max_weight <= 1:
        raise ValueError("Need beta > 0 and max_weight > 1")
    advantage = np.asarray(returns, np.float64) - np.asarray(values, np.float64)
    scale = advantage.std() if advantage.size else 0.0
    if not np.isfinite(advantage).all() or scale < 1e-8:
        raise ValueError("Advantages need finite, nonzero spread")
    raw = np.exp(np.minimum(advantage / scale / beta, np.log(max_weight)))
    return raw / raw.mean(), raw


def weight_statistics(weights, raw, rows, outcomes, max_weight):
    groups = defaultdict(list)
    for w, r, o in zip(weights, rows, outcomes, strict=True):
        groups[action_group(r, o)].append(w)
    return {
        "effective_sample_fraction": float(weights.sum() ** 2 / (weights**2).sum() / len(weights)),
        "clipped_fraction": float(np.mean(raw >= max_weight * (1 - 1e-9))),
        "raw_mean": float(raw.mean()),
        "max_normalized": float(weights.max()),
        "by_action": {
            g: {"mean_weight": float(np.mean(v)), "weight_share": float(np.sum(v) / weights.sum())}
            for g, v in sorted(groups.items())
        },
    }


def auc(scores, labels):
    """Mann-Whitney AUC with half credit for ties; None without both classes."""
    scores, labels = np.asarray(scores, np.float64), np.asarray(labels, bool)
    if scores.shape != labels.shape:
        raise ValueError("Scores and labels differ in shape")
    positive, negative = scores[labels], scores[~labels]
    if not len(positive) or not len(negative):
        return None
    diff = positive[:, None] - negative[None]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def auc_interval(scores, labels, draws=2000, seed=0):
    """95% class-stratified frame bootstrap; ignores within-recording correlation."""
    scores, labels = np.asarray(scores, np.float64), np.asarray(labels, bool)
    positive, negative = np.flatnonzero(labels), np.flatnonzero(~labels)
    if not len(positive) or not len(negative):
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        index = np.concatenate(
            [rng.choice(positive, len(positive)), rng.choice(negative, len(negative))]
        )
        values.append(auc(scores[index], labels[index]))
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def attack_probability(policy, contexts):
    """Model P(key "1") under the canonical sorted key order.

    Keys are encoded ascending, and only "space" sorts before "1", so
    P = P(k0 = "1") + P(k0 = space) * P(k1 = "1" | k0 = space). Probability on
    non-canonical orders is ignored.
    """
    forced = mx.array(np.tile([[SPACE, ATTACK, 0, 0, 0, 0, 0, 0]], (len(contexts), 1)), mx.int32)
    logits = policy.teacher_logits(contexts, forced)
    first, second = mx.softmax(logits[0], -1), mx.softmax(logits[1], -1)
    return first[:, ATTACK] + first[:, SPACE] * second[:, ATTACK]


def predict(policy, split, permutation=None, batch=8):
    """Greedy tokens and P(attack) from one cached-feature forward pass."""
    rows, a = split
    tokens, probabilities = [], []
    for start in range(0, len(rows), batch):
        indices = np.arange(start, min(start + batch, len(rows)))
        images = indices if permutation is None else permutation[indices]
        c = context(policy, mx.array(a["images"][images]), mx.array(a["goals"][indices]))
        greedy, _ = policy.decode(c)
        p = attack_probability(policy, c)
        mx.eval(greedy, p)
        tokens.extend(greedy.tolist())
        probabilities.extend(np.asarray(p.astype(mx.float32)).tolist())
    return tokens, np.array(probabilities)


def outcome_summary(rows, outcomes, p_attack, tokens):
    """Does the model prefer recorded attacks that were followed by damage?"""
    greedy = np.array(
        ["1" in physical_action(t, key_names=KEYS_WITH_TAB)["buttons"] for t in tokens]
    )
    hordes = [i for i, r in enumerate(rows) if r["game"] == HORDES]
    attack = [i for i in hordes if rows[i]["action"]["buttons"] == ["1"]]
    idle = [i for i in hordes if not rows[i]["action"]["buttons"]]
    damage = [i for i in attack if outcomes[i]["target_damage"] > 0]
    missed = [
        i for i in attack if outcomes[i]["target_damage"] == 0 and outcomes[i]["complete_window"]
    ]
    far = [i for i in attack if outcomes[i]["out_of_range"]]

    def mean(indices):
        return float(np.mean(p_attack[indices])) if indices else None

    pair = damage + missed
    both = attack + idle
    return {
        "attack_frames": len(attack),
        "damage_followed": len(damage),
        "no_damage": len(missed),
        "excluded_incomplete_no_damage": len(attack) - len(pair),
        "damage_auc": auc(p_attack[pair], [i in damage for i in pair]) if pair else None,
        "damage_auc_interval": (
            auc_interval(p_attack[pair], [i in damage for i in pair]) if pair else None
        ),
        "attack_vs_idle_auc": auc(p_attack[both], [i in attack for i in both]) if both else None,
        "mean_p_attack": {
            "damage_followed": mean(damage),
            "no_damage": mean(missed),
            "out_of_range_after": mean(far),
            "idle": mean(idle),
            "all_hordes": mean(hordes),
        },
        "greedy_attack_rate": {
            "damage_followed": float(greedy[damage].mean()) if damage else None,
            "no_damage": float(greedy[missed].mean()) if missed else None,
            "idle": float(greedy[idle].mean()) if idle else None,
        },
    }


def evaluate_all(policy, data, outcomes, permutations, splits=SPLITS):
    result = {}
    for split in splits:
        rows, arrays = data[split]
        tokens, p = predict(policy, data[split])
        item = {
            "metrics": metrics(rows, tokens, arrays["tokens"]),
            "outcomes": outcome_summary(rows, outcomes[split], p, tokens),
        }
        if split in permutations:
            tokens, p = predict(policy, data[split], permutation=permutations[split])
            item["shuffled_metrics"] = metrics(rows, tokens, arrays["tokens"])
            item["shuffled_outcomes"] = outcome_summary(rows, outcomes[split], p, tokens)
        result[split] = item
    return result


def compact(evaluation):
    """Key held-out numbers; full metrics stay in the artifact report."""
    result = {}
    for split in ("validation", "test"):
        item = evaluation[split]
        h, s = item["metrics"][HORDES], item["shuffled_metrics"][HORDES]
        result[split] = {
            "hordes_button_f1": h["button_f1"],
            "supported_button_macro_f1": h["supported_button_macro_f1"],
            "exact_button_set_accuracy": h["exact_button_set_accuracy"],
            "idle_false_positive_rate": h["idle_false_positives"] / max(1, h["idle_examples"]),
            "per_button_f1": {b: v["f1"] for b, v in sorted(h["buttons"].items())},
            "public_mean_button_f1": float(
                np.mean([v["button_f1"] for g, v in item["metrics"].items() if g != HORDES])
            ),
            "shuffled_hordes_button_f1": s["button_f1"],
            "damage_auc": item["outcomes"]["damage_auc"],
            "damage_auc_interval": item["outcomes"]["damage_auc_interval"],
            "shuffled_damage_auc": item["shuffled_outcomes"]["damage_auc"],
            "attack_vs_idle_auc": item["outcomes"]["attack_vs_idle_auc"],
            "shuffled_attack_vs_idle_auc": item["shuffled_outcomes"]["attack_vs_idle_auc"],
            "mean_p_attack": item["outcomes"]["mean_p_attack"],
            "greedy_attack_rate": item["outcomes"]["greedy_attack_rate"],
        }
    return result


def seed_means(reports, checkpoint):
    """Per-arm means over training seeds of key held-out numbers."""
    arms = defaultdict(list)
    for report in reports.values():
        arms[report["variant"]].append(compact(report[checkpoint]))
    result = {}
    for arm, items in arms.items():
        result[arm] = {"seeds": len(items)}
        for split in ("validation", "test"):
            values = {k: [c[split][k] for c in items] for k in KEY_FIELDS}
            values["mean_p_attack_idle"] = [c[split]["mean_p_attack"]["idle"] for c in items]
            result[arm][split] = {
                k: None if any(x is None for x in v) else float(np.mean(v))
                for k, v in values.items()
            }
    return result


def train_variant(args, seed, name, sample_weights, data, outcomes, permutations, output):
    """One adapter run; only the per-example CE weights differ between variants."""
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    runtime = LayaP2PRuntime.load(args.bundle)
    if runtime.metadata.get("visual_adapter") or runtime.metadata.get("control_adapter"):
        raise ValueError("Start from the unadapted stitched checkpoint")
    install_control_adapter(runtime.model, rank=args.rank)
    if args.bottleneck:
        install_visual_adapter(runtime.model, args.bottleneck)
    model, policy = runtime.model, runtime.model.policy
    fingerprint = frozen_hash(model)
    parameters = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    baseline = evaluate_all(policy, data, outcomes, permutations)
    base = baseline["validation"]["metrics"]
    base_public = np.mean([v["button_f1"] for g, v in base.items() if g != HORDES])
    base_idle = base[HORDES]["idle_false_positives"] / max(1, base[HORDES]["idle_examples"])

    def score(m, shuffled=None):
        # Same selection rule and shortcut gates as the visual-adapter trainer.
        replay = float(np.mean([v["button_f1"] for g, v in m.items() if g != HORDES]))
        h = m[HORDES]
        if shuffled is not None:
            if h["idle_false_positives"] / max(1, h["idle_examples"]) > base_idle + 0.1:
                return -1.0
            if h["button_f1"] < shuffled[HORDES]["button_f1"] + 0.02:
                return -1.0
        value = (h["button_f1"] + h["supported_button_macro_f1"] + replay) / 3
        return value if replay >= base_public - 0.05 else -1.0

    train_rows, arrays = data["train"]
    x, g, y = [mx.array(arrays[k]) for k in ("images", "goals", "tokens")]
    w = mx.array(sample_weights.astype(np.float32))
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
    idle = [v for (game, b), v in groups.items() if game == HORDES and not b]
    active = [v for (game, b), v in groups.items() if game == HORDES and b]
    public = [v for (game, _), v in groups.items() if game != HORDES]
    replay_mask = mx.array([float(r["game"] != HORDES) for r in train_rows])

    def loss_fn(m, indices):
        logits = m.policy.teacher_logits(context(m.policy, x[indices], g[indices]), y[indices])
        ce, kl = mx.array(0.0), mx.array(0.0)
        for i, logit in enumerate(logits):
            token_ce = nn.losses.cross_entropy(logit, y[indices, i], reduction="none")
            ce = ce + TOKEN_WEIGHTS[i] * (token_ce * w[indices]).mean()
            ref = teacher[i][indices]
            log_ref = ref - mx.logsumexp(ref, axis=-1, keepdims=True)
            log_pred = logit - mx.logsumexp(logit, axis=-1, keepdims=True)
            divergence = (mx.exp(log_ref) * (log_ref - log_pred)).sum(-1)
            kl = kl + (divergence * replay_mask[indices]).mean()
        return ce / sum(TOKEN_WEIGHTS) + args.replay_kl * kl / 8

    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.01)
    value_grad = nn.value_and_grad(model, loss_fn)
    best_score, best_step = score(base), 0
    best = [(k, mx.array(v)) for k, v in tree_flatten(model.trainable_parameters())]
    history = [
        {
            "step": 0,
            "validation": base,
            "validation_outcomes": baseline["validation"]["outcomes"],
            "score": best_score,
        }
    ]
    for step in range(1, args.steps + 1):
        choices = []
        for _ in range(args.batch_size):
            if rng.random() >= 0.5:
                choices.append(public)
            else:
                choices.append(idle if rng.random() < 0.5 else active)
        indices = mx.array([int(rng.choice(c[rng.integers(len(c))])) for c in choices])
        loss, grads = value_grad(model, indices)
        grads, norm = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        mx.eval(loss, norm, model.trainable_parameters(), optimizer.state)
        if not np.isfinite(float(loss)) or not np.isfinite(float(norm)):
            raise FloatingPointError("Nonfinite training loss/gradient")
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate_all(policy, data, outcomes, permutations, splits=("validation",))
            current = result["validation"]
            selection = score(current["metrics"], current["shuffled_metrics"])
            history.append(
                {
                    "step": step,
                    "loss": float(loss),
                    "gradient_norm": float(norm),
                    "validation": current["metrics"],
                    "shuffled_validation": current["shuffled_metrics"],
                    "validation_outcomes": current["outcomes"],
                    "score": selection,
                }
            )
            (output / "progress.json").write_text(json.dumps(history, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "variant": name,
                        "seed": seed,
                        "step": step,
                        "loss": round(float(loss), 4),
                        "score": round(selection, 4),
                        "hordes_f1": round(current["metrics"][HORDES]["button_f1"], 4),
                        "damage_auc": current["outcomes"]["damage_auc"],
                    }
                ),
                flush=True,
            )
            if selection > best_score:
                best_score, best_step = selection, step
                best = [(k, mx.array(v)) for k, v in tree_flatten(model.trainable_parameters())]
    final_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(output / "final-adapters.safetensors"), final_weights)
    final = evaluate_all(policy, data, outcomes, permutations)
    model.load_weights(best, strict=False)
    mx.save_safetensors(str(output / "selected-adapters.safetensors"), dict(best))
    if frozen_hash(model) != fingerprint:
        raise RuntimeError("Frozen pretrained parameters changed")
    selected = evaluate_all(policy, data, outcomes, permutations)
    # Fresh preprocessing + vision + Laya + adapters must match the cached-feature path.
    rows, test_arrays = data["test"]
    with Image.open(rows[0]["frames"][-1]["image"]) as image:
        pixels = mx.array(preprocess(image))
    fresh = runtime.model.step(pixels, *runtime.prepare_goal(rows[0]["goal"]))[0]
    cached = policy.decode(
        context(policy, mx.array(test_arrays["images"][:1]), mx.array(test_arrays["goals"][:1]))
    )[0]
    if fresh.tolist() != cached.tolist():
        raise RuntimeError("Fresh-image action differs from cached-feature path")
    report = {
        "variant": name,
        "seed": seed,
        "updates": args.steps,
        "selected_step": best_step,
        "trainable_parameters": parameters,
        "parents_unchanged": True,
        "frozen_sha256": fingerprint,
        "fresh_image_tokens_match": True,
        "deployment_eligible": False,
        "baseline": baseline,
        "selected": selected,
        "final": final,
        "history": history,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument(
        "--bottleneck", type=int, default=64, help="Zero gives the decoder-only control adapter"
    )
    p.add_argument("--learning-rate", type=float, default=0.0001)
    p.add_argument("--replay-kl", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--betas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--max-weight", type=float, default=20.0)
    p.add_argument("--horizon", type=float, default=1.5)
    p.add_argument("--threshold", type=float, default=0.02)
    p.add_argument("--health-penalty", type=float, default=0.5)
    p.add_argument("--seeds", type=int, nargs="+", default=[20260923])
    p.add_argument(
        "--shuffle-seed", type=int, default=20260923, help="Fixed image-shuffle controls"
    )
    args = p.parse_args()
    if not 1 <= args.steps <= 1000:
        p.error("Keep runs between 1 and 1,000 updates on the shared GPU")
    if (
        args.eval_every < 1
        or any(b <= 0 for b in args.betas)
        or len(set(args.seeds)) != len(args.seeds)
    ):
        p.error("Invalid evaluation interval, beta or seed list")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "args.json").write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2
        )
        + "\n"
    )
    runtime = LayaP2PRuntime.load(args.bundle)
    data = cache_features(runtime, args.data, args.cache, args.bundle)
    del runtime
    settings = {
        "horizon": args.horizon,
        "threshold": args.threshold,
        "health_penalty": args.health_penalty,
    }
    logs, outcomes, rewards = {}, {}, {}
    for split in SPLITS:
        outcomes[split] = row_returns(data[split][0], logs, **settings)
        rewards[split] = reward_statistics(data[split][0], outcomes[split])
    audit_path = args.data / "attack-outcomes.json"
    if audit_path.is_file() and args.horizon == 1.5 and args.threshold == 0.02:
        audit = json.loads(audit_path.read_text())
        rewards["matches_dataset_attack_audit"] = all(
            audit[s]["same_named_target_health_drop"] == rewards[s]["attack"]["damage_followed"]
            and audit[s]["attack_labels"] == rewards[s]["attack"]["labels"]
            for s in SPLITS
        )
    print(json.dumps({"rewards": rewards}), flush=True)

    # V(s): ridge on the frozen image token, fit on train Hordes rows only.
    # The goal is identical for every Hordes row, so it carries no state information.
    def hordes(split):
        rows, arrays = data[split]
        index = np.array([i for i, r in enumerate(rows) if r["game"] == HORDES])
        returns = np.array([outcomes[split][i]["reward"] for i in index])
        return index, arrays["images"][index].astype(np.float64), returns

    train_index, train_x, train_r = hordes("train")
    runs = [data["train"][0][i]["episode"] for i in train_index]
    values, alpha, errors = cross_fitted_values(train_x, train_r, runs)
    fit = ridge_fit(train_x, train_r, alpha)
    value_report = {
        "features": "Frozen 1024D Open-P2P image token (standardized); ridge regression",
        "alpha": alpha,
        "cross_validation_mse": {str(k): v for k, v in errors.items()},
        "fit": "Train Hordes rows only; train advantages use leave-one-run-out predictions",
        "splits": {},
    }
    split_values = {"train": values}
    for split in ("validation", "test"):
        split_values[split] = ridge_predict(fit, hordes(split)[1])
    for split in SPLITS:
        index, _, returns = hordes(split)
        v = split_values[split]
        rows = data[split][0]
        attack = [
            k
            for k, i in enumerate(index)
            if rows[i]["action"]["buttons"] == ["1"]
            and (outcomes[split][i]["target_damage"] > 0 or outcomes[split][i]["complete_window"])
        ]
        labels = [outcomes[split][index[k]]["target_damage"] > 0 for k in attack]
        value_report["splits"][split] = {
            "r2": float(1 - np.mean((returns - v) ** 2) / max(1e-12, returns.var())),
            "value_damage_auc_on_attack_frames": auc(v[attack], labels),
            "value_damage_auc_interval": auc_interval(v[attack], labels),
        }
    print(json.dumps({"value": value_report}), flush=True)

    train_rows = data["train"][0]
    train_outcomes = [outcomes["train"][i] for i in train_index]
    variants = {"bc": np.ones(len(train_rows))}
    weight_report, per_example = {}, {}
    for beta in args.betas:
        weights, raw = awr_weights(train_r, values, beta, args.max_weight)
        full = np.ones(len(train_rows))
        full[train_index] = weights
        variants[f"awr-beta-{beta:g}"] = full
        weight_report[f"{beta:g}"] = weight_statistics(
            weights, raw, [train_rows[i] for i in train_index], train_outcomes, args.max_weight
        )
        for k, i in enumerate(train_index):
            per_example.setdefault(int(i), {})[f"weight_beta_{beta:g}"] = float(weights[k])
    print(json.dumps({"weights": weight_report}), flush=True)
    with (args.output / "rewards.jsonl").open("w") as handle:
        for split in SPLITS:
            index = hordes(split)[0]
            for k, i in enumerate(index):
                row = data[split][0][i]
                item = {
                    "split": split,
                    "id": row["id"],
                    "buttons": row["action"]["buttons"],
                    **outcomes[split][i],
                    "value": float(split_values[split][k]),
                    **(per_example.get(int(i), {}) if split == "train" else {}),
                }
                handle.write(json.dumps(item) + "\n")

    # One within-game shuffle per split for every arm and seed, so controls are matched.
    permutations = {
        "validation": within_game_permutation(data["validation"][0], args.shuffle_seed + 1),
        "test": within_game_permutation(data["test"][0], args.shuffle_seed + 2),
    }
    reports = {}
    for seed in args.seeds:
        for name, weights in variants.items():
            output = args.output / name / f"seed-{seed}"
            reports[f"{name}/seed-{seed}"] = train_variant(
                args, seed, name, weights, data, outcomes, permutations, output
            )
            mx.clear_cache()
    first = next(iter(reports.values()))
    checks = {
        "parents_unchanged": all(r["parents_unchanged"] for r in reports.values()),
        "same_frozen_parents": len({r["frozen_sha256"] for r in reports.values()}) == 1,
        "identical_baselines": all(
            compact(r["baseline"]) == compact(first["baseline"]) for r in reports.values()
        ),
        "fresh_image_tokens_match": all(r["fresh_image_tokens_match"] for r in reports.values()),
    }
    summary = {
        "scope": "Offline development experiment. Logged controller OCR state only weights "
        "training examples and is never a model input. Weak single-frame Hordes labels plus "
        "public human replay; two-recording Hordes validation/test are reused development "
        "holdouts. No live inputs or gameplay claim; nothing is deployment eligible.",
        "settings": json.loads((args.output / "args.json").read_text()),
        "reward_definition": {
            **settings,
            "reward": "target_damage - health_penalty * player_health_loss over (t, t + horizon]",
            "target_damage": "drop of the same-named selected target's OCR health fraction "
            "(minimum later reading); zero at or below threshold, without a named target at t, "
            "or without later readings",
            "player_health_loss": "drop of the player's OCR health fraction; zero at or below "
            "threshold",
            "out_of_range": "flag only: any logged out_of_range state in the window",
        },
        "rewards": rewards,
        "value_baseline": value_report,
        "awr_weights": weight_report,
        "checks": checks,
        "trainable_parameters": first["trainable_parameters"],
        "baseline": compact(first["baseline"]),
        "seed_means": {c: seed_means(reports, c) for c in ("selected", "final")},
        "runs": {
            key: {
                "selected_step": r["selected_step"],
                "selected": compact(r["selected"]),
                "final": compact(r["final"]),
                "validation_history": [
                    {
                        "step": h["step"],
                        "score": h["score"],
                        "hordes_button_f1": h["validation"][HORDES]["button_f1"],
                        "damage_auc": h["validation_outcomes"]["damage_auc"],
                        "mean_p_attack_idle": h["validation_outcomes"]["mean_p_attack"]["idle"],
                    }
                    for h in r["history"]
                ],
            }
            for key, r in reports.items()
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"checks": checks, "output": str(args.output)}), flush=True)
    if not all(checks.values()):
        raise RuntimeError(f"Consistency checks failed: {checks}")


if __name__ == "__main__":
    main()
