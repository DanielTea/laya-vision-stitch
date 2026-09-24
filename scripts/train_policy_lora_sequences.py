"""Train LoRA in the temporal policy and decoder on sequences from many games.

Vision stays frozen, so cached image tokens feed the policy; gradients pass through all
policy layers. Windows are sampled uniformly over games, then sequences. `--policy-rank 0`
trains decoder LoRA through the same loop as the matched control. Validation NLL selects
the checkpoint; held-out games never select weights. `--targets` adds a target encoder fed
with hindsight target points (add_target_labels.py) and audits whether actions use them.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from laya_vision_stitch.laya_p2p import LayaP2PRuntime
from laya_vision_stitch.p2p_adaptation import (
    KEYS_WITH_TAB,
    install_control_adapter,
    install_policy_lora,
)
from laya_vision_stitch.sequence_metrics import (
    baselines,
    evaluate_actions,
    token_buttons,
    token_mouse,
)
from laya_vision_stitch.sequence_policy import guided_decode, sequence_contexts
from laya_vision_stitch.target_conditioning import install_target_encoder, toward_target


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


CAPTIONS = {}  # (data directory name, sequence) -> bridged caption goal vector
USE_TARGETS = False
TARGET_LABELS = "targets"


def load(item):
    cache, split = item.rsplit(":", 1)
    a = dict(np.load(Path(cache) / f"{split}.npz"))
    rows = read(Path(cache) / f"{split}.jsonl")
    data = Path(json.loads((Path(cache) / "metadata.json").read_text())["data"]).name
    a["captioned"] = np.zeros(len(rows), bool)
    a["original_goals"] = a["goals"].copy()
    for i, r in enumerate(rows):
        vector = CAPTIONS.get((data, r["sequence"]))
        if vector is not None:
            a["goals"][i], a["captioned"][i] = vector, True
    groups = defaultdict(list)
    for i, s in enumerate(a["sequence_index"]):
        groups[int(s)].append(i)
    seqs = [np.array(sorted(v, key=lambda i: a["steps"][i])) for v in groups.values()]
    a["target_xy"] = np.full((len(rows), 2), np.nan, np.float32)
    a["target_pointer"] = np.zeros(len(rows), bool)
    labels = Path(cache) / f"{split}.{TARGET_LABELS}.npz"
    if labels.exists():
        a["target_xy"] = np.load(labels)["target_xy"]
        kinds = json.loads((Path(cache) / f"{split}.{TARGET_LABELS}.json").read_text())["games"]
        a["target_pointer"] = np.array(
            [kinds[r["game"]]["targets"] == "press position" for r in rows]
        )
    return rows, a, seqs


def token_nll(policy, contexts, tokens, mask=None):
    logits = policy.teacher_logits(contexts[:, None], tokens)
    loss = sum(
        nn.losses.cross_entropy(logit.astype(mx.float32), tokens[:, j], reduction="none")
        for j, logit in enumerate(logits)
    )
    return loss if mask is None else (loss * mask).sum() / mx.maximum(mask.sum(), 1)


def forward(policy, a, frames, targets=None):
    if targets is None and USE_TARGETS:
        targets = a["target_xy"][frames]
    return sequence_contexts(
        policy,
        mx.array(a["images"][frames]),
        mx.array(a["goals"][frames]),
        mx.array(a["tokens"][frames]),
        target=None if targets is None else mx.array(targets),
    )


def shuffled_targets(a, rows):
    """Each labelled frame gets a target from another sequence of the same game."""
    result = a["target_xy"].copy()
    rng = np.random.default_rng(29)
    by_game = defaultdict(list)
    for i in np.flatnonzero(np.isfinite(a["target_xy"][:, 0])):
        by_game[rows[i]["game"]].append(i)
    for idx in by_game.values():
        idx = np.array(idx)
        order = rng.permutation(len(idx))
        for i, j in zip(idx, idx[order], strict=True):
            if a["sequence_index"][i] == a["sequence_index"][j]:
                j = idx[(np.searchsorted(idx, i) + len(idx) // 2) % len(idx)]
            result[i] = a["target_xy"][j]
    return result


def target_audit(policy, item, limit=None):
    """NLL and greedy behavior on frames with a target: own, another sequence's, or none."""
    rows, a, seqs = load(item)
    if not np.isfinite(a["target_xy"][:, 0]).any():
        return None
    seqs = [g for g in seqs if np.isfinite(a["target_xy"][g, 0]).any()]
    if limit:
        seqs = seqs[:: max(1, len(seqs) // limit)]
    variants = {
        "own": a["target_xy"],
        "shuffled": shuffled_targets(a, rows),
        "none": np.full_like(a["target_xy"], np.nan),
    }
    frames = np.concatenate(seqs)
    has = np.isfinite(a["target_xy"][frames, 0]) & a["label_complete"][frames]
    pointer = a["target_pointer"][frames]
    result = {"frames_with_target": int(has.sum()), "pointer_frames": int((has & pointer).sum())}
    human = token_buttons(a["tokens"][frames])
    result["recorded"] = toward_target(
        [human[i] for i in np.flatnonzero(has & pointer)],
        token_mouse(a["tokens"][frames])[has & pointer],
        a["target_xy"][frames][has & pointer],
        a.get("cursor_xy", np.full((len(rows), 2), np.nan))[frames][has & pointer],
    )
    for name, targets in variants.items():
        losses, greedy = [], []
        for g in seqs:
            c = forward(policy, a, g[None], targets[g][None])[0]
            loss = token_nll(policy, c, mx.array(a["tokens"][g]))
            tokens, _ = guided_decode(policy, c)
            mx.eval(loss, tokens)
            losses.extend(np.asarray(loss).tolist())
            greedy.extend(np.asarray(tokens).tolist())
        losses, greedy = np.array(losses), np.array(greedy)
        buttons = token_buttons(greedy)
        result[name] = {
            "nll": float(losses[has].mean()),
            "nll_pointer_games": float(losses[has & pointer].mean())
            if (has & pointer).any()
            else None,
            # Behavior is scored against the frame's own target whatever the model was given.
            "behavior_pointer_games": toward_target(
                [buttons[i] for i in np.flatnonzero(has & pointer)],
                token_mouse(greedy)[has & pointer],
                a["target_xy"][frames][has & pointer],
                a.get("cursor_xy", np.full((len(rows), 2), np.nan))[frames][has & pointer],
            ),
        }
    result["shuffled_minus_own"] = result["shuffled"]["nll"] - result["own"]["nll"]
    result["none_minus_own"] = result["none"]["nll"] - result["own"]["nll"]
    return result


def goal_audit(policy, a, seqs):
    """Recorded-action NLL with the clip's own caption, another clip's caption, or none."""
    captioned = [g for g in seqs if a["captioned"][g[0]]]
    if len(captioned) < 2:
        return None
    rng = np.random.default_rng(17)
    order = rng.permutation(len(captioned))
    result = {"sequences": len(captioned)}
    for name in ["own", "swapped", "original", "none"]:
        losses = []
        for k, g in enumerate(captioned):
            goals = a["goals"][g]
            if name == "swapped":
                other = (
                    captioned[order[k]] if order[k] != k else captioned[(k + 1) % len(captioned)]
                )
                goals = np.broadcast_to(a["goals"][other[0]], goals.shape)
            elif name == "original":
                goals = a["original_goals"][g]
            elif name == "none":
                goals = np.zeros_like(goals)
            c = sequence_contexts(
                policy,
                mx.array(a["images"][g])[None],
                mx.array(goals)[None],
                mx.array(a["tokens"][g])[None],
            )[0]
            loss = token_nll(policy, c, mx.array(a["tokens"][g]))
            mx.eval(loss)
            losses.extend(np.asarray(loss).tolist())
        result[f"nll_{name}"] = float(np.mean(losses))
    result["swap_minus_own"] = result["nll_swapped"] - result["nll_own"]
    return result


def evaluate(policy, item, samples=1, limit=None):
    rows, a, seqs = load(item)
    audit = goal_audit(policy, a, seqs) if limit is None else None
    if limit:
        seqs = seqs[:: max(1, len(seqs) // limit)]
    losses, greedy, sampled, order = [], [], [[] for _ in range(samples)], []
    key = mx.random.key(3)
    for g in seqs:
        c = forward(policy, a, g[None])[0]
        t = mx.array(a["tokens"][g])
        loss = token_nll(policy, c, t)
        tokens, _ = guided_decode(policy, c)
        drawn = []
        for _ in range(samples):
            key, sub = mx.random.split(key)
            drawn.append(guided_decode(policy, c, temperature=1.0, key=sub)[0])
        mx.eval(loss, tokens, *drawn)
        keep = a["label_complete"][g]
        losses.extend(np.asarray(loss)[keep].tolist())
        greedy.extend(np.asarray(tokens).tolist())
        for s in range(samples):
            sampled[s].extend(np.asarray(drawn[s]).tolist())
        order.extend(g.tolist())
    used = [rows[i] for i in order]
    truth, mouse = token_buttons(a["tokens"][order]), token_mouse(a["tokens"][order])

    def score(tokens):
        return evaluate_actions(used, token_buttons(tokens), truth, token_mouse(tokens), mouse)

    return {
        "frames": len(order),
        "goal_audit": audit,
        "mean_nll": float(np.mean(losses)),
        "greedy": score(greedy),
        "sampled": [score(s) for s in sampled],
        **baselines(used, truth, mouse),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--evaluate", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--policy-rank", type=int, default=8)
    p.add_argument("--decoder-rank", type=int, default=8)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--validation-sequences", type=int, default=120)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--captions", type=Path, help="Hindsight goal captions (caption_clips.py)")
    p.add_argument("--goal-dropout", type=float, default=0.0)
    p.add_argument(
        "--targets", action="store_true", help="Train a target encoder on hindsight targets"
    )
    p.add_argument("--target-dropout", type=float, default=0.15)
    p.add_argument(
        "--target-labels", default="targets", help="Label set from add_target_labels.py --tag"
    )
    p.add_argument(
        "--evaluate-existing",
        action="store_true",
        help="Skip training; evaluate the checkpoint already in --output (history from <output>.log)",
    )
    args = p.parse_args()
    if not args.evaluate_existing:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "args.json").write_text(
            json.dumps(
                {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2
            )
        )
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    runtime = LayaP2PRuntime.load(args.bundle)
    model = runtime.model
    policy = model.policy
    if args.captions:
        vectors = {}
        for record in read(args.captions):
            text = record["caption"]
            if not text:
                continue
            if text not in vectors:
                vector = model.bridge(model.goal_features(*runtime.prepare_goal(text)))
                vectors[text] = np.asarray(vector.astype(mx.float32))[0]
            CAPTIONS[(Path(record["data"]).name, record["sequence"])] = vectors[text]
        print(
            json.dumps({"captioned_sequences": len(CAPTIONS), "unique_captions": len(vectors)}),
            flush=True,
        )
    reference = {item: evaluate(policy, item) for item in args.evaluate}
    for item, r in reference.items():
        print(json.dumps({"pretrained": item, "nll": round(r["mean_nll"], 4)}), flush=True)
    install_control_adapter(model, rank=args.decoder_rank)
    if args.policy_rank:
        install_policy_lora(model, rank=args.policy_rank)
    if args.targets:
        global USE_TARGETS, TARGET_LABELS
        USE_TARGETS, TARGET_LABELS = True, args.target_labels
        install_target_encoder(model)
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    # Game-balanced window sampling across all training caches.
    pools = defaultdict(list)
    data = []
    for item in args.train:
        rows, a, seqs = load(item)
        data.append(a)
        for g in seqs:
            if len(g) >= args.window:
                pools[rows[g[0]]["game"]].append((len(data) - 1, g))
    games = sorted(pools)

    def loss_fn(m, images, goals, tokens, mask, targets=None):
        c = sequence_contexts(m.policy, images, goals, tokens, target=targets)
        b, t = tokens.shape[:2]
        return token_nll(
            m.policy, c.reshape(b * t, 1024), tokens.reshape(b * t, 8), mask.reshape(-1)
        )

    value_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.01)
    if args.evaluate_existing:
        lines = args.output.with_suffix(".log").read_text().splitlines()
        history = [json.loads(x) for x in lines if x.startswith('{"step"')]
        best_step = min(history, key=lambda h: h["validation_nll"])["step"]
        steps = range(0)
    else:
        best = evaluate(policy, args.validation, samples=0, limit=args.validation_sequences)[
            "mean_nll"
        ]
        best_step, history = 0, [{"step": 0, "validation_nll": best}]
        model.save_weights(str(args.output / "model.safetensors"))
        steps = range(1, args.steps + 1)
    for step in steps:
        batch = []
        for game in rng.choice(games, args.batch):
            source, g = pools[game][rng.integers(len(pools[game]))]
            start = rng.integers(0, len(g) - args.window + 1)
            batch.append((source, g[start : start + args.window]))
        stack = {
            k: np.stack([data[s][k][f] for s, f in batch])
            for k in ["images", "goals", "tokens", "label_complete", "target_xy"]
        }
        if args.goal_dropout:
            drop = rng.random(len(batch)) < args.goal_dropout
            stack["goals"][drop] = 0
        if args.target_dropout:
            stack["target_xy"][rng.random(len(batch)) < args.target_dropout] = np.nan
        stack = {k: mx.array(v) for k, v in stack.items()}
        loss, grads = value_grad(
            model,
            stack["images"],
            stack["goals"],
            stack["tokens"],
            stack["label_complete"].astype(mx.float32),
            stack["target_xy"] if args.targets else None,
        )
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        if step % 250 == 0 or step == args.steps:
            v = evaluate(policy, args.validation, samples=0, limit=args.validation_sequences)[
                "mean_nll"
            ]
            history.append({"step": step, "train_loss": float(loss), "validation_nll": v})
            print(json.dumps(history[-1]), flush=True)
            if v < best:
                best, best_step = v, step
                model.save_weights(str(args.output / "model.safetensors"))
    model.load_weights(str(args.output / "model.safetensors"))
    adapted = {item: evaluate(policy, item) for item in args.evaluate}
    if args.targets:
        for item in args.evaluate:
            adapted[item]["target_audit"] = target_audit(policy, item)
            print(
                json.dumps(
                    {
                        "target_audit": item,
                        **{
                            k: v
                            for k, v in (adapted[item]["target_audit"] or {}).items()
                            if "minus" in k
                        },
                    }
                ),
                flush=True,
            )
    report = {
        "trainable_parameters": trainable,
        "games": {g: len(v) for g, v in pools.items()},
        "selected_step": best_step,
        "history": history,
        "pretrained": reference,
        "adapted": adapted,
    }
    for item in args.evaluate:
        b, a = reference[item], adapted[item]
        print(
            json.dumps(
                {
                    item: {
                        "nll": [round(b["mean_nll"], 4), round(a["mean_nll"], 4)],
                        "greedy_f1": [
                            round(b["greedy"]["macro"]["button_f1"], 4),
                            round(a["greedy"]["macro"]["button_f1"], 4),
                        ],
                        "sampled_onset": [
                            round(b["sampled"][0]["macro"]["onset_f1"], 4),
                            round(a["sampled"][0]["macro"]["onset_f1"], 4),
                        ],
                        "repeat_f1": round(a["repeat_previous"]["macro"]["button_f1"], 4),
                    }
                }
            ),
            flush=True,
        )
    config = {
        **runtime.metadata,
        "control_adapter": {"rank": args.decoder_rank},
        "key_names": list(KEYS_WITH_TAB),
        "deployment_eligible": False,
    }
    if args.policy_rank:
        config["policy_lora"] = {"rank": args.policy_rank}
    if args.targets:
        config["target_encoder"] = {}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
