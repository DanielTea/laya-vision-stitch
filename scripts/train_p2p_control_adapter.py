"""Adapt pretrained action decoding on weak Hordes controls plus public human replay."""

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
from laya_vision_stitch.p2p_adaptation import KEYS_WITH_TAB, encode_action, install_control_adapter
from laya_vision_stitch.p2p_pretrained_policy import physical_action
from laya_vision_stitch.p2p_pretrained_vision import preprocess


def frozen_hash(model):
    trainable = dict(tree_flatten(model.trainable_parameters()))
    digest = hashlib.sha256()
    for key, value in tree_flatten(model.parameters()):
        if key not in trainable:
            digest.update(key.encode())
            digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def evaluate(policy, contexts, labels, rows):
    predictions = []
    for start in range(0, len(rows), 16):
        tokens, _ = policy.decode(mx.array(contexts[start : start + 16]))
        mx.eval(tokens)
        predictions.extend(tokens.tolist())
    grouped = defaultdict(lambda: Counter(tp=0, fp=0, fn=0, exact=0, examples=0))
    for row, pred, truth in zip(rows, predictions, labels, strict=True):
        wanted = set(physical_action(truth, key_names=KEYS_WITH_TAB)["buttons"])
        got = set(physical_action(pred, key_names=KEYS_WITH_TAB)["buttons"])
        m = grouped[row["game"]]
        m.update(
            tp=len(got & wanted),
            fp=len(got - wanted),
            fn=len(wanted - got),
            exact=int(got == wanted),
            examples=1,
        )
    return {
        game: {
            **c,
            "button_f1": 2 * c["tp"] / max(1, 2 * c["tp"] + c["fp"] + c["fn"]),
            "exact_button_set_accuracy": c["exact"] / c["examples"],
        }
        for game, c in grouped.items()
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--hordes-fraction", type=float, default=0.5)
    args = p.parse_args()
    if not 0 < args.hordes_fraction < 1:
        p.error("Hordes fraction must be between zero and one")
    args.output.mkdir(parents=True, exist_ok=False)
    mx.random.seed(20260922)
    rng = np.random.default_rng(20260922)
    runtime = LayaP2PRuntime.load(args.bundle)
    data, excluded = {}, Counter()
    for split in ("train", "validation", "test"):
        rows, contexts, labels = [], [], []
        for line in (args.data / (split + ".jsonl")).read_text().splitlines():
            row = json.loads(line)
            try:
                tokens = encode_action(row["action"])
            except ValueError as exc:
                excluded[split + ": " + str(exc)] += 1
                continue
            with Image.open(row["frames"][-1]["image"]) as image:
                pixels = mx.array(preprocess(image))
            text = runtime.model.bridge(
                runtime.model.goal_features(*runtime.prepare_goal(row["goal"]))
            )
            _, vision = runtime.model.policy.vision(pixels)
            context, _ = runtime.model.policy.context(runtime.model.policy.prefix(vision, text))
            mx.eval(context)
            if not np.isfinite(np.asarray(context)).all():
                raise FloatingPointError("Invalid frozen policy context")
            rows.append(row)
            contexts.append(np.asarray(context)[0])
            labels.append(tokens)
            if len(rows) % 200 == 0:
                print(json.dumps({"split": split, "cached": len(rows)}), flush=True)
        data[split] = rows, np.stack(contexts), np.array(labels, np.int32)
        np.savez(
            args.output / (split + "-features.npz"), context=data[split][1], tokens=data[split][2]
        )
        (args.output / (split + "-rows.jsonl")).write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    # Export retains the whole neural path; features are a training optimization only.
    install_control_adapter(runtime.model, rank=4)
    model = runtime.model
    policy = model.policy
    fingerprint = frozen_hash(model)
    trainable = dict(tree_flatten(model.trainable_parameters()))
    parameters = sum(x.size for x in trainable.values())
    context = mx.array(data["train"][1][:3])
    target = mx.array(data["train"][2][:3])
    sequential = policy.decode(context, forced=target)[1]
    parallel = policy.teacher_logits(context, target)
    parity = max(float(mx.abs(a - b).max()) for a, b in zip(sequential, parallel, strict=True))
    if parity > 0.005:
        raise RuntimeError(f"Parallel teacher decoder differs: {parity}")
    rows, x, y = data["train"]
    contexts, tokens = mx.array(x), mx.array(y)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        # Equal active/idle probability within each equally sampled game.
        groups[(row["game"], bool(row["action"]["buttons"]))].append(i)
    hordes_groups = [v for (game, _), v in groups.items() if game == "Hordes.io"]
    replay_groups = [v for (game, _), v in groups.items() if game != "Hordes.io"]
    if not hordes_groups or not replay_groups:
        raise ValueError("Both Hordes and public replay are required")
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.01)

    def loss_fn(m, context, tokens):
        logits = m.policy.teacher_logits(context, tokens)
        return (
            sum(
                nn.losses.cross_entropy(logit, tokens[:, i], reduction="mean")
                for i, logit in enumerate(logits)
            )
            / 8
        )

    value_grad = nn.value_and_grad(model, loss_fn)
    baseline = evaluate(policy, data["validation"][1], data["validation"][2], data["validation"][0])

    def score(result):
        hordes = result["Hordes.io"]["button_f1"]
        replay = np.mean([r["button_f1"] for g, r in result.items() if g != "Hordes.io"])
        return float((hordes + replay) / 2)

    best_score, best_step = score(baseline), 0
    best_weights = [(k, mx.array(v)) for k, v in tree_flatten(model.trainable_parameters())]
    history = [{"step": 0, "validation": baseline, "selection_score": best_score}]
    print(
        json.dumps(
            {
                "trainable_parameters": parameters,
                "parallel_decoder_max_error": parity,
                "baseline": baseline,
            }
        ),
        flush=True,
    )
    for step in range(1, args.steps + 1):
        groups_for_batch = [
            hordes_groups if rng.random() < args.hordes_fraction else replay_groups
            for _ in range(args.batch_size)
        ]
        indices = mx.array(
            [int(rng.choice(groups[rng.integers(len(groups))])) for groups in groups_for_batch]
        )
        loss, grads = value_grad(model, contexts[indices], tokens[indices])
        optimizer.update(model, grads)
        mx.eval(loss, model.trainable_parameters(), optimizer.state)
        if not np.isfinite(float(loss)):
            raise FloatingPointError("Nonfinite training loss")
        if step % 100 == 0 or step == args.steps:
            result = evaluate(
                policy, data["validation"][1], data["validation"][2], data["validation"][0]
            )
            selection = score(result)
            item = {
                "step": step,
                "loss": float(loss),
                "validation": result,
                "selection_score": selection,
            }
            history.append(item)
            print(json.dumps(item), flush=True)
            if selection > best_score:
                best_score, best_step = selection, step
                best_weights = [
                    (k, mx.array(v)) for k, v in tree_flatten(model.trainable_parameters())
                ]
    model.load_weights(best_weights, strict=False)
    parents_unchanged = frozen_hash(model) == fingerprint
    if not parents_unchanged:
        raise RuntimeError("Frozen pretrained weights changed")
    validation = evaluate(
        policy, data["validation"][1], data["validation"][2], data["validation"][0]
    )
    test = evaluate(policy, data["test"][1], data["test"][2], data["test"][0])
    perm = rng.permutation(len(data["test"][0]))
    shuffled = evaluate(policy, data["test"][1][perm], data["test"][2], data["test"][0])
    runtime.metadata.update(
        control_adapter={"rank": 4},
        key_names=list(KEYS_WITH_TAB),
        selected_control_step=best_step,
        control_training="Weak Hordes plus human P2P; single-frame frozen contexts",
        deployment_eligible=False,
    )
    runtime.save(args.output / "bundle")
    loaded = LayaP2PRuntime.load(args.output / "bundle")
    check = loaded.model.policy.teacher_logits(context, target)
    original = model.policy.teacher_logits(context, target)
    reload_error = max(float(mx.abs(a - b).max()) for a, b in zip(check, original, strict=True))
    if reload_error > 1e-5:
        raise RuntimeError("Control adapter reload differs")
    report = {
        "hordes_sampling_fraction": args.hordes_fraction,
        "source": str(args.bundle),
        "data": str(args.data),
        "excluded": dict(excluded),
        "trainable_parameters": parameters,
        "frozen_parameters_unchanged": parents_unchanged,
        "frozen_sha256": fingerprint,
        "parallel_decoder_max_error": parity,
        "reload_max_error": reload_error,
        "selected_step": best_step,
        "history": history,
        "validation": validation,
        "test": test,
        "shuffled_test": shuffled,
        "scope": "Weak labels, whole-session Hordes splits. No live success claim. Single-frame adaptation; temporal-memory deployment remains unvalidated. Shuffled test mixes games and is diagnostic, not a within-game causal score.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"selected_step": best_step, "test": test, "parents_unchanged": parents_unchanged}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
