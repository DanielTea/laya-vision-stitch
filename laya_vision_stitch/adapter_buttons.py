"""Single-objective balanced button training through connector and small LoRA."""

import json
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from .decoder_probe import button_loss, groups


def matched_quartets(rows):
    """Index complete counterfactual pairs for same-wording balanced minibatches.

    Training metadata only. Each batch uses two scenes (one in each reviewed
    state), each shown with both opposing instructions in the same wording.
    """
    variants = {}
    for i, row in enumerate(rows):
        opened, goal = row["_sampling_stratum"]
        variant = row["_pair_variant"]
        scene = tuple((f["sha256"], f["age_seconds"]) for f in row["frames"])
        pair = variants.setdefault(variant, {}).setdefault(opened, {}).setdefault(scene, {})
        if goal in pair:
            raise ValueError("Duplicate scene/goal in matched sampling")
        pair[goal] = i
    result = []
    for states in variants.values():
        if set(states) != {0, 1}:
            raise ValueError("Matched sampling needs both visible states per wording")
        item = []
        for state in (0, 1):
            pairs = []
            for goals in states[state].values():
                if set(goals) != {0, 1}:
                    raise ValueError("Matched sampling needs both goals per scene")
                pairs.append([goals[0], goals[1]])
            item.append(pairs)
        result.append(item)
    if not result:
        raise ValueError("No matched training groups")
    return result


def train_adapter_buttons(
    runtime,
    examples,
    steps,
    learning_rate,
    seed,
    log,
    matched=False,
    batch_size=None,
    positive_weights=None,
):
    parameters = tree_flatten(runtime.module.trainable_parameters())

    def allowed(name):
        return name.startswith(("actions.", "connector.")) or (
            name.startswith("laya.encoder.layers.") and name.endswith((".lora_a", ".lora_b"))
        )

    if not parameters or any(not allowed(k) for k, _ in parameters):
        raise ValueError("Only decoder, connector and LoRA may train")
    rows = [r for r, _ in examples]
    truth = np.array(
        [
            [b in row["action"]["buttons"] for b in runtime.module.policy_config.buttons]
            for row in rows
        ]
    )
    varying = truth.any(0) & ~truth.all(0)
    strata = groups(rows)
    quartets = matched_quartets(rows) if matched else None
    if batch_size is not None and (batch_size < 1 or matched):
        raise ValueError("Explicit batch size must be positive and use unmatched sampling")
    rng = np.random.default_rng(seed)
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0)

    def loss(model, items):
        return sum(
            button_loss(model.from_features(*inputs)["buttons"], target, varying, positive_weights)
            for inputs, target in items
        ) / len(items)

    update = nn.value_and_grad(runtime.module, loss)
    for step in range(steps):
        started = time.perf_counter()
        if quartets is None:
            selected = (
                strata
                if batch_size is None
                else [strata[int(i)] for i in rng.integers(len(strata), size=batch_size)]
            )
            indices = [int(rng.choice(group)) for group in selected]
        else:
            states = quartets[int(rng.integers(len(quartets)))]
            indices = [i for pairs in states for i in pairs[int(rng.integers(len(pairs)))]]
        items = [(examples[i][1], mx.array(truth[i : i + 1], mx.float32)) for i in indices]
        value, gradient = update(runtime.module, items)
        gradient, norm = optim.clip_grad_norm(gradient, 1.0)
        mx.eval(value, norm)
        if not np.isfinite(float(value)) or not np.isfinite(float(norm)):
            raise FloatingPointError("Nonfinite adapter gradient/loss")
        optimizer.update(runtime.module, gradient)
        mx.eval(runtime.module.trainable_parameters(), optimizer.state)
        row = {
            "step": step + 1,
            "loss": float(value),
            "gradient_norm": float(norm),
            "seconds": time.perf_counter() - started,
        }
        log.write(json.dumps(row) + "\n")
        if (step + 1) % 100 == 0 or step == 0:
            log.flush()
            print(json.dumps(row), flush=True)
    runtime.metadata["training_steps"] += steps
    runtime.metadata["action_training_examples"] += steps * (
        4 if matched else batch_size or len(strata)
    )
