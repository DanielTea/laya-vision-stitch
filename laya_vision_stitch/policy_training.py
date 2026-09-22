"""Small-parameter supervised learning and teacher distillation, native MLX."""

import hashlib
import json
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten


class ExampleSubset:
    def __init__(self, examples, indices):
        self.examples, self.indices = examples, list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.examples[self.indices[index]]


def loss_terms(output, row, config):
    terms = {}
    if "answer" in row:
        target = list(row["choices"]).index(row["answer"])
        terms["answer"] = nn.losses.cross_entropy(
            output["choices"], mx.array([target]), reduction="mean"
        )
    if "teacher_probs" in row:
        temperature = row.get("teacher_temperature", 1.0)
        p = mx.array([[row["teacher_probs"][k] for k in row["choices"]]])
        student = output["choices"] / temperature
        log_p = student - mx.logsumexp(student, axis=-1, keepdims=True)
        terms["distillation"] = (p * (mx.log(mx.maximum(p, 1e-8)) - log_p)).sum() * temperature**2
    if "action" in row:
        action = row["action"]
        target = mx.array([[float(b in action["buttons"]) for b in config.buttons]])
        if "_button_positive_weights" in row:
            weights = mx.array([row["_button_positive_weights"]])
            logits = output["buttons"]
            terms["buttons"] = mx.mean(
                target * weights * mx.logaddexp(0, -logits) + (1 - target) * mx.logaddexp(0, logits)
            )
        else:
            terms["buttons"] = nn.losses.binary_cross_entropy(
                output["buttons"], target, with_logits=True, reduction="mean"
            )
        terms["mouse"] = mx.mean((output["mouse"] - mx.array([action["mouse_delta"]])) ** 2)
        duration = list(config.durations).index(action["duration_seconds"])
        terms["duration"] = nn.losses.cross_entropy(
            output["duration"], mx.array([duration]), reduction="mean"
        )
        active = action.get("pointer_xy") is not None
        terms["pointer_active"] = nn.losses.binary_cross_entropy(
            output["pointer_active"], mx.array([float(active)]), with_logits=True, reduction="mean"
        )
        if active:
            terms["pointer"] = mx.mean((output["pointer"] - mx.array([action["pointer_xy"]])) ** 2)
    return terms


def fingerprint(module, frozen_only=False):
    """Exact content digest of pretrained weights, not just a frozen flag."""
    digest = hashlib.sha256()
    excluded = {k for k, _ in tree_flatten(module.trainable_parameters())} if frozen_only else set()
    for name, value in tree_flatten(module.parameters()):
        if name in excluded:
            continue
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(np.asarray(value.view(mx.uint8)).tobytes())
    return digest.hexdigest()


def supervised_terms(model, output, row):
    terms = loss_terms(output, row, model.policy_config)
    if "_alignment_ids" in row:
        target = model.laya.encoder.embeddings.tok_embeddings(
            mx.array([row["_alignment_ids"]])
        ).astype(mx.float32)
        state = output["visual_state"][:, : target.shape[1]]
        terms["alignment"] = (
            10 * mx.mean((state - target) ** 2) / mx.maximum(mx.mean(target**2), 1e-6)
        )
    return terms


def cache_examples(runtime, rows, directory=None):
    if directory is not None:
        from .feature_cache import CachedExamples

        return CachedExamples(runtime, rows, directory)
    # Frozen visual features are identical across goals; reuse only within this run.
    features, examples = {}, []
    for index, row in enumerate(rows):
        if row.get("description"):
            row = dict(row)
            ids = runtime.agent.tok(row["description"], add_special_tokens=False)["input_ids"]
            if len(ids) > runtime.module.policy_config.visual_slots:
                raise ValueError("Description exceeds visual slots; shorten it or increase slots")
            row["_alignment_ids"] = ids
        key = tuple((f["sha256"], f["age_seconds"]) for f in row["frames"])
        if key not in features:
            features[key] = runtime.features(row)
        examples.append((row, (*features[key], *runtime.prepare(row))))
        if (index + 1) % 20 == 0:
            print(f"Prepared {index + 1}/{len(rows)} examples", flush=True)
    return examples


def evaluate(runtime, examples, zero_visual=False):
    correct, labelled, button_matches, actions, mse = 0, 0, 0, 0, []
    losses, predictions = [], []
    for row, inputs in examples:
        if zero_visual:
            inputs = (mx.zeros_like(inputs[0]), *inputs[1:])
        output = runtime.module.from_features(*inputs)
        terms = supervised_terms(runtime.module, output, row)
        total = sum(terms.values(), mx.array(0.0))
        mx.eval(output, total)
        losses.append(float(total.item()))
        item = {"id": row["id"], "loss": losses[-1]}
        if row.get("choices"):
            choice = list(row["choices"])[int(mx.argmax(output["choices"][0]).item())]
            item["choice"] = choice
            if "answer" in row:
                correct += choice == row["answer"]
                labelled += 1
        if "action" in row:
            p = np.asarray(output["buttons"][0]) >= 0
            expected = np.array(
                [b in row["action"]["buttons"] for b in runtime.module.policy_config.buttons]
            )
            button_matches += bool(np.array_equal(p, expected))
            actions += 1
            mse.append(
                float(np.mean((np.asarray(output["mouse"][0]) - row["action"]["mouse_delta"]) ** 2))
            )
        predictions.append(item)
    by_scene = {}
    for (row, _), prediction in zip(examples, predictions, strict=True):
        if "answer" in row:
            key = (
                tuple((f.get("sha256", f["image"]), f["age_seconds"]) for f in row["frames"]),
                tuple(row["choices"].items()),
                row.get("controls", ""),
                json.dumps(row.get("previous_actions", []), sort_keys=True),
            )
            by_scene.setdefault(key, []).append((row, prediction))
    pairs = [
        items
        for items in by_scene.values()
        if len({r["goal"] for r, _ in items}) > 1 and len({r["answer"] for r, _ in items}) > 1
    ]
    return {
        "loss": float(np.mean(losses)),
        "choice_accuracy": correct / labelled if labelled else None,
        "labelled_choices": labelled,
        "button_exact_match": button_matches / actions if actions else None,
        "mouse_mse": float(np.mean(mse)) if mse else None,
        "examples": len(examples),
        "zero_visual": zero_visual,
        "predictions": predictions,
        "goal_counterfactual_groups": len(pairs),
        "goal_counterfactual_accuracy": (
            sum(all(p["choice"] == r["answer"] for r, p in items) for items in pairs) / len(pairs)
            if pairs
            else None
        ),
    }


def train(
    runtime,
    examples,
    steps=50,
    learning_rate=1e-4,
    seed=17,
    log=None,
    shuffle_options=False,
    button_positive_weights=None,
):
    if steps < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Positive steps and finite learning rate required")
    parameters = tree_flatten(runtime.module.trainable_parameters())

    def allowed(name):
        return name.startswith(("connector.", "actions.")) or (
            name.startswith("laya.encoder.layers.") and name.endswith((".lora_a", ".lora_b"))
        )

    if not parameters or any(not allowed(name) for name, _ in parameters):
        raise RuntimeError("Only connector, action heads and optional LoRA may be trained")
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0)
    rng = np.random.default_rng(seed)

    def loss(model, inputs, row):
        output = model.from_features(*inputs)
        return sum(supervised_terms(model, output, row).values(), mx.array(0.0))

    value_and_grad = nn.value_and_grad(runtime.module, loss)
    history, order = [], []
    for step in range(steps):
        if not order:
            order = rng.permutation(len(examples)).tolist()
        row, inputs = examples[order.pop()]
        if button_positive_weights is not None and "action" in row:
            row = dict(row, _button_positive_weights=button_positive_weights)
        if shuffle_options and row.get("choices"):
            labels = list(row["choices"])
            labels = [labels[i] for i in rng.permutation(len(labels))]
            row = dict(row, choices={k: row["choices"][k] for k in labels})
            inputs = (*inputs[:2], *runtime.prepare(row))
        started = time.perf_counter()
        value, grads = value_and_grad(runtime.module, inputs, row)
        grads, norm = optim.clip_grad_norm(grads, 1.0)
        mx.eval(value, norm)
        if not np.isfinite(value.item()) or not np.isfinite(norm.item()):
            raise FloatingPointError("Nonfinite loss/gradient; no update applied")
        optimizer.update(runtime.module, grads)
        mx.eval(runtime.module.trainable_parameters(), optimizer.state)
        record = {
            "step": step + 1,
            "id": row["id"],
            "loss": float(value.item()),
            "gradient_norm": float(norm.item()),
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        if log:
            log(record)
        runtime.metadata["training_steps"] += 1
        runtime.metadata["action_training_examples"] += int("action" in row)
    return history
