"""Gated offline Qwen feature/decision distillation into the existing stitch.

The teacher runs only during target export. A training-only projection aligns
Laya encoder features with Qwen language features; exported policies omit it.
This pilot targets checked menu reasoning, not general reasoning or gameplay.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from .decoder_probe import button_loss
from .policy_training import cache_examples, fingerprint
from .teacher_audit import (
    load_cases,
    qualify,
    read_rows,
    student,
    summarize,
    teacher_prompt,
    write_json,
)
from .trainable_model import TrainableRuntime


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualified_records(data, thinking):
    """Recompute eligibility from complete predictions; never trust a report flag."""
    name = "teacher-thinking" if thinking else "teacher"
    validation = load_cases(data, "validation")
    teacher_path = data / f"{name}-validation.jsonl"
    student_path = data / "student-validation.jsonl"
    t = summarize(validation, read_rows(teacher_path))
    s = summarize(validation, read_rows(student_path))
    gate = qualify(t, s)
    if not gate["menu_distillation_eligible"]:
        raise ValueError(f"Teacher qualification failed: {gate['checks']}")
    training = load_cases(data, "train")
    prediction_path = data / f"{name}-train.jsonl"
    predictions = read_rows(prediction_path)
    summarize(training, predictions)  # Enforce exact ID coverage before filtering.
    by_id = {r["id"]: r for r in predictions}
    scenes = {}
    for row in training:
        scenes.setdefault(row["source_id"], []).append(row)
    # Keep full paired scenes only. Incorrect teacher answers remain in evidence,
    # never relabelled or silently substituted with ground truth.
    accepted = [
        r
        for group in scenes.values()
        if all(by_id[r["id"]]["prediction"] == r["answer"] for r in group)
        for r in group
    ]
    if len({r["source_id"] for r in accepted}) < 8:
        raise ValueError("Need at least eight completely correct training scenes")
    if {r["opened"] for r in accepted if r["task"] == "instruction"} != {False, True}:
        raise ValueError("Both reviewed visual states must remain after filtering")
    return (
        accepted,
        by_id,
        {
            "qualification": gate,
            "validation_teacher_sha256": file_digest(teacher_path),
            "validation_student_sha256": file_digest(student_path),
            "training_teacher_sha256": file_digest(prediction_path),
            "accepted_ids": [r["id"] for r in accepted],
            "rejected_ids": [r["id"] for r in training if r not in accepted],
        },
    )


def export_targets(data, output, thinking):
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs
    from PIL import Image

    from .backends import QwenVision

    rows, records, gate = qualified_records(data, thinking)
    if output.exists():
        raise FileExistsError(output)
    qwen = QwenVision(640)
    output.mkdir(parents=True, exist_ok=False)
    targets = []
    for i, row in enumerate(rows):
        record = records[row["id"]]
        prompt, letters = teacher_prompt(row)
        if record["prompt"] != prompt:
            raise ValueError("Teacher prompt differs from the prepared case")
        images = []
        for frame in row["frames"]:
            with Image.open(frame["image"]) as source:
                image = source.convert("RGB")
            if image.width > 640:
                image = image.resize((640, round(image.height * 640 / image.width)))
            images.append(image)
        formatted = apply_chat_template(
            qwen.processor,
            qwen.model.config,
            prompt,
            num_images=len(images),
            enable_thinking=thinking,
        )
        # Condition the teacher on its own generated reasoning/evidence, excluding
        # the final letter. None of this text is supplied to the student input.
        evidence = record["response"].rsplit("FINAL:", 1)[0] + "FINAL:"
        inputs = prepare_inputs(
            qwen.processor,
            images=images,
            prompts=formatted + evidence,
            image_token_index=qwen.model.config.image_token_index,
        )
        out = qwen.model(
            inputs.pop("input_ids"),
            mask=inputs.pop("attention_mask", None),
            return_hidden=True,
            **inputs,
        )
        ids = [qwen.processor.tokenizer.encode(" " + s, add_special_tokens=False) for s in letters]
        if any(len(ids_) != 1 for ids_ in ids):
            raise ValueError("Teacher labels must each have one token")
        probs = mx.softmax(out.logits[0, -1, mx.array([v[0] for v in ids])].astype(mx.float32))
        feature = out.hidden_states[-1][0, -1].astype(mx.float32)
        mx.eval(probs, feature)
        if list(row["choices"])[int(mx.argmax(probs))] != record["prediction"]:
            raise ValueError(
                "Teacher feature-prefill answer disagrees with its reviewed generation"
            )
        feature_path = output / f"{i:04d}.npy"
        np.save(feature_path, np.asarray(feature))
        targets.append(
            {
                "id": row["id"],
                "feature": feature_path.name,
                "feature_sha256": file_digest(feature_path),
                "probabilities": dict(zip(row["choices"], np.asarray(probs).tolist(), strict=True)),
            }
        )
        print(f"Exported teacher features {i + 1}/{len(rows)}", flush=True)
    (output / "targets.jsonl").write_text("".join(json.dumps(r) + "\n" for r in targets))
    write_json(
        output / "protocol.json",
        {
            **gate,
            "data": str(data.resolve()),
            "thinking": thinking,
            "targets_sha256": file_digest(output / "targets.jsonl"),
            "target": "Full Qwen final language feature after its generated evidence, before answer letter",
            "teacher_used_at_inference": False,
        },
    )


def kl_loss(logits, probabilities):
    log_p = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    p = mx.array([probabilities], mx.float32)
    return mx.sum(p * (mx.log(mx.maximum(p, 1e-8)) - log_p))


def cosine_loss(prediction, target):
    def normalize(value):
        return value / mx.maximum(mx.linalg.norm(value, axis=-1, keepdims=True), 1e-6)

    return mx.mean(1 - mx.sum(normalize(prediction) * normalize(target), axis=-1))


class TransferModel(nn.Module):
    def __init__(self, policy, teacher_width):
        super().__init__()
        self.policy = policy
        self.projection = nn.Linear(policy.laya.encoder.config.hidden_size, teacher_width)

    def __call__(self, inputs):
        patches, coordinates, batch, goal_ids, start = inputs
        model = self.policy
        goal = model.laya.encoder.embeddings.tok_embeddings(goal_ids)
        state = model.connector(patches, coordinates, goal)
        h, choices = model.encode_state(state, batch, start)
        if model.policy_config.action_context_source != "encoder":
            raise ValueError("Feature transfer requires encoder action context")
        return {
            "choices": choices,
            "feature": self.projection(h.mean(axis=1).astype(mx.float32)),
            **model.actions(h[:, 0], h),
        }


def transfer_loss(output, row, target, buttons, control=False):
    """Matched supervised control omits teacher probabilities and feature matching."""
    if row["task"] == "instruction":
        varying = np.array([b == "w" for b in buttons])
        hard = mx.array([[float(b == "w" and row["answer"] == "hold") for b in buttons]])
        loss = button_loss(output["buttons"], hard, varying)
        if not control:
            soft = mx.array([[target["probabilities"]["hold"] if b == "w" else 0 for b in buttons]])
            loss = 0.5 * loss + 0.5 * button_loss(output["buttons"], soft, varying)
    else:
        index = list(row["choices"]).index(row["answer"])
        loss = nn.losses.cross_entropy(output["choices"], mx.array([index]), reduction="mean")
        if not control:
            loss = 0.5 * loss + 0.5 * kl_loss(
                output["choices"], [target["probabilities"][k] for k in row["choices"]]
            )
    if not control:
        loss = loss + 0.1 * cosine_loss(output["feature"], target["feature_array"][None])
    return loss


def train_transfer(args):
    protocol = json.loads((args.targets / "protocol.json").read_text())
    data = Path(protocol["data"])
    rows, _, gate = qualified_records(data, protocol["thinking"])
    for key in (
        "validation_teacher_sha256",
        "validation_student_sha256",
        "training_teacher_sha256",
        "accepted_ids",
    ):
        if gate[key] != protocol[key]:
            raise ValueError("Qualification evidence changed since target export")
    if file_digest(args.targets / "targets.jsonl") != protocol["targets_sha256"]:
        raise ValueError("Teacher targets changed")
    targets = read_rows(args.targets / "targets.jsonl")
    if [r["id"] for r in targets] != [r["id"] for r in rows]:
        raise ValueError("Teacher feature IDs must match training cases in order")
    for target, row in zip(targets, rows, strict=True):
        path = args.targets / target["feature"]
        if file_digest(path) != target["feature_sha256"]:
            raise ValueError("Teacher feature changed")
        feature = np.load(path, allow_pickle=False)
        p = target["probabilities"]
        if (
            feature.ndim != 1
            or not np.isfinite(feature).all()
            or set(p) != set(row["choices"])
            or not np.isfinite(list(p.values())).all()
            or any(not 0 <= v <= 1 for v in p.values())
            or not np.isclose(sum(p.values()), 1)
        ):
            raise ValueError("Invalid teacher feature/distribution")
        target["feature_array"] = mx.array(feature)
    args.output.mkdir(parents=True, exist_ok=False)
    mx.random.seed(args.seed)
    runtime = TrainableRuntime.load(args.bundle)
    for name in ("mouse", "chunk_mouse", "pointer", "pointer_active", "duration"):
        if hasattr(runtime.module.actions, name):
            getattr(runtime.module.actions, name).freeze()
    baseline = {
        k: fingerprint(getattr(runtime.module, k), frozen_only=True) for k in ("vision", "laya")
    }
    wrapper = TransferModel(runtime.module, len(targets[0]["feature_array"]))
    if args.control:
        wrapper.projection.freeze()
    names = [k for k, _ in tree_flatten(wrapper.trainable_parameters())]
    if any(
        not (
            k.startswith(("projection.", "policy.connector.", "policy.actions."))
            or k.startswith("policy.laya.encoder.layers.")
            and k.endswith((".lora_a", ".lora_b"))
        )
        for k in names
    ):
        raise ValueError("Unexpected trainable backbone weights")
    examples = list(cache_examples(runtime, rows, args.cache))
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0)
    rng = np.random.default_rng(args.seed)

    # Cycle all cases: balanced scenes, both goals and option orders, one fixed
    # update budget shared by distillation and the supervised control.
    def loss(model, index):
        row, inputs = examples[index]
        return transfer_loss(
            model(inputs),
            row,
            targets[index],
            runtime.module.policy_config.buttons,
            control=args.control,
        )

    update = nn.value_and_grad(wrapper, loss)
    order = []
    with (args.output / "steps.jsonl").open("x") as log:
        for step in range(args.steps):
            if not order:
                order = rng.permutation(len(rows)).tolist()
            value, gradient = update(wrapper, order.pop())
            gradient, norm = optim.clip_grad_norm(gradient, 1)
            mx.eval(value, norm)
            if not np.isfinite(float(value)) or not np.isfinite(float(norm)):
                raise FloatingPointError("Non-finite transfer gradient/loss")
            optimizer.update(wrapper, gradient)
            mx.eval(wrapper.trainable_parameters(), optimizer.state)
            record = {"step": step + 1, "loss": float(value), "gradient_norm": float(norm)}
            log.write(json.dumps(record) + "\n")
            if step % 50 == 0:
                print(json.dumps(record), flush=True)
    if baseline != {k: fingerprint(getattr(runtime.module, k), frozen_only=True) for k in baseline}:
        raise RuntimeError("Pretrained backbone weights changed")
    runtime.metadata["training_steps"] += args.steps
    runtime.metadata["reasoning_transfer"] = {
        "scope": "menu diagnostic only",
        "steps": args.steps,
        "control": args.control,
        "teacher_projection_deployed": False,
        "source": str(args.bundle),
        "targets": str(args.targets),
    }
    runtime.save(args.output / "bundle")
    validation = load_cases(data, "validation")
    loaded = TrainableRuntime.load(args.output / "bundle")
    original = runtime.predict(validation[0])
    reloaded = loaded.predict(validation[0])
    np.testing.assert_allclose(
        list(original["button_probabilities"].values()),
        list(reloaded["button_probabilities"].values()),
        atol=1e-6,
    )
    del loaded
    student(validation, args.output / "bundle", args.output / "validation.jsonl")
    report = {
        "validation": summarize(validation, read_rows(args.output / "validation.jsonl")),
        "protocol": runtime.metadata["reasoning_transfer"],
        "frozen_backbones_unchanged": True,
        "reload_matches": True,
        "live_inputs_sent": 0,
    }
    write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("export", "train"))
    p.add_argument("--data", type=Path, default=Path("artifacts/teacher-audit-001"))
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--targets", type=Path, default=Path("artifacts/reasoning-targets-001"))
    p.add_argument("--bundle", type=Path, default=Path("artifacts/robust-decoder-007/bundle"))
    p.add_argument("--cache", type=Path, default=Path("artifacts/vision-feature-cache"))
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--learning-rate", type=float, default=0.00003)
    p.add_argument("--seed", type=int, default=53)
    p.add_argument("--control", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.steps < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error("Steps and learning rate must be positive")
    if args.mode == "export":
        export_targets(args.data, args.targets, args.thinking)
    else:
        if args.output is None:
            p.error("--output is required for training")
        train_transfer(args)


if __name__ == "__main__":
    main()
