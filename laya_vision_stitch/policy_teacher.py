"""Offline full-Qwen teacher: distributions over named choices, no live inputs."""

import json
import re
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from .backends import QWEN_ID, QWEN_REVISION, QwenVision
from .trainable_model import context_text


def parse_final(text, labels):
    match = re.search(r"(?:^|\n)FINAL:\s*([A-Z])\s*$", text.strip())
    if not match or match[1] not in labels:
        raise ValueError("Teacher response must end with FINAL: <valid option letter>")
    return match[1]


def label_examples(rows, output, temperature=2.0, width=320, mode="generate", max_tokens=256):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Teacher temperature must be positive and finite")
    if mode not in ("generate", "logits") or max_tokens < 1:
        raise ValueError("Invalid teacher mode/token limit")
    if any(not row.get("choices") or len(row["choices"]) > 26 for row in rows):
        raise ValueError("Teacher requires 2..26 named choices per record")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    qwen = QwenVision(width)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        for index, original in enumerate(rows):
            row = dict(original)
            labels = [chr(65 + i) for i in range(len(row["choices"]))]
            token_ids = [
                qwen.processor.tokenizer.encode(s, add_special_tokens=False) for s in labels
            ]
            if any(len(ids) != 1 for ids in token_ids):
                raise ValueError("Teacher labels must each encode as one token")
            options = "\n".join(
                f"{label}: {description}"
                for label, description in zip(labels, row["choices"].values(), strict=True)
            )
            ages = ", ".join(str(f["age_seconds"]) for f in row["frames"])
            prompt = (
                f"{context_text(row)}\nImages oldest to newest; ages in seconds: {ages}.\n"
                f"Choose the best answer using the images.\n{options}\n"
                + (
                    "Answer with exactly one capital letter."
                    if mode == "logits"
                    else "Briefly describe the relevant visible evidence, then end with a separate line FINAL: <option letter>."
                )
            )
            images = []
            for frame in row["frames"]:
                with Image.open(frame["image"]) as source:
                    image = source.convert("RGB")
                if image.width > width:
                    image = image.resize((width, max(1, round(image.height * width / image.width))))
                images.append(image)
            formatted = apply_chat_template(
                qwen.processor,
                qwen.model.config,
                prompt,
                num_images=len(images),
                enable_thinking=False,
            )
            response = None
            if mode == "logits":
                data = prepare_inputs(
                    qwen.processor,
                    images=images,
                    prompts=formatted,
                    image_token_index=qwen.model.config.image_token_index,
                )
                output_tokens = qwen.model(
                    data.pop("input_ids"), mask=data.pop("attention_mask", None), **data
                )
                logits = output_tokens.logits[0, -1, mx.array([ids[0] for ids in token_ids])]
                probabilities = mx.softmax(logits.astype(mx.float32) / temperature)
                mx.eval(probabilities)
            else:
                response = generate(
                    qwen.model,
                    qwen.processor,
                    formatted,
                    image=images,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    verbose=False,
                ).text
                try:
                    answer = parse_final(response, labels)
                except ValueError:
                    output.with_suffix(".error.json").write_text(
                        json.dumps({"id": row["id"], "response": response}, indent=2)
                    )
                    raise
                probabilities = mx.array([float(label == answer) for label in labels])
            row["teacher_probs"] = dict(
                zip(row["choices"], np.asarray(probabilities).tolist(), strict=True)
            )
            row["teacher_temperature"] = temperature if mode == "logits" else 1.0
            row["teacher"] = {
                "model": QWEN_ID,
                "revision": QWEN_REVISION,
                "method": "restricted first-token choice logits"
                if mode == "logits"
                else "generated evidence and hard choice target",
                "response": response,
                "prompt": prompt,
                "reviewed": False,
                "image_width": width,
            }
            # Keep human/synthetic ground truth untouched; teacher disagreement remains visible.
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            print(f"Teacher labelled {index + 1}/{len(rows)}", flush=True)
    return output
