"""Offline SigLIP dense features or Qwen single-image grounding proposals."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.prepare_spatial_experiments import QUESTION, read

MODEL = "google/siglip2-base-patch16-224"
REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"


def features(data):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, SiglipVisionModel

    torch.set_num_threads(4)
    path = snapshot_download(
        MODEL,
        revision=REVISION,
        local_files_only=True,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
            "README.md",
        ],
    )
    processor = AutoImageProcessor.from_pretrained(path)
    model = SiglipVisionModel.from_pretrained(path).eval().to("mps")
    cache = {}
    for split in ["train", "validation", "test"]:
        rows = read(data / f"{split}.jsonl")
        unique = {r["frames"][0]["sha256"]: r["frames"][0]["image"] for r in rows}
        missing = [(k, v) for k, v in unique.items() if k not in cache]
        for start in range(0, len(missing), 16):
            batch = missing[start : start + 16]
            images = [Image.open(p).convert("RGB") for _, p in batch]
            inputs = processor(images=images, return_tensors="pt").to("mps")
            with torch.inference_mode():
                output = model(**inputs).last_hidden_state
                if output.shape[1:] != (196, 768):
                    raise ValueError("Unexpected teacher patch grid")
                patches = output.reshape(-1, 14, 14, 768).permute(0, 3, 1, 2)
                patches = torch.nn.functional.interpolate(
                    patches, size=(12, 12), mode="bilinear", align_corners=False
                )
                result = patches.permute(0, 2, 3, 1).cpu().numpy().astype(np.float16)
            for (key, _), value in zip(batch, result, strict=True):
                cache[key] = value
            if start % 160 == 0:
                print(
                    json.dumps({"teacher": MODEL, "split": split, "encoded": start + len(batch)}),
                    flush=True,
                )
        np.save(
            data / f"{split}-siglip.npy", np.stack([cache[r["frames"][0]["sha256"]] for r in rows])
        )
    (data / "siglip-teacher.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "revision": REVISION,
                "license": "Apache-2.0",
                "patches": "224px input; 14x14 teacher grid bilinearly aligned to 12x12 student grid",
                "live_teacher": False,
            },
            indent=2,
        )
        + "\n"
    )


def labels(data):
    import mlx.core as mx

    from laya_vision_stitch.backends import QwenVision

    qwen = QwenVision(width=640)
    scores = []
    letters = [qwen.processor.tokenizer.encode(x, add_special_tokens=False) for x in ["A", "B"]]
    if any(len(x) != 1 for x in letters):
        raise ValueError("Expected single-token labels")
    with (data / "qwen-grounding.jsonl").open("x") as log:
        for split in ["train", "validation", "test"]:
            rows = [r for r in read(data / f"{split}.jsonl") if r["kind"] == "grounding"]
            for row in rows:
                probabilities = []
                for reverse in [False, True]:
                    options = "A: no. B: yes." if reverse else "A: yes. B: no."
                    inputs = qwen.inputs(
                        row["frames"][0]["image"],
                        QUESTION + " " + options + " Answer with exactly one letter.",
                    )
                    out = qwen.model(
                        inputs.pop("input_ids"), mask=inputs.pop("attention_mask", None), **inputs
                    )
                    selected = out.logits[0, -1, mx.array([t[0] for t in letters])].astype(
                        mx.float32
                    )
                    probs = np.asarray(mx.softmax(selected))
                    probabilities.append(float(probs[1 if reverse else 0]))
                pred = [int(p >= 0.5) for p in probabilities]
                accepted = pred[0] == pred[1] == row["ground_label"]
                record = {
                    "id": row["id"],
                    "split": split,
                    "image_sha256": row["frames"][0]["sha256"],
                    "reviewed_label": row["ground_label"],
                    "yes_probabilities_by_order": probabilities,
                    "order_consistent": pred[0] == pred[1],
                    "accepted_after_review": accepted,
                    "teacher": qwen.metadata,
                    "single_image": True,
                    "probability_scope": "Two-option normalized scores, not calibrated confidence",
                }
                scores.append(record)
                log.write(json.dumps(record) + "\n")
                log.flush()
                if len(scores) % 16 == 0:
                    print(
                        json.dumps(
                            {
                                "qwen_checked": len(scores),
                                "accepted": sum(r["accepted_after_review"] for r in scores),
                            }
                        ),
                        flush=True,
                    )
    report = {
        split: {
            "examples": sum(r["split"] == split for r in scores),
            "accepted": sum(r["split"] == split and r["accepted_after_review"] for r in scores),
        }
        for split in ["train", "validation", "test"]
    }
    (data / "qwen-grounding-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def generated_labels(data, retry=False):
    import re

    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    from laya_vision_stitch.backends import QwenVision

    qwen = QwenVision(width=640)
    rows = [r for r in read(data / "train.jsonl") if r["kind"] == "grounding"]
    records = []
    previous = {r["id"]: r for r in read(data / "qwen-grounding-generated.jsonl")} if retry else {}
    name = "qwen-grounding-reviewed" if retry else "qwen-grounding-generated"
    with (data / (name + ".jsonl")).open("x") as log:
        for row in rows:
            if retry and previous[row["id"]]["accepted_after_review"]:
                records.append(previous[row["id"]])
                log.write(json.dumps(previous[row["id"]]) + "\n")
                log.flush()
                continue
            image = Image.open(row["frames"][0]["image"]).convert("RGB")
            if image.width > 640:
                image = image.resize((640, round(image.height * 640 / image.width)))
            predictions, responses = [], []
            for reverse in [False, True]:
                options = "A: no. B: yes." if reverse else "A: yes. B: no."
                prompt = (
                    QUESTION
                    + " "
                    + options
                    + " Briefly describe visible evidence, then end with FINAL: A or FINAL: B."
                )
                if retry:
                    prompt = (
                        QUESTION
                        + " "
                        + options
                        + " Start with FINAL: A or FINAL: B. Then give at most 15 words of visible evidence."
                    )
                formatted = apply_chat_template(
                    qwen.processor, qwen.model.config, prompt, num_images=1, enable_thinking=False
                )
                response = generate(
                    qwen.model,
                    qwen.processor,
                    formatted,
                    image=[image],
                    max_tokens=64 if retry else 96,
                    temperature=0,
                    verbose=False,
                ).text
                found = re.findall(r"FINAL:\s*([AB])\b", response)
                prediction = int(found[-1] == ("B" if reverse else "A")) if found else None
                predictions.append(prediction)
                responses.append(response)
            accepted = predictions[0] == predictions[1] == row["ground_label"]
            record = {
                "id": row["id"],
                "split": "train",
                "image_sha256": row["frames"][0]["sha256"],
                "reviewed_label": row["ground_label"],
                "predictions_by_order": predictions,
                "responses": responses,
                "accepted_after_review": accepted,
                "single_image": True,
                "teacher": qwen.metadata,
                "method": "Final-first retry"
                if retry
                else "Generated evidence and parsed final answer, both option orders",
            }
            records.append(record)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if len(records) % 8 == 0:
                print(
                    json.dumps(
                        {
                            "generated_checked": len(records),
                            "accepted": sum(r["accepted_after_review"] for r in records),
                        }
                    ),
                    flush=True,
                )
    report = {
        "examples": len(records),
        "accepted": sum(r["accepted_after_review"] for r in records),
        "accepted_by_label": {
            str(label): sum(
                r["accepted_after_review"] and r["reviewed_label"] == label for r in records
            )
            for label in [0, 1]
        },
        "scope": "Training screenshots only. Every accepted label agrees with prior hash-verified visual review in both option orders. No sequences or intent labels.",
    }
    (data / (name + "-report.json")).write_text(json.dumps(report, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["features", "labels", "labels-generated", "labels-reviewed"])
    p.add_argument("--data", type=Path, required=True)
    a = p.parse_args()
    {
        "features": features,
        "labels": labels,
        "labels-generated": generated_labels,
        "labels-reviewed": lambda d: generated_labels(d, retry=True),
    }[a.mode](a.data)


if __name__ == "__main__":
    main()
