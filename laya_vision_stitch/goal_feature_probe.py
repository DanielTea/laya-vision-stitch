"""Probe frozen Qwen instruction features before considering a new neural bridge."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

from .backends import QwenVision
from .goal_curriculum import DEVELOPMENT, INVERSE_TRAIN, LOGIC_TRAIN, SEALED, TRAIN


def run(output):
    if output.exists():
        raise FileExistsError(output)
    qwen = QwenVision()
    records, features = {}, {}
    for split, templates in (
        ("train", TRAIN + INVERSE_TRAIN + LOGIC_TRAIN),
        ("development", DEVELOPMENT + SEALED),
    ):
        records[split], values = [], []
        for i, template in enumerate(templates):
            for when_open in (False, True):
                state, other = ("open", "closed") if when_open else ("closed", "open")
                goal = template.format(state=state, opposite=other)
                # Generic instruction encoding; no queried menu state or label
                # is provided to Qwen. No text is generated or parsed.
                prompt = "Read the controller instruction:\n" + goal
                formatted = apply_chat_template(
                    qwen.processor,
                    qwen.model.config,
                    prompt,
                    num_images=0,
                    enable_thinking=False,
                )
                data = prepare_inputs(qwen.processor, prompts=formatted)
                result = qwen.model.language_model(
                    data["input_ids"],
                    return_hidden=True,
                    skip_logits=True,
                )
                feature = result.hidden_states[-1][0, -4:].astype(mx.float32).mean(0)
                value = np.asarray(feature)
                values.append(value / max(float(np.linalg.norm(value)), 1e-12))
                records[split].append({"template": i, "goal": goal, "when_open": when_open})
        features[split] = np.stack(values).astype(np.float64)
    # Fixed ridge strength; no hyperparameter search on development labels.
    mean = features["train"].mean(0)
    x = features["train"] - mean
    y = np.array([1 if r["when_open"] else -1 for r in records["train"]])
    alpha = 0.01
    weights = x.T @ np.linalg.solve(x @ x.T + alpha * np.eye(len(x)), y)
    report = {
        "model": {k: qwen.metadata[k] for k in ("model", "revision")},
        "feature_extraction": "Mean last four final-layer language hidden states; L2 normalized",
        "ridge_alpha": alpha,
        "scope": "Goal-only representation probe; does not test vision, actions or gameplay",
        "reserved_v2_wordings_evaluated": False,
        "results": {},
    }
    for split in records:
        predictions = (features[split] - mean) @ weights
        for row, value in zip(records[split], predictions, strict=True):
            row.update(score=float(value), correct=bool(value >= 0) == row["when_open"])
        report["results"][split] = {
            "accuracy": float(np.mean([r["correct"] for r in records[split]])),
            "records": records[split],
        }
    with output.open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({s: r["accuracy"] for s, r in report["results"].items()}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)


if __name__ == "__main__":
    main()
