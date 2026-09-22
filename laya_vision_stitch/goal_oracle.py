"""Known-text-state conditional goal audit; no images, training or live inputs."""

import argparse
import json
from pathlib import Path

import mlx.core as mx

from .backends import QwenVision
from .goal_curriculum import DEVELOPMENT, TRAIN
from .reference_probe import text_choice
from .reference_stitch import load_laya


def cases():
    # Reproduce the original 72-case audit, without consuming reserved wording.
    for i, template in enumerate(TRAIN + DEVELOPMENT):
        for opened in (False, True):
            for when_open in (False, True):
                for reverse in (False, True):
                    state, opposite = ("open", "closed") if when_open else ("closed", "open")
                    yield {
                        "template": i,
                        "opened": opened,
                        "when_open": when_open,
                        "reverse": reverse,
                        "description": (
                            "A large game menu is open."
                            if opened
                            else "The game world is visible. No large game menu is open."
                        ),
                        "goal": template.format(state=state, opposite=opposite),
                        "expected": "hold" if opened == when_open else "release",
                    }


def run(backend, output):
    if output.exists():
        raise FileExistsError(output)
    agent = load_laya("english") if backend == "laya" else QwenVision()
    records = []
    for row in cases():
        choices = {"hold": "Hold the W key.", "release": "Release every key."}
        if row["reverse"]:
            choices = dict(reversed(list(choices.items())))
        if backend == "laya":
            answer = text_choice(agent, row["description"], row["goal"], choices)
        else:
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import prepare_inputs

            labels = ("A", "B")
            prompt = (
                f"Observed state: {row['description']}\nInstruction: {row['goal']}\n"
                "Which action follows the instruction for the observed state?\n"
                + "\n".join(f"{k}: {v}" for k, v in zip(labels, choices.values(), strict=True))
                + "\nAnswer with exactly one letter, A or B."
            )
            formatted = apply_chat_template(
                agent.processor,
                agent.model.config,
                prompt,
                num_images=0,
                enable_thinking=False,
            )
            data = prepare_inputs(agent.processor, prompts=formatted)
            ids = [agent.processor.tokenizer.encode(k, add_special_tokens=False) for k in labels]
            if any(len(x) != 1 for x in ids):
                raise ValueError("Expected single-token answer labels")
            out = agent.model.language_model(data["input_ids"])
            scores = out.logits[0, -1, mx.array([x[0] for x in ids])]
            answer = list(choices)[int(mx.argmax(scores))]
        records.append({**row, "prediction": answer, "correct": answer == row["expected"]})
    report = {
        "backend": backend,
        "scope": "Correct state supplied as text; not a vision/gameplay evaluation",
        "accuracy": sum(r["correct"] for r in records) / len(records),
        "by_template": {
            str(i): sum(r["correct"] for r in records if r["template"] == i) / 8
            for i in range(len(TRAIN + DEVELOPMENT))
        },
        "records": records,
        "live_inputs_sent": 0,
        "generated_tokens": 0,
    }
    with output.open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("laya", "qwen"), default="laya")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.backend, args.output)


if __name__ == "__main__":
    main()
