"""Goal/question audit compared with the same Laya given an exact description."""

import argparse
import json
from pathlib import Path

import mlx.core as mx

from .policy_data import read_manifest
from .policy_training import cache_examples
from .scaling_data import COLORS, SHAPES
from .trainable_model import TrainableRuntime


def variants(row):
    objects = row["scene"]["objects"]
    left = min(objects, key=lambda o: o[2])
    away = dict(
        row,
        goal=row["goal"].replace("Move toward", "Move away from"),
        answer="right" if row["answer"] == "left" else "left",
    )
    color = dict(
        row,
        goal="What color is the object on the left side of the image?",
        choices={c: c for c in COLORS},
        answer=left[0],
    )
    shape = dict(
        row,
        goal="What shape is the object on the left side of the image?",
        choices={s: s for s in SHAPES},
        answer=left[1],
    )
    result = {"move_away": away, "left_color": color, "left_shape": shape}
    for variant in result.values():
        # Teacher scores belong to the original question, never a changed goal.
        for key in ("teacher_probs", "teacher_temperature", "teacher"):
            variant.pop(key, None)
    return result


def run(bundle, manifest, output, cache=None, paraphrase=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    runtime = TrainableRuntime.load(bundle)
    # One question family per unique scene, avoiding duplicate color/shape questions.
    rows = read_manifest(manifest, runtime.module.policy_config)[::2]
    counts = {
        name: {"model_correct": 0, "text_oracle_correct": 0, "zero_visual_correct": 0, "count": 0}
        for name in ("move_away", "left_color", "left_shape")
    }
    for row, inputs in cache_examples(runtime, rows, cache):
        patches, coordinates = inputs[:2]
        ids = runtime.agent.tok(row["description"], add_special_tokens=False)["input_ids"]
        state = runtime.module.laya.encoder.embeddings.tok_embeddings(mx.array([ids]))
        if len(ids) != runtime.module.policy_config.visual_slots:
            raise ValueError("Oracle audit currently requires exact-length descriptions")
        for name, variant in variants(row).items():
            if paraphrase:
                if name == "move_away":
                    variant["goal"] = (
                        variant["goal"]
                        .replace("Move away from", "Retreat from")
                        .replace("Which direction should you move?", "Choose a movement direction.")
                    )
                else:
                    attribute = "color" if name == "left_color" else "shape"
                    variant["goal"] = f"Identify the {attribute} of the leftmost object."
            batch, goal, start = runtime.prepare(variant)
            result = runtime.module.from_features(patches, coordinates, batch, goal, start)[
                "choices"
            ]
            blank = runtime.module.from_features(
                mx.zeros_like(patches), coordinates, batch, goal, start
            )["choices"]
            oracle = runtime.module.from_state(state, batch, start)["choices"]
            mx.eval(result, blank, oracle)
            options = list(variant["choices"])
            for key, value in (
                ("model_correct", result),
                ("text_oracle_correct", oracle),
                ("zero_visual_correct", blank),
            ):
                counts[name][key] += options[int(mx.argmax(value[0]).item())] == variant["answer"]
            counts[name]["count"] += 1
    report = {}
    for name, values in counts.items():
        report[name] = {
            **values,
            "prompt_style": "paraphrase" if paraphrase else "original",
            **{
                key.replace("correct", "accuracy"): value / values["count"]
                for key, value in values.items()
                if key.endswith("correct")
            },
        }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--paraphrase", action="store_true")
    args = parser.parse_args()
    run(args.bundle, args.manifest, args.output, args.feature_cache, args.paraphrase)


if __name__ == "__main__":
    main()
