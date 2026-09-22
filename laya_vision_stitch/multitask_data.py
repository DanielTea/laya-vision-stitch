"""Expand existing grounding scenes into multiple goals without new image rules in inference."""

import argparse
import json
from pathlib import Path

from .goal_transfer import variants
from .policy_data import read_manifest
from .trainable_model import PolicyConfig


def expand(manifest, output, canonical_descriptions=False, paraphrases=False):
    rows = read_manifest(manifest, PolicyConfig())
    result = []
    seen = set()
    for row in rows:
        row = dict(row)
        row["task_family"] = "toward"
        if canonical_descriptions:
            objects = sorted(row["scene"]["objects"], key=lambda o: o[2])
            row["description"] = " ".join(
                f"The {c} {s} is on the {'left' if x < 0.5 else 'right'}."
                for c, s, x, y, r in objects
            )
        result.append(row)
        tasks = variants(row)
        away = tasks["move_away"]
        away["task_family"] = "away"
        away["id"] = row["id"] + "-away"
        if "action" in away:
            away["action"] = dict(
                away["action"], buttons=["a" if away["answer"] == "left" else "d"], pointer_xy=None
            )
        result.append(away)
        key = tuple(f["sha256"] for f in row["frames"])
        if key not in seen:
            for name in ("left_color", "left_shape"):
                question = tasks[name]
                question["task_family"] = name
                if canonical_descriptions:
                    question["goal"] = question["goal"].replace(
                        "object on the left side of the image", "leftmost object"
                    )
                question["id"] = row["id"] + "-" + name
                question.pop("action", None)
                result.append(question)
            seen.add(key)
    if paraphrases:
        templates = {
            "toward": [
                "Approach the {c} {s}. Choose a movement direction.",
                "Get closer to the {c} {s}. Which way should you move?",
                "Reduce your distance to the {c} {s}.",
            ],
            "away": [
                "Back away from the {c} {s}. Choose a movement direction.",
                "Move farther from the {c} {s}. Which way should you move?",
                "Increase your distance from the {c} {s}.",
            ],
            "left_color": [
                "Name the color furthest to the left.",
                "Which hue belongs to the leftmost object?",
                "Select the color of the leftmost object.",
            ],
            "left_shape": [
                "Name the shape furthest to the left.",
                "Which geometric shape appears at the far left?",
                "Select the shape of the leftmost object.",
            ],
        }
        extra = []
        for row in result:
            c, s = row["scene"]["objects"][row["scene"]["target"]][:2]
            for index, template in enumerate(templates[row["task_family"]]):
                alternate = dict(
                    row, id=f"{row['id']}-phrase-{index}", goal=template.format(c=c, s=s)
                )
                for key in ("teacher_probs", "teacher_temperature", "teacher"):
                    alternate.pop(key, None)
                extra.append(alternate)
        result.extend(extra)
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("Expanded examples have duplicate IDs")
    with Path(output).open("x") as handle:
        for row in result:
            handle.write(json.dumps(row) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--canonical-descriptions",
        action="store_true",
        help="Consistent left-to-right description targets and explicit leftmost questions",
    )
    parser.add_argument(
        "--paraphrases",
        action="store_true",
        help="Add three alternate instructions per example, preserving family balance",
    )
    args = parser.parse_args()
    expand(args.manifest, args.output, args.canonical_descriptions, args.paraphrases)


if __name__ == "__main__":
    main()
