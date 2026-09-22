"""Procedural multi-style grounding/action data with composition holdouts.

Ground truth belongs to the training environment, never the inference policy.
This dataset is a controlled research curriculum, not real-game experience.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

COLORS = {
    "red": (215, 38, 45),
    "blue": (35, 90, 225),
    "green": (30, 170, 70),
    "yellow": (235, 195, 30),
}
SHAPES = ("circle", "square", "triangle")
HOLDOUT = {("red", "triangle"), ("blue", "square")}


def render(path, objects, style, rng, player_marker=False):
    size = int(rng.choice([224, 256, 288, 320]))
    backgrounds = {
        "plain": (235, 238, 242),
        "grid": (205, 220, 225),
        "dark": (30, 38, 45),
        "panel": (175, 165, 150),
    }
    image = Image.new("RGB", (size, size), backgrounds[style])
    draw = ImageDraw.Draw(image)
    if style == "grid":
        for i in range(0, size, 24):
            draw.line((i, 0, i, size), fill=(190, 205, 210))
            draw.line((0, i, size, i), fill=(190, 205, 210))
    elif style == "panel":
        draw.rounded_rectangle((6, 6, size - 6, size - 6), radius=14, outline=(80, 70, 60), width=3)
    for color, shape, x, y, radius in objects:
        x, y, radius = x * size, y * size, radius * size
        fill = COLORS[color]
        box = (x - radius, y - radius, x + radius, y + radius)
        if shape == "circle":
            draw.ellipse(box, fill=fill)
        elif shape == "square":
            draw.rectangle(box, fill=fill)
        else:
            draw.polygon(
                [(x, y - radius), (x - radius, y + radius), (x + radius, y + radius)], fill=fill
            )
    if player_marker:
        draw.line((size / 2, size * 0.78, size / 2, size * 0.93), fill=(120, 120, 120), width=3)
    image.save(path)


def create(
    directory, train_scenes=256, validation_scenes=64, test_scenes=64, seed=2026, recovery=False
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    splits = (
        ("train", train_scenes),
        ("validation", validation_scenes),
        ("composition_test", test_scenes),
        ("style_test", test_scenes),
    )
    for split, count in splits:
        rows = []
        for scene in range(count):
            while True:
                colors = rng.choice(list(COLORS), 2, replace=False).tolist()
                shapes = rng.choice(SHAPES, 2).tolist()
                combinations = set(zip(colors, shapes, strict=True))
                if bool(combinations & HOLDOUT) == (split == "composition_test"):
                    break
            # Caption order is shuffled independently of spatial order.
            first_left = bool(rng.integers(2))
            objects = []
            for i, (color, shape) in enumerate(zip(colors, shapes, strict=True)):
                left = first_left == (i == 0)
                x = float(rng.uniform(0.16, 0.36) if left else rng.uniform(0.64, 0.84))
                objects.append(
                    (color, shape, x, float(rng.uniform(0.3, 0.7)), float(rng.uniform(0.065, 0.12)))
                )
            if recovery:
                shift = float(
                    rng.uniform(
                        0.04 - min(o[2] for o in objects), 0.96 - max(o[2] for o in objects)
                    )
                )
                objects = [
                    (c, s, x + shift, 0.5 if rng.random() < 0.7 else y, r)
                    for c, s, x, y, r in objects
                ]
            style = "panel" if split == "style_test" else str(rng.choice(["plain", "grid", "dark"]))
            image = f"{split}-{scene:04d}.png"
            render(directory / image, objects, style, rng, player_marker=recovery)
            description = " ".join(
                f"The {c} {s} is on the {'left' if x < 0.5 else 'right'}."
                for c, s, x, y, r in objects
            )
            for target, (color, shape, x, y, radius) in enumerate(objects):
                direction = "left" if x < 0.5 else "right"
                rows.append(
                    {
                        "id": f"{'recovery-' if recovery else ''}{split}-{scene}-{target}",
                        "game": "procedural-grounding",
                        "episode": f"{'recovery-' if recovery else ''}{split}-{scene}",
                        "frames": [{"image": image, "age_seconds": 0}],
                        "goal": f"Move toward the {color} {shape}. Which direction should you move?",
                        "controls": "A moves left. D moves right.",
                        "previous_actions": (
                            [{"buttons": [str(rng.choice(["a", "d"]))], "duration_seconds": 0.1}]
                            if recovery and rng.random() < 0.7
                            else []
                        ),
                        "choices": {"left": "Move left.", "right": "Move right."},
                        "answer": direction,
                        "description": description,
                        "action": {
                            "buttons": ["a" if direction == "left" else "d"],
                            "mouse_delta": [0.0, 0.0],
                            "pointer_xy": [x, y],
                            "duration_seconds": 0.1,
                        },
                        "provenance": "procedural ground truth, not a human demonstration",
                        "scene": {"objects": objects, "target": target, "style": style},
                    }
                )
        (directory / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (directory / "config.json").write_text(
        json.dumps(
            {"connector_type": "aligned", "connector_width": 256, "visual_slots": 16}, indent=2
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-scenes", type=int, default=256)
    parser.add_argument("--validation-scenes", type=int, default=64)
    parser.add_argument("--test-scenes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--mix-with", type=Path, help="Also create mixed-train.jsonl with this older training set"
    )
    parser.add_argument(
        "--recovery", action="store_true", help="Pan views, near-center targets and prior actions"
    )
    args = parser.parse_args()
    if min(args.train_scenes, args.validation_scenes, args.test_scenes) < 1:
        parser.error("Scene counts must be positive")
    create(
        args.output,
        args.train_scenes,
        args.validation_scenes,
        args.test_scenes,
        args.seed,
        args.recovery,
    )
    if args.mix_with:
        from .policy_data import mix_manifests
        from .trainable_model import PolicyConfig

        mix_manifests(
            [args.mix_with, args.output / "train.jsonl"],
            args.output / "mixed-train.jsonl",
            PolicyConfig(),
        )


if __name__ == "__main__":
    main()
