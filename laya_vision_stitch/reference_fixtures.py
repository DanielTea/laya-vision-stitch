"""Procedural paired data for an offline visual-grounding experiment, not game footage."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

COLORS = {"red": (220, 35, 40), "blue": (35, 80, 220), "green": (30, 165, 65)}
SHAPES = ("square", "circle", "triangle")
SIDES = ("left", "center", "right")
HELD_OUT = {("red", "triangle"), ("blue", "square"), ("green", "circle")}
QUESTIONS = {
    "color": ("What color is the object?", tuple(COLORS)),
    "shape": ("What shape is the object?", SHAPES),
    "position": ("Where is the object horizontally?", SIDES),
}


def render(path, color, shape, side, style, seed):
    rng = np.random.default_rng(seed)
    size = 320
    background = (
        (250, 250, 250)
        if style == "reference"
        else (220, 225, 230)
        if style == "panel"
        else (35, 40, 50)
    )
    image = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(image)
    if style == "panel":
        draw.rounded_rectangle((8, 8, 312, 312), radius=18, outline=(100, 105, 110), width=3)
    elif style == "night":
        for x in range(0, size, 32):
            draw.line((x, 0, x, size), fill=(45, 50, 60))
    x = {"left": 72, "center": 160, "right": 248}[side] + int(rng.integers(-12, 13))
    y = int(rng.integers(128, 192))
    radius = int(rng.integers(28, 45))
    box = (x - radius, y - radius, x + radius, y + radius)
    if shape == "square":
        draw.rectangle(box, fill=COLORS[color])
    elif shape == "circle":
        draw.ellipse(box, fill=COLORS[color])
    else:
        draw.polygon(
            [(x, y - radius), (x - radius, y + radius), (x + radius, y + radius)],
            fill=COLORS[color],
        )
    image.save(path)


def validate_separation(references, tests):
    if {r["sha256"] for r in references} & {r["sha256"] for r in tests}:
        raise ValueError("Reference/test image leakage")
    combinations = {(r["color"], r["shape"]) for r in references}
    for row in tests:
        if row["novel_composition"] != ((row["color"], row["shape"]) not in combinations):
            raise ValueError("Invalid composition holdout")


def create(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    references, tests, index = [], [], 0
    for color in COLORS:
        for shape in SHAPES:
            for side in SIDES:
                caption = f"A {color} {shape} is on the {side} side of the image."
                for style, variants in (("reference", 2), ("panel", 1), ("night", 1)):
                    if style == "reference" and (color, shape) in HELD_OUT:
                        continue
                    for variant in range(variants):
                        name = f"image-{index:04d}.png"
                        render(directory / name, color, shape, side, style, 731 + index)
                        row = dict(
                            id=f"example-{index:04d}",
                            image=name,
                            caption=caption,
                            color=color,
                            shape=shape,
                            position=side,
                            style=style,
                            sha256=hashlib.sha256((directory / name).read_bytes()).hexdigest(),
                        )
                        index += 1
                        if style == "reference":
                            references.append(row)
                        else:
                            row["novel_composition"] = (color, shape) in HELD_OUT
                            tests.append(row)
    validate_separation(references, tests)
    (directory / "references.json").write_text(json.dumps(references, indent=2) + "\n")
    (directory / "tests.json").write_text(json.dumps(tests, indent=2) + "\n")
    return references, tests
