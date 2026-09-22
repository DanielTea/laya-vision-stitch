"""Synthetic input/gradient smoke test. These labels are not gameplay evidence."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def create(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(31)
    for split, count in (("train", 8), ("validation", 4)):
        rows = []
        for scene in range(count):
            image = Image.new("RGB", (256, 256), (230 + scene, 235, 240))
            draw = ImageDraw.Draw(image)
            red_left = scene % 2 == 0
            for left, color in ((red_left, "red"), (not red_left, "blue")):
                x = (58 if left else 198) + int(rng.integers(-8, 9))
                y = 130 + int(rng.integers(-22, 23))
                draw.ellipse((x - 22, y - 22, x + 22, y + 22), fill=color)
            name = f"{split}-{scene}.png"
            image.save(directory / name)
            for color in ("red", "blue"):
                left = red_left == (color == "red")
                rows.append(
                    {
                        "id": f"{split}-{scene}-{color}",
                        "game": "synthetic-dots",
                        "episode": f"{split}-{scene}",
                        "frames": [{"image": name, "age_seconds": 0}],
                        "goal": f"Move toward the {color} circle. Which direction should you move?",
                        "controls": "A moves left. D moves right.",
                        "previous_actions": [],
                        "choices": {"left": "Move left.", "right": "Move right."},
                        "answer": "left" if left else "right",
                        "action": {
                            "buttons": ["a" if left else "d"],
                            "mouse_delta": [0.0, 0.0],
                            "pointer_xy": [0.25 if left else 0.75, 0.5],
                            "duration_seconds": 0.1,
                        },
                        "provenance": "procedural smoke fixture; no human demonstration",
                    }
                )
        (directory / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return directory
