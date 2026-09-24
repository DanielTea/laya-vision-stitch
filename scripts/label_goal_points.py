"""Offline goal-conditioned pointing labels from Molmo-7B (teacher only, never at inference).

Each sampled frame is queried for general object categories ("Point to the doors.") and
the returned points (or "none") become targets for a goal-conditioned pointer. Categories
are world objects common to many games; nothing is game-specific. Frames from held-out
games and Hordes are labelled for evaluation only.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

MOLMO = "mlx-community/Molmo-7B-D-0924-4bit"
MOLMO_REVISION = "5c04b3a418979597b1968e41414ad799c87533e8"
CATEGORIES = {
    "enemies": (
        "Point to the enemies or monsters.",
        ["attack the enemy", "fight the monster", "defeat the nearby monsters"],
    ),
    "doors": ("Point to the doors.", ["go through the door", "open the door"]),
    "characters": (
        "Point to the characters or people.",
        ["talk to the character", "walk to the person"],
    ),
    "animals": ("Point to the animals.", ["go to the animal", "approach the animal"]),
    "vehicles": ("Point to the vehicles.", ["get in the vehicle", "walk to the car"]),
    "containers": ("Point to the chests or containers.", ["open the chest", "loot the container"]),
}


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def parse(text):
    if re.search(r"there (are|is) none", text, re.I):
        return []
    points = re.findall(r'x\d*="([\d.]+)"\s+y\d*="([\d.]+)"', text)
    return [[float(x) / 100, float(y) / 100] for x, y in points] if points else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--presses", type=Path, required=True, help="rows.jsonl of a click-feature cache"
    )
    p.add_argument(
        "--hordes", nargs="*", type=Path, default=[], help="Live-trial frame folders (evaluation)"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-frames", type=int, default=1600)
    p.add_argument("--heldout-per-game", type=int, default=60)
    p.add_argument("--hordes-frames", type=int, default=60)
    p.add_argument("--seed", type=int, default=20260923)
    args = p.parse_args()
    from huggingface_hub import snapshot_download
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    rng = np.random.default_rng(args.seed)
    rows = read(args.presses)
    by_game = defaultdict(list)
    for r in rows:
        by_game[(r["split"], r["game"])].append(r)
    train_games = [g for s, g in by_game if s == "train"]
    per_game = max(1, args.train_frames // max(1, len(train_games)))
    jobs = []
    for (split, game), items in sorted(by_game.items()):
        count = per_game if split == "train" else args.heldout_per_game
        for r in rng.permutation(items)[:count]:
            others = [c for c in CATEGORIES if c != "enemies"]
            for category in ["enemies", str(rng.choice(others))]:
                jobs.append(
                    {"image": r["image"], "game": game, "split": split, "category": category}
                )
    for folder in args.hordes:
        frames = sorted(folder.glob("*.jpg"))
        for f in [
            frames[i]
            for i in rng.permutation(len(frames))[: args.hordes_frames // max(1, len(args.hordes))]
        ]:
            jobs.append(
                {
                    "image": str(f.resolve()),
                    "game": "Hordes.io",
                    "split": "hordes_eval",
                    "category": "enemies",
                }
            )
    model, processor = load(snapshot_download(MOLMO, revision=MOLMO_REVISION))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = (
        {(r["image"], r["category"]) for r in read(args.output)} if args.output.exists() else set()
    )
    with args.output.open("a") as log:
        for job in jobs:
            if (job["image"], job["category"]) in done:
                continue
            image = Image.open(job["image"]).convert("RGB")
            if image.width > 384:
                image = image.resize((384, round(image.height * 384 / image.width)))
            prompt = CATEGORIES[job["category"]][0]
            text = generate(
                model,
                processor,
                apply_chat_template(processor, model.config, prompt, num_images=1),
                image=[image],
                max_tokens=120,
                temperature=0,
                verbose=False,
            ).text
            log.write(
                json.dumps({**job, "prompt": prompt, "points": parse(text), "raw": text[:400]})
                + "\n"
            )
            log.flush()


if __name__ == "__main__":
    main()
