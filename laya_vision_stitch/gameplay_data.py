"""Expand training within existing sessions, with a required visual-content review."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .d2e_data import BUTTONS
from .learning_gate import choose_diverse
from .policy_data import check_separation, read_manifest
from .trainable_model import PolicyConfig


def prepare(source, base, output, additional=128):
    output.mkdir(parents=True, exist_ok=False)
    config = PolicyConfig(buttons=BUTTONS)
    old = read_manifest(base / "train.jsonl", config)
    validation = read_manifest(base / "validation.jsonl", config)
    pool = read_manifest(source, config)
    used = {r["id"] for r in old}
    episodes = {r["episode"] for r in old}
    games = sorted({r["game"] for r in old})
    rng = np.random.default_rng(8129)
    selected = []
    for i, game in enumerate(games):
        candidates = [
            r
            for r in pool
            if r["game"] == game and r["episode"] in episodes and r["id"] not in used
        ]
        count = additional // len(games) + int(i < additional % len(games))
        selected.extend(choose_diverse(candidates, count, rng))
    prepared = []
    for item in selected:
        row = copy.deepcopy(item)
        row["recorded_previous_actions"] = row.pop("previous_actions", [])
        row["previous_actions"] = []
        prepared.append(row)
    check_separation(old + prepared, validation)
    (output / "candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prepared))
    (output / "selection.json").write_text(
        json.dumps(
            {
                "seed": 8129,
                "base": str(base),
                "source": str(source),
                "additional": additional,
                "scope": "Same training sessions; validation unchanged",
            },
            indent=2,
        )
        + "\n"
    )
    for start in range(0, len(prepared), 32):
        sheet = Image.new("RGB", (1280, 8 * 202), "white")
        draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(prepared[start : start + 32]):
            with Image.open(row["frames"][-1]["image"]) as frame:
                frame = frame.convert("RGB")
                frame.thumbnail((320, 180))
                x, y = j % 4 * 320, j // 4 * 202
                sheet.paste(frame, (x, y))
                draw.text((x + 3, y + 182), f"{start + j}: {row['game']} {row['id']}", fill="black")
        sheet.save(output / f"review-{start // 32}.jpg")


def reviewed_candidates(candidates, review):
    labels = {r["id"]: r for r in review["labels"]}
    if (
        len(review["labels"]) != len(candidates)
        or len(labels) != len(candidates)
        or set(labels) != {r["id"] for r in candidates}
    ):
        raise ValueError("Every candidate requires exactly one content review")
    accepted, rejected = [], []
    for row in candidates:
        label = labels[row["id"]]
        if label["image_sha256"] != row["frames"][-1]["sha256"]:
            raise ValueError("Reviewed image changed")
        if not isinstance(label["game_content"], bool):
            raise ValueError("Content review must be boolean")
        (accepted if label["game_content"] else rejected).append(row)
    return accepted, rejected


def finalize(output, annotations):
    selection = json.loads((output / "selection.json").read_text())
    config = PolicyConfig(buttons=BUTTONS)
    candidates = read_manifest(output / "candidates.jsonl", config)
    accepted, rejected = reviewed_candidates(candidates, json.loads(annotations.read_text()))
    base = Path(selection["base"])
    old = read_manifest(base / "train.jsonl", config)
    validation = read_manifest(base / "validation.jsonl", config)
    check_separation(old + accepted, validation)
    for split, rows in (("train", old + accepted), ("validation", validation)):
        with (output / f"{split}.jsonl").open("x") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    report = {
        **selection,
        "training_examples": len(old + accepted),
        "validation_examples": len(validation),
        "rejected": [r["id"] for r in rejected],
        "review_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
        "previous_actions_in_prompt": False,
        "labels": "Recorded next-100ms controls; goals remain generic, expert intent unavailable",
    }
    (output / "protocol.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("artifacts/d2e-pilot-003/train.jsonl"))
    p.add_argument("--base", type=Path, default=Path("artifacts/learning-gate-002"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--review", type=Path)
    args = p.parse_args()
    if args.review:
        finalize(args.output, args.review)
    else:
        prepare(args.source, args.base, args.output)


if __name__ == "__main__":
    main()
