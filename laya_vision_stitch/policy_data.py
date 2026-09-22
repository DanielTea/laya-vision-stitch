"""Validated, game-independent supervision for the stitched model."""

import hashlib
import json
from pathlib import Path

import numpy as np


def vector(value, size, lower, upper, name):
    array = np.asarray(value, dtype=float)
    if (
        array.shape != (size,)
        or not np.isfinite(array).all()
        or np.any(array < lower)
        or np.any(array > upper)
    ):
        raise ValueError(f"Invalid {name}: expected {size} values in [{lower}, {upper}]")


def read_manifest(path, config, supervised=True):
    path = Path(path)
    rows, seen = [], set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for name in ("id", "game", "episode", "goal"):
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError(f"Each record needs a nonempty {name}")
        if row["id"] in seen:
            raise ValueError("Duplicate example ID")
        seen.add(row["id"])
        if not isinstance(row.get("controls", ""), str):
            raise ValueError("controls must be text")
        if not isinstance(row.get("previous_actions", []), list):
            raise ValueError("previous_actions must be a list")
        frames = row.get("frames", [])
        if not 1 <= len(frames) <= config.max_frames:
            raise ValueError("Invalid frame count")
        ages = []
        for frame in frames:
            image = (path.parent / frame["image"]).resolve()
            if not image.is_file():
                raise ValueError(f"Missing image: {image}")
            frame["image"] = str(image)
            frame["sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
            age = frame.get("age_seconds")
            if not isinstance(age, (int, float)) or not np.isfinite(age) or age < 0:
                raise ValueError("Frames need finite, nonnegative age_seconds")
            ages.append(age)
        if ages != sorted(ages, reverse=True) or ages[-1] != 0 or len(set(ages)) != len(ages):
            raise ValueError("Frames must be oldest first with distinct ages ending at zero")
        choices = row.get("choices")
        if choices is not None:
            if not isinstance(choices, dict) or not 2 <= len(choices) <= 32:
                raise ValueError("choices must map 2..32 names to descriptions")
            if any(not isinstance(v, str) or not v.strip() for v in choices.values()):
                raise ValueError("Empty choice description")
        if "answer" in row and (not choices or row["answer"] not in choices):
            raise ValueError("Answer must name a supplied choice")
        if "description" in row and (
            not isinstance(row["description"], str) or not row["description"].strip()
        ):
            raise ValueError("Description must be nonempty text")
        if "teacher_probs" in row:
            if not choices or set(row["teacher_probs"]) != set(choices):
                raise ValueError("Teacher probabilities must match named choices")
            p = list(row["teacher_probs"].values())
            vector(p, len(choices), 0, 1, "teacher probabilities")
            if not np.isclose(sum(p), 1, atol=1e-5):
                raise ValueError("Teacher probabilities must sum to one")
            temperature = row.get("teacher_temperature", 1.0)
            if not np.isfinite(temperature) or temperature <= 0:
                raise ValueError("Invalid teacher temperature")
        actions = [row["action"]] if "action" in row else []
        if "action_chunk" in row:
            chunk = row["action_chunk"]
            if not isinstance(chunk, list) or not 1 <= len(chunk) <= 8 or not actions:
                raise ValueError("Chunks require a first action and 1..8 future steps")
            if chunk[0] != row["action"]:
                raise ValueError("Chunk prefix must match the first action")
            if config.action_chunk_size > 1 and len(chunk) != config.action_chunk_size:
                raise ValueError("Chunk size differs from the model")
            if any(a.get("duration_seconds") != 0.1 for a in chunk):
                raise ValueError("Chunk steps currently use 100 ms intervals")
            actions.extend(chunk)
        for action in actions:
            buttons = action.get("buttons")
            if (
                not isinstance(buttons, list)
                or len(set(buttons)) != len(buttons)
                or set(buttons) - set(config.buttons)
            ):
                raise ValueError("Action buttons must belong to configured vocabulary")
            vector(action.get("mouse_delta"), 2, -1, 1, "normalized mouse delta")
            duration = action.get("duration_seconds")
            if duration not in config.durations:
                raise ValueError("Action duration must match a configured duration bin")
            if action.get("pointer_xy") is not None:
                vector(action["pointer_xy"], 2, 0, 1, "normalized pointer position")
        if supervised and not any(
            k in row for k in ("answer", "teacher_probs", "action", "description")
        ):
            raise ValueError(
                "Training examples need answers, teacher targets, descriptions, or actions"
            )
        rows.append(row)
    if not rows:
        raise ValueError("Empty manifest")
    return rows


def check_separation(train, validation, holdout_games=False):
    for name, getter in (
        ("ID", lambda rows: {r["id"] for r in rows}),
        ("episode", lambda rows: {(r["game"], r["episode"]) for r in rows}),
        ("image", lambda rows: {f["sha256"] for r in rows for f in r["frames"]}),
    ):
        if getter(train) & getter(validation):
            raise ValueError(f"Train/validation {name} leakage")
    if holdout_games and {r["game"] for r in train} & {r["game"] for r in validation}:
        raise ValueError("Game holdout requested but games overlap")


def manifest_digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def mix_manifests(paths, output, config):
    rows = []
    for path in paths:
        rows.extend(read_manifest(path, config))
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Mixed manifests contain duplicate IDs")
    with Path(output).open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
