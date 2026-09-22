"""Weak screenshot/state pairs with run-disjoint chronological splits."""

import hashlib
import json
from pathlib import Path

import numpy as np


def describe(state):
    """Describe measured fields only; never include policy phase or chosen action."""
    hp = state.get("health")
    health = "unknown" if hp is None else "low" if hp < 0.3 else "medium" if hp < 0.7 else "high"
    target = "monster" if state.get("target") and state.get("target_signature_ok") else "unknown"
    ready = state.get("ability_ready")
    ability = "unknown" if ready is None else "ready" if ready else "cooldown"
    enemy_hp = state.get("target_health")
    enemy = "unknown" if enemy_hp is None else "dead" if enemy_hp <= 0 else "alive"
    return (
        f"Health {health}. Target {target}. Enemy {enemy}. Ability {ability}. "
        f"Range warning {'yes' if state.get('out_of_range') else 'no'}. "
        f"Blocked {'yes' if state.get('blocked') else 'no'}."
    )


def collect(root, max_runs=8, per_run=32):
    groups = []
    for log in sorted(Path(root).glob("*-fast-live/events.jsonl")):
        rows, last = [], -float("inf")
        for line in log.read_text().splitlines():
            r = json.loads(line)
            s = r.get("state", {})
            if (
                not r.get("image")
                or not s.get("hud")
                or not s.get("ocr_fresh")
                or "elapsed_s" not in r
            ):
                continue
            when = float(r["elapsed_s"])
            image = (log.parent / r["image"]).resolve()
            if when - last < 1 or not image.is_file():
                continue
            last = when
            rows.append(
                dict(
                    run=log.parent.name,
                    image=str(image),
                    seconds=when,
                    text=describe(s),
                    sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                )
            )
        if len(rows) >= 12:
            indices = np.linspace(0, len(rows) - 1, min(per_run, len(rows))).astype(int)
            groups.append([rows[i] for i in indices])
    groups = groups[-max_runs:]
    if len(groups) < 6:
        raise ValueError("Need at least six runs with >=12 eligible screenshots each")
    held = max(1, len(groups) // 4)
    result, seen = [], set()
    for i, group in enumerate(groups):
        split = (
            "train"
            if i < len(groups) - 2 * held
            else "validation"
            if i < len(groups) - held
            else "test"
        )
        for row in group:
            if row["sha256"] in seen:
                continue
            seen.add(row["sha256"])
            result.append({**row, "split": split})
    validate_splits(result)
    return result


def validate_splits(rows):
    owners, images = {}, set()
    for row in rows:
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        if owners.setdefault(row["run"], row["split"]) != row["split"]:
            raise ValueError("A run crosses split boundaries")
        if row["sha256"] in images:
            raise ValueError("Duplicate screenshot in dataset")
        images.add(row["sha256"])
    if set(owners.values()) != {"train", "validation", "test"}:
        raise ValueError("All three splits must contain data")
