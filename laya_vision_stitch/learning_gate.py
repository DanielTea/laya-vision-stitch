"""Reproducible small-set fit and session-holdout gates for real gameplay."""

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .d2e_data import BUTTONS, Timeline, load_events
from .policy_data import check_separation, read_manifest
from .trainable_model import PolicyConfig


def choose_diverse(rows, count, rng):
    """Training-only label stratification; no ranking on model performance."""
    buckets = defaultdict(list)
    for row in rows:
        action = row["action"]
        mouse = np.array(action["mouse_delta"])
        kind = tuple(action["buttons"]), tuple(np.where(abs(mouse) >= 0.02, np.sign(mouse), 0))
        buckets[kind].append(row)
    groups = list(buckets.values())
    rng.shuffle(groups)
    for group in groups:
        rng.shuffle(group)
    selected = []
    while len(selected) < count and groups:
        for group in list(groups):
            selected.append(group.pop())
            if not group:
                groups.remove(group)
            if len(selected) == count:
                break
    if len(selected) != count:
        raise ValueError("Not enough clips")
    return selected


def create(manifest, source, output, count=64, seed=29, curation=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = read_manifest(manifest, PolicyConfig(buttons=BUTTONS))
    games = defaultdict(lambda: defaultdict(list))
    for row in rows:
        games[row["game"]][row["episode"]].append(row)
    rng = np.random.default_rng(seed)
    train, validation = [], []
    sessions = {}
    for index, (game, episodes) in enumerate(sorted(games.items())):
        names = sorted(episodes)
        if len(names) < 2:
            raise ValueError("Need separate recording sessions per game")
        n = count // len(games) + int(index < count % len(games))
        train.extend(choose_diverse(episodes[names[0]], n, rng))
        # Validation is uniform, not selected by actions or model predictions.
        validation.extend(
            episodes[names[1]][i] for i in rng.choice(len(episodes[names[1]]), n, replace=False)
        )
        sessions[game] = {"train": names[0], "validation": names[1]}
    review = json.loads(Path(curation).read_text()) if curation else None
    if review:
        by_id = {row["id"]: row for row in rows}
        for split, selected in (("train", train), ("validation", validation)):
            for replacement in review.get("replacements", []):
                if replacement["split"] != split:
                    continue
                indexes = [
                    i for i, row in enumerate(selected) if row["id"] == replacement["excluded_id"]
                ]
                if len(indexes) != 1:
                    raise ValueError("Curation exclusion does not match selection")
                new = by_id[replacement["replacement"]]
                if new["episode"] != selected[indexes[0]]["episode"] or new in selected:
                    raise ValueError("Replacement must be a unique clip from the same session")
                selected[indexes[0]] = new
    timeline_cache = {}
    for split, selected in [("train", train), ("validation", validation)]:
        prepared = []
        for original in selected:
            row = copy.deepcopy(original)
            rec = row["provenance"]["recording"]
            if rec not in timeline_cache:
                _, states, mouse, _ = load_events((Path(source) / rec).with_suffix(".mcap"))
                timeline_cache[rec] = Timeline(states, mouse)
            t = row["provenance"]["log_seconds"]
            if t + 0.4 > min(timeline_cache[rec].times[-1], timeline_cache[rec].mouse_times[-1]):
                raise ValueError("Clip lacks a complete future action interval")
            row["action_chunk"] = [
                timeline_cache[rec].action(t + 0.1 * k, 0.1, 512)[0] for k in range(4)
            ]
            row["action"] = row["action_chunk"][0]
            row["recorded_previous_actions"] = row["previous_actions"]
            row["previous_actions"] = []  # Force the learning test to use images.
            prepared.append(row)
        selected[:] = prepared
        (output / (split + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in selected))
    check_separation(train, validation)
    protocol = {
        "seed": seed,
        "training_clips": len(train),
        "validation_clips": len(validation),
        "sessions": sessions,
        "same_games": True,
        "previous_actions_in_prompt": False,
        "train_selection": "round-robin button-set and mouse-direction strata",
        "validation_selection": "uniform by session, no label stratification",
        "small_fit_gate": {
            "button_micro_f1_min": 0.95,
            "button_exact_match_min": 0.90,
            "mouse_direction_accuracy_min": 0.80,
        },
        "generalization_gate": {
            "button_micro_f1_min": 0.70,
            "image_shuffle_f1_margin_min": 0.10,
            "beat_recorded_persistence_f1": True,
        },
        "live_gameplay_authorized_by_results": False,
    }
    if review:
        protocol["review"] = review
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("artifacts/d2e-pilot-003/train.jsonl"))
    p.add_argument("--source", type=Path, default=Path("artifacts/d2e-source"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--curation", type=Path)
    args = p.parse_args()
    print(
        json.dumps(
            create(args.manifest, args.source, args.output, curation=args.curation), indent=2
        )
    )


if __name__ == "__main__":
    main()
