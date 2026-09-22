"""Expand training recordings with goal coverage while preserving evaluation clips."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .p2p_data import CONTROLS, candidates, extract_frames, read_annotation, sha256
from .sequence_data import choose_starts, valid_starts


def covered_starts(rows, count, length, rng):
    """Prioritize distinct available weak instructions, then sample ordinary activity.

    No labels are invented. At most half the requested quota is instruction-prioritized;
    the rest is uniform among remaining nonoverlapping, constant-goal windows.
    """
    lookup = {r["frame_index"]: r for r in rows}
    starts = valid_starts(rows, length)
    by_goal = {}
    for i in starts:
        if lookup[i]["instruction_provenance"]["annotator"]:
            by_goal.setdefault(lookup[i]["goal"], []).append(i)
    chosen = []

    def available(i):
        return all(abs(i - other) > length for other in chosen)

    groups = [rng.permutation(indices).tolist() for indices in by_goal.values()]
    groups = [groups[i] for i in rng.permutation(len(groups))]
    quota = (count + 1) // 2
    while len(chosen) < quota and groups:
        remaining = []
        for group in groups:
            valid = [i for i in group if available(i)]
            if valid and len(chosen) < quota:
                chosen.append(valid[0])
                remaining.append(valid[1:])
        groups = remaining
    for i in rng.permutation(starts):
        if len(chosen) >= count:
            break
        if available(int(i)):
            chosen.append(int(i))
    return sorted(chosen)


def import_recording(entry, output, rng, length):
    source = Path(entry["source"])
    identifier = entry["recording"]
    if entry["split"] not in {"train", "fresh_test"}:
        raise ValueError("Unknown expansion split")
    folder = source / "dataset" / identifier
    for name, key in (("annotation.proto", "annotation_sha256"), ("video.mp4", "video_sha256")):
        if sha256(folder / name) != entry[key]:
            raise ValueError("Recording differs from reviewed source hashes")
    annotation = read_annotation(folder / "annotation.proto")
    rows, excluded = candidates(annotation, stride=1, include_unlabelled=True)
    starts = (
        covered_starts(rows, entry["sequences"], length, rng)
        if entry["split"] == "train"
        else choose_starts(valid_starts(rows, length), entry["sequences"], length, rng)
    )
    lookup = {r["frame_index"]: r for r in rows}
    timing = {}
    extract_frames(
        folder / "video.mp4",
        {i for s in starts for i in range(s, s + length + 1)},
        output / "frames" / identifier,
        len(annotation.frame_annotations),
        timing=timing,
    )
    group = annotation.metadata.group_name or annotation.metadata.id or identifier
    episode = hashlib.sha256(group.encode()).hexdigest()
    game = annotation.metadata.env.env.strip()
    if annotation.metadata.env.env_subtype:
        game += "/" + annotation.metadata.env.env_subtype.strip()
    result, valid = [], []
    for start in starts:
        if any(start <= i <= start + length for i in timing["irregular_intervals"]):
            continue
        valid.append(start)
        sequence = f"p2p-seq-{identifier}-{start}"
        for step in range(length):
            index = start + step
            result.append(
                {
                    **lookup[index],
                    "id": f"{sequence}-{step}",
                    "sequence": sequence,
                    "sequence_step": step,
                    "sequence_length": length,
                    "game": game,
                    "episode": episode,
                    "controls": CONTROLS,
                    "timestamp_seconds": timing["timestamps"][index],
                    "frames": [
                        {
                            "image": str(
                                (output / "frames" / identifier / f"{index:07d}.png").resolve()
                            ),
                            "age_seconds": 0,
                        }
                    ],
                    "future_image": str(
                        (output / "frames" / identifier / f"{index + 1:07d}.png").resolve()
                    ),
                    "source": {
                        "repository": entry["repository"],
                        "revision": entry["revision"],
                        "recording": identifier,
                        "action_annotation_index": index + 1,
                    },
                }
            )
    return result, {
        **entry,
        "game": game,
        "episode": episode,
        "starts": valid,
        "excluded": excluded,
        "irregular_video_intervals": timing["irregular_intervals"],
    }


def preserved_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        for frame in row["frames"]:
            frame["image"] = str((path.parent / frame["image"]).resolve())
        row["future_image"] = str((path.parent / row["future_image"]).resolve())
    return rows


def image_hashes(rows):
    paths = {r["future_image"] for r in rows} | {f["image"] for r in rows for f in r["frames"]}
    return {sha256(Path(p)) for p in paths}


def build(plan, output):
    selection = json.loads(plan.read_text())
    length = selection.get("length", 32)
    if not 2 <= length <= 128:
        raise ValueError("Invalid sequence length")
    entries = selection["recordings"]
    training_entries = [r for r in entries if r["split"] == "train"]
    fresh_entries = [r for r in entries if r["split"] == "fresh_test"]
    if len({r["recording"] for r in entries}) != len(entries) or any(
        r["sequences"] < 1 for r in entries
    ):
        raise ValueError("Duplicate recording or invalid quota")
    preserved = Path(selection["preserve_evaluation"])
    splits = {s: preserved_rows(preserved / f"{s}.jsonl") for s in ("validation", "test")}
    held_recordings = {r["source"]["recording"] for rows in splits.values() for r in rows}
    if held_recordings & {r["recording"] for r in entries}:
        raise ValueError("Training plan contains an evaluation recording")
    output.mkdir(parents=True, exist_ok=False)
    rng, train, audits, removed = np.random.default_rng(selection.get("seed", 97)), [], [], 0
    if fresh_entries:
        splits["fresh_test"] = []
        for entry in fresh_entries:
            rows, audit = import_recording(entry, output, rng, length)
            splits["fresh_test"].extend(rows)
            audits.append(audit)
            print(f"Fresh holdout {audit['game']}: {len(rows) // length} sequences", flush=True)
    held_episodes = {r["episode"] for rows in splits.values() for r in rows}
    test_games = {r["game"] for split in ("test", "fresh_test") for r in splits.get(split, [])}
    held_hashes = image_hashes([r for rows in splits.values() for r in rows])
    for entry in training_entries:
        rows, audit = import_recording(entry, output, rng, length)
        if audit["episode"] in held_episodes or audit["game"] in test_games:
            raise ValueError("Training session/game leaks into held-out evaluation")
        groups = {}
        for row in rows:
            groups.setdefault(row["sequence"], []).append(row)
        for sequence in groups.values():
            if image_hashes(sequence) & held_hashes:
                removed += 1
            else:
                train.extend(sequence)
        audits.append(audit)
        print(f"Expanded {audit['game']}: {len(rows) // length} sequences", flush=True)
    if not train:
        raise ValueError("No training sequences")
    splits["train"] = train
    for split, rows in splits.items():
        (output / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    audit = {
        "plan_sha256": sha256(plan),
        "sampling": "Half goal-coverage quota then uniform remaining training windows; unchanged evaluation",
        "recordings": audits,
        "removed_image_overlap_sequences": removed,
        "rows": {s: len(rows) for s, rows in splits.items()},
        "games": {s: dict(Counter(r["game"] for r in rows)) for s, rows in splits.items()},
        "distinct_goals": {s: len({r["goal"] for r in rows}) for s, rows in splits.items()},
        "weak_instruction_frames": {
            s: sum(bool(r["instruction_provenance"]["annotator"]) for r in rows)
            for s, rows in splits.items()
        },
        "evaluation_semantically_preserved": True,
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        json.dumps({k: audit[k] for k in ("rows", "distinct_goals", "weak_instruction_frames")}),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    build(a.plan, a.output)


if __name__ == "__main__":
    main()
