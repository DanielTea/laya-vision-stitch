"""Build contiguous, recording-disjoint P2P sequence windows from audited local data."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .p2p_data import CONTROLS, candidates, extract_frames, read_annotation, sha256


def valid_starts(rows, length):
    indexed = {r["frame_index"]: r for r in rows}
    return [
        i
        for i, row in indexed.items()
        if all(j in indexed and indexed[j]["goal"] == row["goal"] for j in range(i, i + length))
    ]


def choose_starts(starts, count, length, rng):
    result = []
    for start in rng.permutation(starts):
        # Include the future supervision frame in overlap checks.
        if all(abs(int(start) - other) > length for other in result):
            result.append(int(start))
        if len(result) == count:
            break
    return sorted(result)


def build(source, plan, output, count=8, length=32, seed=83):
    if count < 1 or not 2 <= length <= 128:
        raise ValueError("Invalid sequence count/length")
    selection = json.loads(plan.read_text())
    inventory = json.loads((source / "inventory.json").read_text())
    index = {r["recording"]: r for r in inventory["recordings"]}
    groups, seen = {}, set()
    for entry in selection["recordings"]:
        identifier = entry["recording"]
        if identifier in seen or entry["split"] not in {"train", "validation", "test"}:
            raise ValueError("Invalid or duplicate recording split")
        seen.add(identifier)
        group = index[identifier]["group"]
        if group in groups and groups[group] != entry["split"]:
            raise ValueError("Session group crosses split")
        groups[group] = entry["split"]
    output.mkdir(parents=True, exist_ok=False)
    splits = {s: [] for s in ("train", "validation", "test")}
    rng, audit = np.random.default_rng(seed), []
    for entry in selection["recordings"]:
        identifier = entry["recording"]
        folder = source / "dataset" / identifier
        info = index[identifier]
        for name, key in (("annotation.proto", "annotation_sha256"), ("video.mp4", "video_sha256")):
            if sha256(folder / name) != info[key]:
                raise ValueError("Source differs from audited hash")
        annotation = read_annotation(folder / "annotation.proto")
        rows, excluded = candidates(annotation, stride=1, include_unlabelled=True)
        lookup = {r["frame_index"]: r for r in rows}
        starts = choose_starts(valid_starts(rows, length), count, length, rng)
        if not starts:
            raise ValueError(f"No consecutive valid windows: {identifier}")
        timing = {}
        extract_frames(
            folder / "video.mp4",
            {i for start in starts for i in range(start, start + length + 1)},
            output / "frames" / identifier,
            len(annotation.frame_annotations),
            timing=timing,
        )
        kept = []
        for start in starts:
            if any(start <= i <= start + length for i in timing["irregular_intervals"]):
                continue
            sequence = f"p2p-seq-{identifier}-{start}"
            game = info["game"] + ("/" + info["subtype"] if info.get("subtype") else "")
            for step in range(length):
                t = start + step
                splits[entry["split"]].append(
                    {
                        **lookup[t],
                        "id": f"{sequence}-{step}",
                        "sequence": sequence,
                        "sequence_step": step,
                        "sequence_length": length,
                        "game": game,
                        "episode": info["group"],
                        "controls": CONTROLS,
                        "timestamp_seconds": timing["timestamps"][t],
                        "frames": [{"image": f"frames/{identifier}/{t:07d}.png", "age_seconds": 0}],
                        "future_image": f"frames/{identifier}/{t + 1:07d}.png",
                        "source": {
                            "repository": inventory["repository"],
                            "revision": inventory["revision"],
                            "recording": identifier,
                            "action_annotation_index": t + 1,
                        },
                    }
                )
            kept.append(start)
        audit.append(
            {
                "recording": identifier,
                "split": entry["split"],
                "requested_sequences": count,
                "sequence_length": length,
                "game": game,
                "starts": kept,
                "excluded": excluded,
                "annotation_sha256": info["annotation_sha256"],
                "video_sha256": info["video_sha256"],
                "irregular_video_intervals": timing["irregular_intervals"],
            }
        )
        print(f"Sequences: {game}/{entry['split']}: {len(kept)} x {length}", flush=True)
    held, removed = set(), {}
    for split in ("test", "validation", "train"):
        grouped = {}
        for row in splits[split]:
            grouped.setdefault(row["sequence"], []).append(row)
        kept, hashes = [], set()
        for rows in grouped.values():
            paths = {r["frames"][0]["image"] for r in rows} | {r["future_image"] for r in rows}
            current = {sha256(output / p) for p in paths}
            if current & held:
                continue
            hashes |= current
            kept.extend(rows)
        removed[split] = (len(splits[split]) - len(kept)) // length
        splits[split] = kept
        held |= hashes
        if not kept:
            raise ValueError(f"Empty split after duplicate-image removal: {split}")
        with (output / f"{split}.jsonl").open("x") as f:
            for row in kept:
                f.write(json.dumps(row) + "\n")
    report = {
        "seed": seed,
        "length": length,
        "sampling": "Uniform random valid nonoverlapping windows; constant weak/generic goal per window",
        "frame_action_alignment": "image[t] -> annotation[t+1]; previous action annotation[t]",
        "future_supervision": "image[t+1], training/evaluation target only",
        "removed_duplicate_sequences": removed,
        "recordings": audit,
        "rows": {s: len(r) for s, r in splits.items()},
        "games": {s: dict(Counter(r["game"] for r in rows)) for s, rows in splits.items()},
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--length", type=int, default=32)
    a = p.parse_args()
    build(a.source, a.plan, a.output, a.count, a.length)


if __name__ == "__main__":
    main()
