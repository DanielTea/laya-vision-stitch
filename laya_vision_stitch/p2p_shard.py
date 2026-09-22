"""Safely inventory one pinned public archive and build bounded session splits."""

import argparse
import hashlib
import json
import shutil
import tarfile
import uuid
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np

from .p2p_data import CONTROLS, candidates, extract_frames, read_annotation, sha256

REVISION = "1553d6b8190d270dcd256528ffa2066098225dba"


def member_identity(member):
    parts = PurePosixPath(member.name).parts
    if not member.isfile() or len(parts) < 2 or parts[-1] not in {"annotation.proto", "video.mp4"}:
        return None
    try:
        identifier = str(uuid.UUID(parts[-2]))
    except ValueError:
        return None
    if parts[-2] != identifier or ".." in parts or PurePosixPath(member.name).is_absolute():
        return None
    return identifier, parts[-1]


def inventory(archive, output):
    output.mkdir(parents=True, exist_ok=False)
    rows, video_sizes = [], {}
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            identity = member_identity(member)
            if identity is None:
                continue
            identifier, name = identity
            if name == "video.mp4":
                video_sizes[identifier] = member.size
                continue
            if member.size > 100_000_000:
                raise ValueError("Unexpectedly large annotation")
            target = output / "dataset" / identifier / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination)
            a = read_annotation(target)
            # Grouping IDs are pseudonyms; do not copy player identities into reports.
            group = a.metadata.group_name or a.metadata.id or identifier
            actor = a.metadata.user
            rows.append(
                {
                    "recording": identifier,
                    "game": a.metadata.env.env,
                    "subtype": a.metadata.env.env_subtype,
                    "group": hashlib.sha256(group.encode()).hexdigest(),
                    "actor": hashlib.sha256(actor.encode()).hexdigest() if actor else None,
                    "fps": a.metadata.frames_per_second,
                    "frames": len(a.frame_annotations),
                    "human_frames": sum(f.user_action.is_known for f in a.frame_annotations),
                    "raw_mouse_frames": sum(
                        f.user_action.is_known and f.user_action.mouse.HasField("mouse_delta_px")
                        for f in a.frame_annotations
                    ),
                    "system_frames": sum(f.system_action.is_known for f in a.frame_annotations),
                    "text_segments": sum(len(f.frame_text_annotation) for f in a.frame_annotations),
                }
            )
            if len(rows) % 25 == 0:
                print(f"Indexed {len(rows)} annotations", flush=True)
    for row in rows:
        row["video_bytes"] = video_sizes.get(row["recording"], 0)
    report = {
        "repository": "elefantai/p2p-full-data",
        "revision": REVISION,
        "archive": archive.name,
        "archive_sha256": sha256(archive),
        "recordings": rows,
    }
    (output / "inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "recordings": len(rows),
                "games": Counter(r["game"] for r in rows),
                "eligible_games": Counter(
                    r["game"]
                    for r in rows
                    if r["fps"] == 20
                    and r["human_frames"] >= 0.95 * r["frames"]
                    and r["raw_mouse_frames"] >= 0.95 * r["frames"]
                    and r["frames"] >= 800
                    and not r["system_frames"]
                ),
            },
            indent=2,
        )
    )


def build(archive, source, plan, output, local_source=False):
    """A reviewed JSON plan names complete recordings and their split, never frame ranges."""
    selection = json.loads(plan.read_text())
    inventory = json.loads((source / "inventory.json").read_text())
    if not local_source and (
        inventory["archive"] != archive.name or inventory["archive_sha256"] != sha256(archive)
    ):
        raise ValueError("Archive differs from inventory")
    indexed = {r["recording"]: r for r in inventory["recordings"]}
    chosen = {r["recording"]: r for r in selection["recordings"]}
    if len(chosen) != len(selection["recordings"]) or set(chosen) - set(indexed):
        raise ValueError("Duplicate or unknown recording in plan")
    groups = {}
    for identifier, entry in chosen.items():
        if entry["split"] not in {"train", "validation", "test"} or entry["count"] < 1:
            raise ValueError("Invalid split/sample count")
        group = indexed[identifier]["group"]
        if group in groups and groups[group] != entry["split"]:
            raise ValueError("Session group crosses split boundary")
        groups[group] = entry["split"]
    output.mkdir(parents=True, exist_ok=False)
    if local_source:
        for identifier in chosen:
            for name, key in (
                ("annotation.proto", "annotation_sha256"),
                ("video.mp4", "video_sha256"),
            ):
                if sha256(source / "dataset" / identifier / name) != indexed[identifier][key]:
                    raise ValueError("Local member differs from streamed source audit")
    else:
        extract_videos(archive, source, chosen)
    splits, audit = {s: [] for s in ("train", "validation", "test")}, []
    rng = np.random.default_rng(71)
    for identifier, entry in chosen.items():
        p = source / "dataset" / identifier
        annotation = read_annotation(p / "annotation.proto")
        rows, excluded = candidates(annotation, include_unlabelled=True)
        if not rows:
            raise ValueError(f"No valid human controls: {identifier}")
        indices = sorted(rng.choice(len(rows), min(entry["count"], len(rows)), replace=False))
        rows = [rows[i] for i in indices]
        frames = {i for r in rows for i in (r["frame_index"] - 4, r["frame_index"])}
        timing = {}
        extract_frames(
            p / "video.mp4",
            frames,
            output / "frames" / identifier,
            len(annotation.frame_annotations),
            timing=timing,
        )
        valid = []
        for row in rows:
            index = row["frame_index"]
            if any(index - 4 <= i <= index + 1 for i in timing["irregular_intervals"]):
                excluded["irregular_video_interval"] = (
                    excluded.get("irregular_video_interval", 0) + 1
                )
            else:
                valid.append(row)
        rows = valid
        game = annotation.metadata.env.env
        if annotation.metadata.env.env_subtype:
            game += "/" + annotation.metadata.env.env_subtype
        for row in rows:
            frame_index = row["frame_index"]
            row.update(
                {
                    "id": f"p2p-{identifier}-{frame_index}",
                    "game": game,
                    "episode": indexed[identifier]["group"],
                    "controls": CONTROLS,
                    "frames": [
                        {
                            "image": f"frames/{identifier}/{i:07d}.png",
                            "age_seconds": timing["timestamps"][frame_index]
                            - timing["timestamps"][i],
                        }
                        for i in (frame_index - 4, frame_index)
                    ],
                    "choices": {"act": "Take the next action.", "wait": "Wait."},
                    "action_supervision": ["buttons", "relative_mouse"],
                    "source": {
                        "repository": "elefantai/p2p-full-data",
                        "revision": REVISION,
                        "archive": indexed[identifier].get("archive", archive.name),
                        "action_annotation_index": frame_index + 1,
                    },
                }
            )
            splits[entry["split"]].append(row)
        audit.append(
            {
                **indexed[identifier],
                **entry,
                "selected": len(rows),
                "excluded": excluded,
                "irregular_video_intervals": timing["irregular_intervals"],
                "annotation_sha256": sha256(p / "annotation.proto"),
                "video_sha256": sha256(p / "video.mp4"),
            }
        )
        print(f"Imported {len(rows)}: {game} / {entry['split']}", flush=True)
    # Prefer held-out images; discard overlapping examples from earlier splits.
    held_hashes, removed = set(), {}
    for split in ("test", "validation", "train"):
        kept, split_hashes = [], set()
        for row in splits[split]:
            hashes = {sha256(output / f["image"]) for f in row["frames"]}
            if hashes & held_hashes:
                continue
            kept.append(row)
            split_hashes.update(hashes)
        removed[split] = len(splits[split]) - len(kept)
        splits[split] = kept
        held_hashes.update(split_hashes)
    for split, rows in splits.items():
        if rows:
            (output / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (output / "audit.json").write_text(
        json.dumps(
            {
                "plan": selection,
                "recordings": audit,
                "cross_split_duplicate_examples_removed": removed,
                "source": {k: v for k, v in inventory.items() if k != "recordings"},
            },
            indent=2,
        )
        + "\n"
    )
    (output / "SOURCE_DATASET_CARD.md").write_bytes((source / "README.md").read_bytes())


def extract_videos(archive, source, chosen):
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            identity = member_identity(member)
            if identity and identity[0] in chosen and identity[1] == "video.mp4":
                target = source / "dataset" / identity[0] / "video.mp4"
                if not target.exists():
                    with tar.extractfile(member) as src, target.open("xb") as dest:
                        shutil.copyfileobj(src, dest)
                print(f"Video ready: {identity[0]}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("inventory", "build"))
    p.add_argument(
        "--archive", type=Path, default=Path("artifacts/p2p-full-subset/dataset/batch_00002.tar.gz")
    )
    p.add_argument("--source", type=Path, default=Path("artifacts/p2p-shard-source"))
    p.add_argument("--plan", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--local-source", action="store_true")
    args = p.parse_args()
    if args.mode == "inventory":
        inventory(args.archive, args.source)
    else:
        if args.plan is None or args.output is None:
            p.error("build requires --plan and --output")
        build(args.archive, args.source, args.plan, args.output, args.local_source)


if __name__ == "__main__":
    main()
