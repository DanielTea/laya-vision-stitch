"""Convert downloaded CK3 / rl-game-traces sessions into extended-vocabulary sequence caches.

Splits: CK3 uses its published date split (train to 2026-07-20, validation to 07-27, test
after); other datasets are split chronologically per session (latest test, second latest
validation, rest train). After the cache is built the videos are deleted (re-downloadable);
input logs stay. Run once per dataset after its download has finished.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from laya_vision_stitch.recorded_sequences import build, readable


def plan(dataset, source, sessions=None, all_train=False):
    sessions = sorted(
        p
        for p in source.iterdir()
        if p.is_dir() and not p.name.startswith(".") and (sessions is None or p.name in sessions)
    )
    recorder = "ck3" if dataset == "ck3" else "bgi"
    sessions = [s for s in sessions if readable(s, recorder)]
    if dataset == "ck3":
        split = {"train": [], "validation": [], "test": []}
        for s in sessions:
            date = s.name.split("_")[1]
            key = "train" if date <= "20260720" else "validation" if date <= "20260727" else "test"
            split[key].append(str(s))
        return split
    names = [str(s) for s in sessions]
    if all_train or len(names) < 3:
        return {"train": names}
    if len(names) == 3:
        return {"train": names[:-1], "test": [names[-1]]}
    return {"train": names[:-2], "validation": [names[-2]], "test": [names[-1]]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="ck3 or an rl-game-traces name")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--game", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--windows-per-minute", type=float, default=2.0)
    p.add_argument(
        "--event-windows-per-minute",
        type=float,
        default=0.0,
        help="Extra training windows around drag and wheel events (recorded_sequences.build)",
    )
    p.add_argument("--bundle", type=Path, default=Path("artifacts/laya-p2p-bridge-001/bundle"))
    p.add_argument("--keep-video", action="store_true")
    p.add_argument("--sessions", nargs="+", help="Only these session directory names")
    p.add_argument("--all-train", action="store_true", help="Put every session in train")
    p.add_argument("--split", help="Put every selected session in this split")
    args = p.parse_args()
    splits = plan(args.dataset, args.source, args.sessions, args.all_train)
    if args.split:
        splits = {args.split: [d for dirs in splits.values() for d in dirs]}
    args.data.mkdir(parents=True, exist_ok=False)
    sessions = [(split, d) for split, dirs in splits.items() for d in dirs]
    recorder = "ck3" if args.dataset == "ck3" else "bgi"
    audit = build(
        sessions,
        args.data,
        args.game,
        recorder,
        args.windows_per_minute,
        20260925,
        args.event_windows_per_minute,
    )
    print(json.dumps({"converted": args.game, "frames": audit["frames"]}), flush=True)
    # test_events overlaps test (same held-out sessions), so it gets its own cache: the
    # builder rejects frames shared across splits of one cache.
    groups = [(args.cache, sorted(k for k in audit["frames"] if k != "test_events"))]
    if "test_events" in audit["frames"]:
        groups.append((args.cache.with_name(args.cache.name + "-events"), ["test_events"]))
    for cache, split_names in groups:
        if not split_names:
            continue
        subprocess.run(
            [
                sys.executable,
                "scripts/build_sequence_cache.py",
                "--bundle",
                str(args.bundle),
                "--data",
                str(args.data),
                "--output",
                str(cache),
                "--splits",
                *split_names,
                "--vocabulary",
                "extended",
                "--spatial-press",
            ],
            check=True,
        )
    if not args.keep_video:
        removed = 0
        converted = {Path(d).name for dirs in splits.values() for d in dirs}
        for name in ("screen.mp4", "video.mkv"):
            for video in args.source.glob(f"*/{name}"):
                if args.sessions and video.parent.name not in converted:
                    continue
                removed += video.stat().st_size
                video.unlink()
        print(json.dumps({"deleted_video_gb": round(removed / 1e9, 1)}), flush=True)


if __name__ == "__main__":
    main()
