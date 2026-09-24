"""Add hindsight target points to D2E sequence caches (target-conditioned controller).

For each cached frame, the target is where the player next pressed a mouse button within
`--ahead` seconds (else the last press within `--behind` seconds), from the kept D2E
input logs. Games whose mouse motion drives the camera (see target_conditioning) use the
screen center, where their clicks act. Writes `<split>.targets.npz` and a JSON audit next
to each cache split; caches without D2E sources get no targets.
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from laya_vision_stitch.d2e_pointer import PointerTimeline, load_presses
from laya_vision_stitch.sequence_metrics import token_mouse
from laya_vision_stitch.target_conditioning import (
    MOUSE_LOOK_CORRELATION,
    hindsight_targets,
    mouse_look_correlation,
)


def normalized_presses(mcap):
    try:
        rect, presses = load_presses(mcap)
    except (ValueError, FileNotFoundError):
        return None
    timeline = PointerTimeline(rect, [(0.0, 0, 0)], presses)
    return [(t, *timeline.normalize(x, y), b) for t, x, y, b in presses]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--caches", nargs="+", required=True, help="cache:split")
    p.add_argument("--source", type=Path, default=Path("artifacts/d2e-source"))
    p.add_argument("--ahead", type=float, default=2.0)
    p.add_argument("--behind", type=float, default=1.0)
    p.add_argument("--tag", default="targets", help="Label set name: <split>.<tag>.npz")
    args = p.parse_args()
    timelines = {}
    pool = ProcessPoolExecutor(8)
    for item in args.caches:
        cache, split = item.rsplit(":", 1)
        cache = Path(cache)
        rows = [
            json.loads(line)
            for line in (cache / f"{split}.jsonl").read_text().splitlines()
            if line.strip()
        ]
        arrays = np.load(cache / f"{split}.npz")
        data = Path(json.loads((cache / "metadata.json").read_text())["data"])
        sources = {}
        for line in (data / f"{split}.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                sources[r["id"]] = r.get("source") or {}
        n = len(rows)
        target_xy = np.full((n, 2), np.nan, np.float32)
        target_dt = np.full(n, np.nan, np.float32)
        target_button = np.zeros(n, np.int8)
        games = np.array([r["game"] for r in rows])
        motion = np.abs(np.asarray(token_mouse(arrays["tokens"]), np.float32)).sum(1)
        audit = {"ahead": args.ahead, "behind": args.behind, "games": {}}
        by_recording = defaultdict(list)
        for i, r in enumerate(rows):
            source = sources.get(r["id"], {})
            if source.get("dataset") == "open-world-agents/D2E-480p":
                by_recording[source["recording"]].append((i, source["log_seconds"]))
        todo = [r for r in by_recording if r not in timelines]
        paths = [args.source / Path(r).with_suffix(".mcap") for r in todo]
        timelines.update(zip(todo, pool.map(normalized_presses, paths), strict=True))
        for recording, frames in by_recording.items():
            presses = timelines[recording]
            if presses is None:
                continue
            index = np.array([i for i, _ in frames])
            xy, dt, button = hindsight_targets(
                [t for _, t in frames], presses, args.ahead, args.behind
            )
            target_xy[index], target_dt[index], target_button[index] = xy, dt, button
        for game in sorted(set(games)):
            idx = np.flatnonzero(games == game)
            corr = mouse_look_correlation(
                arrays["images"][idx],
                motion[idx],
                arrays["sequence_index"][idx],
                arrays["steps"][idx],
            )
            mouse_look = bool(np.isfinite(corr) and corr >= MOUSE_LOOK_CORRELATION)
            labelled = idx[np.isfinite(target_xy[idx, 0])]
            if mouse_look:
                target_xy[labelled] = 0.5
            audit["games"][game] = {
                "frames": int(len(idx)),
                "with_target": int(len(labelled)),
                "mouse_look_correlation": None if not np.isfinite(corr) else round(corr, 3),
                "targets": "screen center" if mouse_look else "press position",
            }
        np.savez(
            cache / f"{split}.{args.tag}.npz",
            target_xy=target_xy,
            target_dt=target_dt,
            target_button=target_button,
        )
        (cache / f"{split}.{args.tag}.json").write_text(json.dumps(audit, indent=2) + "\n")
        print(
            json.dumps(
                {
                    item: {
                        g: [v["with_target"], v["frames"], v["targets"]]
                        for g, v in audit["games"].items()
                    }
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
