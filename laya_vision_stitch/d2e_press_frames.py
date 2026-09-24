"""Extract a sharper frame at recorded mouse presses for a general click-position dataset.

For each D2E recording, presses are subsampled evenly over time (capped per game), and the
last video frame shown before each press is saved at 384 px width with its aspect ratio.
Labels are the normalized press position and button. Every game contributes the same kind
of label; nothing here is game-specific. Supervision only, never an inference input.
"""

import argparse
import bisect
import hashlib
import json
from pathlib import Path

import numpy as np

from .d2e_data import load_events
from .d2e_pointer import PointerTimeline

WIDTH = 384
MAX_FRAME_AGE = 0.035


def frames_at(video, stamps):
    """Latest decoded frame at or before each presentation time (<= MAX_FRAME_AGE old)."""
    import av

    result = {}
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for pts in sorted(set(stamps)):
            container.seek(max(0, int((pts - 1.0) / float(stream.time_base))), stream=stream)
            best = None
            for frame in container.decode(stream):
                stamp = float(frame.pts * frame.time_base)
                if stamp > pts:
                    break
                best = (stamp, frame)
            if best is not None and pts - best[0] <= MAX_FRAME_AGE:
                result[pts] = best[1].to_image().convert("RGB")
    return result


def extract(source, game, output, split, per_game, seed):
    rng = np.random.default_rng(seed)
    videos = sorted((Path(source) / game).glob("*.mkv"))
    videos = [v for v in videos if v.with_suffix(".mcap").exists()]
    pointers, total = {}, 0
    for video in videos:
        try:
            pointers[video] = PointerTimeline.from_mcap(video.with_suffix(".mcap"))
            total += len(pointers[video].presses)
        except ValueError:
            continue
    rows = []
    directory = Path(output) / "frames" / game
    directory.mkdir(parents=True, exist_ok=True)
    for video, pointer in pointers.items():
        if not pointer.presses:
            continue
        quota = max(1, round(per_game * len(pointer.presses) / max(total, 1)))
        chosen = sorted(
            rng.choice(len(pointer.presses), min(quota, len(pointer.presses)), replace=False)
        )
        screens = load_events(video.with_suffix(".mcap"))[0]
        times = [s[0] for s in screens]
        wanted = {}
        for k in chosen:
            t, x, y, button = pointer.presses[k]
            i = bisect.bisect_right(times, t) - 1
            if i < 0 or t - times[i] > MAX_FRAME_AGE:
                continue
            wanted[k] = (screens[i][1], pointer.normalize(x, y), button, t)
        images = frames_at(video, [w[0] for w in wanted.values()])
        ident = hashlib.sha256(f"{game}/{video.name}".encode()).hexdigest()[:12]
        for k, (pts, xy, button, t) in wanted.items():
            if pts not in images:
                continue
            image = images[pts]
            height = round(image.height * WIDTH / image.width)
            path = directory / f"{ident}-{k:06d}.jpg"
            image.resize((WIDTH, height)).save(path, quality=90)
            rows.append(
                {
                    "id": f"{game}-{ident}-{k}",
                    "game": game,
                    "split": split,
                    "episode": f"{game}/{video.name.split('_split_')[0]}",
                    "image": str(path.resolve()),
                    "xy": xy,
                    "button": button,
                    "log_seconds": t,
                    "source": {
                        "dataset": "open-world-agents/D2E-480p",
                        "recording": f"{game}/{video.name}",
                    },
                }
            )
    with (Path(output) / "presses.jsonl").open("a") as log:
        log.write("".join(json.dumps(r) + "\n" for r in rows))
    return {"game": game, "split": split, "presses_available": total, "saved": len(rows)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("artifacts/d2e-source"))
    p.add_argument("--games", nargs="+", required=True)
    p.add_argument("--heldout", nargs="*", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-game", type=int, default=1500)
    p.add_argument("--seed", type=int, default=20260923)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for game in args.games:
        summary = extract(
            args.source,
            game,
            args.output,
            "heldout" if game in args.heldout else "train",
            args.per_game,
            args.seed,
        )
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
