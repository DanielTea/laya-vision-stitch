"""Import D2E recordings as consecutive 20 FPS windows in the P2P sequence format.

Frames are stored after the released 192x192 Hamming preprocessing, so the model sees
the same pixels as fresh inference. Sessions are split chronologically within each game
(latest -> test, second latest -> validation). Labels are recorded human controls; goals
are generic game text, not intent annotations. No environment state enters the model.
"""

import argparse
import bisect
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .d2e_data import Timeline, file_digest, load_events
from .d2e_pointer import PointerTimeline
from .p2p_resize import resize_rgb

STEP = 0.05
WINDOW = 32
# Screen capture in some D2E games runs near 50 FPS with jitter; a frame up to 35 ms old
# is still well inside one 50 ms control step.
MAX_FRAME_AGE = 0.035


def sessions(source, games):
    grouped = defaultdict(list)
    for game in games:
        for video in sorted((Path(source) / game).glob("*.mkv")):
            if not video.with_suffix(".mcap").exists():
                continue
            grouped[(game, video.name.split("_split_")[0])].append(video)
    return grouped


def assign(grouped, mode):
    """Chronological per-game session split, or one split for every session."""
    result = {}
    by_game = defaultdict(list)
    for game, session in grouped:
        by_game[game].append(session)
    for game, names in by_game.items():
        names.sort()
        for i, name in enumerate(names):
            if mode != "chronological":
                result[(game, name)] = mode
            elif len(names) >= 3 and i == len(names) - 1:
                result[(game, name)] = "test"
            elif len(names) >= 3 and i == len(names) - 2:
                result[(game, name)] = "validation"
            else:
                result[(game, name)] = "train"
    return result


def decode_window(container, stream, pts_list):
    """Frames at or before each requested PTS (<=MAX_FRAME_AGE old); None if any is missing."""
    container.seek(max(0, int((pts_list[0] - 1.0) / float(stream.time_base))), stream=stream)
    wanted, result, previous = list(pts_list), [], None
    for frame in container.decode(stream):
        stamp = float(frame.pts * frame.time_base)
        while wanted and stamp > wanted[0]:
            if previous is None or wanted[0] - previous[0] > MAX_FRAME_AGE:
                return None
            result.append(previous[1])
            wanted.pop(0)
        if not wanted:
            break
        previous = (stamp, frame)
    while wanted and previous is not None and wanted[0] - previous[0] <= MAX_FRAME_AGE:
        result.append(previous[1])
        wanted.pop(0)
    return result if not wanted else None


def fovea_crop(rgb, fraction=0.5):
    """Central crop covering `fraction` of each dimension, resized like the full frame."""
    h, w = rgb.shape[:2]
    ch, cw = int(round(h * fraction)), int(round(w * fraction))
    top, left = (h - ch) // 2, (w - cw) // 2
    return resize_rgb(np.ascontiguousarray(rgb[top : top + ch, left : left + cw]))


def build(
    source,
    games,
    output,
    windows_per_minute,
    mode,
    seed,
    mouse_scale=512,
    fovea=False,
    max_windows=None,
):
    """Import windows; `max_windows` caps each game's total so no game dominates."""
    import av

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    grouped = sessions(source, games)
    splits = assign(grouped, mode)
    rows = defaultdict(list)
    audit = {
        "games": games,
        "mode": mode,
        "seed": seed,
        "max_frame_age_seconds": MAX_FRAME_AGE,
        "recordings": [],
    }
    loaded, minutes = {}, defaultdict(float)
    for (game, _), videos in grouped.items():
        for video in videos:
            events = load_events(video.with_suffix(".mcap"))
            loaded[video] = events
            if events[0]:
                minutes[game] += (events[0][-1][0] - events[0][0][0]) / 60
    rate = {
        game: windows_per_minute
        if not max_windows or m * windows_per_minute <= max_windows
        else max_windows / m
        for game, m in minutes.items()
    }
    audit["windows_per_minute_by_game"] = rate
    for (game, session), videos in sorted(grouped.items()):
        split = splits[(game, session)]
        for video in videos:
            screens, states, mouse, unknown = loaded.pop(video)
            if screens and states and not mouse:
                # Keyboard-only play: no relative mouse motion over the recording.
                mouse = [(screens[0][0], 0, 0), (screens[-1][0], 0, 0)]
            try:
                pointer = PointerTimeline.from_mcap(video.with_suffix(".mcap"))
            except ValueError:
                pointer = None
            if not states or not mouse or len(screens) < WINDOW + 2:
                audit["recordings"].append(
                    {"video": video.name, "skipped": "no synchronized input"}
                )
                continue
            times = np.array([s[0] for s in screens])
            pts = np.array([s[1] for s in screens])
            timeline = Timeline(states, mouse)
            low = max(times[0], states[0][0], mouse[0][0]) + STEP
            high = min(times[-1], states[-1][0], mouse[-1][0]) - (WINDOW + 1) * STEP
            if high <= low:
                continue
            count = max(1, int((high - low) / 60 * rate[game]))
            edges = np.linspace(low, high, count + 1)
            ident = hashlib.sha256(f"{game}/{video.name}".encode()).hexdigest()[:12]
            directory = output / "frames" / game / ident
            directory.mkdir(parents=True)
            accepted = 0
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                for w, (a, b) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
                    start = rng.uniform(a, max(a, b - WINDOW * STEP))
                    log_times = start + STEP * np.arange(WINDOW)
                    index = [bisect.bisect_right(times, t) - 1 for t in log_times]
                    if min(index) < 0 or np.any(log_times - times[index] > MAX_FRAME_AGE):
                        continue
                    frames = decode_window(container, stream, [float(pts[i]) for i in index])
                    if frames is None:
                        continue
                    name = f"{game}-{ident}-{w}"
                    previous = timeline.action(start - STEP, STEP, mouse_scale)[0]
                    for k, (t, frame) in enumerate(zip(log_times, frames, strict=True)):
                        rgb = np.asarray(frame.to_image().convert("RGB"))
                        path = directory / f"{w:05d}-{k:02d}.png"
                        Image.fromarray(resize_rgb(rgb)).save(path)
                        image = {"image": str(path), "age_seconds": 0}
                        if fovea:
                            crop = directory / f"{w:05d}-{k:02d}-fovea.png"
                            Image.fromarray(fovea_crop(rgb)).save(crop)
                            image["fovea"] = str(crop)
                        action, clipped = timeline.action(float(t), STEP, mouse_scale)
                        rows[split].append(
                            {
                                "id": f"{name}-{k}",
                                "sequence": name,
                                "sequence_step": k,
                                "sequence_length": WINDOW,
                                "game": game,
                                "episode": f"{game}/{session}",
                                "goal": f"Continue playing {game.replace('_', ' ')}.",
                                "controls": "Physical keys and mouse buttons; mouse_delta is raw motion / 512. Act for 50 ms.",
                                "timestamp_seconds": float(t - log_times[0]),
                                "frames": [image],
                                "action": action,
                                "previous_actions": [previous],
                                "mouse_clipped": clipped,
                                "pointer": None
                                if pointer is None
                                else pointer.label(float(t), STEP),
                                "source": {
                                    "dataset": "open-world-agents/D2E-480p",
                                    "recording": f"{game}/{video.name}",
                                    "log_seconds": float(t),
                                },
                            }
                        )
                        previous = action
                    accepted += 1
            audit["recordings"].append(
                {
                    "video": f"{game}/{video.name}",
                    "split": split,
                    "windows": accepted,
                    "requested_windows": count,
                    "duration_seconds": float(times[-1] - times[0]),
                    "unknown_key_state_counts": unknown,
                    "video_sha256": file_digest(video),
                    "mcap_sha256": file_digest(video.with_suffix(".mcap")),
                }
            )
            print(json.dumps(audit["recordings"][-1]), flush=True)
    for split, items in rows.items():
        (output / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items))
    audit["frames"] = {s: len(v) for s, v in rows.items()}
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("artifacts/d2e-source"))
    p.add_argument("--games", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--windows-per-minute", type=float, default=2.0)
    p.add_argument(
        "--mode", default="chronological", help="'chronological' or a single split name for all"
    )
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--fovea", action="store_true", help="Also store a central half-size crop")
    p.add_argument("--max-windows", type=int, help="Cap on windows per game")
    args = p.parse_args()
    audit = build(
        args.source,
        args.games,
        args.output,
        args.windows_per_minute,
        args.mode,
        args.seed,
        fovea=args.fovea,
        max_windows=args.max_windows,
    )
    print(json.dumps(audit["frames"]))


if __name__ == "__main__":
    main()
