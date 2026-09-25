"""Import non-D2E gameplay recordings (video plus input logs) as 20 FPS sequence windows.

Supported recorders:
- `ck3`: Ethosoft CK3 recorder (screen.mp4, input_events.csv, metadata.json); absolute
  cursor only, so relative motion is the cursor displacement.
- `bgi`: the recorder used by the `rl-game-traces-*` datasets (video.mkv,
  video-km-frames.json, videoStartTime.txt); raw mouse motion and absolute cursor.

Rows match `d2e_sequences` (192x192 released preprocessing, majority-time holds, raw
motion / 512) and add mouse-wheel notches as `scroll_up` / `scroll_down` and pointer
labels (cursor and press position, normalized to the screen). No game state is used.
"""

import argparse
import bisect
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .d2e_data import VK, Timeline
from .d2e_pointer import PointerTimeline
from .d2e_sequences import STEP, WINDOW, decode_window
from .p2p_resize import resize_rgb

# These recorders capture at 30 FPS (CK3, with dropped frames) or 60 FPS; a frame up to one
# dropped 30 FPS frame old (70 ms) still describes the 50 ms control step it starts.
MAX_FRAME_AGE = 0.07

PYNPUT = {
    "space": "space",
    "esc": "escape",
    "enter": "enter",
    "tab": "tab",
    "backspace": "backspace",
    "shift": "shift",
    "shift_l": "shift",
    "shift_r": "shift",
    "ctrl": "ctrl",
    "ctrl_l": "ctrl",
    "ctrl_r": "ctrl",
    "alt": "alt",
    "alt_l": "alt",
    "alt_r": "alt",
    "alt_gr": "alt",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
}
BGI_BUTTONS = {
    "MouseLeft": "mouse_left",
    "MouseRight": "mouse_right",
    "MouseMiddle": "mouse_middle",
}


class InputLog:
    """Event streams on the video clock (seconds)."""

    def __init__(self, width, height):
        self.size = (width, height)
        self.changes, self.raw, self.cursor, self.wheel, self.presses = [], [], [], [], []
        self.unknown = Counter()

    def finish(self):
        """Held-control states after each change, for `d2e_data.Timeline`."""
        held, states = set(), []
        for t, name, down in sorted(self.changes, key=lambda c: c[0]):
            (held.add if down else held.discard)(name)
            states.append((t, sorted(held)))
        self.states = states
        self.raw.sort()
        self.cursor.sort()
        self.wheel.sort()
        self.presses.sort()
        self.wheel_times = [w[0] for w in self.wheel]
        self.timeline = Timeline(self.states, self.raw)
        return self

    def action(self, start):
        action, clipped = self.timeline.action(start, STEP, 512)
        lo = bisect.bisect_left(self.wheel_times, start)
        hi = bisect.bisect_left(self.wheel_times, start + STEP)
        notches = [w[1] for w in self.wheel[lo:hi]]
        extra = (["scroll_up"] if any(n > 0 for n in notches) else []) + (
            ["scroll_down"] if any(n < 0 for n in notches) else []
        )
        action["buttons"] = sorted(set(action["buttons"]) | set(extra))
        return action, clipped


def ck3_log(session):
    """CK3 recorder: times shifted onto the video clock with the recorded offset."""
    meta = json.loads((session / "metadata.json").read_text())
    offset = float(meta["video"]["ffmpeg_popen_sec_from_t0"])
    log = InputLog(1920, 1080)
    last = None
    for e in csv.DictReader((session / "input_events.csv").open()):
        t = float(e["t_sec"]) - offset
        kind = e["event"]
        if kind == "mouse_move":
            x, y = float(e["x"]), float(e["y"])
            if last is not None:
                log.raw.append((t, x - last[0], y - last[1]))
            last = (x, y)
            log.cursor.append((t, x, y))
        elif kind == "mouse_click":
            name = "mouse_" + e["button"]
            down = e["pressed"] == "True"
            log.changes.append((t, name, down))
            if down and e["x"]:
                log.presses.append((t, float(e["x"]), float(e["y"]), e["button"]))
        elif kind == "mouse_scroll":
            if e["dy"] and float(e["dy"]):
                log.wheel.append((t, float(e["dy"])))
        elif kind in ("key_press", "key_release"):
            key = e["key"].strip("'")
            name = PYNPUT.get(key.removeprefix("Key.")) if key.startswith("Key.") else key.lower()
            if name is None or (len(name) == 1 and not name.isalnum()):
                log.unknown[key] += 1
                continue
            log.changes.append((t, name, kind == "key_press"))
    return log.finish()


def bgi_log(session):
    """`rl-game-traces` recorder: per-video-frame event lists with epoch timestamps."""
    info = json.loads((session / "systemInfo.json").read_text())
    start = int((session / "videoStartTime.txt").read_text().strip())
    log = InputLog(int(info["width"]), int(info["height"]))
    cursor = None
    for frame in json.loads((session / "video-km-frames.json").read_text()):
        for e in frame["events"]:
            t = (int(e["timestamp_ns"]) - start) / 1e9
            kind = e["type"]
            if kind == "mouse_move_by":
                log.raw.append((t, float(e["dx"]), float(e["dy"])))
            elif kind == "mouse_move_to":
                cursor = (float(e["x"]), float(e["y"]))
                log.cursor.append((t, *cursor))
            elif kind in ("mouse_down", "mouse_up"):
                name = BGI_BUTTONS.get(e.get("key_name"))
                if name is None:
                    log.unknown[e.get("key_name")] += 1
                    continue
                log.changes.append((t, name, kind == "mouse_down"))
                if kind == "mouse_down" and cursor is not None:
                    log.presses.append((t, *cursor, name.removeprefix("mouse_")))
            elif kind == "mouse_wheel":
                log.wheel.append((t, float(e["wheel"]) / 120))
            elif kind in ("key_down", "key_up"):
                name = VK.get(int(e["key"]))
                if name is None:
                    log.unknown[e.get("key_name")] += 1
                    continue
                log.changes.append((t, name, kind == "key_down"))
    return log.finish()


ENCRYPTED = b"BGI_JSH_V1"  # some rl-game-traces sessions ship encrypted input logs


def readable(session, recorder):
    """False when the session's input log is encrypted (it cannot be labelled)."""
    if recorder != "bgi":
        return True
    with (Path(session) / "video-km-frames.json").open("rb") as f:
        return not f.read(len(ENCRYPTED)).startswith(ENCRYPTED)


def control_events(log, min_hold=0.15, min_motion=20.0, gap=1.0):
    """Starts of drags and of mouse-wheel bursts, in seconds.

    A drag is a mouse button held at least `min_hold` seconds with at least `min_motion`
    pixels of motion while held; wheel notches less than `gap` seconds apart are one burst.
    """
    raw_t = np.array([r[0] for r in log.raw])
    moved = np.concatenate([[0.0], np.cumsum([abs(r[1]) + abs(r[2]) for r in log.raw])])
    events, down = [], {}
    for t, name, is_down in sorted(log.changes, key=lambda c: c[0]):
        if not name.startswith("mouse_"):
            continue
        if is_down:
            down[name] = t
        elif name in down:
            start = down.pop(name)
            lo, hi = np.searchsorted(raw_t, [start, t])
            if t - start >= min_hold and moved[hi] - moved[lo] >= min_motion:
                events.append(start)
    last = -np.inf
    for t, _ in log.wheel:
        if t - last > gap:
            events.append(t)
        last = t
    return sorted(events)


RECORDERS = {
    "ck3": (ck3_log, "screen.mp4"),
    "bgi": (bgi_log, "video.mkv"),
}


def build(sessions, output, game, recorder, windows_per_minute, seed, event_windows_per_minute=0.0):
    """sessions: [(split, session_dir)]. Writes frames, `<split>.jsonl` and `audit.json`.

    Windows are spread uniformly at `windows_per_minute`. Training sessions additionally
    get up to `event_windows_per_minute` windows starting 0.25-1 s before drag or wheel
    events (`control_events`), so rare camera controls are seen often enough; those rows
    are tagged `"sampling": "event"`. Test sessions get the same event windows in a
    separate `test_events` split (a targeted camera-control test); `test` and
    `validation` stay uniform.
    """
    import av

    output = Path(output).resolve()
    rng = np.random.default_rng(seed)
    parse, video_name = RECORDERS[recorder]
    rows = defaultdict(list)
    audit = {"game": game, "recorder": recorder, "seed": seed, "sessions": []}
    for split, session in sessions:
        session = Path(session)
        video = session / video_name
        if not video.exists():
            audit["sessions"].append({"session": session.name, "skipped": "no video"})
            continue
        if not readable(session, recorder):
            audit["sessions"].append({"session": session.name, "skipped": "encrypted input log"})
            continue
        log = parse(session)
        directory = output / "frames" / game / session.name
        directory.mkdir(parents=True, exist_ok=True)
        accepted = 0
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            # Cursor positions share the captured screen's pixel space.
            size = (stream.codec_context.width, stream.codec_context.height)
            pointer = PointerTimeline((0, 0, *size), log.cursor or [(0.0, 0, 0)], log.presses)
            duration = (
                float(stream.duration * stream.time_base)
                if stream.duration
                else float(container.duration / 1e6)
            )
            low, high = STEP, duration - (WINDOW + 1) * STEP
            count = max(1, int((high - low) / 60 * windows_per_minute)) if high > low else 0
            edges = np.linspace(low, high, count + 1)
            starts = [
                (rng.uniform(a, max(a, b - WINDOW * STEP)), "uniform")
                for a, b in zip(edges[:-1], edges[1:], strict=True)
            ]
            if split in ("train", "test") and event_windows_per_minute > 0 and high > low:
                chosen, last = [], -np.inf
                for t in control_events(log):
                    start = t - rng.uniform(0.25, 1.0)
                    if low <= start <= high and start - last >= WINDOW * STEP:
                        chosen.append(start)
                        last = start
                limit = int((high - low) / 60 * event_windows_per_minute)
                if len(chosen) > limit:
                    keep = np.linspace(0, len(chosen) - 1, limit).round().astype(int)
                    chosen = [chosen[i] for i in keep]
                starts += [(start, "event") for start in chosen]
            for w, (start, sampling) in enumerate(starts):
                target = "test_events" if split == "test" and sampling == "event" else split
                times = start + STEP * np.arange(WINDOW)
                frames = decode_window(container, stream, [float(t) for t in times], MAX_FRAME_AGE)
                if frames is None:
                    continue
                name = f"{game}-{session.name}-{w}"
                previous = log.action(start - STEP)[0]
                for k, (t, frame) in enumerate(zip(times, frames, strict=True)):
                    path = directory / f"{w:05d}-{k:02d}.png"
                    Image.fromarray(resize_rgb(np.asarray(frame.to_image().convert("RGB")))).save(
                        path
                    )
                    action, clipped = log.action(float(t))
                    rows[target].append(
                        {
                            "id": f"{name}-{k}",
                            "sequence": name,
                            "sequence_step": k,
                            "sequence_length": WINDOW,
                            "game": game,
                            "episode": f"{game}/{session.name}",
                            "goal": f"Continue playing {game.replace('_', ' ')}.",
                            "controls": "Physical keys and mouse buttons; mouse_delta is raw motion / 512. Act for 50 ms.",
                            "timestamp_seconds": float(t - times[0]),
                            "frames": [{"image": str(path), "age_seconds": 0}],
                            "action": action,
                            "previous_actions": [previous],
                            "mouse_clipped": clipped,
                            "pointer": pointer.label(float(t), STEP),
                            "sampling": sampling,
                            "source": {
                                "dataset": recorder,
                                "recording": f"{game}/{session.name}",
                                "log_seconds": float(t),
                            },
                        }
                    )
                    previous = action
                accepted += 1
        audit["sessions"].append(
            {
                "session": session.name,
                "split": split,
                "windows": accepted,
                "requested_windows": len(starts),
                "event_windows": sum(1 for _, kind in starts if kind == "event"),
                "duration_seconds": duration,
                "screen": list(size),
                "unknown_controls": dict(log.unknown),
            }
        )
    for split, split_rows in rows.items():
        (output / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in split_rows))
    counts = {s: Counter(b for r in rs for b in r["action"]["buttons"]) for s, rs in rows.items()}
    audit["frames"] = {s: len(rs) for s, rs in rows.items()}
    audit["control_counts"] = {s: dict(c.most_common()) for s, c in counts.items()}
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recorder", choices=sorted(RECORDERS), required=True)
    p.add_argument("--game", required=True)
    p.add_argument("--sessions", type=Path, required=True, help="JSON {split: [session dirs]}")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--windows-per-minute", type=float, default=4.0)
    p.add_argument("--event-windows-per-minute", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=20260925)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.sessions.read_text())
    sessions = [(split, d) for split, dirs in plan.items() for d in dirs]
    audit = build(
        sessions,
        args.output,
        args.game,
        args.recorder,
        args.windows_per_minute,
        args.seed,
        args.event_windows_per_minute,
    )
    print(json.dumps({"frames": audit["frames"]}))


if __name__ == "__main__":
    main()
