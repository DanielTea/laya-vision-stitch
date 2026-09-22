"""Passive, window-scoped screenshots and physical controls for imitation learning.

No model and no input posting. Raw presentation/event timestamps are preserved;
the exporter labels the 50 ms interval following each screenshot, not its past.
"""

import argparse
import bisect
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .hordes_demonstrations import GOAL, HORDES_CONTROLS

KEYS = {
    13: "w",
    0: "a",
    1: "s",
    2: "d",
    49: "space",
    48: "tab",
    18: "1",
    19: "2",
    20: "3",
    21: "4",
    23: "5",
    22: "6",
}


def interval_action(events, start, end):
    """Retain held time and press edges; do not infer desired gameplay actions."""
    if not start < end:
        raise ValueError("Control interval must be positive")
    held, duration, pressed, released = set(), {}, set(), set()
    delta, pointer, cursor = [0.0, 0.0], None, start
    for event in events:
        timestamp = event["timestamp"]
        if timestamp >= end:
            break
        if timestamp >= start:
            for button in held:
                duration[button] = duration.get(button, 0.0) + timestamp - cursor
            cursor = timestamp
        kind = event["kind"]
        if kind == "state":
            held = set(event["buttons"])
        elif kind in ("down", "up"):
            button = event["button"]
            if kind == "down":
                held.add(button)
                if timestamp >= start:
                    pressed.add(button)
            else:
                held.discard(button)
                if timestamp >= start:
                    released.add(button)
        if timestamp >= start and "delta" in event:
            delta = [a + b for a, b in zip(delta, event["delta"], strict=True)]
        if timestamp >= start and kind == "down" and event.get("pointer_xy") is not None:
            pointer = event["pointer_xy"]
    for button in held:
        duration[button] = duration.get(button, 0.0) + end - cursor
    if any(abs(v) > 512 for v in delta):
        raise ValueError("Raw mouse motion exceeds the model vocabulary; do not silently clip")
    action = {
        "buttons": sorted({b for b, seconds in duration.items() if seconds > 0} | pressed),
        "mouse_delta": [v / 512 for v in delta],
        "duration_seconds": 0.05,
    }
    if pointer is not None:
        action["pointer_xy"] = pointer
    return action, {
        "held_seconds": duration,
        "pressed": sorted(pressed),
        "released": sorted(released),
    }


def export(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    frames = [json.loads(line) for line in (directory / "frames.jsonl").read_text().splitlines()]
    events = sorted(
        [json.loads(line) for line in (directory / "controls.jsonl").read_text().splitlines()],
        key=lambda e: e["timestamp"],
    )
    if not events or events[0]["kind"] != "state":
        raise ValueError("Recording needs an initial physical button state")
    timestamps, states, held = [], [], set()
    for event in events:
        if event["kind"] == "state":
            held = set(event["buttons"])
        elif event["kind"] == "down":
            held.add(event["button"])
        elif event["kind"] == "up":
            held.discard(event["button"])
        timestamps.append(event["timestamp"])
        states.append(sorted(held))
    rows, rejected, previous = [], {}, None
    for index, frame in enumerate(frames):
        start, end = frame["timestamp"], frame["timestamp"] + 0.05
        if start < events[0]["timestamp"] or end > summary["ended_at"]:
            rejected["incomplete_interval"] = rejected.get("incomplete_interval", 0) + 1
            continue
        try:
            left, right = bisect.bisect_left(timestamps, start), bisect.bisect_left(timestamps, end)
            initial = {
                "timestamp": start,
                "kind": "state",
                "buttons": states[left - 1] if left else [],
            }
            action, details = interval_action([initial, *events[left:right]], start, end)
        except ValueError:
            rejected["mouse_out_of_vocabulary"] = rejected.get("mouse_out_of_vocabulary", 0) + 1
            continue
        image = directory / frame["image"]
        row = {
            "id": f"{directory.name}-{index:06d}",
            "game": config["game"],
            "episode": directory.name,
            "goal": config["goal"],
            "controls": config["controls"],
            "frames": [
                {
                    "image": str(image.resolve()),
                    "age_seconds": 0,
                    "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                }
            ],
            "previous_actions": [],
            "action": action,
            "timestamp_seconds": start - events[0]["timestamp"],
            "provenance": {
                "type": "human_physical_controls",
                "recording": str(directory.resolve()),
                "verified_optimal": False,
                "interval_start": start,
                "interval_end": end,
                "frame_sequence": frame["sequence"],
                **details,
            },
        }
        # History must end before this screenshot, including gaps/dropped frames.
        if previous and previous["provenance"]["interval_end"] <= start:
            row["previous_actions"] = [previous["action"]]
        rows.append(row)
        previous = row
    (directory / "demonstrations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    audit = {
        "examples": len(rows),
        "rejected": rejected,
        "expert_quality_reviewed": False,
        "split_rule": "Keep this complete recording in one split; never split adjacent frames randomly.",
        "control_semantics": "Buttons active anywhere in the following 50 ms; exact held durations and edges retained.",
        "timing": "ScreenCaptureKit presentation time and Quartz event time converted to perf_counter; no human reaction-delay correction.",
    }
    (directory / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def record(args):
    sys.path.append(str(args.screenquest_root.resolve()))
    import CoreMedia as C
    import Quartz as Q
    from screenquest import desktop
    from screenquest.controller_lock import ControllerLock
    from screenquest.fast_capture import WindowStream

    if not 1 <= args.seconds <= 600 or not 5 <= args.fps <= 30:
        raise ValueError("Use 1–600 seconds and 5–30 FPS")
    if not Q.CGPreflightListenEventAccess():
        raise RuntimeError("Input Monitoring permission is required for passive controls recording")
    if args.window is None:
        candidates = [
            w for w in desktop.windows("Google Chrome") if w.get("kCGWindowName") == "Hordes.io"
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Keep exactly one regular Chrome window titled Hordes.io visible, or specify --window"
            )
        args.window = int(candidates[0]["kCGWindowNumber"])
        bounds = candidates[0]["kCGWindowBounds"]
        if [int(bounds[k]) for k in ("Width", "Height")] != [1280, 807]:
            raise ValueError(
                "Hordes auto-selection requires the reviewed 1280×807 Chrome window; recalibrate crop after resizing"
            )
    target = desktop.NativeWindow(args.window)
    bounds, crop = target.original["kCGWindowBounds"], args.crop
    width, height = int(bounds["Width"]), int(bounds["Height"])
    x, y, w, h = crop
    if min(x, y) < 0 or min(w, h) < 1 or x + w > width or y + h > height:
        raise ValueError("Crop must lie within the selected window")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(window_bounds=bounds, input_events_sent=0, capture_cursor=False)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    stream = WindowStream(args.window, width, height, fps=args.fps)
    active, stop, tap = False, None, None
    last_allowed = -float("inf")
    events, frames, futures = [], [], []
    stops = [Path("STOP"), args.screenquest_root / "STOP"]
    origin = bounds["X"] + x, bounds["Y"] + y

    def callback(proxy, kind, event, user):
        nonlocal stop
        if kind in (Q.kCGEventTapDisabledByTimeout, Q.kCGEventTapDisabledByUserInput):
            stop = "event_tap_disabled"
            return event
        if not active or time.perf_counter() - last_allowed > 0.1:
            return event
        front = target.workspace.frontmostApplication()
        if front is None or front.processIdentifier() != target.pid:
            stop = "focus_lost"
            return event
        if Q.CGEventGetFlags(event) & (Q.kCGEventFlagMaskCommand | Q.kCGEventFlagMaskControl):
            stop = "system_shortcut"
            return event
        now = time.perf_counter()
        host = C.CMTimeGetSeconds(C.CMClockGetTime(C.CMClockGetHostTimeClock()))
        age = host - Q.CGEventGetTimestamp(event) / 1e9
        if not -0.02 <= age <= 1:
            stop = "invalid_event_timestamp"
            return event
        item = {"timestamp": now - age, "received_at": now}
        if item["timestamp"] < started:
            return event
        if kind in (Q.kCGEventKeyDown, Q.kCGEventKeyUp):
            key = int(Q.CGEventGetIntegerValueField(event, Q.kCGKeyboardEventKeycode))
            if key in (53, 36):  # Escape stops; Enter stops before chat input.
                stop = "escape_or_chat"
                return event
            if key not in KEYS or Q.CGEventGetIntegerValueField(
                event, Q.kCGKeyboardEventAutorepeat
            ):
                return event
            item.update(kind="down" if kind == Q.kCGEventKeyDown else "up", button=KEYS[key])
        else:
            point = Q.CGEventGetLocation(event)
            px, py = (point.x - origin[0]) / w, (point.y - origin[1]) / h
            if not (0 <= px <= 1 and 0 <= py <= 1):
                stop = "pointer_left_viewport"
                return event
            item["pointer_xy"] = [px, py]
            if kind in (
                Q.kCGEventLeftMouseDown,
                Q.kCGEventLeftMouseUp,
                Q.kCGEventRightMouseDown,
                Q.kCGEventRightMouseUp,
            ):
                item.update(
                    kind="down"
                    if kind in (Q.kCGEventLeftMouseDown, Q.kCGEventRightMouseDown)
                    else "up",
                    button="mouse_left"
                    if kind in (Q.kCGEventLeftMouseDown, Q.kCGEventLeftMouseUp)
                    else "mouse_right",
                )
            else:
                item.update(
                    kind="move",
                    delta=[
                        int(Q.CGEventGetIntegerValueField(event, field))
                        for field in (Q.kCGMouseEventDeltaX, Q.kCGMouseEventDeltaY)
                    ],
                )
        events.append(item)
        return event

    types = (
        Q.kCGEventKeyDown,
        Q.kCGEventKeyUp,
        Q.kCGEventLeftMouseDown,
        Q.kCGEventLeftMouseUp,
        Q.kCGEventRightMouseDown,
        Q.kCGEventRightMouseUp,
        Q.kCGEventMouseMoved,
        Q.kCGEventLeftMouseDragged,
        Q.kCGEventRightMouseDragged,
    )
    started, ended = None, time.perf_counter()
    with ControllerLock(), ThreadPoolExecutor(max_workers=1) as writer:
        try:
            if any(p.exists() for p in stops):
                raise RuntimeError("STOP exists")
            stream.start()
            # Matches live inference: complete viewport with the OS cursor hidden.
            # Click coordinates remain explicit targets in the event trace.
            tap = Q.CGEventTapCreate(
                Q.kCGSessionEventTap,
                Q.kCGHeadInsertEventTap,
                Q.kCGEventTapOptionListenOnly,
                sum(1 << int(t) for t in types),
                callback,
                None,
            )
            if tap is None:
                raise RuntimeError(
                    "Cannot create passive event tap; check Input Monitoring permission"
                )
            source = Q.CFMachPortCreateRunLoopSource(None, tap, 0)
            Q.CFRunLoopAddSource(Q.CFRunLoopGetCurrent(), source, Q.kCFRunLoopCommonModes)
            Q.CGEventTapEnable(tap, True)
            print(
                "Passive recorder ready. Click the game viewport within 60 seconds. Escape stops; no inputs will be sent.",
                flush=True,
            )
            until = time.perf_counter() + 60
            while time.perf_counter() < until:
                desktop.refresh_app_events()
                if (
                    target.focused()
                    and not target.text_input_focused()
                    and stream.latest() is not None
                ):
                    point = Q.CGEventGetLocation(Q.CGEventCreate(None))
                    if (
                        origin[0] <= point.x <= origin[0] + w
                        and origin[1] <= point.y <= origin[1] + h
                    ):
                        break
                if any(p.exists() for p in stops):
                    raise RuntimeError("STOP requested")
                time.sleep(0.005)
            else:
                raise RuntimeError("Focus wait timed out")
            started = time.perf_counter()
            held = [
                name
                for code, name in KEYS.items()
                if Q.CGEventSourceKeyState(Q.kCGEventSourceStateCombinedSessionState, code)
            ]
            held += [
                name
                for button, name in ((0, "mouse_left"), (1, "mouse_right"))
                if Q.CGEventSourceButtonState(Q.kCGEventSourceStateCombinedSessionState, button)
            ]
            events.append({"timestamp": started, "kind": "state", "buttons": held})
            active, last_allowed, sequence = True, started, -1
            print(
                f"Recording your controls for up to {args.seconds:g} seconds: {args.output}",
                flush=True,
            )
            while time.perf_counter() - started < args.seconds and stop is None:
                desktop.refresh_app_events()
                if not target.focused() or not target.unchanged() or target.text_input_focused():
                    stop = "focus_or_window_changed"
                    break
                if any(p.exists() for p in stops):
                    stop = "STOP_requested"
                    break
                last_allowed = time.perf_counter()
                frame = stream.latest()
                if frame is None or frame.sequence == sequence or frame.captured_at < started:
                    time.sleep(0.002)
                    continue
                if time.perf_counter() - frame.captured_at > 0.15:
                    raise RuntimeError("Stale screenshot")
                sequence = frame.sequence
                name = f"frames/{len(frames):06d}.jpg"
                image = desktop.viewport(frame.image, crop)
                futures.append(writer.submit(image.save, args.output / name, quality=90))
                if len(futures) > 8:
                    futures.pop(0).result()
                frames.append({"timestamp": frame.captured_at, "sequence": sequence, "image": name})
            ended = time.perf_counter()
        except (RuntimeError, KeyboardInterrupt) as error:
            stop, ended = str(error) or "interrupted", time.perf_counter()
        finally:
            active = False
            if tap is not None:
                Q.CGEventTapEnable(tap, False)
                Q.CFMachPortInvalidate(tap)
            stream.close()
            for future in futures:
                future.result()
            for name, items in (("frames", frames), ("controls", events)):
                (args.output / f"{name}.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in items)
                )
            summary = {
                "started_at": started,
                "ended_at": ended,
                "stop_reason": stop or "duration_complete",
                "frames": len(frames),
                "events": len(events),
                "input_events_sent": 0,
            }
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if frames and events:
        summary["export"] = export(args.output)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    recorder = sub.add_parser("record")
    recorder.add_argument("--window", type=int)
    recorder.add_argument("--screenquest-root", type=Path, required=True)
    recorder.add_argument("--crop", type=int, nargs=4, default=[0, 87, 1280, 720])
    recorder.add_argument("--output", type=Path, required=True)
    recorder.add_argument("--seconds", type=float, default=300)
    recorder.add_argument("--fps", type=int, default=20)
    recorder.add_argument("--game", default="Hordes.io")
    recorder.add_argument("--goal", default=GOAL)
    recorder.add_argument("--controls", default=HORDES_CONTROLS)
    exporter = sub.add_parser("export")
    exporter.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "record":
        record(args)
    else:
        print(json.dumps(export(args.directory), indent=2))


if __name__ == "__main__":
    main()
