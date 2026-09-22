"""Bounded experimental native trial; neural proposals with no gameplay heuristics.

Uses ScreenQuest's capture/input transport only. Does not import its live policy,
planner, targeting, OCR or combat controller. Default is observation only.
"""

import argparse
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .p2p_data import CONTROLS
from .temporal_runtime import TemporalRuntime

KEYS = {"w": 13, "a": 0, "s": 1, "d": 2, "space": 49, "tab": 48, "1": 18, "2": 19, "3": 20, "4": 21}
ALLOWED = set(KEYS) | {"mouse_left", "mouse_right"}


def bounded_action(proposal):
    """Limit transport scope, without substituting or inventing model actions."""
    delta = np.asarray(proposal["mouse_delta"], dtype=float)
    if delta.shape != (2,) or not np.isfinite(delta).all():
        raise ValueError("Invalid mouse proposal")
    raw = np.rint(delta * 512).astype(int)
    bounded = np.clip(raw, -64, 64)
    return {
        "buttons": sorted(set(proposal["buttons"]) & ALLOWED),
        "mouse_delta": (bounded / 512).tolist(),
        "blocked_buttons": sorted(set(proposal["buttons"]) - ALLOWED),
        "mouse_clamped": bool(np.any(raw != bounded)),
    }


class Pulse:
    """Release after 50 ms independently of the next model inference."""

    def __init__(self, target, crop):
        self.target, self.crop = target, crop
        self.lock = threading.Lock()
        self.timer = None
        b = target.original["kCGWindowBounds"]
        self.origin = (b["X"] + crop[0], b["Y"] + crop[1])
        self.point = (self.origin[0] + crop[2] / 2, self.origin[1] + crop[3] / 2)

    def release(self):
        with self.lock:
            self.target.release()

    def finish(self):
        if self.timer:
            self.timer.join()
            self.timer = None
        self.release()

    def apply(self, action, deadline):
        self.finish()
        target, Q = self.target, self.target.Q
        with self.lock:
            try:
                if not target.focused() or not target.unchanged():
                    raise RuntimeError("Game window lost focus or moved")
                if target.text_input_focused():
                    raise RuntimeError("Text input focused")
                if time.perf_counter() >= deadline:
                    raise RuntimeError("Trial deadline reached")
                start = time.perf_counter()
                buttons = action["buttons"]
                for key in buttons:
                    if key in KEYS:
                        target.event(key, True)
                if "mouse_right" in buttons:
                    target.mouse_event("down", self.point)
                if "mouse_left" in buttons:
                    target.left_event("down", self.point)
                dx, dy = [round(v * 512) for v in action["mouse_delta"]]
                x, y = self.point
                # Cursor stays within the central playfield, clear of browser/HUD controls.
                nx = float(
                    np.clip(
                        x + dx,
                        self.origin[0] + 0.36 * self.crop[2],
                        self.origin[0] + 0.72 * self.crop[2],
                    )
                )
                ny = float(
                    np.clip(
                        y + dy,
                        self.origin[1] + 0.22 * self.crop[3],
                        self.origin[1] + 0.63 * self.crop[3],
                    )
                )
                dx, dy = round(nx - x), round(ny - y)
                action["mouse_delta"] = [dx / 512, dy / 512]
                if dx or dy:
                    kind = (
                        Q.kCGEventRightMouseDragged
                        if "mouse_right" in buttons
                        else Q.kCGEventLeftMouseDragged
                        if "mouse_left" in buttons
                        else Q.kCGEventMouseMoved
                    )
                    button = (
                        Q.kCGMouseButtonRight if "mouse_right" in buttons else Q.kCGMouseButtonLeft
                    )
                    event = Q.CGEventCreateMouseEvent(None, kind, (nx, ny), button)
                    Q.CGEventSetIntegerValueField(event, Q.kCGMouseEventDeltaX, dx)
                    Q.CGEventSetIntegerValueField(event, Q.kCGMouseEventDeltaY, dy)
                    Q.CGEventPost(Q.kCGHIDEventTap, event)
                    self.point = (nx, ny)
                    if target.mouse_pressed:
                        target.mouse_point = self.point
                    if target.left_pressed:
                        target.left_point = self.point
                elapsed = time.perf_counter() - start
                self.timer = threading.Timer(
                    max(0, min(0.05 - elapsed, deadline - time.perf_counter())), self.release
                )
                self.timer.start()
                return start
            except BaseException:
                target.release()
                raise


def run(args):
    sys.path.append(str(args.screenquest_root.resolve()))
    import ApplicationServices as A
    from screenquest import desktop
    from screenquest.controller_lock import ControllerLock
    from screenquest.fast_capture import WindowStream
    from screenquest.hordes import ScreenGate

    if not 1 <= args.seconds <= 60:
        raise ValueError("Trial duration must be 1–60 seconds")
    stops = [Path("STOP"), args.screenquest_root / "STOP"]
    if any(p.exists() for p in stops):
        raise RuntimeError("STOP exists; no trial started")
    if args.execute and not A.AXIsProcessTrusted():
        raise RuntimeError("Host Accessibility permission is missing")
    target = desktop.NativeWindow(args.window)
    if target.original["kCGWindowOwnerName"] != "Google Chrome":
        raise ValueError("Only the selected regular Chrome window is supported")
    desktop.KEYCODES.update(KEYS)
    bounds = target.original["kCGWindowBounds"]
    if [int(bounds[k]) for k in ("Width", "Height")] != [1280, 807]:
        raise ValueError("Reviewed calibration requires a 1280×807 window")
    crop = [0, 87, 1280, 720]
    gate = ScreenGate(str(args.reference), [1184, 4, 23, 28])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    (args.output / "config.json").write_text(
        json.dumps(
            {
                **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "crop": crop,
                "allowed_buttons": sorted(ALLOWED),
                "max_delta_px": 64,
                "pulse_seconds": 0.05,
                "policy": "temporal checkpoint only",
                "limitations": "Input restrictions logged. Event posting is not game acknowledgement.",
            },
            indent=2,
        )
        + "\n"
    )
    print("Loading experimental temporal checkpoint...", flush=True)
    runtime = TemporalRuntime.load(args.bundle)
    from PIL import Image

    warm_row = {
        "frames": [{"image": Image.open(args.reference).convert("RGB"), "age_seconds": 0}],
        "goal": args.goal,
        "controls": CONTROLS,
        "previous_actions": [],
    }
    for i in range(3):
        runtime.predict(warm_row, session_id="warmup", timestamp_seconds=i * 0.05)
    runtime.reset()
    events, futures, previous = [], [], []
    stream = WindowStream(args.window, 1280, 807, fps=30)
    pulse = Pulse(target, crop)
    start, stop_reason = None, "duration_complete"
    with ControllerLock(), ThreadPoolExecutor(max_workers=1) as writer:
        try:
            stream.start()
            until = time.perf_counter() + 5
            while stream.latest() is None:
                desktop.refresh_app_events()
                if time.perf_counter() > until:
                    raise RuntimeError("No capture frame")
                time.sleep(0.005)
            if args.execute and not target.focused():
                raise RuntimeError("Selected game window is not focused")
            initial = desktop.viewport(stream.latest().image, crop)
            if not gate.check(initial)["verified"]:
                raise RuntimeError("HUD does not match reviewed calibration")
            initial.save(args.output / "before.png")
            # Neutral pointer setup only; no target-selection click or game action.
            if args.execute:
                target.mouse_event("move", pulse.point)
            start = time.perf_counter()
            deadline, last_sequence = start + args.seconds, -1
            print(
                f"Starting {'input' if args.execute else 'observation'} trial: {args.output}",
                flush=True,
            )
            with (args.output / "events.jsonl").open("w") as log:
                while time.perf_counter() < deadline:
                    desktop.refresh_app_events()
                    if any(p.exists() for p in stops):
                        raise RuntimeError("STOP requested")
                    if not target.unchanged() or (args.execute and not target.focused()):
                        raise RuntimeError("Window changed or focus lost")
                    frame = stream.latest()
                    if frame.sequence == last_sequence:
                        time.sleep(0.002)
                        continue
                    last_sequence = frame.sequence
                    if time.perf_counter() - frame.captured_at > 0.18:
                        raise RuntimeError("Stale capture")
                    image = desktop.viewport(frame.image, crop)
                    if not gate.check(image)["verified"]:
                        raise RuntimeError("HUD changed")
                    proposal = runtime.predict(
                        {
                            "frames": [{"image": image, "age_seconds": 0}],
                            "goal": args.goal,
                            "controls": CONTROLS,
                            "previous_actions": previous,
                        },
                        session_id=str(args.output),
                        timestamp_seconds=frame.captured_at,
                    )
                    action = bounded_action(proposal)
                    if time.perf_counter() >= deadline:
                        break
                    age = (time.perf_counter() - frame.captured_at) * 1000
                    applied = args.execute and age <= 180
                    fresh = stream.latest()
                    if not gate.check(desktop.viewport(fresh.image, crop))["verified"]:
                        raise RuntimeError("HUD changed during inference")
                    posted = pulse.apply(action, deadline) if applied else None
                    previous = (
                        [{"buttons": action["buttons"], "mouse_delta": action["mouse_delta"]}]
                        if applied
                        else []
                    )
                    name = f"frames/{len(events):05d}.jpg"
                    futures.append(writer.submit(image.save, args.output / name, quality=85))
                    record = {
                        "step": len(events),
                        "elapsed_s": frame.captured_at - start,
                        "image": name,
                        "proposal": proposal,
                        "bounded_action": action,
                        "applied": applied,
                        "frame_age_after_inference_ms": age,
                        "screenshot_to_dispatch_start_ms": (posted - frame.captured_at) * 1000
                        if posted
                        else None,
                    }
                    events.append(record)
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    if len(events) == 1 or len(events) % 50 == 0:
                        print(
                            json.dumps(
                                {
                                    "step": len(events),
                                    "buttons": proposal["buttons"],
                                    "mouse": proposal["mouse_delta"],
                                    "inference_ms": proposal["image_to_outputs_ms"],
                                }
                            ),
                            flush=True,
                        )
            pulse.finish()
            desktop.viewport(stream.latest().image, crop).save(args.output / "after.png")
        except (RuntimeError, KeyboardInterrupt) as error:
            stop_reason = str(error) or "interrupted"
        finally:
            pulse.finish()
            stream.close()
            for future in futures:
                future.result()
    inference = [e["proposal"]["image_to_outputs_ms"] for e in events]
    latency = [e["screenshot_to_dispatch_start_ms"] for e in events if e["applied"]]
    summary = {
        "stop_reason": stop_reason,
        "observations": len(events),
        "applied_steps": sum(e["applied"] for e in events),
        "steps_with_input_events": sum(
            e["applied"]
            and bool(e["bounded_action"]["buttons"] or any(e["bounded_action"]["mouse_delta"]))
            for e in events
        ),
        "nonempty_button_steps": sum(
            bool(e["bounded_action"]["buttons"]) and e["applied"] for e in events
        ),
        "proposed_buttons": dict(Counter(b for e in events for b in e["proposal"]["buttons"])),
        "applied_buttons": dict(
            Counter(b for e in events if e["applied"] for b in e["bounded_action"]["buttons"])
        ),
        "state_resets": sum(e["proposal"]["state_reset"] for e in events),
        "inference_p50_ms": float(np.median(inference)) if inference else None,
        "inference_p95_ms": float(np.percentile(inference, 95)) if inference else None,
        "screenshot_to_dispatch_start_p50_ms": float(np.median(latency)) if latency else None,
        "screenshot_to_dispatch_start_p95_ms": float(np.percentile(latency, 95))
        if latency
        else None,
        "gameplay_success": None,
        "note": "Latency ends at dispatch start, not event posting or game acknowledgement. Requires visual review. No gameplay heuristics or teacher actions.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--screenquest-root", type=Path, required=True)
    p.add_argument("--window", type=int, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument(
        "--goal",
        default="Select a nearby Young Grub with Tab, approach with WASD, and attack with ability 1. Avoid other players. Retreat if health is low.",
    )
    p.add_argument("--execute", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
