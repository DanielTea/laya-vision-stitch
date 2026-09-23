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


class LiveRuntime:
    """Select the checkpoint's trained neural path; no gameplay decisions here."""

    def __init__(self, bundle):
        metadata = json.loads((Path(bundle) / "config.json").read_text())
        if metadata.get("format") == "laya-p2p-1":
            from .laya_p2p_stream import LayaP2PStream

            self.model = LayaP2PStream.load(bundle)
            self.temporal = True
            return
        from .trainable_model import TrainableRuntime

        model = TrainableRuntime.load(bundle)
        self.temporal = model.module.policy_config.temporal_adapter != "none"
        if not self.temporal and not model.metadata.get("first_step_control_only"):
            raise ValueError("Live runner requires explicitly supervised control outputs")
        self.model = TemporalRuntime(model) if self.temporal else model

    def reset(self):
        if self.temporal:
            self.model.reset()

    def predict(self, row, **session):
        if self.temporal:
            return self.model.predict(row, **session)
        # This model was trained without previous-control inputs.
        result = self.model.predict({**row, "previous_actions": []})
        return {
            **result,
            "mouse_delta": result["action_chunk"][0]["mouse_delta"],
            "state_reset": False,
        }


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
        wait_started = time.perf_counter()
        self.finish()
        wait_finished = time.perf_counter()
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
                first_post = None
                buttons = action["buttons"]
                for key in buttons:
                    if key in KEYS:
                        target.event(key, True)
                        if first_post is None:
                            first_post = time.perf_counter()
                if "mouse_right" in buttons:
                    target.mouse_event("down", self.point)
                    if first_post is None:
                        first_post = time.perf_counter()
                if "mouse_left" in buttons:
                    target.left_event("down", self.point)
                    if first_post is None:
                        first_post = time.perf_counter()
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
                    if first_post is None:
                        first_post = time.perf_counter()
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
                return {
                    "dispatch_start": start,
                    "first_event_posted": first_post,
                    "pulse_wait_ms": (wait_finished - wait_started) * 1000,
                    "dispatch_guard_ms": (start - wait_finished) * 1000,
                }
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
    if args.capture_fps not in (30, 60, 120):
        raise ValueError("Capture rate must be 30, 60 or 120 FPS")
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
    width, height = [int(bounds[k]) for k in ("Width", "Height")]
    crop = args.crop
    x, y, crop_width, crop_height = crop
    if (
        (crop_width, crop_height) != (1280, 720)
        or min(x, y) < 0
        or x + crop_width > width
        or y + crop_height > height
    ):
        raise ValueError(
            "Reviewed calibration requires a complete 1280×720 viewport inside the window"
        )
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
                "policy": "checkpoint neural outputs only",
                "limitations": "Input restrictions logged. Event posting is not game acknowledgement.",
            },
            indent=2,
        )
        + "\n"
    )
    print("Loading experimental stitched checkpoint...", flush=True)
    runtime = LiveRuntime(args.bundle)
    from PIL import Image

    warm_row = {
        "frames": [{"image": Image.open(args.reference).convert("RGB"), "age_seconds": 0}],
        "goal": args.goal,
        "controls": args.controls,
        "previous_actions": [],
    }
    for i in range(3):
        runtime.predict(warm_row, session_id="warmup", timestamp_seconds=i * 0.05)
    runtime.reset()
    events, futures, previous = [], [], []
    stream = WindowStream(args.window, width, height, fps=args.capture_fps)
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
                    inference_started = time.perf_counter()
                    proposal = runtime.predict(
                        {
                            "frames": [{"image": image, "age_seconds": 0}],
                            "goal": args.goal,
                            "controls": args.controls,
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
                    hud_check_started = time.perf_counter()
                    if not gate.check(desktop.viewport(fresh.image, crop))["verified"]:
                        raise RuntimeError("HUD changed during inference")
                    hud_check_ms = (time.perf_counter() - hud_check_started) * 1000
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
                        "frame_age_before_inference_ms": (inference_started - frame.captured_at)
                        * 1000,
                        "post_inference_hud_check_ms": hud_check_ms,
                        "pulse_wait_ms": posted["pulse_wait_ms"] if posted else None,
                        "dispatch_guard_ms": posted["dispatch_guard_ms"] if posted else None,
                        "screenshot_to_dispatch_start_ms": (
                            posted["dispatch_start"] - frame.captured_at
                        )
                        * 1000
                        if posted
                        else None,
                        "screenshot_to_first_event_ms": (
                            posted["first_event_posted"] - frame.captured_at
                        )
                        * 1000
                        if posted and posted["first_event_posted"] is not None
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
    event_latency = [
        e["screenshot_to_first_event_ms"]
        for e in events
        if e["screenshot_to_first_event_ms"] is not None
    ]
    summary = {
        "stop_reason": stop_reason,
        "requested_seconds": args.seconds,
        "last_observation_elapsed_s": events[-1]["elapsed_s"] if events else None,
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
        "screenshot_to_first_event_p50_ms": float(np.median(event_latency))
        if event_latency
        else None,
        "screenshot_to_first_event_p95_ms": float(np.percentile(event_latency, 95))
        if event_latency
        else None,
        "screenshot_to_dispatch_start_p50_ms": float(np.median(latency)) if latency else None,
        "screenshot_to_dispatch_start_p95_ms": float(np.percentile(latency, 95))
        if latency
        else None,
        "gameplay_success": None,
        "note": "Dispatch-start and first-posted-event latency are separate. Idle steps have no event latency. Neither measures game acknowledgement. Requires visual review. No gameplay heuristics or teacher actions.",
    }
    summary["timing_stages_ms"] = {}
    for field in (
        "frame_age_before_inference_ms",
        "post_inference_hud_check_ms",
        "pulse_wait_ms",
        "dispatch_guard_ms",
    ):
        values = [e[field] for e in events if e[field] is not None]
        summary["timing_stages_ms"][field] = {
            "samples": len(values),
            "p50": float(np.median(values)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
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
    p.add_argument("--crop", type=int, nargs=4, default=[0, 87, 1280, 720])
    p.add_argument("--capture-fps", type=int, default=30)
    p.add_argument(
        "--goal",
        default="Select a nearby Young Grub with Tab, approach with WASD, and attack with ability 1. Avoid other players. Retreat if health is low.",
    )
    p.add_argument("--execute", action="store_true")
    p.add_argument("--controls", default=CONTROLS)
    run(p.parse_args())


if __name__ == "__main__":
    main()
