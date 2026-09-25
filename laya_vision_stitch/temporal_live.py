"""Bounded experimental native trial; neural proposals with no gameplay heuristics.

Uses ScreenQuest's capture/input transport only. Does not import its live policy,
planner, targeting, OCR or combat controller. Default is observation only: without
`--execute` it captures, runs the model and logs proposals, and posts no events.
`--planner` adds the Molmo slow planner (one selection click per target); `--planner-act`
also lets it approach the target with W/A/S/D and click the skill Molmo points to on the
skill bar (general control conventions, see planner_actions.py).
`--pipeline` selects the pipelined scheduler in `live_pipeline.py` (asynchronous guards,
non-waiting pulses); the serial loop remains the default.
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

from .hold_transport import HoldTransport
from .p2p_data import CONTROLS
from .temporal_runtime import TemporalRuntime

KEYS = {"w": 13, "a": 0, "s": 1, "d": 2, "space": 49, "tab": 48, "1": 18, "2": 19, "3": 20, "4": 21}
ALLOWED = set(KEYS) | {"mouse_left", "mouse_right"}
# macOS virtual key codes for every keyboard control in the extended vocabulary
# (p2p_adaptation.EXTENDED_KEYS); `--all-keys` allows them with the middle button and wheel.
KEY_CODES = {
    **KEYS,
    "e": 14,
    "f": 3,
    "q": 12,
    "z": 6,
    "r": 15,
    "c": 8,
    "x": 7,
    "v": 9,
    "g": 5,
    "i": 34,
    "m": 46,
    "b": 11,
    "t": 17,
    "h": 4,
    "5": 23,
    "6": 22,
    "7": 26,
    "8": 28,
    "9": 25,
    "0": 29,
    "up": 126,
    "down": 125,
    "left": 123,
    "right": 124,
    "shift": 56,
    "ctrl": 59,
    "alt": 58,
    "escape": 53,
    "enter": 36,
}
EXTENDED_ALLOWED = set(KEY_CODES) | {
    "mouse_left",
    "mouse_right",
    "mouse_middle",
    "scroll_up",
    "scroll_down",
}
MOVEMENT = {"w", "a", "s", "d"}


class LiveRuntime:
    """Select the checkpoint's trained neural path; no gameplay decisions here."""

    def __init__(self, bundle, stream_options=None):
        metadata = json.loads((Path(bundle) / "config.json").read_text())
        if metadata.get("format") == "laya-p2p-1":
            from .laya_p2p_stream import LayaP2PStream

            self.model = LayaP2PStream.load(bundle, **(stream_options or {}))
            self.temporal = True
            return
        if stream_options:
            raise ValueError("Streaming options apply only to laya-p2p-1 checkpoints")
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
    action = {
        "buttons": sorted(set(proposal["buttons"]) & ALLOWED),
        "mouse_delta": (bounded / 512).tolist(),
        "blocked_buttons": sorted(set(proposal["buttons"]) - ALLOWED),
        "mouse_clamped": bool(np.any(raw != bounded)),
    }
    if proposal.get("pointer_xy") is not None:
        xy = np.asarray(proposal["pointer_xy"], dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("Invalid pointer proposal")
        action["pointer_xy"] = np.clip(xy, 0, 1).tolist()
    hold = proposal.get("planner_hold")
    if hold is not None:
        if not set(hold) <= set(KEYS):
            raise ValueError("Invalid planner movement")
        # The planner moves the character: its keys replace the policy's movement keys.
        action["buttons"] = sorted((set(action["buttons"]) - MOVEMENT) | set(hold))
        action["planner_hold"] = sorted(hold)
    click = proposal.get("planner_click")
    if click is not None:
        xy = np.asarray(click["xy"], dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or click.get("button") != "mouse_left":
            raise ValueError("Invalid planner click")
        # The slow planner's high-level action: one left click on its target or skill button.
        action["planner_click"] = {
            "xy": np.clip(xy, 0, 1).tolist(),
            "button": "mouse_left",
            "kind": click.get("kind", "select"),
        }
        action["pointer_xy"] = action["planner_click"]["xy"]
        action["buttons"] = sorted(set(action["buttons"]) | {"mouse_left"})
    return action


class Pulse:
    """Release after 50 ms independently of the next model inference."""

    # Cursor bounds as fractions of the viewport (x0, x1, y0, y1). The default central
    # playfield suits relative camera control; pointer mode allows most of the viewport
    # but keeps clear of the top menu band and the outer margins.
    CENTRAL, POINTER = (0.36, 0.72, 0.22, 0.63), (0.03, 0.97, 0.08, 0.97)

    def __init__(self, target, crop, pointer=False):
        self.target, self.crop = target, crop
        self.lock = threading.Lock()
        self.timer = None
        b = target.original["kCGWindowBounds"]
        self.origin = (b["X"] + crop[0], b["Y"] + crop[1])
        self.point = (self.origin[0] + crop[2] / 2, self.origin[1] + crop[3] / 2)
        self.pointer, self.bounds, self.held = (
            pointer,
            self.POINTER if pointer else self.CENTRAL,
            set(),
        )

    def clamp(self, x, y):
        x0, x1, y0, y1 = self.bounds
        return (
            float(
                np.clip(x, self.origin[0] + x0 * self.crop[2], self.origin[0] + x1 * self.crop[2])
            ),
            float(
                np.clip(y, self.origin[1] + y0 * self.crop[3], self.origin[1] + y1 * self.crop[3])
            ),
        )

    def release(self):
        with self.lock:
            self.target.release()

    def finish(self):
        if self.timer:
            self.timer.join()
            self.timer = None
        self.release()

    def settle(self):
        """End the previous pulse before the next one; serial runner waits for it."""
        self.finish()

    def check(self, deadline):
        """Synchronous guards, called under the transport lock just before posting."""
        target = self.target
        if not target.focused() or not target.unchanged():
            raise RuntimeError("Game window lost focus or moved")
        if target.text_input_focused():
            raise RuntimeError("Text input focused")
        if time.perf_counter() >= deadline:
            raise RuntimeError("Trial deadline reached")

    def schedule_release(self, delay):
        self.timer = threading.Timer(delay, self.release)
        self.timer.start()

    def apply(self, action, deadline):
        wait_started = time.perf_counter()
        self.settle()
        wait_finished = time.perf_counter()
        target, Q = self.target, self.target.Q
        with self.lock:
            try:
                self.check(deadline)
                start = time.perf_counter()
                first_post = None
                buttons = action["buttons"]
                mouse = {"mouse_left", "mouse_right"} & set(buttons)
                onset = mouse - self.held
                if action.get("planner_click"):
                    onset = onset | {"mouse_left"}  # a planner click is always a new press
                self.held = mouse
                if self.pointer and onset and action.get("pointer_xy") is not None:
                    # Absolute pointing only when a button is newly pressed; holds keep
                    # relative drags so camera control is unchanged.
                    px, py = action["pointer_xy"]
                    self.point = self.clamp(
                        self.origin[0] + px * self.crop[2], self.origin[1] + py * self.crop[3]
                    )
                    target.mouse_event("move", self.point)
                    first_post = time.perf_counter()
                    action["pointer_applied"] = [
                        (self.point[0] - self.origin[0]) / self.crop[2],
                        (self.point[1] - self.origin[1]) / self.crop[3],
                    ]
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
                nx, ny = self.clamp(x + dx, y + dy)
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
                self.schedule_release(max(0, min(0.05 - elapsed, deadline - time.perf_counter())))
                return {
                    "dispatch_start": start,
                    "first_event_posted": first_post,
                    "pulse_wait_ms": (wait_finished - wait_started) * 1000,
                    "dispatch_guard_ms": (start - wait_finished) * 1000,
                }
            except BaseException:
                target.release()
                raise


class PreemptingPulse(Pulse):
    """Pipelined transport: a new pulse supersedes the active one instead of waiting.

    The active pulse is released at once, so the event sequence (key-up, then key-down)
    matches the serial runner but an earlier pulse can be shorter than 50 ms. Focus,
    layout, text-input and HUD checks run in the pipeline guard loop; `permit()` returns
    its fail-closed denial and is read under the transport lock right before posting.
    ScreenQuest's mouse-button helpers still re-check focus per event.
    """

    def __init__(self, target, crop, permit, pointer=False):
        super().__init__(target, crop, pointer)
        self.permit, self.generation = permit, 0

    def settle(self):
        with self.lock:
            # A release timer that already fired but waits for the lock must not end
            # the pulse posted after this point.
            self.generation += 1
        if self.timer:
            self.timer.cancel()
            self.timer = None
        self.release()

    def check(self, deadline):
        reason = self.permit()
        if reason:
            raise RuntimeError(reason)
        if time.perf_counter() >= deadline:
            raise RuntimeError("Trial deadline reached")

    def schedule_release(self, delay):
        generation = self.generation

        def release():
            with self.lock:
                if generation == self.generation:
                    self.target.release()

        self.timer = threading.Timer(delay, release)
        self.timer.start()


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
    if args.precommit and not args.pipeline:
        raise ValueError("--precommit requires --pipeline")
    if not 0 <= args.wait_for_focus <= 60:
        raise ValueError("Focus wait must be 0-60 seconds")
    if not 0.005 <= args.guard_interval <= 0.5:
        raise ValueError("Guard interval must be 5-500 ms")
    stream_options = {}
    if args.compile:
        stream_options["compile"] = True
    if args.max_memory_gap is not None:
        stream_options["max_gap_seconds"] = args.max_memory_gap
    if args.hold:
        # Held presses start drags; one on the avatar would select the player's character.
        stream_options["avoid_avatar"] = True
    planner = None
    if args.planner:
        if not args.pointer:
            raise ValueError("--planner requires --pointer")
        from .molmo_planner import AsyncPlanner

        print("Starting Molmo planner process...", flush=True)
        planner = AsyncPlanner()
        stream_options["planner"] = planner
        stream_options["act"] = bool(args.planner_act)
    elif args.planner_act:
        raise ValueError("--planner-act requires --planner")
    stops = [Path("STOP"), args.screenquest_root / "STOP"]
    if any(p.exists() for p in stops):
        raise RuntimeError("STOP exists; no trial started")
    if args.execute and not A.AXIsProcessTrusted():
        raise RuntimeError("Host Accessibility permission is missing")
    target = desktop.NativeWindow(args.window)
    if target.original["kCGWindowOwnerName"] != "Google Chrome":
        raise ValueError("Only the selected regular Chrome window is supported")
    if args.all_keys:
        KEYS.update(KEY_CODES)
        ALLOWED.update(EXTENDED_ALLOWED)
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
    runtime = LiveRuntime(args.bundle, stream_options)
    if args.precommit and not hasattr(runtime.model, "commit"):
        raise ValueError("--precommit requires a laya-p2p-1 checkpoint")
    if args.compile:
        print(f"Compile status: {runtime.model.compile_status}", flush=True)
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
    if args.pipeline:
        from .live_pipeline import GuardVeto

        veto = GuardVeto(max(0.1, 2 * args.guard_interval))
        pulse = (
            HoldTransport(target, crop, set(KEYS), args.pointer, permit=veto.denial)
            if args.hold
            else PreemptingPulse(target, crop, veto.denial, args.pointer)
        )
    else:
        pulse = (
            HoldTransport(target, crop, set(KEYS), args.pointer)
            if args.hold
            else Pulse(target, crop, args.pointer)
        )
    start, stop_reason, pipeline_summary = None, "duration_complete", None
    with ControllerLock(), ThreadPoolExecutor(max_workers=1) as writer:
        try:
            stream.start()
            until = time.perf_counter() + 5
            while stream.latest() is None:
                desktop.refresh_app_events()
                if time.perf_counter() > until:
                    raise RuntimeError("No capture frame")
                time.sleep(0.005)
            if args.execute and args.wait_for_focus and not target.focused():
                print(
                    f"Click inside the game window to start (waiting {args.wait_for_focus:g} s)...",
                    flush=True,
                )
                until = time.perf_counter() + args.wait_for_focus
                while not target.focused() and time.perf_counter() < until:
                    desktop.refresh_app_events()
                    time.sleep(0.05)
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
                if args.pipeline:
                    from .live_pipeline import LivePipeline, PollingCapture

                    def predict(image, frame, previous):
                        row = {
                            "frames": [{"image": image, "age_seconds": 0}],
                            "goal": args.goal,
                            "controls": args.controls,
                            "previous_actions": previous,
                        }
                        return runtime.predict(
                            row, session_id=str(args.output), timestamp_seconds=frame.captured_at
                        )

                    def on_record(record, image):
                        record["image"] = f"frames/{record['step']:05d}.jpg"
                        futures.append(
                            writer.submit(image.save, args.output / record["image"], quality=85)
                        )
                        events.append(record)
                        log.write(json.dumps(record) + "\n")
                        log.flush()

                    def stop_file():
                        return "STOP requested" if any(p.exists() for p in stops) else None

                    def focus_layout():
                        if not target.unchanged() or (args.execute and not target.focused()):
                            return "Window changed or focus lost"
                        return None

                    def text_input():
                        return (
                            "Text input focused"
                            if args.execute and target.text_input_focused()
                            else None
                        )

                    def hud():
                        image = desktop.viewport(stream.latest().image, crop)
                        return None if gate.check(image)["verified"] else "HUD changed"

                    guards = [
                        ("stop", stop_file),
                        ("focus_layout", focus_layout),
                        ("text_input", text_input),
                        ("hud", hud),
                    ]
                    pipeline_summary = LivePipeline(
                        capture=PollingCapture(stream),
                        prepare=lambda frame: desktop.viewport(frame.image, crop),
                        predict=predict,
                        dispatcher=pulse,
                        guards=guards,
                        deadline=deadline,
                        execute=args.execute,
                        veto=veto,
                        commit=runtime.model.commit if args.precommit else None,
                        guard_interval=args.guard_interval,
                        on_record=on_record,
                        pump=desktop.refresh_app_events,
                        start=start,
                    ).run()
                    if pipeline_summary["stop_reason"] != "duration_complete":
                        raise RuntimeError(pipeline_summary["stop_reason"])
                # Serial loop (default); skipped entirely when the pipeline ran.
                while not args.pipeline and time.perf_counter() < deadline:
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
            if planner is not None:
                planner.close()
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
    if pipeline_summary is not None:
        summary["pipeline"] = pipeline_summary
    if stream_options or args.pipeline:
        summary["runtime_options"] = {
            **{k: v for k, v in stream_options.items() if k != "planner"},
            "planner": "molmo" if stream_options.get("planner") is not None else None,
            "pipeline": args.pipeline,
            "precommit": args.precommit,
            "guard_interval_s": args.guard_interval if args.pipeline else None,
            "compile_status": getattr(runtime.model, "compile_status", None),
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
    # Experimental options; defaults keep the reviewed serial runner unchanged.
    p.add_argument("--pipeline", action="store_true", help="asynchronous guards, no pulse wait")
    p.add_argument("--guard-interval", type=float, default=0.05, help="pipeline guard period (s)")
    p.add_argument("--precommit", action="store_true", help="commit feedback before next frame")
    p.add_argument("--compile", action="store_true", help="mx.compile the laya-p2p-1 stream")
    p.add_argument("--max-memory-gap", type=float, help="laya-p2p-1 memory gap bound (s)")
    p.add_argument(
        "--planner",
        action="store_true",
        help="run the Molmo planner: it targets goal objects and issues one click per new target",
    )
    p.add_argument(
        "--hold",
        action="store_true",
        help="stateful transport: controls stay down across steps, drags, clutching, wheel",
    )
    p.add_argument(
        "--all-keys",
        action="store_true",
        help="allow every extended-vocabulary key plus the middle button and mouse wheel",
    )
    p.add_argument(
        "--planner-act",
        action="store_true",
        help="let the planner also approach its target (W/A/S/D) and click the skill Molmo points to",
    )
    p.add_argument(
        "--pointer",
        action="store_true",
        help="move the cursor to the checkpoint's absolute pointer output on new mouse presses",
    )
    p.add_argument(
        "--wait-for-focus",
        type=float,
        default=0,
        help="seconds to wait for the user to focus the game before an --execute trial",
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
