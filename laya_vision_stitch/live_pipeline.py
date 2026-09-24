"""Pipelined live loop: newest-frame slot, inference worker, dispatch worker, async guards.

Transport scheduling only. Proposals pass through the same allowed-key filter and mouse
bound as the serial runner; there are no gameplay rules. Capture, preparation, prediction,
guards and dispatch are injected, so the loop runs offline with fakes.

Safety semantics compared with the serial runner:
- Guards (STOP file, focus/layout, text input, HUD) run on the calling thread every
  `guard_interval` seconds. A failing guard stops the run and releases inputs at once.
- Dispatch is fail-closed: it posts only while no guard has failed and the last complete
  guard pass started at most `max_guard_age` seconds earlier; otherwise the decision is
  logged as not applied (which resets model memory, as in the serial runner).
- Residual risk: a focus, layout or text-input change is noticed up to one guard interval
  plus one guard pass later than in the serial runner, which re-checks before posting.
- A frame older than `max_frame_age` at pickup stops the run (stale capture). A proposal
  whose frame is older than `max_apply_age` at dispatch is logged but not applied.
- The dispatcher must not wait for the previous pulse. The model memory still receives only
  the action that was actually dispatched: the inference worker waits for dispatch feedback
  (or, with `commit`, commits it) before predicting from the next frame.
"""

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from .temporal_live import bounded_action


@dataclass(frozen=True)
class Frame:
    image: object
    captured_at: float
    sequence: int


class LatestFrameSlot:
    """Single-slot mailbox. A new frame replaces an unconsumed one; nothing queues."""

    def __init__(self):
        self._cond = threading.Condition()
        self._frame, self._closed, self._consumed = None, False, -1
        self.published = self.dropped = 0

    def put(self, frame):
        with self._cond:
            if self._closed:
                return
            if self._frame is not None and self._frame.sequence > self._consumed:
                self.dropped += 1
            self._frame = frame
            self.published += 1
            self._cond.notify_all()

    def latest(self):
        with self._cond:
            return self._frame

    def wait_newer(self, sequence, timeout):
        """Newest frame whose sequence exceeds `sequence`; None on timeout or close."""
        deadline = time.perf_counter() + timeout
        with self._cond:
            while not self._closed and (self._frame is None or self._frame.sequence <= sequence):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            if self._closed:
                return None
            self._consumed = self._frame.sequence
            return self._frame

    def close(self):
        with self._cond:
            self._closed = True
            self._cond.notify_all()


class PollingCapture:
    """`wait_newer` over a source that only exposes `latest()` (ScreenCaptureKit stream)."""

    def __init__(self, source, poll_seconds=0.001):
        if not 0 < poll_seconds <= 0.01:
            raise ValueError("Poll interval must be in (0, 10] ms")
        self.source, self.poll_seconds = source, poll_seconds

    def wait_newer(self, sequence, timeout):
        deadline = time.perf_counter() + timeout
        while True:
            frame = self.source.latest()
            if frame is not None and frame.sequence > sequence:
                return frame
            if time.perf_counter() >= deadline:
                return None
            time.sleep(self.poll_seconds)


class GuardVeto:
    """Fail-closed dispatch permission written by the guard loop, read by dispatch."""

    def __init__(self, max_age, clock=time.perf_counter):
        if not 0 < max_age <= 1:
            raise ValueError("Guard freshness bound must be in (0, 1] seconds")
        self.max_age, self.clock = max_age, clock
        self.lock = threading.Lock()
        self.reason, self.checked_at = None, None

    def trip(self, reason):
        with self.lock:
            if self.reason is None:
                self.reason = reason

    def passed(self, started):
        with self.lock:
            self.checked_at = started

    def denial(self):
        """None when posting is permitted, otherwise the reason it is not."""
        with self.lock:
            if self.reason is not None:
                return self.reason
            if self.checked_at is None:
                return "No completed guard pass"
            age = self.clock() - self.checked_at
            return f"Guard pass is stale ({1000 * age:.1f} ms)" if age > self.max_age else None


def percentiles(values):
    values = [v for v in values if v is not None]
    return {
        "samples": len(values),
        "p50": float(np.median(values)) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
    }


class LivePipeline:
    """Run capture -> inference -> dispatch with guards on the calling (main) thread.

    Interfaces:
    - capture.wait_newer(sequence, timeout) -> Frame | None
    - prepare(frame) -> model image (for example the calibrated viewport crop)
    - predict(image, frame, previous) -> proposal; `previous` is [] or [applied action]
    - commit(applied) optional: feed back the applied action before the next frame
    - dispatcher.apply(action, deadline) -> timing dict, non-blocking; raises to stop;
      dispatcher.release() and dispatcher.finish() release held inputs
    - guards: [(name, check)], check() -> None or a stop reason
    - pump() optional, called every loop tick (Cocoa run loop on macOS)

    The guard loop must run on the thread that owns the platform event loop, so `run`
    blocks the caller; inference and dispatch run on worker threads.
    """

    def __init__(
        self,
        *,
        capture,
        prepare,
        predict,
        dispatcher,
        guards,
        deadline,
        execute,
        veto=None,
        commit=None,
        guard_interval=0.05,
        max_frame_age=0.18,
        max_apply_age=0.18,
        feedback_timeout=0.5,
        on_record=None,
        pump=None,
        start=None,
        clock=time.perf_counter,
    ):
        if not 0.005 <= guard_interval <= 0.5:
            raise ValueError("Guard interval must be 5-500 ms")
        if veto is None:
            veto = GuardVeto(max(0.1, 2 * guard_interval), clock)
        if veto.max_age < guard_interval:
            raise ValueError("Guard freshness bound must cover one guard interval")
        if not 0 < max_apply_age <= max_frame_age <= 1:
            raise ValueError("Frame age bounds must satisfy 0 < apply <= pickup <= 1 s")
        if not guards:
            raise ValueError("At least one guard is required")
        self.capture, self.prepare, self.predict, self.commit = capture, prepare, predict, commit
        self.dispatcher, self.guards, self.veto = dispatcher, list(guards), veto
        self.deadline, self.execute, self.clock = deadline, bool(execute), clock
        self.start = clock() if start is None else start
        self.guard_interval, self.feedback_timeout = guard_interval, feedback_timeout
        self.max_frame_age, self.max_apply_age = max_frame_age, max_apply_age
        self.on_record, self.pump = on_record, pump
        self.stopped, self.lock = threading.Event(), threading.Lock()
        self.outbox, self.feedback = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
        self.stop_reason, self.error = None, None
        self.records, self.guard_passes, self.skips = [], [], {}

    def stop(self, reason):
        with self.lock:
            if self.stop_reason is None:
                self.stop_reason = reason
        # Deny first, then release: a dispatch past its permission check still holds the
        # transport lock, so release happens after it rather than before.
        self.veto.trip(reason)
        self.stopped.set()
        self.dispatcher.release()

    def _fail(self, error):
        if not isinstance(error, (RuntimeError, KeyboardInterrupt)):
            self.error = error
        self.stop(str(error) or type(error).__name__)

    def run(self):
        workers = [
            threading.Thread(target=self._infer, name="pipeline-inference", daemon=True),
            threading.Thread(target=self._dispatch, name="pipeline-dispatch", daemon=True),
        ]
        next_pass = self.clock()
        try:
            for worker in workers:
                worker.start()
            while not self.stopped.is_set():
                if self.pump is not None:
                    self.pump()
                now = self.clock()
                if now >= self.deadline:
                    self.stop("duration_complete")
                    break
                if now >= next_pass:
                    for _, check in self.guards:
                        reason = check()
                        if reason:
                            raise RuntimeError(reason)
                    self.veto.passed(now)
                    self.guard_passes.append(1000 * (self.clock() - now))
                    next_pass = now + self.guard_interval
                self.stopped.wait(max(0.0, min(0.002, next_pass - self.clock())))
        except BaseException as error:
            self._fail(error)
        finally:
            self.stop("stopped")
            for worker in workers:
                if worker.is_alive():
                    worker.join(5)
            self.dispatcher.finish()
        if self.error is not None:
            raise self.error
        return self.summary()

    def _infer(self):
        sequence, previous = -1, []
        try:
            while not self.stopped.is_set():
                frame = self.capture.wait_newer(sequence, 0.05)
                if frame is None:
                    continue
                sequence, picked = frame.sequence, self.clock()
                if picked - frame.captured_at > self.max_frame_age:
                    raise RuntimeError("Stale capture")
                image = self.prepare(frame)
                started = self.clock()
                proposal = self.predict(image, frame, previous)
                action = bounded_action(proposal)
                times = {"picked": picked, "started": started, "finished": self.clock()}
                self.outbox.put((frame, image, proposal, action, times))
                applied = self._await_feedback()
                if applied is None:
                    previous = []
                elif self.commit is not None:
                    # Memory update runs before the next frame is picked, not after it.
                    self.commit(applied)
                    previous = []
                else:
                    previous = [applied]
        except BaseException as error:
            self._fail(error)

    def _await_feedback(self):
        until = self.clock() + self.feedback_timeout
        while not self.stopped.is_set():
            try:
                return self.feedback.get(timeout=0.01)
            except queue.Empty:
                if self.clock() > until:
                    raise RuntimeError("Dispatch feedback timed out") from None
        raise RuntimeError(self.stop_reason or "stopped")

    def _dispatch(self):
        try:
            while not self.stopped.is_set():
                try:
                    item = self.outbox.get(timeout=0.01)
                except queue.Empty:
                    continue
                frame, image, proposal, action, times = item
                now = times["dispatch"] = self.clock()
                if now >= self.deadline:
                    self.stop("duration_complete")
                    break
                skip = None if self.execute else "observation_only"
                if skip is None and now - frame.captured_at > self.max_apply_age:
                    skip = "frame_too_old"
                if skip is None:
                    skip = self.veto.denial()
                times["checked"] = self.clock()
                posted = self.dispatcher.apply(action, self.deadline) if skip is None else None
                applied = (
                    {"buttons": action["buttons"], "mouse_delta": action["mouse_delta"]}
                    if posted is not None
                    else None
                )
                self.feedback.put(applied)
                if skip is not None:
                    self.skips[skip] = self.skips.get(skip, 0) + 1
                self._record(frame, image, proposal, action, posted, skip, times)
        except BaseException as error:
            self._fail(error)

    def _record(self, frame, image, proposal, action, posted, skip, times):
        def since_capture(t):
            return None if t is None else 1000 * (t - frame.captured_at)

        record = {
            "step": len(self.records),
            "elapsed_s": frame.captured_at - self.start,
            "sequence": frame.sequence,
            "proposal": proposal,
            "bounded_action": action,
            "applied": posted is not None,
            "skip_reason": skip,
            "frame_age_at_pickup_ms": since_capture(times["picked"]),
            "frame_age_before_inference_ms": since_capture(times["started"]),
            "frame_age_after_inference_ms": since_capture(times["finished"]),
            "post_inference_hud_check_ms": None,
            "dispatch_queue_ms": 1000 * (times["dispatch"] - times["finished"]),
            "veto_check_ms": 1000 * (times["checked"] - times["dispatch"]),
            "pulse_wait_ms": posted["pulse_wait_ms"] if posted else None,
            "dispatch_guard_ms": posted["dispatch_guard_ms"] if posted else None,
            "screenshot_to_dispatch_start_ms": since_capture(posted["dispatch_start"])
            if posted
            else None,
            "screenshot_to_first_event_ms": since_capture(posted.get("first_event_posted"))
            if posted
            else None,
        }
        self.records.append(record)
        if self.on_record is not None:
            self.on_record(record, image)

    def summary(self):
        records = self.records
        elapsed = [r["elapsed_s"] for r in records]
        span = elapsed[-1] - elapsed[0] if len(elapsed) > 1 else None
        return {
            "stop_reason": self.stop_reason,
            "decisions": len(records),
            "applied": sum(r["applied"] for r in records),
            "decisions_per_second": (len(records) - 1) / span if span else None,
            "skipped": dict(self.skips),
            "guard_pass_ms": percentiles(self.guard_passes),
            "screenshot_to_dispatch_start_ms": percentiles(
                [r["screenshot_to_dispatch_start_ms"] for r in records]
            ),
            "frame_age_before_inference_ms": percentiles(
                [r["frame_age_before_inference_ms"] for r in records]
            ),
            "inference_ms": percentiles([r["proposal"]["image_to_outputs_ms"] for r in records]),
            "dispatch_queue_ms": percentiles([r["dispatch_queue_ms"] for r in records]),
            "state_resets": sum(bool(r["proposal"].get("state_reset")) for r in records),
        }
