import threading
import time
from types import SimpleNamespace

import pytest

from laya_vision_stitch.live_pipeline import (
    Frame,
    GuardVeto,
    LatestFrameSlot,
    LivePipeline,
)
from laya_vision_stitch.temporal_live import PreemptingPulse, Pulse


class Replay(threading.Thread):
    def __init__(self, slot, fps=200):
        super().__init__(daemon=True)
        self.slot, self.period, self.stopped = slot, 1 / fps, threading.Event()

    def run(self):
        i = 0
        while not self.stopped.is_set():
            self.slot.put(Frame(f"image-{i}", time.perf_counter(), i))
            i += 1
            time.sleep(self.period)


class Dispatcher:
    def __init__(self, veto=None):
        self.applied, self.released, self.veto = [], [], veto
        self.lock = threading.Lock()

    def apply(self, action, deadline):
        with self.lock:
            if self.veto is not None and self.veto.denial():
                raise RuntimeError(self.veto.denial())
            start = time.perf_counter()
            self.applied.append((start, dict(action)))
        return {"dispatch_start": start, "pulse_wait_ms": 0.0, "dispatch_guard_ms": 0.0}

    def release(self):
        with self.lock:
            self.released.append(time.perf_counter())

    def finish(self):
        self.release()


def proposal(buttons=("w",), mouse=(0.0, 0.0)):
    return {
        "buttons": list(buttons),
        "mouse_delta": list(mouse),
        "image_to_outputs_ms": 1.0,
        "state_reset": False,
    }


def run(predict=None, guards=None, seconds=0.3, execute=True, commit=None, **options):
    slot = LatestFrameSlot()
    replay = Replay(slot)
    replay.start()
    dispatcher = Dispatcher(options.pop("veto", None))
    calls = []

    def default_predict(image, frame, previous):
        calls.append(previous)
        time.sleep(0.005)
        return proposal()

    pipeline = LivePipeline(
        capture=slot,
        prepare=lambda frame: frame.image,
        predict=predict or default_predict,
        dispatcher=dispatcher,
        guards=guards or [("ok", lambda: None)],
        deadline=time.perf_counter() + seconds,
        execute=execute,
        commit=commit,
        veto=dispatcher.veto,
        **options,
    )
    summary = pipeline.run()
    replay.stopped.set()
    return pipeline, summary, dispatcher, calls


def test_slot_keeps_only_the_newest_frame():
    slot = LatestFrameSlot()
    for i in range(3):
        slot.put(Frame(i, 0.0, i))
    assert slot.wait_newer(-1, 0.01).sequence == 2
    assert slot.dropped == 2 and slot.published == 3
    assert slot.wait_newer(2, 0.01) is None
    slot.close()
    slot.put(Frame(9, 0.0, 9))
    assert slot.wait_newer(2, 0.01) is None


def test_veto_is_fail_closed_until_fresh_pass_and_sticky_after_trip():
    now = [0.0]
    veto = GuardVeto(0.1, clock=lambda: now[0])
    assert veto.denial() == "No completed guard pass"
    veto.passed(0.0)
    assert veto.denial() is None
    now[0] = 0.2
    assert "stale" in veto.denial()
    veto.passed(0.2)
    veto.trip("Window changed or focus lost")
    veto.passed(0.2)
    assert veto.denial() == "Window changed or focus lost"


def test_pipeline_filters_keys_bounds_mouse_and_feeds_back_dispatched_action():
    seen = []

    def predict(image, frame, previous):
        seen.append(previous)
        time.sleep(0.005)
        return proposal(buttons=("w", "enter", "ctrl"), mouse=(0.5, -0.01))

    pipeline, summary, dispatcher, _ = run(predict)
    assert summary["stop_reason"] == "duration_complete"
    assert dispatcher.applied
    for _, action in dispatcher.applied:
        assert action["buttons"] == ["w"] and action["mouse_delta"] == [64 / 512, -5 / 512]
    assert seen[0] == []
    assert all(p == [{"buttons": ["w"], "mouse_delta": [64 / 512, -5 / 512]}] for p in seen[1:])
    record = pipeline.records[0]
    assert record["bounded_action"]["blocked_buttons"] == ["ctrl", "enter"]
    assert record["applied"] and record["screenshot_to_dispatch_start_ms"] >= 0


def test_guard_failure_stops_releases_and_blocks_later_dispatch():
    started = time.perf_counter()
    state = {"focused": True}

    def focus():
        return None if state["focused"] else "Window changed or focus lost"

    def lose_focus():
        time.sleep(0.1)
        state["focused"] = False

    threading.Thread(target=lose_focus, daemon=True).start()
    veto = GuardVeto(0.1)
    pipeline, summary, dispatcher, _ = run(
        guards=[("focus", focus)], seconds=2, guard_interval=0.01, veto=veto
    )
    assert summary["stop_reason"] == "Window changed or focus lost"
    assert time.perf_counter() - started < 1.5
    tripped = dispatcher.released[0]
    assert all(t < tripped for t, _ in dispatcher.applied)
    assert veto.denial() == "Window changed or focus lost"


def test_stop_file_guard(tmp_path):
    stop = tmp_path / "STOP"
    threading.Timer(0.05, stop.touch).start()
    _, summary, _, _ = run(
        guards=[("stop", lambda: "STOP requested" if stop.exists() else None)], seconds=2
    )
    assert summary["stop_reason"] == "STOP requested"


def test_observation_mode_never_dispatches():
    pipeline, summary, dispatcher, calls = run(execute=False)
    assert dispatcher.applied == [] and pipeline.records
    assert all(
        r["skip_reason"] == "observation_only" and not r["applied"] for r in pipeline.records
    )
    assert all(p == [] for p in calls)


def test_stale_guard_pass_skips_dispatch_without_stopping():
    def slow():
        time.sleep(0.25)
        return None

    pipeline, summary, dispatcher, _ = run(
        guards=[("slow", slow)], seconds=0.6, veto=GuardVeto(0.1)
    )
    assert summary["stop_reason"] == "duration_complete"
    assert any(r["skip_reason"] and "stale" in r["skip_reason"] for r in pipeline.records)


def test_precommit_feeds_dispatched_action_before_next_frame():
    committed, previous = [], []

    def predict(image, frame, prev):
        previous.append(prev)
        time.sleep(0.005)
        return proposal()

    pipeline, _, dispatcher, _ = run(predict, commit=committed.append)
    assert len(committed) >= len(dispatcher.applied) - 1 > 0
    assert all(c == {"buttons": ["w"], "mouse_delta": [0.0, 0.0]} for c in committed)
    assert all(p == [] for p in previous)


def test_invalid_configuration_rejected():
    options = dict(
        capture=LatestFrameSlot(),
        prepare=None,
        predict=None,
        dispatcher=Dispatcher(),
        guards=[("ok", lambda: None)],
        deadline=0,
        execute=False,
    )
    with pytest.raises(ValueError, match="Guard interval"):
        LivePipeline(**options, guard_interval=1)
    with pytest.raises(ValueError, match="cover one guard interval"):
        LivePipeline(**options, guard_interval=0.2, veto=GuardVeto(0.1))
    with pytest.raises(ValueError, match="Frame age"):
        LivePipeline(**options, max_apply_age=0.5, max_frame_age=0.18)
    with pytest.raises(ValueError, match="guard is required"):
        LivePipeline(**{**options, "guards": []})


def target(events, focused=True):
    return SimpleNamespace(
        original={"kCGWindowBounds": {"X": 0, "Y": 33}},
        Q=None,
        focused=lambda: focused,
        unchanged=lambda: True,
        text_input_focused=lambda: False,
        release=lambda: events.append(("release", time.perf_counter())),
        event=lambda key, down: events.append((key, time.perf_counter())),
        mouse_pressed=False,
        left_pressed=False,
    )


def test_preempting_pulse_does_not_wait_and_stale_timer_cannot_end_new_pulse():
    events = []
    pulse = PreemptingPulse(target(events), [0, 87, 1280, 720], lambda: None)
    deadline = time.perf_counter() + 1
    first = pulse.apply({"buttons": ["w"], "mouse_delta": [0, 0]}, deadline)
    second = pulse.apply({"buttons": ["a"], "mouse_delta": [0, 0]}, deadline)
    assert second["pulse_wait_ms"] < 20
    assert second["dispatch_start"] - first["dispatch_start"] < 0.04
    time.sleep(0.12)
    after_w = events[[name for name, _ in events].index("w") + 1 :]
    # The first pulse is released before "a"; its stale timer does not end the new pulse.
    assert [name for name, _ in after_w] == ["release", "a", "release"]
    assert after_w[2][1] - after_w[1][1] >= 0.045
    pulse.finish()


def test_serial_pulse_still_waits_for_previous_pulse():
    events = []
    pulse = Pulse(target(events), [0, 87, 1280, 720])
    deadline = time.perf_counter() + 1
    first = pulse.apply({"buttons": ["w"], "mouse_delta": [0, 0]}, deadline)
    second = pulse.apply({"buttons": ["a"], "mouse_delta": [0, 0]}, deadline)
    assert second["dispatch_start"] - first["dispatch_start"] >= 0.045
    pulse.finish()


def test_preempting_pulse_posts_nothing_when_vetoed():
    events = []
    pulse = PreemptingPulse(target(events), [0, 87, 1280, 720], lambda: "Text input focused")
    with pytest.raises(RuntimeError, match="Text input"):
        pulse.apply({"buttons": ["w"], "mouse_delta": [0, 0]}, time.perf_counter() + 1)
    assert "w" not in [name for name, _ in events]
