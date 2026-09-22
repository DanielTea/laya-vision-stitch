from types import SimpleNamespace

import pytest

from laya_vision_stitch.temporal_live import Pulse, bounded_action


def test_transport_does_not_invent_actions_or_allow_chat_shortcuts():
    action = bounded_action(
        {"buttons": ["w", "enter", "ctrl", "r", "mouse_right"], "mouse_delta": [0.5, -0.01]}
    )
    assert action["buttons"] == ["mouse_right", "w"]
    assert action["blocked_buttons"] == ["ctrl", "enter", "r"]
    assert action["mouse_delta"] == [64 / 512, -5 / 512]
    assert action["mouse_clamped"]
    with pytest.raises(ValueError, match="Invalid mouse"):
        bounded_action({"buttons": [], "mouse_delta": [float("nan"), 0]})


def test_no_post_when_game_loses_focus_and_cleanup_runs():
    released = []
    target = SimpleNamespace(
        original={"kCGWindowBounds": {"X": 0, "Y": 33}},
        Q=None,
        focused=lambda: False,
        release=lambda: released.append(True),
    )
    pulse = Pulse(target, [0, 87, 1280, 720])
    with pytest.raises(RuntimeError, match="lost focus"):
        pulse.apply({"buttons": ["w"], "mouse_delta": [0, 0]}, float("inf"))
    assert released


def test_keys_release_even_when_following_inference_has_not_finished():
    import threading
    import time

    released = threading.Event()
    pressed = []
    target = SimpleNamespace(
        original={"kCGWindowBounds": {"X": 0, "Y": 33}},
        Q=None,
        focused=lambda: True,
        unchanged=lambda: True,
        text_input_focused=lambda: False,
        release=lambda: released.set(),
        event=lambda key, down: pressed.append((key, down)),
    )
    pulse = Pulse(target, [0, 87, 1280, 720])
    pulse.apply({"buttons": ["w"], "mouse_delta": [0, 0]}, time.perf_counter() + 1)
    released.clear()
    assert released.wait(0.5)
    assert pressed == [("w", True)]
    pulse.finish()
