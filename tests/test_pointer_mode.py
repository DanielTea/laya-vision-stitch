import time

import mlx.core as mx
import numpy as np
import pytest

from laya_vision_stitch.pointer_head import GRID, PointerHead
from laya_vision_stitch.temporal_live import Pulse, bounded_action


class FakeTarget:
    Q = None

    def __init__(self):
        self.original = {
            "kCGWindowBounds": {"X": 100.0, "Y": 50.0, "Width": 1300.0, "Height": 900.0}
        }
        self.calls, self.mouse_pressed, self.left_pressed = [], False, False

    def focused(self):
        return True

    def unchanged(self):
        return True

    def text_input_focused(self):
        return False

    def event(self, key, down):
        self.calls.append(("key", key, down))

    def mouse_event(self, kind, point):
        self.calls.append(("right" if kind != "move" else "move", kind, point))

    def left_event(self, kind, point):
        self.calls.append(("left", kind, point))

    def release(self):
        self.calls.append(("release",))


def act(pulse, buttons, xy):
    action = bounded_action({"buttons": buttons, "mouse_delta": [0, 0], "pointer_xy": xy})
    pulse.apply(action, time.perf_counter() + 5)
    pulse.finish()
    return action


def test_pointer_moves_only_on_new_press_and_stays_in_safe_area():
    target = FakeTarget()
    pulse = Pulse(target, [0, 87, 1280, 720], pointer=True)
    first = act(pulse, ["mouse_left"], [0.9, 0.01])
    assert first["pointer_applied"] == pytest.approx([0.9, 0.08])  # kept below the top band
    moves = [c for c in target.calls if c[0] == "move"]
    assert moves[-1][2] == pytest.approx((100 + 0.9 * 1280, 50 + 87 + 0.08 * 720))
    target.calls.clear()
    held = act(pulse, ["mouse_left"], [0.2, 0.5])  # still held: no absolute jump
    assert "pointer_applied" not in held and not [c for c in target.calls if c[0] == "move"]


def test_pointer_is_ignored_without_pointer_mode_and_validated():
    target = FakeTarget()
    pulse = Pulse(target, [0, 87, 1280, 720])
    action = act(pulse, ["mouse_left"], [0.9, 0.9])
    assert "pointer_applied" not in action and pulse.point == (100 + 640, 50 + 87 + 360)
    with pytest.raises(ValueError):
        bounded_action({"buttons": [], "mouse_delta": [0, 0], "pointer_xy": [float("nan"), 0]})


def test_pointer_head_targets_round_trip_and_predict_shape():
    xy = mx.array([[0.0, 0.0], [0.51, 0.26], [0.999, 0.999]])
    index, offset = PointerHead(width=32).targets(xy)
    cell = np.stack([np.asarray(index) % GRID, np.asarray(index) // GRID], 1)
    np.testing.assert_allclose((cell + np.asarray(offset)) / GRID, np.asarray(xy), atol=1e-6)
    head = PointerHead(width=32)
    out = head.predict(mx.random.normal((2, 12, 12, 112)), mx.random.normal((2, 1024)))
    assert out.shape == (2, 2) and float(mx.min(out)) >= 0 and float(mx.max(out)) <= 1
    assert float(head.loss(mx.random.normal((3, 12, 12, 112)), mx.random.normal((3, 1024)), xy)) > 0


def test_planner_click_forces_a_new_left_press_at_its_target():
    target = FakeTarget()
    pulse = Pulse(target, [0, 87, 1280, 720], pointer=True)
    act(pulse, ["mouse_left"], [0.5, 0.5])  # left already held from the policy
    target.calls.clear()
    action = bounded_action(
        {
            "buttons": ["mouse_left"],
            "mouse_delta": [0, 0],
            "planner_click": {"xy": [0.2, 0.3], "button": "mouse_left"},
        }
    )
    pulse.apply(action, time.perf_counter() + 5)
    pulse.finish()
    assert action["pointer_applied"] == pytest.approx([0.2, 0.3])
    assert [c for c in target.calls if c[0] == "left"][0][1] == "down"
    with pytest.raises(ValueError):
        bounded_action(
            {
                "buttons": [],
                "mouse_delta": [0, 0],
                "planner_click": {"xy": [0.2, 0.3], "button": "mouse_right"},
            }
        )
