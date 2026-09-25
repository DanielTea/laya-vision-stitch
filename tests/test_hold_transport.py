import time

import pytest

from laya_vision_stitch.hold_transport import HoldTransport
from tests.test_pointer_mode import FakeTarget


class Recording(HoldTransport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = []

    def post_key(self, key, down):
        self.log.append(("key", key, down))

    def post_mouse(self, kind, button, point, delta=(0, 0)):
        self.log.append(("mouse", kind, button, tuple(round(v) for v in point), tuple(delta)))

    def post_wheel(self, notches, point):
        self.log.append(("wheel", notches))


def step(transport, buttons, delta=(0, 0), **extra):
    transport.log.clear()
    action = {"buttons": buttons, "mouse_delta": [delta[0] / 512, delta[1] / 512], **extra}
    transport.apply(action, time.perf_counter() + 5)
    return transport.log, action


def transport(**kwargs):
    # Viewport 1280x720 at window (100, 50) + crop offset (0, 87).
    return Recording(FakeTarget(), [0, 87, 1280, 720], {"w", "a", "space"}, **kwargs)


def test_controls_stay_down_until_released_and_only_changes_are_sent():
    t = transport()
    log, action = step(t, ["w", "mouse_right"])
    assert ("key", "w", True) in log and any(e[:3] == ("mouse", "down", 1) for e in log)
    assert action["held"] == ["mouse_right", "w"]
    log, _ = step(t, ["w", "mouse_right"], delta=(20, 0))
    assert [e[:3] for e in log] == [("mouse", "drag", 1)]  # no re-press, just the drag
    assert log[0][4] == (20, 0)
    log, _ = step(t, ["a"])
    assert ("key", "w", False) in log and ("key", "a", True) in log
    assert any(e[:3] == ("mouse", "up", 1) for e in log)


def test_middle_button_drag_and_wheel_notches():
    t = transport()
    step(t, ["mouse_middle"])
    log, _ = step(t, ["mouse_middle", "scroll_down"], delta=(-8, 4))
    assert log[0][:3] == ("mouse", "drag", 2) and ("wheel", -1) in log


def test_drag_clutches_at_the_edge_instead_of_stopping():
    t = transport()
    step(t, ["mouse_right"])
    start = t.point
    for _ in range(6):
        log, _ = step(t, ["mouse_right"], delta=(64, 0))
    assert t.clutches >= 1
    clutch = [e[:3] for e in log] if t.clutches == 1 else None
    assert t.point[0] > start[0] or clutch is not None
    # Every drag event carries the model's full delta.
    assert all(e[4] == (64, 0) for e in t.log if e[1] == "drag")


def test_planner_click_is_a_fresh_left_press_at_its_target():
    t = transport(pointer=True)
    step(t, ["mouse_left"])
    log, action = step(
        t,
        ["mouse_left"],
        planner_click={"xy": [0.2, 0.3], "button": "mouse_left"},
        pointer_xy=[0.2, 0.3],
    )
    kinds = [e[1] for e in log if e[0] == "mouse"]
    assert kinds[:3] == ["up", "move", "down"]
    assert action["pointer_applied"] == pytest.approx([0.2, 0.3])


def test_watchdog_releases_everything_when_actions_stop():
    t = transport(watchdog=0.05)
    step(t, ["w", "mouse_left"])
    t.log.clear()
    time.sleep(0.15)
    assert ("key", "w", False) in t.log and any(e[:3] == ("mouse", "up", 0) for e in t.log)
    assert not t.keys_down and not t.buttons_down


def test_unknown_keys_are_ignored_and_errors_release_all():
    t = transport()
    log, action = step(t, ["q", "w"])
    assert action["held"] == ["w"]
    t.target.focused = lambda: False
    with pytest.raises(RuntimeError):
        step(t, ["w"])
    assert not t.keys_down
