import time
from types import SimpleNamespace

import mlx.core as mx
import pytest
from PIL import Image

from laya_vision_stitch.laya_p2p_stream import LayaP2PStream
from laya_vision_stitch.planner_actions import TargetActions, direction_keys
from laya_vision_stitch.temporal_live import Pulse, bounded_action
from tests.test_pointer_mode import FakeTarget


def test_direction_keys_cover_eight_sectors():
    assert direction_keys(0, -1) == ["w"] and direction_keys(0, 1) == ["s"]
    assert direction_keys(-1, 0) == ["a"] and direction_keys(1, 0) == ["d"]
    assert direction_keys(1, -1) == ["w", "d"] and direction_keys(-1, 1) == ["s", "a"]
    assert direction_keys(1, -0.2) == ["d"]  # within 22.5 degrees of the axis
    assert direction_keys(0, 0) == []


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_target_actions_approach_reselect_then_use_skill():
    clock = Clock()
    actions = TargetActions(near=0.12, reselect=4.0, act_every=0.8, clock=clock)
    assert actions.step([0.8, 0.2]) == {}  # nothing before the target was selected
    actions.selected()
    assert actions.step([0.8, 0.2]) == {"hold": ["w", "d"]}
    clock.t = 4.0
    step = actions.step([0.8, 0.2])
    assert step["click"] == {"xy": [0.8, 0.2], "kind": "select"} and step["hold"] == ["w", "d"]
    assert actions.step([0.55, 0.5]) == {"hold": []}  # near, but no skill button known
    actions.skill_xy = [0.3, 0.93]
    assert actions.step([0.55, 0.5])["click"] == {"xy": [0.3, 0.93], "kind": "skill"}
    assert "click" not in actions.step([0.55, 0.5])  # waits act_every between skill clicks
    clock.t = 5.0
    assert actions.step([0.55, 0.5])["click"]["kind"] == "skill"
    actions.reset()
    assert actions.step([0.55, 0.5]) == {}


def test_planner_movement_replaces_policy_movement_and_skill_click_is_applied():
    action = bounded_action(
        {
            "buttons": ["a", "s", "space"],
            "mouse_delta": [0, 0],
            "planner_hold": ["w", "d"],
            "planner_click": {"xy": [0.3, 0.93], "button": "mouse_left", "kind": "skill"},
        }
    )
    assert action["buttons"] == ["d", "mouse_left", "space", "w"]
    assert action["planner_click"]["kind"] == "skill" and action["pointer_xy"] == [0.3, 0.93]
    target = FakeTarget()
    pulse = Pulse(target, [0, 87, 1289, 785], pointer=True)
    pulse.apply(action, time.perf_counter() + 5)
    pulse.finish()
    assert action["pointer_applied"] == pytest.approx([0.3, 0.93])
    assert ("key", "w", True) in target.calls and ("key", "a", True) not in target.calls
    with pytest.raises(ValueError):
        bounded_action({"buttons": [], "mouse_delta": [0, 0], "planner_hold": ["q"]})


class StubPlanner:
    def __init__(self):
        self.answer, self.submitted = None, []

    def poll(self):
        answer, self.answer = self.answer, None
        return answer

    def submit(self, image, goal, skill=False, point=True):
        self.submitted.append((skill, point))
        return len(self.submitted)


class StubTracker:
    def __init__(self):
        self.xy = None

    def set_target(self, image, xy):
        self.xy = list(xy)

    def track(self, image, window=None):
        return list(self.xy), 0.9


def test_stream_selects_approaches_and_uses_the_skill_molmo_found():
    planner = StubPlanner()
    policy = SimpleNamespace(
        vision=lambda pixels: (None, mx.zeros((1, 1024))),
        prefix=lambda image, goal, spatial=None: mx.zeros((1, 4, 1024)),
        context=lambda prefix, actions=None, caches=None, position=0: (
            mx.zeros((1, 1, 1024)),
            None,
        ),
        decode=lambda context, temperature: (
            mx.array([[6, 0, 0, 0, 0, 0, 11, 8]]),
            (mx.zeros((1, 20)),),
        ),
        pointer_encoder=object(),
    )
    runtime = SimpleNamespace(
        model=SimpleNamespace(
            policy=policy, bridge=lambda x: x, goal_features=lambda *a: mx.ones((1, 768))
        ),
        prepare_goal=lambda goal: (mx.ones((1, 1), mx.int32), mx.ones((1, 1), mx.bool_)),
        metadata={},
    )
    stream = LayaP2PStream(runtime, planner=planner, act=True, track_every=1)
    stream.tracker = StubTracker()
    image = Image.new("RGB", (64, 36))
    row = {"goal": "Go", "frames": [{"image": image, "age_seconds": 0}], "previous_actions": []}
    out = stream.predict(row, session_id="s", timestamp_seconds=0)
    assert planner.submitted == [(False, True)]  # no target yet: look for targets only
    planner.answer = {
        "frame_id": 1,
        "phrase": "monsters",
        "points": [[0.8, 0.2]],
        "planner_seconds": 3,
        "error": None,
        "image": image,
    }
    feedback = {**row, "previous_actions": [{"buttons": ["a"], "mouse_delta": [0, 0]}]}
    out = stream.predict(feedback, session_id="s", timestamp_seconds=0.05)
    assert out["planner_click"]["kind"] == "select" and out["planner_click"]["xy"] == [0.8, 0.2]
    assert planner.submitted[-1] == (True, False)  # following a target: ask for the skill
    for skill in ([0.3, 0.93], [0.31, 0.93]):  # two agreeing skill-only answers
        planner.answer = {
            "frame_id": 2,
            "phrase": None,
            "points": None,
            "planner_seconds": 2,
            "error": None,
            "image": image,
            "skill": skill,
        }
        out = stream.predict(
            feedback, session_id="s", timestamp_seconds=0.1 + len(planner.submitted) * 0.01
        )
    assert out["planner"]["skill_confirmed"] == [0.305, 0.93]
    assert out["planner_hold"] == ["w", "d"] and "planner_click" not in out
    assert stream.tracker.xy == [0.8, 0.2]  # skill answers leave the target alone
    stream.tracker.xy = [0.52, 0.5]  # the character reached the target
    out = stream.predict(feedback, session_id="s", timestamp_seconds=0.5)
    assert out["planner_hold"] == [] and out["planner_click"] == {
        "xy": [0.305, 0.93],
        "kind": "skill",
        "button": "mouse_left",
    }


def test_skill_button_needs_two_agreeing_answers_in_the_safe_area():
    actions = TargetActions()
    assert not actions.skill_answer([0.3, 0.93]) and actions.skill_xy is None
    assert not actions.skill_answer([0.6, 0.93])  # disagrees: becomes the new candidate
    assert actions.skill_answer([0.61, 0.94]) and actions.skill_xy == pytest.approx([0.605, 0.935])
    assert not actions.skill_answer([0.95, 0.02]) and not actions.skill_answer(
        [0.95, 0.02]
    )  # top menu band
    assert actions.skill_xy == pytest.approx([0.605, 0.935])  # a confirmed button is kept
