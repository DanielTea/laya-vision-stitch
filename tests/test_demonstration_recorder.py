import json

import pytest
from PIL import Image

from laya_vision_stitch.demonstration_recorder import export, interval_action


def test_past_button_is_not_future_action_and_future_press_is_preserved():
    events = [
        {"timestamp": 0, "kind": "state", "buttons": ["w"]},
        {"timestamp": 0.09, "kind": "up", "button": "w"},
        {"timestamp": 0.11, "kind": "down", "button": "1"},
        {"timestamp": 0.12, "kind": "up", "button": "1"},
        {"timestamp": 0.15, "kind": "down", "button": "s"},
    ]
    action, detail = interval_action(events, 0.1, 0.15)
    assert action["buttons"] == ["1"]
    assert detail["held_seconds"]["1"] == pytest.approx(0.01)
    assert detail["pressed"] == ["1"]


def test_held_mouse_and_only_future_relative_motion_are_recorded():
    events = [
        {"timestamp": 0, "kind": "state", "buttons": ["mouse_right"]},
        {"timestamp": 0.01, "kind": "move", "delta": [99, 99]},
        {"timestamp": 0.11, "kind": "move", "delta": [4, -2]},
        {"timestamp": 0.12, "kind": "move", "delta": [6, 1]},
    ]
    action, detail = interval_action(events, 0.1, 0.15)
    assert action["mouse_delta"] == [10 / 512, -1 / 512]
    assert action["buttons"] == ["mouse_right"]
    assert detail["held_seconds"]["mouse_right"] == pytest.approx(0.05)


def test_click_position_comes_from_press_and_out_of_range_motion_is_rejected():
    action, _ = interval_action(
        [
            {"timestamp": 0.01, "kind": "down", "button": "mouse_left", "pointer_xy": [0.2, 0.3]},
            {"timestamp": 0.02, "kind": "move", "pointer_xy": [0.8, 0.9], "delta": [2, 0]},
        ],
        0,
        0.05,
    )
    assert action["pointer_xy"] == [0.2, 0.3]
    with pytest.raises(ValueError, match="silently clip"):
        interval_action([{"timestamp": 0.01, "kind": "move", "delta": [513, 0]}], 0, 0.05)


def test_export_drops_incomplete_intervals_and_does_not_leak_future_history(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"game": "test", "goal": "act", "controls": "W"})
    )
    (tmp_path / "summary.json").write_text(json.dumps({"ended_at": 0.2}))
    Image.new("RGB", (16, 16)).save(tmp_path / "frame.jpg")
    frames = [
        {"timestamp": t, "sequence": i, "image": "frame.jpg"}
        for i, t in enumerate([0.01, 0.04, 0.1, 0.18])
    ]
    events = [
        {"timestamp": 0, "kind": "state", "buttons": []},
        {"timestamp": 0.02, "kind": "down", "button": "w"},
        {"timestamp": 0.07, "kind": "up", "button": "w"},
        {"timestamp": 0.1, "kind": "down", "button": "1"},
    ]
    for name, items in (("frames", frames), ("controls", events)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items))
    audit = export(tmp_path)
    assert audit["examples"] == 3
    assert audit["rejected"] == {"incomplete_interval": 1}
    rows = [
        json.loads(line) for line in (tmp_path / "demonstrations.jsonl").read_text().splitlines()
    ]
    assert rows[0]["action"]["buttons"] == ["w"]
    assert rows[1]["previous_actions"] == []  # Last label extends past this screenshot.
    assert rows[2]["previous_actions"] == [rows[1]["action"]]
    assert rows[2]["action"]["buttons"] == ["1"]
