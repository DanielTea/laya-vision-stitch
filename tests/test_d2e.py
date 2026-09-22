from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from laya_vision_stitch.d2e_data import BUTTONS, Timeline, load_events  # noqa: E402
from laya_vision_stitch.trainable_model import PolicyConfig, TrainableRuntime  # noqa: E402


def test_timeline_is_causal_and_integrates_holds():
    timeline = Timeline(
        [(0.0, ["w"]), (0.16, ["d"]), (0.24, [])],
        [(0.05, 1, 2), (0.1, 10, -5), (0.18, 20, -5), (0.2, 100, 100)],
    )
    past, _ = timeline.action(0, 0.1, 100)
    future, _ = timeline.action(0.1, 0.1, 100)
    assert past["buttons"] == ["w"]
    assert past["mouse_delta"] == [0.01, 0.02]
    assert future["buttons"] == ["w"]  # 60% of interval, despite a later D press
    np.testing.assert_allclose(future["mouse_delta"], [0.3, -0.1])
    clipped, saturated = timeline.action(0.2, 0.1, 50)
    assert saturated
    assert clipped["mouse_delta"] == [1, 1]


def test_expand_buttons_preserves_old_logits():
    config = PolicyConfig()
    actions = SimpleNamespace(buttons=nn.Linear(4, len(config.buttons)))
    runtime = TrainableRuntime(
        SimpleNamespace(policy_config=config, actions=actions), None, None, {}
    )
    x = mx.array([[1.0, 2.0, 3.0, 4.0]])
    before = np.asarray(actions.buttons(x))
    runtime.expand_buttons(BUTTONS)
    np.testing.assert_array_equal(np.asarray(actions.buttons(x))[:, :7], before)
    assert len(config.buttons) == len(BUTTONS)
    with pytest.raises(ValueError):
        runtime.expand_buttons(["w"])


def test_mcap_states_and_events_use_log_clock(tmp_path):
    import json

    pytest.importorskip("mcap")
    from mcap.writer import Writer

    p = tmp_path / "sample.mcap"
    with p.open("wb") as f:
        writer = Writer(f)
        writer.start()
        schema = writer.register_schema("test", "jsonschema", b"{}")
        channels = {
            t: writer.register_channel(t, "json", schema)
            for t in ["keyboard/state", "keyboard", "mouse/state", "mouse/raw", "screen"]
        }
        rows = [
            ("keyboard/state", {"buttons": [87]}),
            ("mouse/state", {"buttons": ["left"]}),
            ("keyboard", {"event_type": "press", "vk": 69}),
            ("mouse/raw", {"last_x": 3, "last_y": -2, "timestamp": 99999999}),
            ("screen", {"media_ref": {"pts_ns": 123000000, "uri": "sample.mkv"}}),
        ]
        for i, (topic, value) in enumerate(rows):
            writer.add_message(channels[topic], (i + 1) * 100000000, json.dumps(value).encode(), 0)
        writer.finish()
    screens, states, mouse, unknown = load_events(p)
    assert screens == [(0.5, 0.123)]
    assert states[-1] == (0.3, ["e", "mouse_left", "w"])
    assert mouse == [(0.4, 3, -2)]
    assert unknown == {}


def test_metrics_do_not_reward_all_idle_with_high_f1():
    from laya_vision_stitch.game_eval import metrics

    rows = [
        {"action": {"buttons": ["w"], "mouse_delta": [0.1, -0.2]}},
        {"action": {"buttons": [], "mouse_delta": [0, 0]}},
    ]
    idle = [{"buttons": [], "mouse_delta": [0, 0]}] * 2
    result = metrics(rows, idle, BUTTONS)
    assert result["button_exact_match"] == 0.5
    assert result["button_micro_f1"] == 0
    assert result["mouse_direction_accuracy"] == 0


def test_video_alignment_never_uses_future_frame(tmp_path):
    import fractions

    av = pytest.importorskip("av")
    from PIL import Image

    from laya_vision_stitch.d2e_data import extract_frames

    path = tmp_path / "video.mkv"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("ffv1", rate=60)
        stream.width = stream.height = 32
        stream.pix_fmt = "bgr0"
        for pts, color in [(0, "red"), (17, "green"), (100, "blue")]:
            frame = av.VideoFrame.from_image(Image.new("RGB", (32, 32), color))
            frame.pts = pts
            frame.time_base = fractions.Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    output = tmp_path / "frames"
    output.mkdir()
    paths, offsets = extract_frames(path, [0.01, 0.025, 0.08], output)
    assert 0.08 not in paths  # Gap >20ms must not substitute the future blue frame.
    assert Image.open(paths[0.01]).getpixel((0, 0)) == (255, 0, 0)
    assert Image.open(paths[0.025]).getpixel((0, 0)) == (0, 128, 0)
    assert 0 <= offsets[0.025] <= 0.02
