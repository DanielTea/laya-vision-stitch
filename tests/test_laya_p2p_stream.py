from types import SimpleNamespace

import mlx.core as mx
import pytest
from PIL import Image

from laya_vision_stitch.laya_p2p_stream import LayaP2PStream


def fixture():
    committed = []

    def context(prefix, actions=None, caches=None, position=0):
        if actions is not None:
            committed.append((position, actions.tolist()[0]))
        return mx.zeros((1, 1, 1024)), None

    policy = SimpleNamespace(
        vision=lambda pixels: (None, mx.zeros((1, 1024))),
        prefix=lambda image, goal: mx.zeros((1, 4, 1024)),
        context=context,
        decode=lambda context, temperature: (
            mx.array([[11, 0, 0, 0, 0, 0, 11, 8]]),
            (mx.zeros((1, 20)),),
        ),
    )
    runtime = SimpleNamespace(
        model=SimpleNamespace(
            policy=policy, bridge=lambda x: x, goal_features=lambda *args: mx.zeros((1, 768))
        ),
        prepare_goal=lambda goal: (mx.ones((1, 1), mx.int32), mx.ones((1, 1), mx.bool_)),
        metadata={},
    )
    row = {
        "goal": "Move forward",
        "frames": [{"image": Image.new("RGB", (1280, 720)), "age_seconds": 0}],
        "previous_actions": [],
    }
    return LayaP2PStream(runtime), row, committed


def test_memory_commits_applied_action_instead_of_proposed_action():
    stream, row, committed = fixture()
    first = stream.predict(row, session_id="test", timestamp_seconds=0)
    assert first["buttons"] == ["w"]
    assert first["state_reset"]
    assert committed == []
    row["previous_actions"] = [{"buttons": ["d"], "mouse_delta": [0, 0]}]
    second = stream.predict(row, session_id="test", timestamp_seconds=0.05)
    assert not second["state_reset"]
    assert committed == [(0, [7, 0, 0, 0, 0, 0, 11, 8])]
    assert stream.position == 12


def test_unknown_feedback_or_gap_resets_memory_and_reversed_time_rejects():
    stream, row, committed = fixture()
    stream.predict(row, session_id="test", timestamp_seconds=0)
    result = stream.predict(row, session_id="test", timestamp_seconds=0.05)
    assert result["state_reset"] and committed == []
    row["previous_actions"] = [{"buttons": [], "mouse_delta": [0, 0]}]
    result = stream.predict(row, session_id="test", timestamp_seconds=0.5)
    assert result["state_reset"] and committed == []
    with pytest.raises(ValueError, match="increase"):
        stream.predict(row, session_id="test", timestamp_seconds=0.4)
