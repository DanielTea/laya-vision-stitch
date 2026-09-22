from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from laya_vision_stitch.temporal_adapter import TemporalActionAdapter
from laya_vision_stitch.temporal_runtime import TemporalRuntime, frame_inputs
from laya_vision_stitch.trainable_model import PolicyConfig, TrainableStitch


@pytest.fixture
def runtime(monkeypatch):
    import laya_vision_stitch.temporal_runtime as module

    mx.random.seed(92)
    config = replace(PolicyConfig(buttons=("w", "a")), temporal_adapter="mamba3", temporal_width=32)
    policy = TemporalActionAdapter(12, 16, config)
    visual, language = mx.random.normal((16, 12)), mx.random.normal((2, 16))
    monkeypatch.setattr(module, "frame_inputs", lambda _, row: (visual, language))
    return TemporalRuntime(
        SimpleNamespace(module=SimpleNamespace(policy_config=config, temporal_actions=policy))
    )


def test_session_goal_gap_and_explicit_reset(runtime):
    row = {"goal": "move", "previous_actions": []}
    assert runtime.predict(row, session_id="a", timestamp_seconds=0)["state_reset"]
    assert not runtime.predict(row, session_id="a", timestamp_seconds=0.05)["state_reset"]
    with pytest.raises(ValueError, match="increasing"):
        runtime.predict(row, session_id="a", timestamp_seconds=0.04)
    assert runtime.predict(row, session_id="b", timestamp_seconds=0)["state_reset"]
    assert runtime.predict({**row, "goal": "stop"}, session_id="b", timestamp_seconds=0.05)[
        "state_reset"
    ]
    assert runtime.predict({**row, "controls": "changed"}, session_id="b", timestamp_seconds=0.1)[
        "state_reset"
    ]
    assert runtime.predict(row, session_id="b", timestamp_seconds=1)["state_reset"]
    assert runtime.predict(row, session_id="b", timestamp_seconds=0, reset=True)["state_reset"]
    runtime.reset()
    assert runtime.state is None and runtime.timestamp is None


def test_runtime_outputs_do_not_depend_on_target_labels(runtime):
    row = {"goal": "move", "previous_actions": [{"buttons": ["w"], "mouse_delta": [0, 0]}]}
    before = runtime.predict(row, session_id="a", timestamp_seconds=0)
    after = runtime.predict(
        {**row, "action": {"buttons": ["a"]}, "future_image": "/does-not-exist"},
        session_id="a",
        timestamp_seconds=0,
        reset=True,
    )
    for key in ("buttons", "mouse_delta", "duration_seconds"):
        assert before[key] == after[key]
    assert after["input_events_sent"] == 0 and not after["deployment_eligible"]


def test_frame_context_receives_no_previous_controls_or_future_image():
    observed = []
    config = PolicyConfig(visual_slots=2)
    model = SimpleNamespace(
        policy_config=config,
        action_context=lambda patches, coords, *prepared: (mx.ones((1, 6, 16)), None),
    )

    def features(row):
        observed.append(row["frames"])
        return mx.ones((16, 12)), mx.array(
            [[i / 2 - 1, j / 2 - 1, 0, 0] for i in range(4) for j in range(4)]
        )

    def prepare(row):
        assert row["previous_actions"] == []
        assert "future_image" not in row and "action" not in row
        return {}, None, 2

    runtime = SimpleNamespace(features=features, prepare=prepare, module=model)
    row = {
        "goal": "move",
        "previous_actions": [{"buttons": ["w"]}],
        "frames": [{"image": "current", "age_seconds": 0}],
        "future_image": "forbidden",
    }
    visual, language = frame_inputs(runtime, row)
    assert observed == [row["frames"]]
    assert visual.shape == (16, 12) and language.shape == (2, 16)
    with pytest.raises(ValueError, match="exactly one"):
        frame_inputs(runtime, {**row, "frames": row["frames"] * 2})


def test_temporal_checkpoint_cannot_silently_use_old_stateless_head():
    stub = SimpleNamespace(policy_config=SimpleNamespace(temporal_adapter="mamba3"))
    with pytest.raises(ValueError, match="TemporalRuntime"):
        TrainableStitch.from_state(stub, None, None, None)


def test_training_batch_reaches_memory_and_action_heads():
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from laya_vision_stitch.temporal_training import loss_terms

    config = replace(PolicyConfig(buttons=("w", "a")), temporal_adapter="mamba3", temporal_width=32)
    model = TemporalActionAdapter(12, 16, config)
    batch = {
        "visual": mx.random.normal((2, 6, 16, 12)),
        "language": mx.random.normal((2, 6, 2, 16)),
        "history": mx.zeros((2, 6, 9)),
        "buttons": mx.ones((2, 6, 2)),
        "mouse": mx.zeros((2, 6, 2)),
        "transition": mx.ones((2, 6)),
        "future_delta": mx.ones((2, 6, 64)) * 0.01,
    }
    stats = {
        "varying": mx.ones(2),
        "camera_weights": mx.ones((2, len(config.mouse_bins))),
        "delta_scale": mx.ones(64) * 0.01,
    }
    value, grad = nn.value_and_grad(model, lambda m: loss_terms(m, batch, stats, config, 0.1))(
        model
    )
    assert np.isfinite(float(value))
    for name in ("blocks", "visual", "language", "buttons", "camera", "dynamics"):
        assert any(float(mx.abs(g).sum()) > 0 for _, g in tree_flatten(grad[name]))
