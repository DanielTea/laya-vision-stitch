from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from laya_vision_stitch.trainable_model import PolicyConfig, TrainableStitch
from laya_vision_stitch.visual_action_adapter import VisualActionAdapter


def fixture():
    mx.random.seed(53)
    config = PolicyConfig(connector_width=8, heads=2, action_chunk_size=2, buttons=("w", "a"))
    model = VisualActionAdapter(12, 16, config)
    patches = mx.random.normal((8, 12))
    coords = mx.random.normal((8, 4))
    context = mx.random.normal((1, 5, 16))
    original = {
        "chunk_buttons": mx.random.normal((1, 2, 2)),
        "chunk_mouse": mx.random.normal((1, 2, 2, len(config.mouse_bins))),
    }
    original["buttons"] = original["chunk_buttons"][:, 0]
    return model, patches, coords, context, original


def test_zero_initialization_preserves_parent_outputs():
    model, patches, coords, context, original = fixture()
    result = model.fuse(original, patches, coords, context)
    for key in original:
        np.testing.assert_array_equal(np.asarray(original[key]), np.asarray(result[key]))


def test_learned_visual_branch_can_use_pixels_and_laya_context():
    model, patches, coords, context, _ = fixture()
    model.buttons.output.weight = mx.random.normal(model.buttons.output.weight.shape)
    first = np.asarray(model.buttons(patches, coords, context))
    assert not np.allclose(first, np.asarray(model.buttons(-patches, coords, context)))
    assert not np.allclose(first, np.asarray(model.buttons(patches, coords, -context)))


def test_camera_gradient_cannot_update_button_branch():
    model, patches, coords, context, original = fixture()
    before = np.asarray(model.fuse(original, patches, coords, context)["buttons"])

    def loss(adapter):
        logits = adapter.camera(patches, coords, context)
        return ((logits - 1) ** 2).mean()

    _, grads = nn.value_and_grad(model, loss)(model)
    from mlx.utils import tree_flatten

    assert all(float(mx.abs(g).sum()) == 0 for _, g in tree_flatten(grads["buttons"]))
    assert any(float(mx.abs(g).sum()) > 0 for _, g in tree_flatten(grads["camera"]))
    model.camera.output.bias = model.camera.output.bias - 0.1 * grads["camera"]["output"]["bias"]
    after = np.asarray(model.fuse(original, patches, coords, context)["buttons"])
    np.testing.assert_array_equal(before, after)


def test_enabled_adapter_cannot_be_silently_skipped_by_state_only_call():
    stub = SimpleNamespace(
        policy_config=SimpleNamespace(visual_action_adapter=True),
        encode_state=lambda *_: (mx.zeros((1, 2, 8)), mx.zeros((1, 2))),
        actions=lambda *_: {},
    )
    with pytest.raises(ValueError, match="requires raw visual"):
        TrainableStitch.from_state(stub, mx.zeros((1, 2, 8)), {}, 0)


def test_training_with_dropped_history_updates_only_visual_adapter(tmp_path):
    from laya_vision_stitch.visual_adapter_training import train

    model, patches, coords, context, original = fixture()
    config = PolicyConfig(connector_width=8, heads=2, action_chunk_size=2, buttons=("w", "a"))
    runtime = SimpleNamespace(
        module=SimpleNamespace(policy_config=config, visual_actions=model),
        metadata={"training_steps": 0, "action_training_examples": 0},
    )
    rows = [
        {"game": "test", "action": {"buttons": [b], "mouse_delta": [d, 0]}}
        for b, d in (("w", 0.1), ("a", -0.1))
    ]
    examples = [(r, (patches, coords)) for r in rows]
    contexts = [[(context, original), (-context, original)] for _ in rows]
    with (tmp_path / "steps.jsonl").open("w") as log:
        train(runtime, rows, examples, contexts, 3, 1e-3, 53, 1.0, log)
    assert float(mx.abs(model.buttons.output.weight).sum()) > 0
    assert float(mx.abs(model.camera.output.weight).sum()) > 0
    assert runtime.metadata["action_training_examples"] == 12


def test_numeric_history_distinguishes_unknown_from_idle_and_ignores_target():
    from laya_vision_stitch.visual_action_adapter import encode_action_history

    config = PolicyConfig(buttons=("w", "a"))
    unknown = np.asarray(encode_action_history({}, config))
    idle = np.asarray(
        encode_action_history(
            {"previous_actions": [{"buttons": [], "mouse_delta": [0, 0]}]}, config
        )
    )
    assert np.all(unknown == 0)
    np.testing.assert_array_equal(idle[0, :3], [1, 1, 1])
    row = {"previous_actions": [{"buttons": ["a"], "mouse_delta": [-0.25, 0.5]}]}
    observed = np.asarray(encode_action_history(row, config))
    np.testing.assert_array_equal(observed[0, 3:5], [0, 1])
    np.testing.assert_array_equal(observed[0, -4:-2], [-0.25, 0.5])
    assert observed[0, -2] < 0 < observed[0, -1]
    row["action"] = {"buttons": ["w"], "mouse_delta": [1, 1]}
    np.testing.assert_array_equal(observed, np.asarray(encode_action_history(row, config)))
    row["previous_actions"][0]["mouse_delta"] = [float("nan"), 0]
    with pytest.raises(ValueError, match="previous mouse"):
        encode_action_history(row, config)


def test_numeric_adapter_requires_history_and_can_learn_from_it():
    from laya_vision_stitch.visual_action_adapter import encode_action_history

    config = PolicyConfig(
        connector_width=8,
        heads=2,
        action_chunk_size=2,
        visual_action_adapter=True,
        numeric_action_history=True,
        buttons=("w", "a"),
    )
    model = VisualActionAdapter(12, 16, config)
    _, patches, coords, context, _ = fixture()
    with pytest.raises(ValueError, match="history input is required"):
        model.camera(patches, coords, context)
    model.camera.output.weight = mx.random.normal(model.camera.output.weight.shape)
    left = encode_action_history({"previous_actions": [{"mouse_delta": [-1, 0]}]}, config)
    right = encode_action_history({"previous_actions": [{"mouse_delta": [1, 0]}]}, config)
    assert not np.allclose(
        np.asarray(model.camera(patches, coords, context, left)),
        np.asarray(model.camera(patches, coords, context, right)),
    )


def test_prediction_api_exposes_only_supervised_pilot_controls():
    from laya_vision_stitch.trainable_model import apply_supervision_scope

    raw = {
        "action_chunk": [{"buttons": ["w"]}, {"buttons": ["mouse_left"]}],
        "chunk_step_seconds": 0.1,
        "duration_seconds": 0.4,
        "pointer_xy_normalized": [0.7, 0.8],
        "pointer_active_probability": 0.9,
    }
    scoped = apply_supervision_scope(raw, {"visual_adapter_experiment": {"scope": "first step"}})
    assert len(scoped["action_chunk"]) == 1
    assert "pointer_xy_normalized" not in scoped and "pointer_active_probability" not in scoped
    assert scoped["duration_seconds"] == scoped["chunk_step_seconds"] == 0.05
    assert scoped["deployment_eligible"] is False
    assert len(raw["action_chunk"]) == 2
    assert apply_supervision_scope(raw, {}) == raw
