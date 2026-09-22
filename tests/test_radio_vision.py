import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from laya_vision_stitch.hordes_demonstrations import label
from laya_vision_stitch.radio_vision import RadioConfig, RadioProcessor, RadioVision
from laya_vision_stitch.trainable_model import apply_supervision_scope


def test_rectangular_position_encoding_uses_reference_square_then_top_left_crop():
    model = RadioVision(RadioConfig(out_hidden_size=4, heads=1, depth=0, position_side=4))
    grid = np.arange(16, dtype=np.float32).reshape(1, 16, 1)
    model.positions = mx.array(np.repeat(grid, 4, axis=-1))
    # Square interpolation samples original coordinates 0, 1.5, 3 on both axes;
    # the rectangle keeps the first two rows rather than stretching its height.
    actual = np.asarray(model.position_encoding(2, 3))[0, :, 0]
    np.testing.assert_allclose(actual, [0, 1.5, 3, 6, 7.5, 9])


def test_rgb_patch_channel_order_and_no_summary_tokens_in_dense_output():
    model = RadioVision(
        RadioConfig(out_hidden_size=12, heads=1, depth=0, patch_size=2, position_side=2)
    )
    model.embedder.weight = mx.eye(12)
    pixels = np.arange(24, dtype=np.float32).reshape(1, 2, 4, 3)
    dense, _ = model(mx.array(pixels))
    expected = np.stack(
        [
            pixels[0, :, :2].transpose(2, 0, 1).reshape(-1),
            pixels[0, :, 2:].transpose(2, 0, 1).reshape(-1),
        ]
    )
    np.testing.assert_array_equal(np.asarray(dense), expected)
    with pytest.raises(ValueError, match="divisible"):
        model(mx.zeros((1, 3, 4, 3)))


def test_processor_keeps_complete_image_and_returns_consistent_patch_grid():
    processor = RadioProcessor()
    image = Image.new("RGB", (320, 180), (255, 128, 0))
    data = processor(images=[image])
    assert data["pixel_values"].shape == (1, 176, 320, 3)
    np.testing.assert_array_equal(data["image_grid_thw"], [[1, 11, 20]])
    np.testing.assert_allclose(data["pixel_values"][0, 0, 0], [1, 128 / 255, 0])


def test_demonstration_labels_exclude_failed_proposals_and_do_not_use_phase():
    assert label({"action": "G", "status": "fresh_state_rejected", "input_applied": False}) is None
    record = {"input_applied": True, "status": "applied", "key": "1", "phase": "unknown"}
    assert label(record)["buttons"] == ["1"]
    assert label({**record, "phase": "cooldown", "state": {"target": None}}) == label(record)
    assert label({"input_applied": True, "status": "loot_hovered"}) is None


def test_control_only_exports_do_not_expose_unsupervised_output_heads():
    result = apply_supervision_scope(
        {
            "action_chunk": [{"buttons": ["1"]}, {"buttons": ["w"]}],
            "pointer_xy_normalized": [0.2, 0.3],
            "pointer_active_probability": 0.9,
            "duration_seconds": 0.4,
        },
        {"first_step_control_only": True},
    )
    assert len(result["action_chunk"]) == 1
    assert "pointer_xy_normalized" not in result
    assert result["duration_seconds"] == 0.05


def test_visual_fusion_starts_as_identity_and_can_carry_image_gradients():
    import mlx.nn as nn

    from laya_vision_stitch.trainable_model import GatedVisualFusion

    layer = GatedVisualFusion(12, 16, width=8)
    hidden = mx.random.normal((1, 5, 16))
    patches, coords = mx.random.normal((7, 12)), mx.zeros((7, 4))
    np.testing.assert_array_equal(np.asarray(layer(hidden, patches, coords)), np.asarray(hidden))
    _, grads = nn.value_and_grad(layer, lambda m: m(hidden, patches, coords).sum())(layer)
    assert abs(float(grads["gate"])) > 1e-7
    layer.gate = mx.array(0.2)
    gradient = mx.grad(lambda p: layer(hidden, p, coords).sum())(patches)
    assert float(mx.abs(gradient).sum()) > 1e-7
