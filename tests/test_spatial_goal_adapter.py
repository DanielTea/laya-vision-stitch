import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from laya_vision_stitch.p2p_pretrained_policy import OpenP2PPolicy
from laya_vision_stitch.spatial_goal_adapter import SpatialGoalAdapter


def test_identity_start_and_independent_batched_goals():
    mx.random.seed(9)
    model = SpatialGoalAdapter()
    spatial = mx.random.normal((2, 12, 12, 112))
    goals = mx.random.normal((2, 768))
    images = mx.random.normal((2, 1024))
    np.testing.assert_array_equal(np.asarray(model(spatial, goals, images)), np.asarray(images))
    batch = model.features(spatial, goals)
    for i in range(2):
        single = model.features(spatial[i : i + 1], goals[i : i + 1])
        for b, s in zip(batch, single, strict=True):
            np.testing.assert_allclose(np.asarray(b[i : i + 1]), np.asarray(s), atol=2e-6)
    np.testing.assert_allclose(np.asarray(batch[2].sum(-1)), 1, atol=1e-6)
    swapped = model.features(spatial, goals[::-1])[2]
    assert not np.allclose(np.asarray(batch[2]), np.asarray(swapped))


def test_goal_and_image_content_are_both_used_after_training_output():
    model = SpatialGoalAdapter()
    model.output.weight = mx.random.normal(model.output.weight.shape) * 0.01
    spatial = mx.random.normal((1, 12, 12, 112))
    goal = mx.random.normal((1, 768))
    image = mx.zeros((1, 1024))
    first = np.asarray(model(spatial, goal, image))
    assert not np.allclose(first, np.asarray(model(spatial, -goal, image)))
    assert not np.allclose(first, np.asarray(model(-spatial, goal, image)))
    with pytest.raises(ValueError):
        model(spatial[:, :6], goal, image)


def test_spatial_checkpoint_cannot_silently_skip_its_visual_path():
    policy = OpenP2PPolicy.__new__(OpenP2PPolicy)
    nn.Module.__init__(policy)
    policy.spatial_adapter = SpatialGoalAdapter()
    with pytest.raises(ValueError, match="requires spatial"):
        policy.prefix(mx.zeros((1, 1024)), mx.zeros((1, 768)))
