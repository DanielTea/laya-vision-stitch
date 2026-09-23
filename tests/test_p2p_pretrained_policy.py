import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from laya_vision_stitch.p2p_pretrained_policy import (
    Layer,
    OpenP2PPolicy,
    physical_action,
    policy_mask,
)


def test_target_actions_cannot_leak_to_current_decision():
    positions = mx.arange(24)
    mask = np.asarray(policy_mask(positions, positions))
    assert not mask[:12, 12:].any()  # No future-frame input.
    assert not mask[3, 4:12].any()  # Current target actions are inaccessible.
    assert not mask[15, 16:24].any()
    assert mask[15, [4, 5, 6, 7, 8, 9, 10, 11]].all()  # Prior real actions ARE accessible.
    assert not mask[15, 3]  # Prior decision-placeholder is excluded.
    assert mask[16, 23] and not mask[16, 15]  # Grouped action history, no placeholder.
    assert not mask[12:15, 15:].any()


def test_history_window_mask():
    q = mx.array([201 * 12 + 3])
    k = mx.array([0, 12, 12 + 3, 12 + 4, 200 * 12, 201 * 12 + 4])
    assert np.asarray(policy_mask(q, k)).tolist() == [[False, True, False, True, True, False]]


def test_attention_targets_do_not_change_decision_but_change_history():
    mx.random.seed(23)
    layers = [Layer(width=16, heads=2) for _ in range(2)]
    x = mx.random.normal((1, 12, 16))
    changed = mx.concatenate([x[:, :4], x[:, 4:] + mx.random.normal((1, 8, 16)) * 5], 1)
    mask = policy_mask(mx.arange(12), mx.arange(12))
    for layer in layers:
        x, _ = layer(x, 0, None, mask)
        changed, _ = layer(changed, 0, None, mask)
    np.testing.assert_allclose(np.asarray(x[:, :4]), np.asarray(changed[:, :4]), atol=1e-6)
    assert not np.allclose(np.asarray(x[:, 4:]), np.asarray(changed[:, 4:]))


def test_action_vocabulary_is_not_silently_remapped():
    assert physical_action([0, 0, 0, 0, 0, 0, 11, 8])["buttons"] == []
    a = physical_action([11, 2, 0, 0, 2, 0, 12, 7])
    assert a["buttons"] == ["1", "mouse_right", "w"]
    assert a["mouse_delta"] == [1 / 512, -1 / 512]
    for tokens in ([20, 0, 0, 0, 0, 0, 11, 8], [0] * 7, [0.5] * 8):
        with pytest.raises(ValueError):
            physical_action(tokens)


def test_full_cache_rolls_exactly_one_frame_and_keeps_absolute_positions():
    calls = []

    def echo(x, position, caches, mask):
        calls.append((position, caches, np.asarray(mask)))
        return x, caches

    model = OpenP2PPolicy.__new__(OpenP2PPolicy)
    nn.Module.__init__(model)
    model.policy = echo
    cache = [(mx.arange(2400).reshape(1, 1, 2400, 1), mx.zeros((1, 1, 2400, 1))) for _ in range(10)]
    model.context(mx.zeros((1, 4, 1024)), caches=cache, position=2400)
    position, used, mask = calls[0]
    assert position == 2400
    assert used[0][0].shape[2] == 2388
    assert int(used[0][0][0, 0, 0, 0]) == 12
    assert mask.shape == (12, 2400)
    assert not mask[:, 3].any()  # First retained frame's placeholder remains masked.
    assert not mask[3, -8:].any()  # Current decision cannot see target actions.
    with pytest.raises(ValueError, match="before the start"):
        model.context(mx.zeros((1, 4, 1024)), caches=cache, position=0)


def test_batched_prefix_and_context_match_independent_examples():
    mx.random.seed(48)
    model = OpenP2PPolicy.__new__(OpenP2PPolicy)
    nn.Module.__init__(model)
    model.text_projection = nn.Linear(768, 1024, bias=False)
    for name in ("no_text", "image_position", "text_position", "thinking", "action_start"):
        setattr(model, name, mx.random.normal((1, 1, 1024)))
    layer = Layer(1024, 16)
    model.policy = lambda x, position, caches, mask: layer(x, position, None, mask)
    images, goals = mx.random.normal((2, 1024)), mx.random.normal((2, 768))
    batch = model.prefix(images, goals)
    contexts = model.context(batch)[0]
    for i in range(2):
        single = model.prefix(images[i : i + 1], goals[i : i + 1])
        np.testing.assert_allclose(np.asarray(batch[i : i + 1]), np.asarray(single), atol=1e-5)
        np.testing.assert_allclose(
            np.asarray(contexts[i : i + 1]), np.asarray(model.context(single)[0]), atol=2e-5
        )
    assert model.prefix(images).shape == (2, 4, 1024)
