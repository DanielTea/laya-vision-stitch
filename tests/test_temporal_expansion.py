from dataclasses import replace

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from laya_vision_stitch.sequence_expansion import covered_starts
from laya_vision_stitch.temporal_adapter import TemporalActionAdapter
from laya_vision_stitch.temporal_training import augmented_batch, selection_loss, train
from laya_vision_stitch.trainable_model import PolicyConfig


def test_coverage_sampler_keeps_rare_existing_goals_without_overlap():
    rows = []
    for t in range(140):
        goal = "first" if 30 <= t < 42 else "second" if 85 <= t < 97 else "generic"
        rows.append(
            {
                "frame_index": t,
                "goal": goal,
                "instruction_provenance": {"annotator": None if goal == "generic" else "weak"},
            }
        )
    indices = covered_starts(rows, 8, 4, np.random.default_rng(97))
    assert len(indices) == 8
    assert {rows[i]["goal"] for i in indices} == {"generic", "first", "second"}
    assert all(b - a > 4 for a, b in zip(indices, indices[1:]))
    assert all(len({r["goal"] for r in rows[i : i + 4]}) == 1 for i in indices)


def training_fixture():
    config = replace(
        PolicyConfig(buttons=("w", "a")), temporal_adapter="attention", temporal_width=32
    )
    adapter = TemporalActionAdapter(12, 16, config)
    item = {
        "visual": mx.random.normal((8, 16, 12)),
        "language": mx.random.normal((8, 2, 16)),
        "history": mx.zeros((8, 9)),
        "buttons": mx.ones((8, 2)),
        "mouse": mx.zeros((8, 2)),
        "transition": mx.ones(8),
        "future_delta": mx.ones((8, 64)) * 0.01,
    }
    stats = {
        "varying": mx.ones(2),
        "camera_weights": mx.ones((2, len(config.mouse_bins))),
        "delta_scale": mx.ones(64) * 0.01,
    }
    return adapter, item, config, stats


def test_random_crops_preserve_action_target_alignment_and_temporal_mask():
    _, item, _, _ = training_fixture()
    for key, value in item.items():
        shape = (8,) + (1,) * (value.ndim - 1)
        item[key] = mx.broadcast_to(mx.arange(1, 9).reshape(shape), value.shape).astype(mx.float32)
    batch = augmented_batch([item], [0, 0], np.random.default_rng(97), True)
    assert batch["visual"].shape == (2, 6, 16, 12)
    for i in range(2):
        clock = np.asarray(batch["buttons"][i, :, 0])
        np.testing.assert_array_equal(np.diff(clock), np.ones(5))
        for key in ("language", "mouse", "transition", "future_delta"):
            observed = np.asarray(batch[key][i]).reshape(6, -1)[:, 0]
            np.testing.assert_array_equal(observed, clock)
        masks = np.asarray(batch["visual"][i, :, :, 0]) != 0
        np.testing.assert_array_equal(masks, np.broadcast_to(masks[0], masks.shape))


def test_checkpoint_selection_cannot_read_future_targets():
    adapter, item, config, stats = training_fixture()
    initial = selection_loss(adapter, [item], stats, config)
    del item["future_delta"]
    np.testing.assert_allclose(initial, selection_loss(adapter, [item], stats, config))


def test_training_restores_validation_selected_weights(monkeypatch, tmp_path):
    import laya_vision_stitch.temporal_training as module

    adapter, item, config, stats = training_fixture()
    snapshots = []

    def validation(model, data, stats, config):
        snapshots.append({k: np.asarray(v).copy() for k, v in tree_flatten(model.parameters())})
        return float(len(snapshots))

    monkeypatch.setattr(module, "selection_loss", validation)
    result = train(
        adapter,
        [item],
        config,
        2,
        0.001,
        83,
        stats,
        tmp_path,
        0.1,
        validation=[item],
        validation_every=1,
        regularize=True,
    )
    assert result["selected_step"] == 1
    final = dict(tree_flatten(adapter.parameters()))
    for key, value in snapshots[0].items():
        np.testing.assert_array_equal(np.asarray(final[key]), value)
    assert any(not np.array_equal(a, snapshots[1][k]) for k, a in snapshots[0].items())


def test_regularization_preserves_sampling_and_history_dropout_rng():
    _, item, _, _ = training_fixture()
    plain, regularized = np.random.default_rng(83), np.random.default_rng(83)
    for _ in range(5):
        a = augmented_batch([item], [0, 0], plain, False)
        b = augmented_batch([item], [0, 0], regularized, True, np.random.default_rng(84))
        assert a["buttons"].shape[1] == 8 and b["buttons"].shape[1] == 6
        assert plain.integers(100000) == regularized.integers(100000)


def test_generic_only_recording_is_not_given_invented_instructions():
    rows = [
        {"frame_index": t, "goal": "generic", "instruction_provenance": {"annotator": None}}
        for t in range(100)
    ]
    starts = covered_starts(rows, 5, 8, np.random.default_rng(17))
    assert len(starts) == 5
    assert all(rows[i]["goal"] == "generic" for i in starts)
    assert all(b - a > 8 for a, b in zip(starts, starts[1:]))
