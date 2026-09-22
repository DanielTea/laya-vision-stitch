import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("laya_mlx")

from laya_vision_stitch.learning_gate import choose_diverse  # noqa: E402
from laya_vision_stitch.trainable_model import (  # noqa: E402
    AlignedConnector,
    PolicyConfig,
    TemporalConnector,
    context_text,
    decode_action_chunk,
)


def test_diverse_selection_is_fixed_and_covers_actions():
    rows = [
        {"id": str(i), "action": {"buttons": ["w"] if i < 8 else ["e"], "mouse_delta": [0, 0]}}
        for i in range(10)
    ]
    selected = choose_diverse(rows, 4, np.random.default_rng(17))
    assert len({r["id"] for r in selected}) == 4
    assert any(r["action"]["buttons"] == ["e"] for r in selected)
    assert selected == choose_diverse(rows, 4, np.random.default_rng(17))


def test_temporal_conversion_preserves_initial_state_and_can_see_frame_order():
    config = PolicyConfig(connector_width=8, heads=2, visual_slots=4)
    base = AlignedConnector(8, 12, config)
    temporal = TemporalConnector(8, 12, config)
    temporal.base = base
    patches = mx.random.normal((8, 8))
    coords = mx.array(
        [
            [-1, -1, 0.2, 0],
            [1, -1, 0.2, 0],
            [-1, 1, 0.2, 0],
            [1, 1, 0.2, 0],
            [-1, -1, 0, 1],
            [1, -1, 0, 1],
            [-1, 1, 0, 1],
            [1, 1, 0, 1],
        ]
    )
    goal = mx.random.normal((1, 3, 12))
    np.testing.assert_array_equal(
        np.asarray(base(patches, coords, goal)), np.asarray(temporal(patches, coords, goal))
    )
    temporal.output.weight = mx.random.normal(temporal.output.weight.shape) * 0.1
    actual = temporal(patches, coords, goal)
    swapped = temporal(mx.concatenate([patches[4:], patches[:4]]), coords, goal)
    assert not np.allclose(np.asarray(actual), np.asarray(swapped), atol=1e-7)


def test_chunk_modes_do_not_average_opposite_mouse_turns():
    config = PolicyConfig(action_chunk_size=4)
    logits = mx.full((1, 4, 2, len(config.mouse_bins)), -10.0)
    logits[0, :, 0, 0] = 2.0
    logits[0, :, 0, -1] = 1.9
    logits[0, :, 1, 8] = 3.0
    output = {"chunk_buttons": mx.full((1, 4, len(config.buttons)), -10.0), "chunk_mouse": logits}
    chunks = decode_action_chunk(output, config)
    assert len(chunks) == 4
    assert chunks[0]["mouse_delta"] == [-1.0, 0.0]
    assert chunks[0]["buttons"] == []


def test_labels_never_enter_prompt():
    row = {
        "goal": "Continue playing.",
        "controls": "Physical keys.",
        "previous_actions": [],
        "recorded_previous_actions": [{"buttons": ["SECRET"]}],
        "action_chunk": [{"buttons": ["SECRET"]}],
        "_preserve_choices": ["SECRET"],
    }
    assert "SECRET" not in context_text(row)


def test_chunk_decoder_reads_tokens_beyond_cls():
    from laya_vision_stitch.trainable_model import ActionHeads

    config = PolicyConfig(connector_width=8, heads=2, action_chunk_size=4)
    heads = ActionHeads(12, config)
    heads.readout.weight = mx.random.normal(heads.readout.weight.shape) * 0.1
    state = mx.random.normal((1, 12))
    first = heads(state, mx.random.normal((1, 6, 12)))
    second = heads(state, mx.random.normal((1, 6, 12)))
    assert first["chunk_buttons"].shape == (1, 4, len(config.buttons))
    assert first["chunk_mouse"].shape == (1, 4, 2, len(config.mouse_bins))
    assert not np.allclose(np.asarray(first["buttons"]), np.asarray(second["buttons"]))


def test_grounding_labels_are_review_bound_and_remove_old_targets(tmp_path):
    import hashlib
    import json

    from laya_vision_stitch.grounding_data import expand

    image = tmp_path / "image.png"
    image.write_bytes(b"reviewed image")
    source = {
        "id": "sample",
        "frames": [{"image": str(image)}],
        "provenance": {},
        "action": {"buttons": ["SECRET"]},
        "action_chunk": [{"buttons": ["SECRET"]}],
        "recorded_previous_actions": [{"buttons": ["SECRET"]}],
        "description": "SECRET",
        "teacher_probs": {"SECRET": 1},
    }
    label = {
        "id": "sample",
        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "large_menu_open": True,
    }
    annotation = tmp_path / "review.json"
    annotation.write_text(json.dumps({"splits": {s: [label] for s in ("train", "validation")}}))
    for split in ("train", "validation"):
        (tmp_path / (split + ".jsonl")).write_text(json.dumps(source) + "\n")
    expand(tmp_path, annotation)
    result = (tmp_path / "train-grounding.jsonl").read_text()
    assert "SECRET" not in result
    rows = [json.loads(line) for line in result.splitlines()]
    assert [row["answer"] for row in rows] == ["yes", "no", "release", "hold"]
    assert rows[2]["action"]["buttons"] == []
    assert rows[3]["action"]["buttons"] == ["w"]
    image.write_bytes(b"different screenshot")
    with pytest.raises(ValueError, match="Reviewed image changed"):
        expand(tmp_path, annotation)


def test_chunk_validation_checks_future_labels(tmp_path):
    import copy
    import json

    from laya_vision_stitch.policy_data import read_manifest

    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    action = {"buttons": ["w"], "mouse_delta": [0, 0], "duration_seconds": 0.1}
    row = {
        "id": "clip",
        "game": "game",
        "episode": "session",
        "goal": "Play.",
        "frames": [{"image": str(image), "age_seconds": 0}],
        "action": action,
        "action_chunk": [copy.deepcopy(action) for _ in range(4)],
    }
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(row))
    config = PolicyConfig(action_chunk_size=4)
    assert len(read_manifest(path, config)) == 1
    row["action_chunk"][3]["buttons"] = ["INVALID"]
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="vocabulary"):
        read_manifest(path, config)
    row["action_chunk"][0]["mouse_delta"] = [1, 1]
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="prefix"):
        read_manifest(path, config)


def test_single_step_labels_supervise_categorical_mouse_in_chunk_model():
    from laya_vision_stitch.policy_training import loss_terms
    from laya_vision_stitch.trainable_model import ActionHeads

    config = PolicyConfig(connector_width=8, heads=2, action_chunk_size=4)
    heads = ActionHeads(12, config)
    output = heads(mx.zeros((1, 12)), mx.ones((1, 4, 12)))
    row = {"action": {"buttons": [], "mouse_delta": [0, 0], "duration_seconds": 0.1}}
    terms = loss_terms(output, row, config)
    assert "chunk_mouse" in terms
    assert float(terms["chunk_mouse"]) > 0
