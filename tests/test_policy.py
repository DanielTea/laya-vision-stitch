import json
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("laya_mlx")

from laya_mlx.model import DecisionModel, EncoderConfig  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from laya_vision_stitch.policy_data import check_separation, read_manifest  # noqa: E402
from laya_vision_stitch.policy_fixtures import create  # noqa: E402
from laya_vision_stitch.policy_teacher import parse_final  # noqa: E402
from laya_vision_stitch.policy_training import fingerprint, loss_terms, train  # noqa: E402
from laya_vision_stitch.trainable_model import (  # noqa: E402
    LoRALinear,
    PolicyConfig,
    TrainableStitch,
)


class SmallVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(out_hidden_size=8)
        self.linear = nn.Linear(8, 8)


def small_model(lora_rank=0):
    mx.random.seed(5)
    config = PolicyConfig(visual_slots=2, connector_width=16, heads=2, lora_rank=lora_rank)
    encoder = EncoderConfig.from_dict(
        dict(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
        )
    )
    model = TrainableStitch(SmallVision(), DecisionModel(encoder, {"head_layers": 1}), config)
    batch = dict(
        input_ids=mx.array([[1, 2, 3, 4, 0, 0, 5]]),
        attention_mask=mx.ones((1, 7), mx.bool_),
        marker_pos=mx.array([[1, 3]]),
        marker_mask=mx.ones((1, 2), mx.bool_),
        qtype=mx.array([0]),
    )
    inputs = (mx.random.normal((4, 8)), mx.zeros((4, 4)), batch, mx.array([[1, 2, 3]]), 4)
    return model, inputs


def test_gradient_updates_only_connector_and_action_heads():
    model, inputs = small_model()
    before = {
        key: fingerprint(getattr(model, key)) for key in ("vision", "laya", "connector", "actions")
    }
    row = {
        "id": "one",
        "choices": {"a": "left", "b": "right"},
        "answer": "a",
        "action": {
            "buttons": ["a"],
            "mouse_delta": [0.2, -0.1],
            "duration_seconds": 0.1,
            "pointer_xy": [0.4, 0.7],
        },
    }
    runtime = SimpleNamespace(
        module=model, metadata={"training_steps": 0, "action_training_examples": 0}
    )
    trace = train(runtime, [(row, inputs)], steps=3)
    after = {key: fingerprint(getattr(model, key)) for key in before}
    assert before["vision"] == after["vision"] and before["laya"] == after["laya"]
    assert before["connector"] != after["connector"] and before["actions"] != after["actions"]
    assert all(step["gradient_norm"] > 0 for step in trace)
    assert {k.split(".")[0] for k, _ in tree_flatten(model.trainable_parameters())} == {
        "connector",
        "actions",
    }


def test_answer_only_loss_backpropagates_through_frozen_laya():
    model, inputs = small_model()
    row = {"choices": {"a": "left", "b": "right"}, "answer": "a"}
    _, grads = nn.value_and_grad(
        model, lambda m: loss_terms(m.from_features(*inputs), row, m.policy_config)["answer"]
    )(model)
    values = [
        float(mx.sum(mx.abs(v)).item())
        for k, v in tree_flatten(grads)
        if k.startswith("connector.")
    ]
    assert sum(values) > 0


def test_goal_and_image_both_reach_outputs():
    model, inputs = small_model()
    original = model.from_features(*inputs)["choices"]
    new_goal = (*inputs[:3], mx.array([[6, 7, 8]]), inputs[4])
    new_image = (inputs[0] + 2 * mx.random.normal(inputs[0].shape), *inputs[1:])
    assert not np.allclose(
        np.asarray(original), np.asarray(model.from_features(*new_goal)["choices"])
    )
    assert not np.allclose(
        np.asarray(original), np.asarray(model.from_features(*new_image)["choices"])
    )


def test_distillation_uses_names_and_temperature():
    row = {
        "choices": {"second": "B", "first": "A"},
        "teacher_probs": {"first": 0.75, "second": 0.25},
        "teacher_temperature": 2.0,
    }
    logits = 2 * mx.log(mx.array([[0.25, 0.75]]))
    loss = loss_terms({"choices": logits}, row, PolicyConfig())["distillation"]
    assert abs(float(loss.item())) < 1e-6


def test_manifest_separation_and_action_validation(tmp_path):
    create(tmp_path / "data")
    config = PolicyConfig()
    train_rows = read_manifest(tmp_path / "data/train.jsonl", config)
    val_rows = read_manifest(tmp_path / "data/validation.jsonl", config)
    check_separation(train_rows, val_rows)
    with pytest.raises(ValueError, match="games overlap"):
        check_separation(train_rows, val_rows, holdout_games=True)
    val_rows[0]["frames"][0]["sha256"] = train_rows[0]["frames"][0]["sha256"]
    with pytest.raises(ValueError, match="image leakage"):
        check_separation(train_rows, val_rows)
    row = train_rows[0]
    row["action"]["buttons"] = ["unknown"]
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="vocabulary"):
        read_manifest(path, config)


def test_frames_must_be_chronological_and_teacher_targets_complete(tmp_path):
    create(tmp_path / "data")
    rows = read_manifest(tmp_path / "data/train.jsonl", PolicyConfig())
    row = rows[0]
    row["frames"] = [dict(row["frames"][0], age_seconds=0), dict(row["frames"][0], age_seconds=1)]
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="oldest first"):
        read_manifest(path, PolicyConfig())
    row["frames"] = [row["frames"][0]]
    row["teacher_probs"] = {"left": 1.0}
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="match named choices"):
        read_manifest(path, PolicyConfig())


def test_fingerprint_supports_bfloat16():
    model = nn.Linear(2, 2)
    model.weight = model.weight.astype(mx.bfloat16)
    first = fingerprint(model)
    model.weight = model.weight + 1
    assert fingerprint(model) != first


def test_lora_starts_identical_and_preserves_frozen_base():
    layer = LoRALinear(nn.Linear(4, 3), 2)
    x = mx.ones((1, 4))
    np.testing.assert_array_equal(np.asarray(layer(x)), np.asarray(layer.base(x)))
    frozen = fingerprint(layer, frozen_only=True)
    layer.lora_b = mx.ones_like(layer.lora_b)
    assert fingerprint(layer, frozen_only=True) == frozen
    assert set(dict(tree_flatten(layer.trainable_parameters()))) == {"lora_a", "lora_b"}


def test_teacher_parser_rejects_missing_or_invalid_final_answer():
    assert parse_final("The target is to the left.\n\nFINAL: A", ["A", "B"]) == "A"
    for text in ("Maybe A", "FINAL: C", "FINAL: A\nBut perhaps B"):
        with pytest.raises(ValueError):
            parse_final(text, ["A", "B"])


def test_lora_training_updates_adapters_but_not_pretrained_weights():
    model, inputs = small_model(lora_rank=2)
    original = fingerprint(model.laya, frozen_only=True)
    initial_adapters = {
        k: np.asarray(v).copy() for k, v in tree_flatten(model.laya.trainable_parameters())
    }
    row = {"id": "one", "choices": {"a": "left", "b": "right"}, "answer": "a"}
    runtime = SimpleNamespace(
        module=model, metadata={"training_steps": 0, "action_training_examples": 0}
    )
    train(runtime, [(row, inputs)], steps=3)
    assert fingerprint(model.laya, frozen_only=True) == original
    assert any(
        not np.array_equal(initial_adapters[k], np.asarray(v))
        for k, v in tree_flatten(model.laya.trainable_parameters())
    )
