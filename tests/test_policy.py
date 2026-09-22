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


def test_decoder_context_matches_normal_forward_and_freezes_entire_upstream():
    from laya_vision_stitch.decoder_probe import freeze_decoder_only

    model, inputs = small_model(lora_rank=2)
    runtime = SimpleNamespace(module=model)
    ordinary = model.from_features(*inputs)
    h, choices = model.action_context(*inputs)
    cached = model.actions(h[:, 0], h)
    np.testing.assert_array_equal(np.asarray(ordinary["buttons"]), np.asarray(cached["buttons"]))
    np.testing.assert_array_equal(np.asarray(ordinary["choices"]), np.asarray(choices))
    freeze_decoder_only(runtime)
    trainable = [k for k, _ in tree_flatten(model.trainable_parameters())]
    assert trainable and all(k.startswith("actions.") for k in trainable)
    assert not any(k.startswith("actions.mouse.") for k in trainable)
    assert not any(k.startswith("actions.pointer") for k in trainable)
    assert all(
        not tree_flatten(getattr(model, k).trainable_parameters())
        for k in ("vision", "connector", "laya")
    )


@pytest.mark.parametrize("batch_size", [1, 4])
def test_decoder_training_updates_buttons_without_changing_context(batch_size):
    import io

    from laya_vision_stitch.decoder_probe import freeze_decoder_only, train_decoder

    model, inputs = small_model(lora_rank=2)
    runtime = SimpleNamespace(
        module=model, metadata={"training_steps": 0, "action_training_examples": 0}
    )
    freeze_decoder_only(runtime)
    before = {k: fingerprint(getattr(model, k)) for k in ("vision", "connector", "laya", "actions")}
    mouse = fingerprint(model.actions.mouse)
    h, _ = model.action_context(*inputs)
    rows = [
        {"goal": "Hold W.", "action": {"buttons": ["w"]}},
        {"goal": "Release W.", "action": {"buttons": []}},
    ]
    train_decoder(runtime, rows, [h, h + 1], 4, 0.0001, 17, io.StringIO(), batch_size=batch_size)
    after = {k: fingerprint(getattr(model, k)) for k in before}
    assert all(before[k] == after[k] for k in ("vision", "connector", "laya"))
    assert before["actions"] != after["actions"]
    assert mouse == fingerprint(model.actions.mouse)


def test_decoder_button_signal_is_not_diluted_by_unused_keys():
    from laya_vision_stitch.decoder_probe import button_loss

    def gradient(width):
        target = mx.array([[1.0] + [0.0] * (width - 1)])
        return mx.grad(lambda logits: button_loss(logits, target, [True] + [False] * (width - 1)))(
            mx.zeros((1, width))
        )

    a, b = gradient(7), gradient(51)
    np.testing.assert_allclose(np.asarray(a)[0, 0], np.asarray(b)[0, 0])
    assert float(a[0, 0]) < 0


def test_action_context_normalization_prevents_scale_driven_attention_saturation():
    from laya_vision_stitch.trainable_model import ActionHeads

    config = PolicyConfig(
        connector_width=16, heads=2, action_chunk_size=4, normalize_action_context=True
    )
    head = ActionHeads(16, config)
    head.readout.weight = mx.random.normal(head.readout.weight.shape) * 0.1
    tokens = mx.random.normal((1, 9, 16))
    small = head(tokens[:, 0], tokens)["buttons"]
    large = head(tokens[:, 0] * 1000, tokens * 1000)["buttons"]
    np.testing.assert_allclose(np.asarray(small), np.asarray(large), atol=1e-3)
    _, gradients = nn.value_and_grad(
        head, lambda h: mx.mean(h(tokens[:, 0] * 1000, tokens * 1000)["buttons"] ** 2)
    )(head)
    relevant = [v for k, v in tree_flatten(gradients) if k.startswith("attention.query_proj")]
    assert relevant
    assert sum(float(mx.abs(v).sum()) for v in relevant) > 1e-6
    assert all(np.isfinite(np.asarray(v)).all() for _, v in tree_flatten(gradients))


def test_goal_curriculum_keeps_labels_out_of_inputs(tmp_path):
    import hashlib

    from laya_vision_stitch.goal_curriculum import DEVELOPMENT, SEALED, TRAIN, expand
    from laya_vision_stitch.trainable_model import context_text

    assert not (
        set(TRAIN) & set(DEVELOPMENT) or set(TRAIN) & set(SEALED) or set(DEVELOPMENT) & set(SEALED)
    )
    image = tmp_path / "reviewed.png"
    image.write_bytes(b"reviewed")
    source = {
        "id": "one",
        "game": "a",
        "episode": "e",
        "frames": [{"image": str(image), "age_seconds": 0}],
        "provenance": {},
        "description": "SECRET",
        "answer": "SECRET",
        "previous_actions": ["SECRET"],
        "action": {"buttons": ["SECRET"]},
    }
    labels = {
        "one": {
            "large_menu_open": True,
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        }
    }
    rows = expand([source], labels, [("train-0", TRAIN[0])])
    assert len(rows) == 4
    assert rows[0]["action"]["buttons"] == [] and rows[1]["action"]["buttons"] == ["w"]
    assert list(rows[0]["choices"]) == list(reversed(rows[2]["choices"]))
    assert all("SECRET" not in context_text(r) and "answer" not in r for r in rows)
    assert all(
        "menu_state" not in context_text(r) and "_sampling_stratum" not in context_text(r)
        for r in rows
    )


def test_encoder_action_context_keeps_choice_predictions_unchanged():
    model, inputs = small_model(lora_rank=2)
    decision_context, choices = model.action_context(*inputs)
    model.policy_config.action_context_source = "encoder"
    encoder_context, second_choices = model.action_context(*inputs)
    np.testing.assert_array_equal(np.asarray(choices), np.asarray(second_choices))
    assert not np.allclose(np.asarray(decision_context), np.asarray(encoder_context))
    direct = model.from_features(*inputs)
    cached = model.actions(encoder_context[:, 0], encoder_context)
    np.testing.assert_array_equal(np.asarray(direct["buttons"]), np.asarray(cached["buttons"]))


@pytest.mark.parametrize("batch_size", [None, 4])
def test_single_objective_adapter_training_protects_base_weights(batch_size):
    import io

    from laya_vision_stitch.adapter_buttons import train_adapter_buttons

    model, inputs = small_model(lora_rank=2)
    runtime = SimpleNamespace(
        module=model, metadata={"training_steps": 0, "action_training_examples": 0}
    )
    before = {k: fingerprint(getattr(model, k), frozen_only=True) for k in ("vision", "laya")}
    connector_before = fingerprint(model.connector)
    examples = [
        ({"goal": "Hold W.", "action": {"buttons": ["w"]}}, inputs),
        ({"goal": "Release W.", "action": {"buttons": []}}, (inputs[0] + 1, *inputs[1:])),
    ]
    train_adapter_buttons(runtime, examples, 3, 0.0001, 17, io.StringIO(), batch_size=batch_size)
    assert before == {k: fingerprint(getattr(model, k), frozen_only=True) for k in before}
    assert connector_before != fingerprint(model.connector)
    assert runtime.metadata["action_training_examples"] == 3 * (batch_size or 2)


def test_fresh_game_audit_requires_every_wording_to_pass():
    from laya_vision_stitch.game_goal_audit import wording_passed

    good = {
        "button_exact_match": 0.9,
        "balanced_button_exact_match": 0.9,
        "both_goals_correct": 0.9,
    }
    assert not wording_passed("Not evaluated")
    assert not wording_passed({"by_template": {}})
    assert wording_passed({"by_template": {"one": good}})
    assert not wording_passed(
        {"by_template": {"one": good, "two": {**good, "both_goals_correct": 0.5}}}
    )


def test_clause_order_curriculum_retains_disjoint_new_wording():
    from laya_vision_stitch.goal_curriculum import (
        DEVELOPMENT,
        INVERSE_TRAIN,
        LOGIC_TRAIN,
        SEALED,
        SEALED_V2,
        TRAIN,
    )

    assert len(INVERSE_TRAIN) == len(TRAIN)
    assert all(
        text.index("{opposite}") < text.index("{state}") if "{state}" in text else True
        for text in INVERSE_TRAIN
    )
    assert not set(TRAIN + INVERSE_TRAIN + LOGIC_TRAIN) & set(DEVELOPMENT + SEALED + SEALED_V2)
    assert not set(SEALED_V2) & set(DEVELOPMENT + SEALED)


def test_matched_sampling_holds_scene_and_wording_constant_across_opposing_goals():
    from laya_vision_stitch.adapter_buttons import matched_quartets

    rows = [
        {
            "_sampling_stratum": [state, goal],
            "_pair_variant": wording,
            "frames": [{"sha256": f"scene-{state}-{scene}", "age_seconds": 0}],
        }
        for wording in ("a", "b")
        for state in (0, 1)
        for scene in (0, 1)
        for goal in (0, 1)
    ]
    variants = matched_quartets(rows)
    assert len(variants) == 2
    for states in variants:
        all_indices = [i for pairs in states for pair in pairs for i in pair]
        assert len({rows[i]["_pair_variant"] for i in all_indices}) == 1
        for state, pairs in enumerate(states):
            for a, b in pairs:
                assert rows[a]["frames"] == rows[b]["frames"]
                assert rows[a]["_sampling_stratum"] == [state, 0]
                assert rows[b]["_sampling_stratum"] == [state, 1]
    with pytest.raises(ValueError, match="both goals"):
        matched_quartets(rows[1:])
    with pytest.raises(ValueError, match="Duplicate"):
        matched_quartets(rows + rows[:1])


def test_gameplay_button_metrics_distinguish_empty_majority_from_action_learning():
    from laya_vision_stitch.gameplay_buttons import button_metrics

    rows = [{"action": {"buttons": []}} for _ in range(3)] + [
        {"action": {"buttons": ["w", "shift"]}}
    ]
    empty = button_metrics(rows, [[], [], [], []], ["w", "shift"])
    assert empty["button_exact_match"] == 0.75
    assert empty["button_micro_f1"] == 0
    assert empty["action_set_balanced_exact"] == 0.5
    perfect = button_metrics(rows, [[], [], [], ["w", "shift"]], ["w", "shift"])
    assert perfect["button_exact_match"] == perfect["button_micro_f1"] == 1


def test_text_oracle_is_balanced_and_keeps_reserved_wording_unconsumed():
    from laya_vision_stitch.goal_curriculum import SEALED, SEALED_V2
    from laya_vision_stitch.goal_oracle import cases

    rows = list(cases())
    assert len(rows) == 72
    assert sum(r["expected"] == "hold" for r in rows) == 36
    reserved = {
        t.format(state=s, opposite=o)
        for t in SEALED + SEALED_V2
        for s, o in (("open", "closed"), ("closed", "open"))
    }
    assert not {r["goal"] for r in rows} & reserved


def test_positive_weights_balance_rare_key_gradients_under_stratum_sampler():
    from laya_vision_stitch.decoder_probe import button_loss, stratum_positive_weights

    # Three equiprobable strata; W appears in one. Repeating rows in another
    # stratum must not change weighting because strata, not rows, are sampled.
    rows = [
        {"goal": goal, "action": {"buttons": buttons}}
        for goal, buttons in (("one", ["w"]), ("two", []), ("three", []))
    ]
    weights = stratum_positive_weights(rows, ["w", "unused"])
    np.testing.assert_allclose(weights, [2, 1])
    np.testing.assert_array_equal(
        weights, stratum_positive_weights(rows + rows[1:2] * 10, ["w", "unused"])
    )
    target = [mx.array([[float(i == 0), 0]]) for i in range(3)]
    gradient = mx.grad(
        lambda logits: (
            sum(button_loss(logits, t, [True, False], mx.array(weights)) for t in target) / 3
        )
    )(mx.zeros((1, 2)))
    assert abs(float(gradient[0, 0])) < 1e-7
    assert float(gradient[0, 1]) > 0


def test_gameplay_content_review_rejects_missing_duplicate_and_stale_labels():
    from laya_vision_stitch.gameplay_data import reviewed_candidates

    rows = [{"id": name, "frames": [{"sha256": name}]} for name in ("game", "browser")]
    labels = [
        {"id": r["id"], "image_sha256": r["id"], "game_content": r["id"] == "game"} for r in rows
    ]
    accepted, rejected = reviewed_candidates(rows, {"labels": labels})
    assert accepted == rows[:1] and rejected == rows[1:]
    for changed in (labels[:1], labels + labels[:1]):
        with pytest.raises(ValueError, match="exactly one"):
            reviewed_candidates(rows, {"labels": changed})
    labels[0]["image_sha256"] = "changed"
    with pytest.raises(ValueError, match="changed"):
        reviewed_candidates(rows, {"labels": labels})
