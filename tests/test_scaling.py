import json
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("laya_mlx")

from laya_vision_stitch.feature_cache import CachedExamples  # noqa: E402
from laya_vision_stitch.multitask_data import expand  # noqa: E402
from laya_vision_stitch.policy_data import check_separation, read_manifest  # noqa: E402
from laya_vision_stitch.scaling_data import HOLDOUT, create  # noqa: E402
from laya_vision_stitch.trainable_model import (  # noqa: E402
    AlignedConnector,
    PolicyConfig,
    TrainableRuntime,
)


class Tokenizer:
    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 1, 2, 3, 0
    mask_token = "[MASK]"

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) % 28 + 4 for c in text]}


def test_supervision_never_enters_inference_prompt():
    runtime = TrainableRuntime.__new__(TrainableRuntime)
    runtime.agent = SimpleNamespace(tok=Tokenizer(), cfg={"max_len": 2048})
    runtime.module = SimpleNamespace(policy_config=PolicyConfig())
    row = {"goal": "Find the red object", "choices": {"a": "left", "b": "right"}}
    first = runtime.prepare(row)
    second = runtime.prepare(
        dict(
            row,
            description="SECRET ANSWER IS RIGHT",
            answer="b",
            action={"buttons": ["d"]},
            scene={"target": "right"},
        )
    )
    assert first[2] == second[2]
    for key in first[0]:
        np.testing.assert_array_equal(np.asarray(first[0][key]), np.asarray(second[0][key]))
    np.testing.assert_array_equal(np.asarray(first[1]), np.asarray(second[1]))


def test_cache_reuses_images_across_goals_and_bounds_residency(tmp_path):
    calls = []
    runtime = SimpleNamespace(
        module=SimpleNamespace(vision=nn.Linear(2, 2), policy_config=PolicyConfig()),
        processor=SimpleNamespace(to_dict=lambda: {"version": 1}),
        features=lambda row: (calls.append(row["id"]) or mx.ones((2, 8)), mx.zeros((2, 4))),
        prepare=lambda row: (row["goal"],),
    )
    rows = [
        {"id": str(i), "goal": str(i), "frames": [{"sha256": str(i // 2), "age_seconds": 0}]}
        for i in range(6)
    ]
    examples = CachedExamples(runtime, rows, tmp_path, resident=1)
    assert len(calls) == 3
    assert examples[0][1][-1] == "0" and examples[1][1][-1] == "1"
    assert len(list(tmp_path.glob("*.npz"))) == 3
    for i in range(6):
        examples[i]
        assert len(examples.memory) == 1
    CachedExamples(runtime, rows, tmp_path)
    assert len(calls) == 3
    runtime.module.policy_config.image_width = 512
    CachedExamples(runtime, rows, tmp_path)
    assert len(calls) == 6


def test_aligned_connector_starts_at_learned_anchors_then_receives_gradients():
    config = PolicyConfig(connector_type="aligned", connector_width=16, visual_slots=4)
    connector = AlignedConnector(8, 16, config)
    connector.anchors = mx.random.normal((4, 16))
    inputs = (mx.random.normal((8, 8)), mx.zeros((8, 4)), mx.zeros((1, 3, 16)))
    np.testing.assert_array_equal(np.asarray(connector(*inputs)[0]), np.asarray(connector.anchors))
    _, grads = nn.value_and_grad(connector, lambda m: mx.mean((m(*inputs) - 1) ** 2))(connector)
    assert float(mx.abs(grads["output"]["weight"]).sum().item()) > 0


def test_compositions_styles_and_images_are_held_out(tmp_path):
    create(tmp_path / "data", train_scenes=12, validation_scenes=6, test_scenes=6)
    root = tmp_path / "data"
    splits = {
        s: read_manifest(root / f"{s}.jsonl", PolicyConfig())
        for s in ("train", "validation", "composition_test", "style_test")
    }
    for name, rows in splits.items():
        if name != "train":
            check_separation(splits["train"], rows)
        for row in rows:
            combinations = {(o[0], o[1]) for o in row["scene"]["objects"]}
            assert bool(combinations & HOLDOUT) == (name == "composition_test")
            assert (row["scene"]["style"] == "panel") == (name == "style_test")
    # Each unique image has opposite goals/labels: no constant policy can exceed 50%.
    for rows in splits.values():
        for i in range(0, len(rows), 2):
            assert rows[i]["answer"] != rows[i + 1]["answer"]
    bad = json.loads((root / "train.jsonl").read_text().splitlines()[0])
    bad["description"] = 123
    (root / "bad.jsonl").write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="Description"):
        read_manifest(root / "bad.jsonl", PolicyConfig())


def test_new_goals_receive_new_supervision_without_stale_teacher_targets(tmp_path):
    create(tmp_path / "data", train_scenes=2, validation_scenes=2, test_scenes=2)
    path = tmp_path / "data/train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["teacher_probs"] = {"left": 0.75, "right": 0.25}
        row["teacher_temperature"] = 2.0
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    expanded = expand(path, tmp_path / "expanded.jsonl")
    assert len(expanded) == 12
    by_id = {r["id"]: r for r in expanded}
    for original in rows:
        away = by_id[original["id"] + "-away"]
        assert away["answer"] != original["answer"]
        assert away["action"]["buttons"] != original["action"]["buttons"]
        assert "teacher_probs" not in away
        assert by_id[original["id"]]["teacher_probs"] == original["teacher_probs"]
    read_manifest(tmp_path / "expanded.jsonl", PolicyConfig())


def test_canonical_targets_and_paraphrases_preserve_task_balance(tmp_path):
    create(tmp_path / "data", train_scenes=2, validation_scenes=2, test_scenes=2)
    expanded = expand(
        tmp_path / "data/train.jsonl",
        tmp_path / "expanded.jsonl",
        canonical_descriptions=True,
        paraphrases=True,
    )
    assert len(expanded) == 48
    families = {
        name: sum(r["task_family"] == name for r in expanded)
        for name in ("toward", "away", "left_color", "left_shape")
    }
    assert families == {"toward": 16, "away": 16, "left_color": 8, "left_shape": 8}
    descriptions = {}
    for row in expanded:
        left = min(row["scene"]["objects"], key=lambda o: o[2])
        assert row["description"].startswith(f"The {left[0]} {left[1]}")
        descriptions.setdefault(row["frames"][0]["sha256"], set()).add(row["description"])
    assert all(len(values) == 1 for values in descriptions.values())
    read_manifest(tmp_path / "expanded.jsonl", PolicyConfig())
