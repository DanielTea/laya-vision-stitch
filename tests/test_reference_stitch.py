import json
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("laya_mlx")

from mlx.utils import tree_flatten  # noqa: E402

from laya_vision_stitch.clip_backend import legacy_weights  # noqa: E402
from laya_vision_stitch.reference_fixtures import create, validate_separation  # noqa: E402
from laya_vision_stitch.reference_stitch import ReferenceStitch, read_references  # noqa: E402


class IdentityVision(nn.Module):
    def __call__(self, image):
        return SimpleNamespace(pooler_output=image)


class CaptionReader(nn.Module):
    def __call__(self, ids, mask, markers, marker_mask, qtype):
        # Test double: each stored caption chooses its encoded class.
        second = ids[:, -1] == 2
        return mx.stack([mx.where(second, -6.0, 6.0), mx.where(second, 6.0, -6.0)], axis=-1), None


def test_reference_attention_uses_paired_captions_and_remains_frozen():
    model = ReferenceStitch(
        IdentityVision(),
        nn.Identity(),
        CaptionReader(),
        mx.eye(2),
        mx.eye(2),
        mx.array([[1], [2]]),
        mx.ones((2, 1), dtype=mx.bool_),
        top_k=2,
    )
    args = (mx.array([[1.0, 0.0]]), mx.array([0]), mx.array([0, 0]), mx.array([True, True]))
    before, indices, weights, _ = model(*args)
    assert int(mx.argmax(before).item()) == 0
    assert indices.tolist() == [0, 1]
    assert np.isclose(float(weights.sum().item()), 1)
    model.reference_ids = model.reference_ids[::-1]
    after, *_ = model(*args)
    assert int(mx.argmax(after).item()) == 1
    assert tree_flatten(model.trainable_parameters()) == []


def test_fixtures_hold_out_compositions_and_exact_images(tmp_path):
    references, tests = create(tmp_path / "fixtures")
    assert len(references) == 36 and len(tests) == 54
    assert sum(t["novel_composition"] for t in tests) == 18
    validate_separation(references, tests)
    leaked = [dict(tests[0], sha256=references[0]["sha256"])]
    with pytest.raises(ValueError, match="leakage"):
        validate_separation(references, leaked)
    manifest = tmp_path / "fixtures/references.json"
    rows = json.loads(manifest.read_text())
    rows[1]["image"] = rows[0]["image"]
    manifest.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="unique"):
        read_references(manifest)


def test_legacy_clip_conversion_only_renames_tensors():
    value = mx.array([[1.0, 2.0]])
    converted = legacy_weights(
        {
            "text_model.layers.0.attention.query_proj.weight": value,
            "vision_model.position_embedding": value,
        }
    )
    assert converted["text_model.encoder.layers.0.self_attn.q_proj.weight"] is value
    assert converted["vision_model.embeddings.position_embedding.weight"] is value
