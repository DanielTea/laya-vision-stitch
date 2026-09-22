import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("laya_mlx")

from laya_mlx.model import DecisionModel, EncoderConfig  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from laya_vision_stitch.lexical_stitch import FrozenStitch, LexicalBridge  # noqa: E402


def test_fixed_word_identity_transfer_and_freezing():
    bridge = LexicalBridge(mx.eye(3), mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), top_k=1)
    mapped, indices, weights = bridge(mx.array([[0.0, 1.0, 0.0]]))
    np.testing.assert_allclose(np.asarray(mapped), [[3.0, 4.0]])
    assert int(indices.item()) == 1
    np.testing.assert_array_equal(np.asarray(weights), [[1.0]])
    assert bridge.trainable_parameters() == {}


def test_mlx_embedding_path_preserves_original_laya_logits():
    cfg = EncoderConfig.from_dict(
        dict(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
        )
    )
    laya = DecisionModel(cfg, {"head_layers": 1})
    model = FrozenStitch(None, laya, LexicalBridge(mx.eye(3), mx.ones((3, 16)), top_k=1))
    batch = dict(
        input_ids=mx.array([[1, 2, 3, 4, 0]]),
        attention_mask=mx.array([[True, True, True, True, False]]),
        marker_pos=mx.array([[1, 3]]),
        marker_mask=mx.array([[True, True]]),
        qtype=mx.array([0]),
    )
    original, _ = laya(**batch)
    embedded = laya.encoder.embeddings.tok_embeddings(batch["input_ids"])
    stitched = model.score_embeddings(
        embedded, **{k: v for k, v in batch.items() if k != "input_ids"}
    )
    np.testing.assert_allclose(np.asarray(stitched), np.asarray(original), atol=1e-6)
    assert tree_flatten(model.trainable_parameters()) == []
