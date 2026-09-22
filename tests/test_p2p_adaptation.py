import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from laya_vision_stitch.p2p_adaptation import (
    ExtendedEmbedding,
    ExtendedOutput,
    LoRALinear,
    encode_action,
)
from laya_vision_stitch.p2p_pretrained_policy import Layer, OpenP2PPolicy


class TinyStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [Layer(16, 2, 1) for _ in range(2)]

    def __call__(self, x, offset=0, caches=None, mask=None):
        updated = []
        for i, layer in enumerate(self.layers):
            x, cache = layer(x, offset, None if caches is None else caches[i], mask)
            updated.append(cache)
        return x, updated


class TinyPolicy(OpenP2PPolicy):
    def __init__(self):
        nn.Module.__init__(self)
        self.decoder = TinyStack()
        self.decoder_projection = nn.Linear(16, 16)
        self.decoder_position = mx.random.normal((9, 16))
        self.embeddings = [nn.Embedding(n, 16) for n in (20, 4, 23, 17)]
        self.outputs = [nn.Linear(16, n) for n in (20, 4, 23, 17)]


def test_parallel_teacher_decoder_matches_autoregressive_forced_batch():
    mx.random.seed(87)
    model = TinyPolicy()
    context = mx.random.normal((3, 1, 16))
    tokens = mx.array(
        [[11, 0, 0, 0, 1, 0, 12, 8], [6, 7, 0, 0, 2, 0, 7, 3], [2, 3, 4, 0, 0, 0, 11, 8]]
    )
    _, sequential = model.decode(context, forced=tokens)
    parallel = model.teacher_logits(context, tokens)
    for actual, expected in zip(parallel, sequential, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
    changed = mx.concatenate([tokens[:, :3], mx.zeros((3, 5), mx.int32)], 1)
    changed_logits = model.teacher_logits(context, changed)
    for i in range(4):
        np.testing.assert_allclose(
            np.asarray(parallel[i]), np.asarray(changed_logits[i]), atol=1e-5
        )


def test_zero_lora_preserves_frozen_base_and_only_exposes_adapter():
    base = nn.Linear(8, 12)
    layer = LoRALinear(base, 4)
    x = mx.random.normal((2, 8))
    np.testing.assert_array_equal(np.asarray(layer(x)), np.asarray(base(x)))
    assert set(dict(tree_flatten(layer.trainable_parameters()))) == {"a", "b"}


def test_tab_extension_preserves_original_embeddings_and_logits():
    embedding = nn.Embedding(20, 16)
    extended = ExtendedEmbedding(embedding)
    ids = mx.arange(20)
    np.testing.assert_array_equal(np.asarray(extended(ids)), np.asarray(embedding(ids)))
    base = nn.Linear(16, 20)
    output = ExtendedOutput(base)
    x = mx.random.normal((3, 16))
    np.testing.assert_array_equal(np.asarray(output(x)[:, :20]), np.asarray(base(x)))
    assert output(x).shape == (3, 21)
    assert set(dict(tree_flatten(output.trainable_parameters()))) == {"extra.weight", "extra.bias"}


def test_training_action_mapping_preserves_idle_and_quantization_boundaries():
    assert encode_action({"buttons": [], "mouse_delta": [0, 0]}) == [0, 0, 0, 0, 0, 0, 11, 8]
    assert encode_action(
        {"buttons": ["tab", "w", "mouse_right"], "mouse_delta": [1 / 512, -1 / 512]}
    ) == [11, 20, 0, 0, 2, 0, 12, 7]
    with pytest.raises(ValueError, match="Unsupported controls"):
        encode_action({"buttons": ["enter"], "mouse_delta": [0, 0]})
