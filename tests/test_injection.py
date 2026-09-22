from types import SimpleNamespace

import numpy as np
import pytest

from laya_vision_stitch.backends import LayaEmbeddings


def test_visual_injection_preserves_question_tokens_masks_and_template():
    seen = []

    def predict(inputs):
        seen.append(inputs)
        return {"scores": np.zeros((1, 1, 1, 32))}

    model = LayaEmbeddings.__new__(LayaEmbeddings)
    model.slots, model.width, model.start = 2, 3, 1
    model.inputs = {
        "embeddings": np.ones((1, 3, 1, 6), dtype=np.float16),
        "full_mask": np.zeros((1, 6, 1, 6), dtype=np.float16),
    }
    model.agent = SimpleNamespace(model=SimpleNamespace(predict=predict))
    model.predict(np.full((2, 3), 9))
    np.testing.assert_array_equal(seen[0]["embeddings"][0, :, 0, 1:3], 9)
    np.testing.assert_array_equal(seen[0]["embeddings"][0, :, 0, 0], 1)
    np.testing.assert_array_equal(seen[0]["embeddings"][0, :, 0, 3:], 1)
    np.testing.assert_array_equal(model.inputs["embeddings"], 1)
    assert seen[0]["full_mask"] is model.inputs["full_mask"]
    with pytest.raises(ValueError):
        model.predict(np.full((2, 3), np.nan))
