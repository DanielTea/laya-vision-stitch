import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from laya_vision_stitch.p2p_pretrained_vision import MBConv, OpenP2PVision, preprocess


def test_p2p_preprocessing_preserves_rgb_scale_and_full_frame():
    pixels = preprocess(Image.new("RGB", (320, 180), (255, 128, 0)))
    assert pixels.shape == (1, 192, 192, 3)
    np.testing.assert_allclose(pixels[0, 80, 80], [1, 128 / 255, 0])


def test_overflowing_fp16_conversion_is_explicitly_rejected():
    with pytest.raises(ValueError, match="FP16 overflows"):
        OpenP2PVision.from_state({}, mx.float16)


def test_mbconv_residual_preserves_input_when_projection_is_zero():
    block = MBConv(8, 8, 6, 3, 1)
    block.eval()
    block.project.conv.weight = mx.zeros_like(block.project.conv.weight)
    x = mx.random.normal((1, 8, 8, 8))
    np.testing.assert_array_equal(np.asarray(block(x)), np.asarray(x))
