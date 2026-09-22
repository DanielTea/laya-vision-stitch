"""Golden bytes from the pinned upstream Rust resizer, not a second Python port."""

import hashlib

import numpy as np
import pytest

from laya_vision_stitch.p2p_resize import resize_rgb


def test_upstream_rust_golden_bytes():
    rng = np.random.default_rng(61)
    cases = [
        ((1280, 720, 192, 192), "ab420b03745c7ada456ee696750ea00e9aed5bd40bd60f18f36d0bb45bd2b4c2"),
        ((192, 192, 192, 192), "44e69c9169cbb40688c66eb63b55325d4bbc69755791f27307792ad76bae7863"),
        ((31, 17, 192, 192), "f380a571f44affd529786062aa662f10954bb7c9968f066f8f0a35573ebd5dfb"),
        ((1, 1, 192, 192), "589be11216d3650cbce7ef0968c8c00eeac5e604ababd97048787b077b82e33e"),
        ((517, 383, 81, 53), "3d2868e0d5362614b60ee9e186db808e221cb15c4216774f7fb2d76ea4db0340"),
        ((111, 51, 1, 1), "7cb8255bc229306df576f473b5deb473d26a625d4b7023ffdfddf4a54a55f70a"),
        ((2, 5, 7, 3), "63b4d0b91eb5ef9804f20dcbe788486009697b3fa1f6790b319a974a84e7702c"),
    ]
    for (w, h, ow, oh), digest in cases:
        pixels = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        output = resize_rgb(pixels, ow, oh)
        assert hashlib.sha256(output.tobytes()).hexdigest() == digest


def test_reject_non_rgb_and_invalid_dimensions():
    with pytest.raises(ValueError):
        resize_rgb(np.zeros((10, 10, 4), np.uint8))
    with pytest.raises(ValueError):
        resize_rgb(np.zeros((10, 10, 3), np.float32))
    with pytest.raises(ValueError):
        resize_rgb(np.zeros((10, 10, 3), np.uint8), 0, 192)
