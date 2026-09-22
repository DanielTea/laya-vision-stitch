# SPDX-License-Identifier: MIT
# Algorithm adapted from fast_image_resize 5.1.4; see third_party/fast-image-resize.NOTICE.
"""Open-P2P Hamming interpolation, including uint8 rounding after each pass.

Unlike Pillow downsampling this does not widen the filter kernel. The pinned
Rust helper is an offline reference, not a runtime subprocess/dependency.
"""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=32)
def coefficients(source, destination):
    center = (np.arange(destination, dtype=np.float64) + 0.5) * (source / destination) - 0.5
    lower = np.floor(center).astype(np.int32)
    indices = np.stack([lower, lower + 1], -1)
    distance = np.abs(indices - center[:, None])
    angle = np.pi * distance
    sinc = np.ones_like(angle)
    np.divide(np.sin(angle), angle, out=sinc, where=angle != 0)
    weights = (0.54 + 0.46 * np.cos(angle)) * sinc
    weights[(distance >= 1) | (indices < 0) | (indices >= source)] = 0
    weights /= weights.sum(-1, keepdims=True)
    maximum = weights.max()
    precision = 0
    for precision in range(22):
        if np.floor(maximum * (1 << (precision + 1)) + 0.5) >= (1 << 15):
            break
    fixed = np.floor(weights * (1 << precision) + 0.5).astype(np.int32)
    return np.clip(indices, 0, source - 1), fixed, precision


def resize_rgb(pixels, width=192, height=192):
    pixels = np.asarray(pixels)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise ValueError("Expected uint8 HWC RGB pixels")
    if min(*pixels.shape[:2], width, height) < 1 or max(*pixels.shape[:2], width, height) > 16384:
        raise ValueError("Invalid image dimensions")
    result = pixels
    if pixels.shape[0] != height:
        indices, weights, bits = coefficients(pixels.shape[0], height)
        accum = (pixels[indices].astype(np.int32) * weights[:, :, None, None]).sum(1)
        result = np.clip((accum + (1 << (bits - 1))) >> bits, 0, 255).astype(np.uint8)
    if pixels.shape[1] != width:
        indices, weights, bits = coefficients(pixels.shape[1], width)
        accum = (result[:, indices].astype(np.int32) * weights[None, :, :, None]).sum(2)
        result = np.clip((accum + (1 << (bits - 1))) >> bits, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(result)
