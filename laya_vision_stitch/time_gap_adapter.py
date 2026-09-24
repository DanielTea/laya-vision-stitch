"""Elapsed-time input so temporal memory can survive irregular screenshot gaps.

A zero-initialized residual on the image token encodes the time since the previous
processed frame. At the nominal 50 ms interval the pretrained behavior is unchanged at
initialization. This replaces memory resets with a learned input, not a timing rule.
"""

import math

import mlx.core as mx
import mlx.nn as nn

NOMINAL_SECONDS = 0.05


class TimeGapAdapter(nn.Module):
    def __init__(self, hidden=256, frequencies=8):
        super().__init__()
        self.frequencies = frequencies
        self.hidden = nn.Linear(2 + 2 * frequencies, hidden)
        self.output = nn.Linear(hidden, 1024)
        self.output.weight = mx.zeros_like(self.output.weight)
        self.output.bias = mx.zeros_like(self.output.bias)

    def features(self, dt):
        ratio = mx.maximum(dt.astype(mx.float32), 1e-3) / NOMINAL_SECONDS
        log_ratio = mx.log(ratio)
        scales = 2.0 ** mx.arange(self.frequencies) * math.pi / 8
        angles = log_ratio[..., None] * scales
        return mx.concatenate(
            [
                log_ratio[..., None],
                mx.minimum(ratio, 40.0)[..., None] / 40.0,
                mx.sin(angles),
                mx.cos(angles),
            ],
            -1,
        )

    def __call__(self, dt):
        """dt seconds [...] -> residual [..., 1024]; zero exactly at the nominal interval."""
        f = self.features(dt) - self.features(mx.full(dt.shape, NOMINAL_SECONDS))
        return self.output(nn.silu(self.hidden(f)) - nn.silu(self.hidden(mx.zeros_like(f))))


def install_time_gap_adapter(model, hidden=256, frequencies=8):
    if hasattr(model.policy, "time_gap_adapter"):
        raise ValueError("Time-gap adapter already installed")
    model.policy.time_gap_adapter = TimeGapAdapter(hidden, frequencies)
