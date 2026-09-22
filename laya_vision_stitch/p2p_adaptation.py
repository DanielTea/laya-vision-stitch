"""Small learned control adapters; frozen pretrained weights stay intact."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .p2p_pretrained_policy import KEY_NAMES, MOUSE_NAMES

KEYS_WITH_TAB = (*KEY_NAMES, "tab")
X_EDGES = [
    -500.1,
    -144.1,
    -76.1,
    -46.1,
    -29.1,
    -18.1,
    -11.1,
    -6.1,
    -3.1,
    -1.1,
    -0.1,
    0,
    1,
    3,
    6,
    11,
    18,
    29,
    46,
    76,
    144,
    500,
]
Y_EDGES = [-150.1, -24.1, -12.1, -7.1, -4.1, -3.1, -1.1, -0.1, 0, 1, 3, 4, 7, 12, 24, 150]


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.base.freeze()
        outputs, inputs = base.weight.shape
        self.a = mx.random.normal((inputs, rank)) * 0.01
        self.b = mx.zeros((rank, outputs))

    def __call__(self, x):
        return self.base(x) + (x @ self.a) @ self.b


class ExtendedEmbedding(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.base.freeze()
        self.extra = mx.mean(base.weight, axis=0, keepdims=True)

    def __call__(self, ids):
        return mx.concatenate([self.base.weight, self.extra], 0)[ids]


class ExtendedOutput(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.base.freeze()
        self.extra = nn.Linear(base.weight.shape[1], 1)
        self.extra.weight = mx.zeros_like(self.extra.weight)
        self.extra.bias = mx.array([-8.0])

    def __call__(self, x):
        return mx.concatenate([self.base(x), self.extra(x)], -1)


def install_control_adapter(model, rank=4):
    if rank < 1 or rank > 64:
        raise ValueError("Invalid control LoRA rank")
    model.freeze()
    for layer in model.policy.decoder.layers:
        layer.attention.qkv = LoRALinear(layer.attention.qkv, rank)
        layer.attention.output = LoRALinear(layer.attention.output, rank)
        layer.w2 = LoRALinear(layer.w2, rank)
    model.policy.embeddings[0] = ExtendedEmbedding(model.policy.embeddings[0])
    model.policy.outputs[0] = ExtendedOutput(model.policy.outputs[0])


def encode_action(action):
    """Original ordered token mapping plus Tab; reject unknown labels explicitly."""
    buttons = action["buttons"]
    unsupported = set(buttons) - set(KEYS_WITH_TAB) - set(MOUSE_NAMES)
    if unsupported:
        raise ValueError(f"Unsupported controls: {sorted(unsupported)}")
    keys = sorted({KEYS_WITH_TAB.index(x) for x in buttons if x in KEYS_WITH_TAB})
    mouse = sorted({MOUSE_NAMES.index(x) for x in buttons if x in MOUSE_NAMES})
    if len(keys) > 4 or len(mouse) > 2:
        raise ValueError("Too many simultaneous controls")
    delta = np.asarray(action["mouse_delta"], dtype=float) * 512
    if delta.shape != (2,) or not np.isfinite(delta).all():
        raise ValueError("Invalid mouse delta")
    bins = [
        int(np.searchsorted(edges, value, side="left"))
        for edges, value in zip((X_EDGES, Y_EDGES), np.rint(delta), strict=True)
    ]
    return keys + [0] * (4 - len(keys)) + mouse + [0] * (2 - len(mouse)) + bins
