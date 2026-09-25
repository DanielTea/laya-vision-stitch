"""Small learned control adapters; frozen pretrained weights stay intact."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .p2p_pretrained_policy import KEY_NAMES, MOUSE_NAMES

KEYS_WITH_TAB = (*KEY_NAMES, "tab")
# Controls beyond the released vocabulary, appended so earlier token ids are unchanged.
# A mouse-wheel notch is a key pressed for one step (camera zoom in strategy games,
# MMOs and ARPGs); the rest are keys common in D2E, CK3 and Baldur's Gate 3 logs.
EXTRA_CONTROLS = (
    "scroll_up",
    "scroll_down",
    "ctrl",
    "alt",
    "escape",
    "enter",
    "r",
    "c",
    "x",
    "v",
    "g",
    "i",
    "m",
    "b",
    "t",
    "h",
    "5",
    "6",
    "7",
    "8",
    "9",
    "0",
)
EXTENDED_KEYS = (*KEYS_WITH_TAB, *EXTRA_CONTROLS)
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
    def __init__(self, base, extra=1):
        super().__init__()
        self.base = base
        self.base.freeze()
        mean = mx.mean(base.weight, axis=0, keepdims=True)
        # Small noise breaks the symmetry between new controls; one extra stays the mean.
        noise = mx.random.normal((extra, base.weight.shape[1])) * mx.std(base.weight) * 0.01
        self.extra = mean + (noise if extra > 1 else 0)

    def __call__(self, ids):
        return mx.concatenate([self.base.weight, self.extra], 0)[ids]


class ExtendedOutput(nn.Module):
    def __init__(self, base, extra=1):
        super().__init__()
        self.base = base
        self.base.freeze()
        self.extra = nn.Linear(base.weight.shape[1], extra)
        self.extra.weight = mx.zeros_like(self.extra.weight)
        self.extra.bias = mx.full((extra,), -8.0)

    def __call__(self, x):
        return mx.concatenate([self.base(x), self.extra(x)], -1)


class VisualResidual(nn.Module):
    """Identity-initialized bottleneck trained through the frozen temporal policy."""

    def __init__(self, width=1024, bottleneck=64):
        super().__init__()
        if not 1 <= bottleneck <= width:
            raise ValueError("Invalid visual bottleneck")
        self.norm = nn.LayerNorm(width, affine=False)
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width, bias=False)
        self.up.weight = mx.zeros_like(self.up.weight)

    def __call__(self, x):
        return x + self.up(nn.silu(self.down(self.norm(x))))


def install_visual_adapter(model, bottleneck=64):
    if hasattr(model.policy, "visual_adapter"):
        raise ValueError("Visual adapter already installed")
    # Preserve the trainability of any separately installed control adapters.
    model.policy.visual_adapter = VisualResidual(bottleneck=bottleneck)


def install_control_adapter(model, rank=4, extra_keys=1):
    """Decoder LoRA plus `extra_keys` controls after the released key vocabulary.

    extra_keys=1 is Tab (KEYS_WITH_TAB, earlier checkpoints); len(EXTRA_CONTROLS) + 1
    gives EXTENDED_KEYS.
    """
    if rank < 1 or rank > 64:
        raise ValueError("Invalid control LoRA rank")
    if not 1 <= extra_keys <= 64:
        raise ValueError("Invalid number of extra controls")
    model.freeze()
    for layer in model.policy.decoder.layers:
        layer.attention.qkv = LoRALinear(layer.attention.qkv, rank)
        layer.attention.output = LoRALinear(layer.attention.output, rank)
        layer.w2 = LoRALinear(layer.w2, rank)
    model.policy.embeddings[0] = ExtendedEmbedding(model.policy.embeddings[0], extra_keys)
    model.policy.outputs[0] = ExtendedOutput(model.policy.outputs[0], extra_keys)


def restrict_controls(action, key_names=KEY_NAMES):
    """Drop controls outside `key_names` and mouse buttons; returns (action, complete).

    At most four keys fit one action; extra keys are dropped in vocabulary order. The
    frame is flagged incomplete whenever anything was dropped.
    """
    known = (set(key_names) | set(MOUSE_NAMES)) - {None}
    buttons = [b for b in action["buttons"] if b in known]
    keys = sorted((b for b in buttons if b in key_names), key=list(key_names).index)
    kept = set(keys[:4]) | {b for b in buttons if b in MOUSE_NAMES}
    buttons = [b for b in buttons if b in kept]
    return {**action, "buttons": buttons}, len(buttons) == len(action["buttons"])


def encode_action(action, key_names=KEYS_WITH_TAB):
    """Original ordered token mapping plus appended controls; reject unknown labels."""
    buttons = action["buttons"]
    unsupported = set(buttons) - set(key_names) - set(MOUSE_NAMES)
    if unsupported:
        raise ValueError(f"Unsupported controls: {sorted(unsupported)}")
    keys = sorted({list(key_names).index(x) for x in buttons if x in key_names})
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


def install_policy_lora(model, rank=8):
    """LoRA in every temporal-policy layer; call after `install_control_adapter`."""
    if rank < 1 or rank > 64:
        raise ValueError("Invalid policy LoRA rank")
    for layer in model.policy.policy.layers:
        if isinstance(layer.attention.qkv, LoRALinear):
            raise ValueError("Policy LoRA already installed")
        layer.attention.qkv = LoRALinear(layer.attention.qkv, rank)
        layer.attention.output = LoRALinear(layer.attention.output, rank)
        layer.w2 = LoRALinear(layer.w2, rank)
