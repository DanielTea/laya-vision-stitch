"""Flow-matching action-chunk head over the frozen P2P policy context (pi0-style).

One chunk holds H consecutive 50 ms steps. Each step has binary controls in {-1, +1}
and symlog-scaled relative mouse motion. The same module without noise/time inputs is a
deterministic ablation. This is a learned output head, not a gameplay rule.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .p2p_adaptation import KEYS_WITH_TAB
from .p2p_pretrained_policy import MOUSE_NAMES, MOUSE_X, MOUSE_Y

CONTROLS = tuple(dict.fromkeys(k for k in (*KEYS_WITH_TAB, *MOUSE_NAMES) if k is not None))
BINARY = len(CONTROLS)
DIM = BINARY + 2
MOUSE_LOG = math.log1p(501.0)


def tokens_to_vectors(tokens):
    """[N,8] P2P tokens -> [N,DIM] continuous action vectors."""
    tokens = np.asarray(tokens).reshape(-1, 8)
    out = -np.ones((len(tokens), DIM), np.float32)
    index = {c: i for i, c in enumerate(CONTROLS)}
    for n, row in enumerate(tokens):
        for t in row[:4]:
            if KEYS_WITH_TAB[t] is not None:
                out[n, index[KEYS_WITH_TAB[t]]] = 1
        for t in row[4:6]:
            if MOUSE_NAMES[t] is not None:
                out[n, index[MOUSE_NAMES[t]]] = 1
    px = np.stack([np.asarray(MOUSE_X)[tokens[:, 6]], np.asarray(MOUSE_Y)[tokens[:, 7]]], 1)
    out[:, BINARY:] = np.sign(px) * np.log1p(np.abs(px)) / MOUSE_LOG
    return out


def vectors_to_actions(vectors):
    """[N,DIM] -> (button sets, mouse pixels); binary threshold at zero."""
    vectors = np.asarray(vectors).reshape(-1, DIM)
    buttons = [frozenset(CONTROLS[i] for i in np.flatnonzero(v[:BINARY] > 0)) for v in vectors]
    m = np.clip(vectors[:, BINARY:], -1, 1)
    px = np.sign(m) * np.expm1(np.abs(m) * MOUSE_LOG)
    # Snap to the released bin centers so errors match token-based evaluation.
    centers = [np.asarray(MOUSE_X, float), np.asarray(MOUSE_Y, float)]
    snapped = np.stack(
        [centers[k][np.abs(px[:, k : k + 1] - centers[k][None]).argmin(1)] for k in (0, 1)], 1
    )
    return buttons, snapped


def chunk_targets(vectors, sequence_index, steps, horizon):
    """Targets [N,H,DIM] and validity mask [N,H]; chunks never cross sequences."""
    n = len(vectors)
    targets = np.zeros((n, horizon, DIM), np.float32)
    mask = np.zeros((n, horizon), np.float32)
    for i in range(n):
        for h in range(horizon):
            j = i + h
            if j < n and sequence_index[j] == sequence_index[i] and steps[j] == steps[i] + h:
                targets[i, h], mask[i, h] = vectors[j], 1
    return targets, mask


def timestep_embedding(t, width):
    half = width // 2
    freqs = mx.exp(-math.log(10000.0) * mx.arange(half) / half)
    angles = t[:, None] * 1000.0 * freqs[None]
    return mx.concatenate([mx.sin(angles), mx.cos(angles)], -1)


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, affine=False)
        self.norm2 = nn.LayerNorm(width, affine=False)
        self.attention = nn.MultiHeadAttention(width, heads)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )
        self.modulation = nn.Linear(width, 6 * width)
        self.modulation.weight = mx.zeros_like(self.modulation.weight)
        self.modulation.bias = mx.zeros_like(self.modulation.bias)

    def __call__(self, x, condition):
        s1, b1, g1, s2, b2, g2 = mx.split(self.modulation(condition)[:, None], 6, -1)
        h = self.norm1(x) * (1 + s1) + b1
        x = x + g1 * self.attention(h, h, h)
        h = self.norm2(x) * (1 + s2) + b2
        return x + g2 * self.mlp(h)


class ChunkHead(nn.Module):
    """Flow velocity field (flow=True) or deterministic chunk predictor (flow=False)."""

    def __init__(self, horizon=8, width=256, depth=4, heads=4, flow=True, context_width=1024):
        super().__init__()
        self.horizon, self.flow = horizon, flow
        self.context = nn.Sequential(
            nn.LayerNorm(context_width),
            nn.Linear(context_width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.input = nn.Linear(DIM, width)
        self.position = mx.random.normal((1, horizon, width)) * 0.02
        self.blocks = [Block(width, heads) for _ in range(depth)]
        self.final_norm = nn.LayerNorm(width, affine=False)
        self.output = nn.Linear(width, DIM)
        self.width = width

    def __call__(self, context, noisy=None, t=None):
        condition = self.context(context.astype(mx.float32))
        if self.flow:
            condition = condition + self.time(timestep_embedding(t, self.width))
            x = self.input(noisy) + self.position
        else:
            x = mx.broadcast_to(self.position, (context.shape[0], self.horizon, self.width))
        condition = nn.silu(condition)
        for block in self.blocks:
            x = block(x, condition)
        return self.output(self.final_norm(x))


def flow_loss(head, context, target, mask, key):
    """Rectified flow: x_t = t*a + (1-t)*eps, velocity a - eps; noisier t emphasized."""
    k1, k2 = mx.random.split(key)
    b = target.shape[0]
    u = mx.random.beta(1.5, 1.0, (b,), key=k1) if hasattr(mx.random, "beta") else None
    if u is None:
        # Beta(1.5, 1) via inverse CDF: F(u) = u^1.5.
        u = mx.random.uniform(shape=(b,), key=k1) ** (1 / 1.5)
    t = 0.999 * (1 - u)
    noise = mx.random.normal(target.shape, key=k2)
    noisy = t[:, None, None] * target + (1 - t[:, None, None]) * noise
    velocity = head(context, noisy, t)
    error = ((velocity - (target - noise)) ** 2).mean(-1)
    return (error * mask).sum() / mx.maximum(mask.sum(), 1)


def deterministic_loss(head, context, target, mask):
    out = head(context)
    binary = nn.losses.binary_cross_entropy(
        out[..., :BINARY], (target[..., :BINARY] > 0).astype(mx.float32), reduction="none"
    ).mean(-1)
    mouse = ((mx.tanh(out[..., BINARY:]) - target[..., BINARY:]) ** 2).mean(-1)
    return ((binary + mouse) * mask).sum() / mx.maximum(mask.sum(), 1)


def deterministic_predict(head, context):
    out = head(context)
    return mx.concatenate(
        [mx.where(out[..., :BINARY] > 0, 1.0, -1.0), mx.tanh(out[..., BINARY:])], -1
    )


def sample(head, context, steps=10, key=None, unconditional=None, scale=1.0, noise=None):
    """Euler integration from noise (t=0) to actions (t=1); optional CFG on the context."""
    b = context.shape[0]
    x = noise if noise is not None else mx.random.normal((b, head.horizon, DIM), key=key)
    for i in range(steps):
        t = mx.full((b,), i / steps)
        v = head(context, x, t)
        if unconditional is not None and scale != 1.0:
            v = head(unconditional, x, t) + scale * (v - head(unconditional, x, t))
        x = x + v / steps
    return x


def realtime_sample(head, context, committed, delay, steps=10, key=None, beta=5.0, overlap=None):
    """Real-time chunking: inpaint a new chunk consistent with already committed actions.

    `committed` [B,H,DIM] holds the previous chunk shifted to the new start time; the first
    `delay` steps will execute before this chunk is ready and are frozen (weight 1). Weights
    decay exponentially over the remaining overlap, then zero. Guidance follows the
    pseudo-inverse correction on the one-step denoised estimate, clipped by `beta`.
    """
    b, h = context.shape[0], head.horizon
    overlap = h if overlap is None else overlap
    weights = np.zeros(h, np.float32)
    weights[:delay] = 1
    span = max(1, overlap - delay)
    for i in range(delay, overlap):
        c = (overlap - i) / (span + 1)
        weights[i] = c * (math.e**c - 1) / (math.e - 1)
    w = mx.array(weights)[None, :, None]
    x = mx.random.normal((b, h, DIM), key=key)
    for i in range(steps):
        tau = i / steps
        t = mx.full((b,), tau)

        def denoised(z):
            return z + (1 - tau) * head(context, z, t)

        estimate, vjp = mx.vjp(denoised, [x], [w * (committed - denoised(x))])
        v = head(context, x, t)
        r2 = ((1 - tau) ** 2) / (tau**2 + (1 - tau) ** 2)
        gain = min(beta, (1 - tau) / max(tau, 1e-3) / max(r2, 1e-6)) if tau > 0 else beta
        x = x + (v + gain * vjp[0]) / steps
    return x
