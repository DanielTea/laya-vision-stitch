# SPDX-License-Identifier: Apache-2.0
# Mamba-3 equations adapted from state-spaces/mamba; see third_party/mamba3.NOTICE.
"""Functional, causal memory with identical training and streaming computations."""

import math

import mlx.core as mx
import mlx.nn as nn


def affine_scan(decay, inputs):
    """Inclusive affine prefix scan, supporting gradients through every timestep."""
    a, b = decay, inputs
    gap = 1
    while gap < b.shape[1]:
        a, b = (
            mx.concatenate([a[:, :gap], a[:, gap:] * a[:, :-gap]], 1),
            mx.concatenate([b[:, :gap], b[:, gap:] + a[:, gap:] * b[:, :-gap]], 1),
        )
        gap *= 2
    return a, b


class Mamba3SISO(nn.Module):
    """Mamba-3 SISO: input-dependent decay, trapezoidal update and complex state.

    Pure MLX, float32 recurrent state. No CUDA kernel or pretrained Mamba weights.
    State is returned explicitly, never stored/mutated on the model.
    """

    def __init__(self, width, state_width=32, head_width=32):
        super().__init__()
        inner = 2 * width
        if inner % head_width or state_width % 4:
            raise ValueError("Invalid Mamba dimensions")
        self.inner, self.heads, self.head_width = inner, inner // head_width, head_width
        self.state_width, self.angles = state_width, state_width // 4
        self.sizes = [
            inner,
            inner,
            state_width,
            state_width,
            self.heads,
            self.heads,
            self.heads,
            self.angles,
        ]
        self.projection = nn.Linear(width, sum(self.sizes), bias=False)
        self.output = nn.Linear(inner, width, bias=False)
        dt = mx.exp(mx.random.uniform(shape=(self.heads,)) * math.log(100) + math.log(0.001))
        self.dt_bias = dt + mx.log(-mx.expm1(-dt))
        self.b_norm, self.c_norm = nn.RMSNorm(state_width), nn.RMSNorm(state_width)
        self.b_bias = mx.ones((self.heads, state_width))
        self.c_bias = mx.ones((self.heads, state_width))
        self.skip = mx.ones((self.heads,))

    def __call__(self, u, state=None):
        batch, length, _ = u.shape
        split = []
        n = 0
        for size in self.sizes[:-1]:
            n += size
            split.append(n)
        z, x, b, c, dt, a, trap, angle = mx.split(self.projection(u), split, -1)
        x, z = [v.reshape(batch, length, self.heads, self.head_width) for v in (x, z)]
        dt = nn.softplus(dt + self.dt_bias)
        # Official heavy-tail activation, not softplus(A).
        a = -(mx.maximum(a, 0) + 1 / (1 - mx.minimum(a, 0)))
        decay = mx.exp(mx.minimum(a, -1e-4) * dt)[..., None, None]
        gate = mx.sigmoid(trap)[..., None, None]
        dt = dt[..., None, None]
        phases = mx.cumsum(mx.tanh(angle)[:, :, None] * math.pi * dt[..., 0], axis=1)
        if state is not None:
            phases = phases + state[0][:, None]
        # Match official bounded phase; paired dimensions rotate only the first half.
        phases = phases - 2 * math.pi * mx.floor(phases / (2 * math.pi))

        def rotate(value, norm, bias):
            value = norm(value)[:, :, None] + bias
            r, tail = value[..., : 2 * self.angles], value[..., 2 * self.angles :]
            even, odd = r[..., 0::2], r[..., 1::2]
            r = mx.stack(
                [
                    even * mx.cos(phases) - odd * mx.sin(phases),
                    even * mx.sin(phases) + odd * mx.cos(phases),
                ],
                -1,
            )
            return mx.concatenate([r.reshape(*r.shape[:-2], -1), tail], -1)

        b, c = rotate(b, self.b_norm, self.b_bias), rotate(c, self.c_norm, self.c_bias)
        bx = x[..., None] * b[..., None, :]
        previous = mx.zeros_like(bx[:, :1]) if state is None else state[2][:, None]
        previous = mx.concatenate([previous, bx[:, :-1]], 1)
        # h_t = alpha*h_prev + alpha*dt*(1-gate)*Bx_prev + dt*gate*Bx_t.
        prefix, hidden = affine_scan(decay, dt * (gate * bx + (1 - gate) * decay * previous))
        if state is not None:
            hidden = hidden + prefix * state[1][:, None]
        y = (hidden * c[..., None, :]).sum(-1) + self.skip[None, None, :, None] * x
        y = self.output((y * nn.silu(z)).reshape(batch, length, -1))
        return y, (phases[:, -1], hidden[:, -1], bx[:, -1])


class WindowAttention(nn.Module):
    """Causal local attention with relative position bias and a bounded KV cache."""

    def __init__(self, width, heads=4, window=32):
        super().__init__()
        self.heads, self.window = heads, window
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.relative_bias = mx.zeros((heads, window))

    def __call__(self, x, state=None):
        batch, length, width = x.shape
        q, k, v = [
            a.reshape(batch, length, self.heads, -1).transpose(0, 2, 1, 3)
            for a in mx.split(self.qkv(x), 3, -1)
        ]
        past = 0
        if state is not None:
            past = state[0].shape[2]
            k, v = mx.concatenate([state[0], k], 2), mx.concatenate([state[1], v], 2)
        distance = mx.arange(past, past + length)[:, None] - mx.arange(k.shape[2])[None]
        valid = (distance >= 0) & (distance < self.window)
        bias = self.relative_bias[:, mx.clip(distance, 0, self.window - 1)]
        mask = mx.where(valid[None], bias, -1e9)[None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask=mask)
        y = self.output(y.transpose(0, 2, 1, 3).reshape(batch, length, width))
        keep = self.window - 1
        return y, (k[:, :, -keep:], v[:, :, -keep:])


class MemoryBlock(nn.Module):
    def __init__(self, width, kind, window):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.mixer = Mamba3SISO(width) if kind == "mamba3" else WindowAttention(width, 4, window)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )

    def __call__(self, x, state=None):
        y, state = self.mixer(self.norm(x), state)
        x = x + y
        return x + self.ffn(self.ffn_norm(x)), state
