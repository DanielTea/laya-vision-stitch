import math
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from laya_vision_stitch.temporal_adapter import TemporalActionAdapter
from laya_vision_stitch.temporal_memory import Mamba3SISO, MemoryBlock
from laya_vision_stitch.trainable_model import PolicyConfig


def numpy_mamba(model, values):
    """Independent serial reference for the official SISO recurrence."""

    def norm(x, module):
        return (
            x / np.sqrt(np.mean(x * x, -1, keepdims=True) + module.eps) * np.asarray(module.weight)
        )

    def softplus(x):
        return np.logaddexp(0, x)

    def rotate(v, angle):
        r = v[..., : 2 * model.angles].reshape(*v.shape[:-1], -1, 2)
        a, b = r[..., 0], r[..., 1]
        r = np.stack(
            [a * np.cos(angle) - b * np.sin(angle), a * np.sin(angle) + b * np.cos(angle)], -1
        )
        return np.concatenate([r.reshape(*v.shape[:-1], -1), v[..., 2 * model.angles :]], -1)

    projection = values @ np.asarray(model.projection.weight).T
    phase = np.zeros((values.shape[0], model.heads, model.angles))
    hidden = np.zeros((values.shape[0], model.heads, model.head_width, model.state_width))
    previous = np.zeros_like(hidden)
    result = []
    for t in range(values.shape[1]):
        z, x, b, c, dt, a, trap, angle = np.split(projection[:, t], np.cumsum(model.sizes)[:-1], -1)
        z, x = [v.reshape(-1, model.heads, model.head_width) for v in (z, x)]
        dt = softplus(dt + np.asarray(model.dt_bias))
        decay = np.exp(-np.maximum(np.maximum(a, 0) + 1 / (1 - np.minimum(a, 0)), 1e-4) * dt)
        phase = (phase + np.tanh(angle)[:, None] * math.pi * dt[..., None]) % (2 * math.pi)
        b = rotate(norm(b, model.b_norm)[:, None] + np.asarray(model.b_bias), phase)
        c = rotate(norm(c, model.c_norm)[:, None] + np.asarray(model.c_bias), phase)
        bx = x[..., None] * b[..., None, :]
        gate = 1 / (1 + np.exp(-trap))
        alpha, beta, gamma = [
            v[..., None, None] for v in (decay, decay * dt * (1 - gate), dt * gate)
        ]
        hidden = alpha * hidden + beta * previous + gamma * bx
        y = (
            ((hidden * c[..., None, :]).sum(-1) + np.asarray(model.skip)[None, :, None] * x)
            * z
            / (1 + np.exp(-z))
        )
        result.append(y.reshape(values.shape[0], -1) @ np.asarray(model.output.weight).T)
        previous = bx
    return np.stack(result, 1)


def test_mamba_matches_independent_serial_equations():
    mx.random.seed(83)
    model = Mamba3SISO(32, state_width=16, head_width=16)
    x = mx.random.normal((2, 17, 32))
    y, _ = model(x)
    np.testing.assert_allclose(
        np.asarray(y), numpy_mamba(model, np.asarray(x)), rtol=2e-4, atol=2e-5
    )


@pytest.mark.parametrize("kind", ["mamba3", "attention"])
def test_streaming_chunking_causality_and_reset(kind):
    mx.random.seed(85)
    model = MemoryBlock(32, kind, 8)
    x = mx.random.normal((2, 23, 32))
    y, final = model(x)
    state, parts = None, []
    for i in range(x.shape[1]):
        out, state = model(x[:, i : i + 1], state)
        parts.append(out)
    np.testing.assert_allclose(
        np.asarray(y), np.asarray(mx.concatenate(parts, 1)), rtol=2e-4, atol=2e-5
    )
    for (_, a), (_, b) in zip(tree_flatten(final), tree_flatten(state), strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=3e-4, atol=3e-5)
    a, state = model(x[:, :7])
    b, _ = model(x[:, 7:], state)
    np.testing.assert_allclose(
        np.asarray(y), np.asarray(mx.concatenate([a, b], 1)), rtol=2e-4, atol=2e-5
    )
    changed, _ = model(mx.concatenate([x[:, :10], -10 * x[:, 10:]], 1))
    np.testing.assert_allclose(
        np.asarray(y[:, :10]), np.asarray(changed[:, :10]), rtol=1e-5, atol=1e-5
    )
    fresh, _ = model(x)
    np.testing.assert_array_equal(np.asarray(y), np.asarray(fresh))


@pytest.mark.parametrize("kind", ["mamba3", "attention"])
def test_gradient_reaches_earlier_frames(kind):
    model = MemoryBlock(32, kind, 8)
    x = mx.random.normal((1, 6, 32))
    grad = mx.grad(lambda v: model(v)[0][:, -1].square().mean())(x)
    assert float(mx.abs(grad[:, 0]).sum()) > 1e-8
    assert np.isfinite(np.asarray(grad)).all()


def test_temporal_adapter_has_no_future_or_action_target_input():
    config = replace(PolicyConfig(buttons=("w", "a")), temporal_adapter="mamba3", temporal_width=32)
    model = TemporalActionAdapter(12, 16, config)
    visual, language, history = (
        mx.random.normal((1, 5, 16, 12)),
        mx.random.normal((1, 5, 2, 16)),
        mx.zeros((1, 5, 9)),
    )
    output, _ = model(visual, language, history)
    # Dynamics receives labels through a separate training-only call, not action prediction.
    model.future_delta(output["hidden"], mx.ones((1, 5, 4)))
    after, _ = model(visual, language, history)
    np.testing.assert_array_equal(np.asarray(output["buttons"]), np.asarray(after["buttons"]))
    _, gradients = nn.value_and_grad(
        model,
        lambda m: (
            m.future_delta(m(visual, language, history)[0]["hidden"], mx.ones((1, 5, 4)))
            .square()
            .mean()
        ),
    )(model)
    assert any(float(mx.abs(g).sum()) > 0 for _, g in tree_flatten(gradients["blocks"]))


def test_sequence_selection_excludes_gaps_goal_changes_and_overlap():
    from laya_vision_stitch.sequence_data import choose_starts, valid_starts

    rows = [{"frame_index": i, "goal": "a" if i < 5 else "b"} for i in range(12) if i != 9]
    assert valid_starts(rows, 3) == [0, 1, 2, 5, 6]
    starts = choose_starts(list(range(100)), 8, 5, np.random.default_rng(7))
    assert all(b - a > 5 for a, b in zip(starts, starts[1:]))
