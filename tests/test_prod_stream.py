from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from laya_vision_stitch.laya_p2p_stream import (
    MEMORY_TOKENS,
    NOMINAL_GAP_SECONDS,
    LayaP2PStream,
    full_cache_context,
)
from laya_vision_stitch.p2p_pretrained_policy import OpenP2PPolicy


def fixture(adapter=None, metadata=None):
    calls = {"goal": 0, "committed": [], "images": [], "dt": []}

    def context(prefix, actions=None, caches=None, position=0):
        if actions is not None:
            calls["committed"].append((position, actions.tolist()[0]))
        return mx.zeros((1, 1, 1024)), None

    def goal_features(*args):
        calls["goal"] += 1
        return mx.ones((1, 768))

    def prefix(image, goal, spatial=None):
        calls["images"].append(np.asarray(image))
        return mx.zeros((1, 4, 1024))

    policy = SimpleNamespace(
        vision=lambda pixels: (None, mx.zeros((1, 1024))),
        prefix=prefix,
        context=context,
        decode=lambda context, temperature: (
            mx.array([[11, 0, 0, 0, 0, 0, 11, 8]]),
            (mx.zeros((1, 20)),),
        ),
    )
    if adapter is not None:

        def time_gap_adapter(dt):
            calls["dt"].append(dt.tolist())
            return adapter(dt)

        policy.time_gap_adapter = time_gap_adapter
    runtime = SimpleNamespace(
        model=SimpleNamespace(policy=policy, bridge=lambda x: x, goal_features=goal_features),
        prepare_goal=lambda goal: (mx.ones((1, 1), mx.int32), mx.ones((1, 1), mx.bool_)),
        metadata=metadata or {},
    )
    row = {
        "goal": "Move forward",
        "frames": [{"image": Image.new("RGB", (1280, 720)), "age_seconds": 0}],
        "previous_actions": [],
    }
    return runtime, row, calls


APPLIED = [{"buttons": ["d"], "mouse_delta": [0, 0]}]


def test_goal_vector_is_encoded_once_per_goal_unless_disabled():
    runtime, row, calls = fixture()
    stream = LayaP2PStream(runtime)
    for i in range(3):
        stream.predict(row, session_id="s", timestamp_seconds=0.05 * i)
    assert calls["goal"] == 1
    stream.predict({**row, "goal": "Jump"}, session_id="s", timestamp_seconds=1)
    assert calls["goal"] == 2
    runtime, row, calls = fixture()
    stream = LayaP2PStream(runtime, cache_goal=False)
    for i in range(3):
        stream.predict(row, session_id="s", timestamp_seconds=0.05 * i)
    assert calls["goal"] == 3


def test_default_gap_bound_is_unchanged_and_larger_bound_keeps_memory():
    runtime, row, calls = fixture()
    stream = LayaP2PStream(runtime)
    stream.predict(row, session_id="s", timestamp_seconds=0)
    result = stream.predict(
        {**row, "previous_actions": APPLIED}, session_id="s", timestamp_seconds=0.5
    )
    assert result["state_reset"] and calls["committed"] == []
    runtime, row, calls = fixture()
    stream = LayaP2PStream(runtime, max_gap_seconds=1.0)
    stream.predict(row, session_id="s", timestamp_seconds=0)
    result = stream.predict(
        {**row, "previous_actions": APPLIED}, session_id="s", timestamp_seconds=0.5
    )
    assert not result["state_reset"] and calls["committed"] == [(0, [7, 0, 0, 0, 0, 0, 11, 8])]
    result = stream.predict(
        {**row, "previous_actions": APPLIED}, session_id="s", timestamp_seconds=2
    )
    assert result["state_reset"]
    assert "time_gap_seconds" not in result


@pytest.mark.parametrize("bound", [0, -1, 11, float("nan")])
def test_invalid_gap_bound_rejected(bound):
    runtime, _, _ = fixture()
    with pytest.raises(ValueError, match="gap"):
        LayaP2PStream(runtime, max_gap_seconds=bound)


def test_time_gap_adapter_receives_elapsed_time_and_adds_residual_before_prefix():
    runtime, row, calls = fixture(adapter=lambda dt: mx.broadcast_to(dt[:, None], (1, 1024)))
    stream = LayaP2PStream(runtime, max_gap_seconds=1.0)
    first = stream.predict(row, session_id="s", timestamp_seconds=10)
    second = stream.predict(
        {**row, "previous_actions": APPLIED}, session_id="s", timestamp_seconds=10.4
    )
    assert first["time_gap_seconds"] == NOMINAL_GAP_SECONDS
    assert second["time_gap_seconds"] == pytest.approx(0.4)
    assert np.allclose(calls["dt"], [[NOMINAL_GAP_SECONDS], [0.4]])
    # Residual is added to the image token that `policy.prefix` receives.
    assert np.allclose(calls["images"][0], NOMINAL_GAP_SECONDS)
    assert np.allclose(calls["images"][1], 0.4)
    # A reset (new goal) restarts at the nominal step.
    third = stream.predict(
        {**row, "goal": "Jump", "previous_actions": APPLIED},
        session_id="s",
        timestamp_seconds=10.45,
    )
    assert third["state_reset"] and third["time_gap_seconds"] == NOMINAL_GAP_SECONDS


def test_absent_adapter_is_a_no_op_but_declared_missing_adapter_is_rejected():
    runtime, row, calls = fixture()
    LayaP2PStream(runtime).predict(row, session_id="s", timestamp_seconds=0)
    assert np.all(calls["images"][0] == 0) and calls["dt"] == []
    runtime, _, _ = fixture(metadata={"time_gap_adapter": {"hidden": 256}})
    with pytest.raises(ValueError, match="not installed"):
        LayaP2PStream(runtime)


def test_adapter_shape_mismatch_rejected_and_memory_reset():
    runtime, row, _ = fixture(adapter=lambda dt: mx.zeros((1, 7)))
    stream = LayaP2PStream(runtime)
    with pytest.raises(ValueError, match="residual"):
        stream.predict(row, session_id="s", timestamp_seconds=0)
    assert stream.pending is None and stream.timestamp is None


def test_early_commit_matches_feedback_with_next_frame():
    runtime, row, calls = fixture()
    stream = LayaP2PStream(runtime)
    with pytest.raises(ValueError, match="awaiting"):
        stream.commit(APPLIED[0])
    stream.predict(row, session_id="s", timestamp_seconds=0)
    stream.commit(APPLIED[0])
    assert calls["committed"] == [(0, [7, 0, 0, 0, 0, 0, 11, 8])] and stream.position == 12
    with pytest.raises(ValueError, match="already committed"):
        stream.predict({**row, "previous_actions": APPLIED}, session_id="s", timestamp_seconds=0.05)
    result = stream.predict(row, session_id="s", timestamp_seconds=0.05)
    assert not result["state_reset"] and len(calls["committed"]) == 1
    # Over-long gaps still reset after an early commit.
    stream.commit(APPLIED[0])
    assert stream.predict(row, session_id="s", timestamp_seconds=1)["state_reset"]


def test_compile_falls_back_cleanly_for_non_mlx_policy():
    runtime, row, _ = fixture()
    stream = LayaP2PStream(runtime, compile=True)
    assert stream.compiled is None and stream.compile_status.startswith("fallback")
    assert stream.predict(row, session_id="s", timestamp_seconds=0)["buttons"] == ["w"]


@pytest.fixture(scope="module")
def policy():
    mx.random.seed(0)
    model = OpenP2PPolicy()
    model.freeze()
    mx.eval(model.parameters())
    return model


def test_full_cache_context_matches_released_context_at_any_position(policy):
    rng = np.random.default_rng(1)
    prefix = mx.array(rng.standard_normal((1, 4, 1024), dtype=np.float32))
    caches = [
        tuple(
            mx.array(rng.standard_normal((1, 16, MEMORY_TOKENS, 64), dtype=np.float32))
            for _ in range(2)
        )
        for _ in range(10)
    ]
    actions = mx.array([[11, 0, 0, 0, 1, 0, 3, 8]], mx.int32)
    for position in (MEMORY_TOKENS, MEMORY_TOKENS + 12 * 37):
        for acts in (None, actions):
            eager, eager_caches = policy.context(prefix, acts, caches, position)
            ours, our_caches = full_cache_context(policy, prefix, acts, caches, mx.array(position))
            assert mx.array_equal(eager, ours).item()
            assert all(mx.array_equal(a[0], b[0]).item() for a, b in zip(eager_caches, our_caches))
    with pytest.raises(ValueError, match="full"):
        full_cache_context(
            policy, prefix, None, [(k[:, :, 12:], v) for k, v in caches], mx.array(0)
        )


def test_compiled_stream_passes_parity_self_check(policy):
    runtime = SimpleNamespace(
        model=SimpleNamespace(
            policy=policy, bridge=lambda x: x, goal_features=lambda *a: mx.zeros((1, 768))
        ),
        prepare_goal=lambda goal: (mx.ones((1, 1), mx.int32), mx.ones((1, 1), mx.bool_)),
        metadata={},
    )
    stream = LayaP2PStream(runtime, compile=True, temperature=0.0)
    assert stream.compile_status == "compiled", stream.compile_status
    assert max(stream.compile_parity.values()) < 1e-4
