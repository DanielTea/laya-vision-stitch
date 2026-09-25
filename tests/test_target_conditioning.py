import mlx.core as mx
import numpy as np

from laya_vision_stitch.sequence_policy import sequence_contexts
from laya_vision_stitch.target_conditioning import (
    TargetEncoder,
    hindsight_targets,
    mouse_look_correlation,
    toward_target,
)
from tests.test_sequence_experiments import small_policy


def test_untrained_or_missing_target_leaves_policy_unchanged():
    policy = small_policy()
    images, goals = mx.random.normal((1, 3, 1024)), mx.random.normal((1, 3, 768))
    tokens = mx.array([[[11, 0, 0, 0, 1, 0, 12, 8]] * 3])
    base = sequence_contexts(policy, images, goals, tokens)
    policy.target_encoder = TargetEncoder()
    target = mx.array([[[0.2, 0.7], [np.nan, np.nan], [0.9, 0.1]]])
    same = sequence_contexts(policy, images, goals, tokens, target=target)
    np.testing.assert_allclose(np.asarray(base), np.asarray(same), atol=1e-6)
    policy.target_encoder.out.weight = mx.random.normal(policy.target_encoder.out.weight.shape)
    moved = np.asarray(sequence_contexts(policy, images, goals, tokens, target=target))
    # Frame 1 has no target but sees frame 0's target through causal attention.
    assert not np.allclose(moved[0, 0], np.asarray(base)[0, 0], atol=1e-4)
    first = sequence_contexts(
        policy, images[:, :1], goals[:, :1], tokens[:, :1], target=mx.array([[[np.nan, np.nan]]])
    )
    np.testing.assert_allclose(np.asarray(first)[0, 0], np.asarray(base)[0, 0], atol=1e-5)


def test_streaming_prefix_matches_sequence_prefix_with_target():
    policy = small_policy()
    policy.target_encoder = TargetEncoder()
    policy.target_encoder.out.weight = (
        mx.random.normal(policy.target_encoder.out.weight.shape) * 0.1
    )
    images, goals = mx.random.normal((1, 2, 1024)), mx.random.normal((1, 2, 768))
    target = mx.array([[[0.3, 0.4], [0.6, 0.2]]])
    tokens = mx.array([[[11, 0, 0, 0, 1, 0, 12, 8]] * 2])
    parallel = sequence_contexts(policy, images, goals, tokens, target=target)[0]
    caches, stream = None, []
    for t in range(2):
        prefix = policy.prefix(
            images[0, t : t + 1], goals[0, t : t + 1], None, target[0, t : t + 1]
        )
        context, _ = policy.context(prefix, caches=caches, position=12 * t)
        stream.append(context[:, 0])
        _, caches = policy.context(prefix, tokens[0, t : t + 1], caches, 12 * t)
    np.testing.assert_allclose(
        np.asarray(parallel), np.asarray(mx.concatenate(stream)), atol=2e-4, rtol=1e-4
    )


def test_hindsight_targets_prefer_next_press_then_recent_one():
    presses = [(1.0, 0.2, 0.3, "left"), (5.0, 0.8, 0.9, "right")]
    xy, dt, button = hindsight_targets([0.5, 1.5, 2.5, 3.5, 4.0], presses, ahead=2.0, behind=1.0)
    assert np.allclose(xy[0], [0.2, 0.3]) and np.isclose(dt[0], 0.5) and button[0] == 1
    assert np.allclose(xy[1], [0.2, 0.3]) and np.isclose(dt[1], -0.5)  # just pressed
    assert np.isnan(xy[2]).all() and button[2] == 0  # nothing within the horizons
    assert np.allclose(xy[3], [0.8, 0.9]) and button[3] == 2
    assert np.allclose(xy[4], [0.8, 0.9])


def test_mouse_look_correlation_detects_camera_driven_by_mouse():
    rng = np.random.default_rng(0)
    motion = rng.random(200)
    images = np.cumsum(rng.normal(size=(200, 8)) * 0.01, 0)
    look = images.copy()
    for i in range(1, 200):
        look[i] = look[i - 1] + motion[i - 1] * np.ones(8)
    sequence, steps = np.zeros(200, int), np.arange(200)
    assert mouse_look_correlation(look, motion, sequence, steps) > 0.9
    assert abs(mouse_look_correlation(images, motion, sequence, steps)) < 0.3


def test_toward_target_scores_direction():
    targets = np.array([[0.9, 0.5], [0.5, 0.1], [0.5, 0.5]])
    good = toward_target(
        [{"d"}, {"w"}, {"a"}], np.array([[5, 0], [0, -5], [0, 0]]), targets, np.full((3, 2), 0.5)
    )
    bad = toward_target(
        [{"a"}, {"s"}, {"a"}], np.array([[-5, 0], [0, 5], [0, 0]]), targets, np.full((3, 2), 0.5)
    )
    assert good["move_cos"] == 1.0 and bad["move_cos"] == -1.0 and good["move_frames"] == 2
    assert good["mouse_cos"] == 1.0 and bad["mouse_cos"] == -1.0


def stream_fixture(tokens=(11, 0, 0, 0, 0, 0, 11, 8), pointer=(0.5, 0.5)):
    from types import SimpleNamespace

    from PIL import Image

    from laya_vision_stitch.laya_p2p_stream import LayaP2PStream

    seen = []

    def prefix(image, goal, spatial=None, target=None):
        seen.append(None if target is None else np.asarray(target).tolist())
        return mx.zeros((1, 4, 1024))

    policy = SimpleNamespace(
        vision=lambda pixels: (mx.zeros((1, 6, 6, 112)), mx.zeros((1, 1024))),
        prefix=prefix,
        context=lambda prefix, actions=None, caches=None, position=0: (
            mx.zeros((1, 1, 1024)),
            None,
        ),
        decode=lambda context, temperature: (mx.array([list(tokens)]), (mx.zeros((1, 20)),)),
        target_encoder=object(),
        pointer_head=SimpleNamespace(predict=lambda spatial, context: mx.array([list(pointer)])),
    )
    runtime = SimpleNamespace(
        model=SimpleNamespace(
            policy=policy, bridge=lambda x: x, goal_features=lambda *a: mx.ones((1, 768))
        ),
        prepare_goal=lambda goal: (mx.ones((1, 1), mx.int32), mx.ones((1, 1), mx.bool_)),
        metadata={},
    )
    row = {
        "goal": "Go",
        "frames": [{"image": Image.new("RGB", (64, 36)), "age_seconds": 0}],
        "previous_actions": [],
    }
    return LayaP2PStream(runtime), row, seen


def test_stream_feeds_tracked_target_or_none_to_the_policy():
    import time
    from types import SimpleNamespace

    stream, row, seen = stream_fixture()
    out = stream.predict(row, session_id="s", timestamp_seconds=0)
    assert np.isnan(seen[-1]).all() and out["target_input"] is None
    stream.tracker, stream.target_time = SimpleNamespace(xy=[0.7, 0.3]), time.perf_counter()
    out = stream.predict(
        {**row, "previous_actions": [{"buttons": ["w"], "mouse_delta": [0, 0]}]},
        session_id="s",
        timestamp_seconds=0.05,
    )
    assert np.allclose(seen[-1], [[0.7, 0.3]]) and np.allclose(out["target_input"], [0.7, 0.3])


def test_pointer_head_click_on_the_avatar_moves_off_it_when_avoiding():
    from types import SimpleNamespace

    press = (11, 0, 0, 0, 1, 0, 11, 8)
    stream, row, _ = stream_fixture(tokens=press, pointer=(0.52, 0.49))
    out = stream.predict(row, session_id="s", timestamp_seconds=0)
    assert out["pointer_source"] == "pointer_head" and np.allclose(out["pointer_xy"], [0.52, 0.49])
    # Planner mode (on by default when a planner is given): the press moves 0.15 out.
    stream, row, _ = stream_fixture(tokens=press, pointer=(0.52, 0.49))
    stream.planner = SimpleNamespace(poll=lambda: None, submit=lambda image, goal: None)
    stream.avoid_avatar = True
    out = stream.predict(row, session_id="s", timestamp_seconds=0)
    assert out["pointer_source"] == "avatar_zone_shifted"
    assert np.allclose(out["pointer_xy"], [0.5 + 0.15 * 2 / 5**0.5, 0.5 - 0.15 / 5**0.5])
    # Without a planner (hold mode), explicitly enabled; a dead-center click goes straight up.
    stream, row, _ = stream_fixture(tokens=press, pointer=(0.5, 0.5))
    stream.avoid_avatar = True
    out = stream.predict(row, session_id="s", timestamp_seconds=0)
    assert np.allclose(out["pointer_xy"], [0.5, 0.35])
    # Clicks away from the avatar are untouched.
    stream, row, _ = stream_fixture(tokens=press, pointer=(0.8, 0.3))
    stream.avoid_avatar = True
    assert np.allclose(
        stream.predict(row, session_id="s", timestamp_seconds=0)["pointer_xy"], [0.8, 0.3]
    )


def test_compiled_stream_with_target_encoder_passes_parity():
    from types import SimpleNamespace

    from laya_vision_stitch.laya_p2p_stream import LayaP2PStream
    from laya_vision_stitch.p2p_pretrained_policy import OpenP2PPolicy

    mx.random.seed(0)
    policy = OpenP2PPolicy()
    policy.target_encoder = TargetEncoder()
    policy.target_encoder.out.weight = (
        mx.random.normal(policy.target_encoder.out.weight.shape) * 0.01
    )
    policy.freeze()
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
