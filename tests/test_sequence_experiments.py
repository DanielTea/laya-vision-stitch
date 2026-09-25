import mlx.core as mx
import numpy as np

from laya_vision_stitch.d2e_sequences import assign
from laya_vision_stitch.flow_action_head import (
    DIM,
    ChunkHead,
    chunk_targets,
    realtime_sample,
    sample,
    tokens_to_vectors,
    vectors_to_actions,
)
from laya_vision_stitch.p2p_pretrained_policy import OpenP2PPolicy, Stack
from laya_vision_stitch.sequence_metrics import baselines, evaluate_actions, token_buttons
from laya_vision_stitch.sequence_policy import guided_decode, sequence_contexts
from laya_vision_stitch.slow_planner import SlowPlanner
from laya_vision_stitch.time_gap_adapter import TimeGapAdapter


def small_policy():
    mx.random.seed(3)
    policy = OpenP2PPolicy()
    policy.policy = Stack(2, 16)
    policy.decoder = Stack(1, 8, sinks=1)
    for name in ["no_text", "text_position", "image_position", "thinking", "action_start"]:
        setattr(policy, name, mx.random.normal((1, 1, 1024)) * 0.1)
    policy.action_position = mx.random.normal((1, 8, 1024)) * 0.1
    policy.decoder_position = mx.random.normal((9, 1024)) * 0.1
    return policy


def test_parallel_sequence_matches_streaming_context():
    policy = small_policy()
    images = mx.random.normal((1, 4, 1024))
    goals = mx.random.normal((1, 4, 768))
    tokens = mx.array([[[11, 0, 0, 0, 1, 0, 12, 8], [6, 0, 0, 0, 0, 0, 11, 8]] * 2])
    parallel = sequence_contexts(policy, images, goals, tokens)[0]
    caches, stream = None, []
    for t in range(4):
        prefix = policy.prefix(images[0, t : t + 1], goals[0, t : t + 1])
        context, _ = policy.context(prefix, caches=caches, position=12 * t)
        stream.append(context[:, 0])
        _, caches = policy.context(prefix, tokens[0, t : t + 1], caches, 12 * t)
    np.testing.assert_allclose(
        np.asarray(parallel), np.asarray(mx.concatenate(stream)), atol=2e-4, rtol=1e-4
    )


def test_guided_decode_without_guidance_matches_released_decoder():
    policy = small_policy()
    context = mx.random.normal((3, 1024))
    ours, _ = guided_decode(policy, context)
    released, _ = policy.decode(context[:, None])
    assert np.array_equal(np.asarray(ours), np.asarray(released))
    same, _ = guided_decode(policy, context, context, scale=5.0)
    assert np.array_equal(np.asarray(same), np.asarray(released))


def test_onset_metric_gives_repeat_previous_no_credit():
    rows = [{"game": "g", "sequence": "s", "step": k} for k in range(4)]
    truth = [frozenset(), frozenset({"w"}), frozenset({"w"}), frozenset()]
    base = baselines(rows, truth, np.zeros((4, 2)))
    assert base["repeat_previous"]["g"]["onset_f1"] == 0
    early = [frozenset({"w"}), frozenset({"w"}), frozenset({"w"}), frozenset()]
    assert evaluate_actions(rows, early, truth)["g"]["onset_recall"] == 1
    late = [frozenset(), frozenset(), frozenset(), frozenset({"w"})]
    report = evaluate_actions(rows, late, truth, tolerance=1)
    assert report["g"]["onset_recall"] == 0 and report["g"]["onset_precision"] == 0


def test_action_vectors_round_trip_and_chunks_stay_in_sequence():
    tokens = np.array([[11, 20, 0, 0, 1, 0, 12, 8], [0, 0, 0, 0, 0, 0, 22, 0]])
    buttons, mouse = vectors_to_actions(tokens_to_vectors(tokens))
    assert buttons == token_buttons(tokens)
    assert mouse.tolist() == [[1, 0], [501, -151]]
    vectors = tokens_to_vectors(np.zeros((5, 8), int))
    targets, mask = chunk_targets(vectors, np.array([0, 0, 0, 1, 1]), np.array([0, 1, 2, 0, 1]), 3)
    assert mask.tolist() == [[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0]]
    assert targets.shape == (5, 3, DIM)


def test_flow_head_shapes_and_realtime_prefix_is_pulled_toward_committed():
    mx.random.seed(0)
    head = ChunkHead(horizon=4, width=32, depth=1, heads=2)
    context = mx.random.normal((2, 1024))
    assert sample(head, context, steps=3, key=mx.random.key(1)).shape == (2, 4, DIM)
    committed = mx.ones((2, 4, DIM))
    free = sample(head, context, steps=4, key=mx.random.key(2))
    guided = realtime_sample(head, context, committed, delay=2, steps=4, key=mx.random.key(2))
    free_error = float(mx.abs(free[:, :2] - committed[:, :2]).mean())
    guided_error = float(mx.abs(guided[:, :2] - committed[:, :2]).mean())
    assert guided_error < free_error


def test_time_gap_adapter_is_zero_at_nominal_interval_and_planner_starts_at_zero():
    adapter = TimeGapAdapter()
    adapter.output.weight = mx.random.normal(adapter.output.weight.shape)
    assert float(mx.abs(adapter(mx.array([0.05, 0.05]))).max()) < 1e-5
    assert float(mx.abs(adapter(mx.array([0.3]))).max()) > 0
    language, image = SlowPlanner(feature_width=16, width=8)(
        mx.random.normal((2, 5, 16)), mx.random.normal((2, 768))
    )
    assert float(mx.abs(language).max()) == 0 and float(mx.abs(image).max()) == 0


def test_d2e_sessions_split_chronologically_per_game():
    grouped = {("G", f"recording_2025090{i}"): [] for i in range(1, 5)}
    grouped[("H", "recording_1")] = []
    splits = assign(grouped, "chronological")
    assert [splits[("G", f"recording_2025090{i}")] for i in range(1, 5)] == [
        "train",
        "train",
        "validation",
        "test",
    ]
    assert splits[("H", "recording_1")] == "train"
    assert set(assign(grouped, "fresh_games").values()) == {"fresh_games"}


def test_control_report_scores_scroll_onsets_and_drags():
    from laya_vision_stitch.sequence_metrics import control_report

    rows = [{"game": "g", "sequence": "s", "step": k} for k in range(6)]
    truth = [
        frozenset(),
        frozenset(),
        frozenset({"scroll_up"}),
        frozenset(),
        frozenset({"mouse_right"}),
        frozenset({"mouse_right"}),
    ]
    motion = [[0, 0]] * 4 + [[10, 0], [8, 2]]
    same = control_report(rows, truth, truth, motion, motion)
    assert same["scroll_up"]["onset_f1"] == 1.0 and same["mouse_right"]["onset_f1"] == 1.0
    assert same["drag"]["f1"] == 1.0 and abs(same["drag"]["cosine"] - 1) < 1e-9
    idle = control_report(rows, [frozenset()] * 6, truth, [[0, 0]] * 6, motion)
    assert (
        idle["scroll_up"]["onset_f1"] == 0.0
        and idle["drag"]["f1"] == 0.0
        and idle["drag"]["cosine"] is None
    )
    reversed_drag = control_report(rows, truth, truth, [[0, 0]] * 4 + [[-10, 0], [-8, -2]], motion)
    assert reversed_drag["drag"]["cosine"] < -0.9
