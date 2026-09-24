import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from PIL import Image

from laya_vision_stitch.flow_action_head import DIM
from laya_vision_stitch.latent_actions import (
    ActionHead,
    LatentActionModel,
    LogisticProbe,
    Standardizer,
    VectorQuantizer,
    action_facts,
    action_loss,
    binary_scores,
    chance_nmi,
    code_loss,
    combine_codes,
    lagged_facts,
    lam_codes,
    lam_errors,
    lam_loss,
    mouse_direction,
    normalized_mutual_information,
    offset_index,
    read_trial,
    replace_dead_codes,
    successors,
    train_lam,
)
from laya_vision_stitch.latent_world_model import (
    LatentWorldModel,
    action_index,
    controllability,
    fit_ridge_world_model,
    future_index,
    shuffled_starts,
    train_world_model,
)


def test_pairs_never_cross_sequences():
    seq = np.array([0, 0, 0, 1, 1, 2])
    steps = np.array([0, 1, 2, 0, 1, 0])
    assert successors(seq, steps).tolist() == [0, 1, 3]
    assert successors(seq, steps, gap=2).tolist() == [0]
    target, valid = offset_index(seq, steps, np.array([0, 3, 4]), -1)
    assert valid.tolist() == [False, False, True] and target[2] == 3
    with pytest.raises(ValueError, match="Gap"):
        successors(seq, steps, gap=0)


def test_standardizer_uses_given_statistics(tmp_path):
    x = np.random.default_rng(0).normal(3, 2, (200, 5))
    std = Standardizer.fit(x)
    z = std(x)
    assert np.allclose(z.mean(0), 0, atol=1e-5) and np.allclose(z.std(0), 1, atol=1e-3)
    std.save(tmp_path / "s.npz")
    assert np.allclose(Standardizer.load(tmp_path / "s.npz")(x + 1), std(x + 1))
    with pytest.raises(ValueError):
        Standardizer.fit(x[:1])


def test_quantizer_straight_through_and_nearest_code():
    vq = VectorQuantizer(tokens=2, codes=3, dim=4)
    vq.codebook = mx.array(np.stack([np.eye(3, 4) * 5, -np.eye(3, 4) * 5]).astype(np.float32))
    z = mx.array(np.array([[[5, 0.1, 0, 0], [0, -5, 0.2, 0]]], np.float32))
    q, index, codebook_loss, commitment = vq(z)
    assert index.tolist() == [[0, 1]]
    assert np.allclose(np.array(q), np.array(vq.lookup(index)), atol=1e-6)
    grad = mx.grad(lambda x: vq(x)[0].sum())(z)
    assert np.allclose(np.array(grad), 1.0)
    with pytest.raises(ValueError, match="tokens"):
        vq(mx.zeros((1, 3, 4)))


def test_dead_codes_move_onto_encodings():
    vq = VectorQuantizer(tokens=1, codes=4, dim=16)
    usage = np.array([[0.97, 0.01, 0.01, 0.01]])
    encoded = np.full((8, 1, 16), 7.0, np.float32)
    replaced = replace_dead_codes(vq, usage, encoded, np.random.default_rng(0), noise=0.0)
    book = np.array(vq.codebook)
    assert replaced == 3 and np.allclose(book[0, 1:], 7.0)
    assert np.allclose(usage[0, 1:], 0.25)


def test_lam_starts_at_copy_last_and_zeroed_latents_differ_after_training():
    rng = np.random.default_rng(0)
    mx.random.seed(0)
    n, d = 1280, 32
    z0 = rng.normal(size=(n, d)).astype(np.float32)
    # Next token depends on a hidden 2-bit "action" that only the pair reveals.
    action = rng.integers(0, 4, n)
    z1 = z0 + np.eye(4, d)[action].astype(np.float32) * 3
    train, held = slice(0, 1024), slice(1024, n)
    model = LatentActionModel(features=d, tokens=2, codes=4, dim=16, width=64)
    before = lam_errors(model, z0[held], z1[held])
    assert np.allclose(before["full"], before["copy_last"])
    report = train_lam(
        model,
        (z0[train], z1[train]),
        (z0[held], z1[held]),
        steps=400,
        batch=64,
        learning_rate=3e-3,
        every=100,
    )
    errors = lam_errors(model, z0[held], z1[held])
    assert report["validation"] < 0.1 * float(np.mean(before["full"]))
    # Without latents the held-out action is unknowable; zeroed latents stay near copy-last.
    assert errors["full"].mean() < 0.2 * errors["zeroed_latents"].mean()
    codes = lam_codes(model, z0[held], z1[held])
    assert codes.shape == (256, 2) and codes.max() < 4
    combo = combine_codes(codes, 4)
    purity = sum(np.bincount(action[held][combo == k]).max() for k in np.unique(combo)) / 256
    assert purity > 0.95
    loss, (mse, index, encoded) = lam_loss(model, mx.array(z0[:4]), mx.array(z1[:4]), mx.ones(4))
    assert index.shape == (4, 2) and encoded.shape == (4, 2, 16) and float(loss) >= float(mse)
    with pytest.raises(ValueError, match="16-32D"):
        LatentActionModel(dim=8)


def test_decoder_context_limits_view_of_current_token():
    rng = np.random.default_rng(1)
    latents = mx.array(rng.normal(size=(3, 2, 16)).astype(np.float32))
    a, b = (mx.array(rng.normal(size=(3, 8)).astype(np.float32)) for _ in range(2))
    for context in (0, 4, -1):
        model = LatentActionModel(features=8, tokens=2, codes=4, dim=16, width=16, context=context)
        last = model.decoder.layers[-1]
        last.weight = mx.array(rng.normal(size=last.weight.shape).astype(np.float32))
        change_a = np.array(model.decode(a, latents) - a)
        change_b = np.array(model.decode(b, latents) - b)
        # With no decoder context the predicted change cannot depend on z_t.
        assert np.allclose(change_a, change_b, atol=1e-5) == (context == 0)
    with pytest.raises(ValueError, match="context"):
        LatentActionModel(features=8, context=9)


def test_nmi_properties():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 5, 2000)
    assert normalized_mutual_information(a, a) == pytest.approx(1.0)
    assert normalized_mutual_information(a, (a + 2) % 5) == pytest.approx(1.0)
    assert normalized_mutual_information(a, rng.integers(0, 5, 2000)) < 0.01
    assert normalized_mutual_information(np.zeros(5), np.zeros(5)) == 1.0
    many = rng.integers(0, 500, 300)
    # Many-valued codes are biased upward on few samples; the permutation reference shows it.
    assert chance_nmi(many, rng.integers(0, 3, 300), rng, repeats=5) > 0.1
    assert combine_codes(np.array([[1, 2], [0, 0]]), 3).tolist() == [7, 0]


def test_mouse_direction_sectors():
    mouse = np.array([[0, 0], [5, 0], [5, 5], [0, 5], [-5, 0], [0, -5], [5, -0.1]])
    assert mouse_direction(mouse).tolist() == [0, 1, 2, 3, 5, 7, 1]


def test_facts_and_lags_respect_history():
    buttons = [frozenset(), frozenset({"w"}), frozenset({"w", "mouse_left"}), frozenset({"a"})]
    mouse = np.array([[0, 0], [3, 0], [-2, 1], [0, 0]])
    facts = action_facts(buttons, mouse, [None, *buttons[:-1]], groups={"move": ("a", "d")})
    assert facts["w_held"][0].tolist() == [False, True, True, False]
    assert facts["move"][0].tolist() == [False, False, False, True]
    assert facts["mouse_x_positive"][1].tolist() == [False, True, True, False]
    # mouse_left is not a keyboard key; w at step 1 and a at step 3 are key onsets.
    assert facts["key_onset"][0].tolist() == [False, True, False, True]
    assert facts["key_onset"][1].tolist() == [False, True, True, True]
    seq, steps = np.zeros(4, int), np.arange(4)
    lagged, target, valid = lagged_facts(buttons, mouse, seq, steps, np.array([0, 1, 2]), 1)
    assert valid.tolist() == [False, True, True] and target[1:].tolist() == [0, 1]
    assert lagged["key_onset"][1].tolist() == [False, False, True]


def test_probe_learns_separable_fact_and_reports_chance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 6))
    y = x[:, 0] + 0.1 * rng.normal(size=400) > 0.5
    probe = LogisticProbe(1e-3).fit(x[:300], y[:300])
    scores = binary_scores(y[300:], probe.probability(x[300:]))
    assert scores["accuracy"] > 0.9 and scores["balanced_accuracy"] > 0.85
    assert scores["majority_accuracy"] == max(y[300:].mean(), 1 - y[300:].mean())
    constant = LogisticProbe().fit(x[:10], np.ones(10))
    assert np.all(constant.probability(x[:3]) == 1.0)
    with pytest.raises(ValueError):
        LogisticProbe(0)


def test_action_head_losses_and_shapes():
    head = ActionHead(inputs=8, width=16, tokens=2, codes=3)
    x = mx.zeros((5, 8))
    assert head(x).shape == (5, DIM) and head.code_logits(x).shape == (5, 2, 3)
    target = mx.array(np.tile(np.r_[-np.ones(DIM - 2), 0, 0], (5, 1)).astype(np.float32))
    assert float(action_loss(head, x, target, mx.zeros(5))) == 0.0
    assert np.isfinite(float(code_loss(head, x, mx.zeros((5, 2), mx.int32))))


def test_world_model_chunks_and_shuffles():
    seq = np.array([0, 0, 0, 0, 1, 1])
    steps = np.array([0, 1, 2, 3, 0, 1])
    index, mask = future_index(seq, steps, 3)
    assert mask.tolist()[0] == [True, True, True] and mask.tolist()[2] == [True, False, False]
    assert not mask[3].any() and mask[4].tolist() == [True, False, False]
    assert action_index(index)[0].tolist() == [0, 1, 2]
    groups = np.array(["a"] * 4 + ["b"] * 4)
    valid = np.array([True] * 8)
    rng = np.random.default_rng(0)
    for _ in range(20):
        pick = shuffled_starts(groups, valid, rng)
        assert np.all(pick != np.arange(8)) and np.all(groups[pick] == groups)
    with pytest.raises(ValueError, match="two complete"):
        shuffled_starts(groups, np.array([True] + [False] * 7), rng)


def test_world_model_learns_action_effect_and_detects_controllability():
    rng = np.random.default_rng(0)
    mx.random.seed(0)
    n, d, h = 600, 16, 2
    z = rng.normal(size=(n, d)).astype(np.float32)
    actions = np.where(rng.random((n, h, DIM)) < 0.5, -1.0, 1.0).astype(np.float32)
    effect = np.cumsum(actions[:, :, :d] * 0.5, 1)
    data = {
        "context": rng.normal(size=(n, 8)).astype(np.float32),
        "z": z,
        "actions": actions,
        "targets": (z[:, None] + effect).astype(np.float32),
        "mask": np.ones((n, h), bool),
    }
    model = LatentWorldModel(features=d, context=8, width=64)
    split = {k: v[:500] for k, v in data.items()}, {k: v[500:] for k, v in data.items()}
    report = train_world_model(model, *split, steps=300, batch=64, learning_rate=3e-3, every=100)
    assert report["selected_step"] > 0
    groups, sequences = np.zeros(100), np.arange(100) // 10
    result = controllability(model, split[1], groups, sequences, rng, draws=2)
    assert result["h1"]["true_actions"] < 0.5 * result["h1"]["shuffled_actions"]
    assert result["h2"]["relative_gain_vs_shuffled"] > 0.3
    assert result["h1"]["shuffled_minus_true_ci95"][0] > 0
    with pytest.raises(ValueError, match="actions"):
        model(mx.zeros((2, 8)), mx.zeros((2, d)), mx.zeros((2, DIM)))


def test_ridge_world_model_selects_penalty_and_blind_twin_ignores_actions():
    rng = np.random.default_rng(2)
    n, d, h = 400, 6, 2
    z = rng.normal(size=(n, d)).astype(np.float32)
    actions = rng.normal(size=(n, h, DIM)).astype(np.float32)
    targets = z[:, None] + np.cumsum(actions[:, :, :d], 1)
    data = {
        "context": rng.normal(5, 3, (n, 4)).astype(np.float32),
        "z": z,
        "actions": actions,
        "targets": targets.astype(np.float32),
        "mask": np.ones((n, h), bool),
    }
    train, held = ({k: v[s] for k, v in data.items()} for s in (slice(0, 300), slice(300, n)))
    model, info = fit_ridge_world_model(train, held, penalties=(1e-3, 1e4))
    assert info["penalty"] == 1e-3 and info["validation"] < 1e-3
    blind, _ = fit_ridge_world_model(train, held, penalties=(1e-3,), blind=True)
    shuffled = held["actions"][::-1].copy()
    same = [
        np.array(blind(mx.array(held["context"]), mx.array(held["z"]), mx.array(a)))
        for a in (held["actions"], shuffled)
    ]
    assert np.allclose(*same)
    result = controllability(model, held, np.zeros(100), np.arange(100) // 10, rng, draws=1)
    assert result["h2"]["relative_gain_vs_shuffled"] > 0.9
    with pytest.raises(ValueError, match="one action"):
        model(mx.array(held["context"]), mx.array(held["z"]), mx.array(held["actions"][:, :1]))


def test_action_only_world_model_ignores_content():
    model = LatentWorldModel(features=4, context=3, width=8, content=0)
    model.output.layers[-1].weight = mx.ones_like(model.output.layers[-1].weight)
    actions = mx.ones((2, 2, DIM))
    a = model(mx.zeros((2, 3)), mx.zeros((2, 4)), actions)
    b = model(mx.ones((2, 3)) * 9, mx.zeros((2, 4)), actions)
    assert np.allclose(np.array(a), np.array(b))
    with pytest.raises(ValueError, match="Content"):
        LatentWorldModel(content=-2)


def trial(tmp_path, events):
    (tmp_path / "frames").mkdir()
    for e in events:
        Image.new("RGB", (4, 4)).save(tmp_path / e["image"])
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return tmp_path


def event(step, applied=True, buttons=("w",), delta=(0.01, 0.0), elapsed=None):
    return {
        "step": step,
        "elapsed_s": 0.07 * step if elapsed is None else elapsed,
        "image": f"frames/{step:05d}.jpg",
        "proposal": {"buttons": ["w", "f"], "mouse_delta": [0.5, 0.0]},
        "bounded_action": {"buttons": list(buttons), "mouse_delta": list(delta)},
        "applied": applied,
    }


def test_trial_reader_uses_dispatched_actions(tmp_path):
    rows = read_trial(trial(tmp_path, [event(0), event(1, applied=False)]))
    assert rows[0]["action"] == {"buttons": ["w"], "mouse_delta": [0.01, 0.0]}
    assert rows[1]["action"] == {"buttons": [], "mouse_delta": [0.0, 0.0]}


@pytest.mark.parametrize(
    ("events", "message"),
    [
        ([event(0), event(2)], "consecutive"),
        ([event(0), event(1, elapsed=0.0)], "increase"),
        ([event(0), event(1, delta=(float("nan"), 0.0))], "mouse delta"),
    ],
)
def test_trial_reader_rejects_malformed_logs(tmp_path, events, message):
    with pytest.raises(ValueError, match=message):
        read_trial(trial(tmp_path, events))


def test_trial_reader_rejects_missing_frames(tmp_path):
    directory = trial(tmp_path, [event(0), event(1)])
    (directory / "frames/00001.jpg").unlink()
    with pytest.raises(ValueError, match="Missing frame"):
        read_trial(directory)


def test_value_and_grad_ignores_nothing_trainable():
    model = LatentActionModel(features=4, tokens=1, codes=2, dim=16, width=8)
    grads = nn.value_and_grad(model, lam_loss)(model, mx.ones((2, 4)), mx.ones((2, 4)), mx.ones(2))[
        1
    ]
    assert "codebook" in grads["quantizer"]
