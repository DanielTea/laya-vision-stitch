import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from laya_vision_stitch.p2p_pretrained_policy import Layer, OpenP2PPolicy
from scripts.train_hordes_awr import (
    ATTACK,
    SPACE,
    attack_probability,
    auc,
    auc_interval,
    awr_weights,
    cross_fitted_values,
    example_return,
    outcome_summary,
    reward_statistics,
    row_returns,
)

IDLE = [0, 0, 0, 0, 0, 0, 11, 8]
PRESS = [ATTACK, 0, 0, 0, 0, 0, 11, 8]


def record(step, t, target=None, hp=None, health=1.0, panel=None, out_of_range=False):
    return {
        "step": step,
        "elapsed_s": t,
        "image": f"frames/{step:05d}.jpg",
        "state": {
            "target": target,
            "target_health": hp,
            "target_panel": bool(target) if panel is None else panel,
            "health": health,
            "out_of_range": out_of_range,
        },
    }


def test_return_uses_same_named_target_within_horizon_and_counts_missing_ocr():
    sequence = [
        record(0, 0.0, "Grub", 0.9),
        record(1, 0.5, "Fox", 0.1),  # different name: ignored
        record(2, 1.0, "Grub", 0.89),  # below threshold
        record(3, 1.4, "Grub", 0.6, health=0.9, out_of_range=True),
        record(4, 1.6, "Grub", 0.1),  # beyond horizon
    ]
    out = example_return(sequence, 0, health_penalty=0.5)
    assert out["target_damage"] == pytest.approx(0.3)
    assert out["player_health_loss"] == pytest.approx(0.1)
    assert out["reward"] == pytest.approx(0.25)
    assert out["target_status"] == "observed" and out["out_of_range"] and out["complete_window"]
    small = [record(0, 0.0, "Grub", 0.9), record(1, 0.5, "Grub", 0.89)]
    assert (
        example_return(small, 0)["reward"] == 0 and not example_return(small, 0)["complete_window"]
    )
    unseen = [record(0, 0.0, "Grub", 0.9), record(1, 0.5), record(2, 2.0)]
    assert example_return(unseen, 0)["target_status"] == "unobserved"
    panel = [record(0, 0.0, panel=True), record(1, 0.5, "Grub", 0.1)]
    assert example_return(panel, 0)["target_status"] == "unidentified"
    assert example_return(panel, 0)["reward"] == 0
    assert example_return([record(0, 0.0), record(1, 2.0)], 0)["target_status"] == "no_target"
    with pytest.raises(ValueError):
        example_return(sequence, 0, horizon=0)


def test_rows_join_logged_step_and_screenshot(tmp_path):
    log = tmp_path / "events.jsonl"
    log.write_text(
        "".join(
            json.dumps(r) + "\n" for r in [record(3, 0.0, "Grub", 1.0), record(4, 1.0, "Grub", 0.5)]
        )
    )
    row = {
        "id": "r",
        "game": "Hordes.io",
        "action": {"buttons": ["1"]},
        "frames": [{"image": str(tmp_path / "frames/00003.jpg")}],
        "provenance": {"log": str(log), "step": 3},
    }
    public = {"id": "p", "game": "doom", "action": {"buttons": []}}
    result = row_returns([row, public])
    assert result[1] is None and result[0]["target_damage"] == pytest.approx(0.5)
    stats = reward_statistics([row, public], result)
    assert stats["examples"] == 1 and stats["attack"]["damage_followed"] == 1
    with pytest.raises(ValueError, match="screenshot"):
        row_returns([{**row, "frames": [{"image": str(tmp_path / "frames/00004.jpg")}]}])
    with pytest.raises(ValueError, match="missing"):
        row_returns([{**row, "provenance": {"log": str(log), "step": 9}}])


def test_auc_handles_ties_and_missing_classes():
    assert auc([0.9, 0.2, 0.5], [1, 0, 0]) == 1.0
    assert auc([0.5, 0.5], [1, 0]) == 0.5
    assert auc([0.1, 0.9], [1, 0]) == 0.0
    assert auc([0.1, 0.2], [0, 0]) is None
    rng = np.random.default_rng(1)
    labels = np.arange(30) < 15
    scores = labels + rng.normal(size=30)
    low, high = auc_interval(scores, labels)
    assert low <= auc(scores, labels) <= high and 0 <= low < high <= 1
    assert auc_interval(scores, labels) == [low, high]
    assert auc_interval([0.1, 0.2], [1, 1]) is None


def test_value_cross_fitting_never_sees_the_held_out_run():
    rng = np.random.default_rng(0)
    x, groups = rng.normal(size=(60, 5)), np.repeat([0, 1, 2], 20)
    y = x[:, 0] + 0.1 * rng.normal(size=60)
    before, _, _ = cross_fitted_values(x, y, groups, alphas=(1.0,))
    changed = y.copy()
    changed[groups == 1] += 100
    after, _, _ = cross_fitted_values(x, changed, groups, alphas=(1.0,))
    np.testing.assert_allclose(before[groups == 1], after[groups == 1])
    assert not np.allclose(before[groups == 0], after[groups == 0])
    values, alpha, errors = cross_fitted_values(x, y, groups)
    assert errors[alpha] == min(errors.values()) and np.corrcoef(values, y)[0, 1] > 0.9
    with pytest.raises(ValueError):
        cross_fitted_values(x, y, np.zeros(60))


def test_awr_weights_are_clipped_mean_one_and_scale_invariant():
    returns = np.array([0.0, 0.0, 0.0, 0.1, 1.0])
    values = np.full(5, 0.05)
    weights, raw = awr_weights(returns, values, beta=0.5, max_weight=20)
    assert weights.mean() == pytest.approx(1.0) and raw.max() == pytest.approx(20)
    assert np.all(np.diff(weights) >= 0)
    scaled, _ = awr_weights(10 * returns, 10 * values, beta=0.5, max_weight=20)
    np.testing.assert_allclose(weights, scaled)
    softer, _ = awr_weights(returns, values, beta=2.0)
    assert softer.max() < weights.max()
    with pytest.raises(ValueError):
        awr_weights(np.zeros(3), np.zeros(3), beta=1)
    with pytest.raises(ValueError):
        awr_weights(returns, values, beta=0)


class TinyStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [Layer(16, 2, 1) for _ in range(2)]

    def __call__(self, x, offset=0, caches=None, mask=None):
        updated = []
        for i, layer in enumerate(self.layers):
            x, cache = layer(x, offset, None if caches is None else caches[i], mask)
            updated.append(cache)
        return x, updated


class TinyPolicy(OpenP2PPolicy):
    def __init__(self):
        nn.Module.__init__(self)
        self.decoder = TinyStack()
        self.decoder_projection = nn.Linear(16, 16)
        self.decoder_position = mx.random.normal((9, 16))
        self.embeddings = [nn.Embedding(n, 16) for n in (21, 4, 23, 17)]
        self.outputs = [nn.Linear(16, n) for n in (21, 4, 23, 17)]


def test_attack_probability_is_canonical_order_marginal():
    mx.random.seed(5)
    policy = TinyPolicy()
    contexts = mx.random.normal((3, 1, 16))
    forced = mx.array([[SPACE, ATTACK, 0, 0, 0, 0, 0, 0]] * 3)
    _, logits = policy.decode(contexts, forced=forced)
    first, second = np.asarray(mx.softmax(logits[0], -1)), np.asarray(mx.softmax(logits[1], -1))
    expected = first[:, ATTACK] + first[:, SPACE] * second[:, ATTACK]
    actual = np.asarray(attack_probability(policy, contexts))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert np.all((actual > 0) & (actual < 1))


def test_outcome_summary_ranks_damaging_attacks_and_excludes_unknown_windows():
    def row(buttons, game="Hordes.io"):
        return {"game": game, "action": {"buttons": buttons}}

    def outcome(damage, complete=True, far=False):
        return {"target_damage": damage, "complete_window": complete, "out_of_range": far}

    rows = [row(["1"]), row(["1"]), row(["1"]), row(["1"]), row([]), row(["w"]), row([], "doom")]
    outcomes = [outcome(0.3), outcome(0, far=True), outcome(0, complete=False), outcome(0.1)]
    outcomes += [outcome(0), outcome(0), None]
    p = np.array([0.9, 0.2, 0.5, 0.8, 0.1, 0.4, 0.99])
    tokens = [PRESS, IDLE, IDLE, PRESS, PRESS, IDLE, PRESS]
    result = outcome_summary(rows, outcomes, p, tokens)
    assert (result["damage_followed"], result["no_damage"]) == (2, 1)
    assert result["excluded_incomplete_no_damage"] == 1
    assert result["damage_auc"] == 1.0 and result["attack_vs_idle_auc"] == 1.0
    assert result["mean_p_attack"]["out_of_range_after"] == pytest.approx(0.2)
    assert result["mean_p_attack"]["all_hordes"] == pytest.approx(np.mean(p[:6]))
    assert result["greedy_attack_rate"] == {"damage_followed": 1.0, "no_damage": 0.0, "idle": 1.0}
