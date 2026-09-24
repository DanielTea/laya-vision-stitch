"""Action-conditioned latent dynamics over frozen P2P features; offline evaluation only.

From the frozen policy context c_t and the standardized image token z_t, a residual MLP
state is rolled forward with executed 24-D action vectors to predict z_t+1..z_t+H as
residuals on z_t. Controllability compares true actions with actions copied from a random
other frame of the same game. This measures whether imagined futures respond to actions;
it is not a simulator qualified for policy optimization, and nothing here sends input.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .flow_action_head import DIM
from .latent_actions import batched, fit, offset_index


class LatentWorldModel(nn.Module):
    """s_0 = f(c_t, z_t); s_h = s_h-1 + g(s_h-1, a_t+h-1); z_t+h = z_t + out(s_h).

    `content` limits what s_0 sees of (c_t, z_t): -1 everything, 0 nothing (a learned
    start state; changes come from actions only), k > 0 a learned k-D bottleneck.
    """

    def __init__(self, features=1024, context=1024, action=DIM, width=512, content=-1):
        super().__init__()
        if content < -1:
            raise ValueError("Content must be -1, 0 or a bottleneck width")
        self.content = content
        if content == -1:
            self.context_in = nn.Sequential(nn.LayerNorm(context), nn.Linear(context, width))
            self.image_in = nn.Linear(features, width)
        elif content > 0:
            self.context_in = nn.Sequential(nn.LayerNorm(context), nn.Linear(context, content))
            self.image_in = nn.Linear(features, content)
            self.bottleneck = nn.Linear(content, width)
        else:
            self.initial = mx.zeros((width,))
        self.start = nn.Sequential(nn.GELU(), nn.Linear(width, width))
        self.transition = nn.Sequential(
            nn.LayerNorm(width + action),
            nn.Linear(width + action, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, features))
        last = self.output.layers[-1]
        last.weight, last.bias = mx.zeros_like(last.weight), mx.zeros_like(last.bias)

    def __call__(self, context, z, actions):
        if actions.ndim != 3 or actions.shape[0] != z.shape[0]:
            raise ValueError("Expected actions [B,H,action]")
        if self.content == -1:
            s = self.context_in(context.astype(mx.float32)) + self.image_in(z)
        elif self.content > 0:
            s = self.bottleneck(self.context_in(context.astype(mx.float32)) + self.image_in(z))
        else:
            s = mx.broadcast_to(self.initial, (z.shape[0], self.initial.shape[0]))
        s = self.start(s)
        out = []
        for h in range(actions.shape[1]):
            s = s + self.transition(mx.concatenate([s, actions[:, h]], -1))
            out.append(z + self.output(s))
        return mx.stack(out, 1)


def future_index(sequence_index, steps, horizon):
    """[N,H] frames t+1..t+H and a prefix-closed validity mask; never crosses sequences."""
    n = len(steps)
    index = np.zeros((n, horizon), np.int64)
    mask = np.zeros((n, horizon), bool)
    for h in range(1, horizon + 1):
        index[:, h - 1], valid = offset_index(sequence_index, steps, np.arange(n), h)
        mask[:, h - 1] = valid if h == 1 else valid & mask[:, h - 2]
    return index, mask


def action_index(index):
    """Action frames t..t+H-1 for each start t, from future_index output."""
    return np.concatenate([np.arange(len(index))[:, None], index[:, :-1]], 1)


def wm_loss(model, context, z, actions, targets, mask):
    error = ((model(context, z, actions) - targets) ** 2).mean(-1)
    return (error * mask).sum() / mx.maximum(mask.sum(), 1)


def rollout_errors(model, context, z, actions, targets):
    """Per-frame, per-horizon MSE [N,H]."""
    return batched(
        lambda c, z0, a, t: ((model(c, z0, a) - t) ** 2).mean(-1),
        context,
        z,
        actions,
        targets,
        size=512,
    )


def shuffled_starts(groups, valid, rng):
    """Random other start j != i in the same group, drawn from starts with full chunks."""
    groups = np.asarray(groups)
    out = np.arange(len(groups))
    for g in np.unique(groups):
        members = np.flatnonzero(groups == g)
        pool = members[valid[members]]
        if len(pool) < 2:
            raise ValueError(f"Group {g} has fewer than two complete chunks")
        pick = rng.choice(pool, len(members))
        same = pick == members
        while same.any():
            pick[same] = rng.choice(pool, int(same.sum()))
            same = pick == members
        out[members] = pick
    return out


def train_world_model(
    model,
    train,
    validation,
    steps=3000,
    batch=256,
    learning_rate=3e-4,
    seed=0,
    every=250,
    blind=False,
    weight_decay=1e-4,
):
    """train/validation: dicts with context, z, actions [N,H,A], targets [N,H,F], mask [N,H].

    `blind=True` zeroes actions in training and validation (action-free reference).
    """
    rng = np.random.default_rng(seed)
    keep = 0.0 if blind else 1.0

    def batches(step):
        idx = rng.integers(0, len(train["z"]), batch)
        return (
            mx.array(train["context"][idx]),
            mx.array(train["z"][idx]),
            mx.array(train["actions"][idx] * keep),
            mx.array(train["targets"][idx]),
            mx.array(train["mask"][idx].astype(np.float32)),
        )

    def validate():
        v = validation
        errors = rollout_errors(model, v["context"], v["z"], v["actions"] * keep, v["targets"])
        return float((errors * v["mask"]).sum() / v["mask"].sum())

    return fit(model, wm_loss, batches, validate, steps, learning_rate, every, weight_decay)


class RidgeWorldModel(nn.Module):
    """Closed-form linear reference: z_t+h - z_t = [c_t, z_t, a_t..a_t+h-1] W_h + b_h.

    Contexts are standardized with training statistics. `blind` drops action features.
    """

    def __init__(self, weights, biases, means, context_mean, context_scale, blind=False):
        super().__init__()
        self.weights = [mx.array(w, mx.float32) for w in weights]
        self.biases = [mx.array(b, mx.float32) for b in biases]
        self.means = [mx.array(m, mx.float32) for m in means]
        self.context_mean = mx.array(context_mean, mx.float32)
        self.context_scale = mx.array(context_scale, mx.float32)
        self.blind = blind

    def __call__(self, context, z, actions):
        if actions.ndim != 3 or actions.shape[1] < len(self.weights):
            raise ValueError("Ridge model needs one action per predicted step")
        c = (context.astype(mx.float32) - self.context_mean) / self.context_scale
        out = []
        for h, (w, b, m) in enumerate(zip(self.weights, self.biases, self.means, strict=True)):
            parts = [c, z] if self.blind else [c, z, actions[:, : h + 1].reshape(z.shape[0], -1)]
            out.append(z + (mx.concatenate(parts, -1) - m) @ w + b)
        return mx.stack(out, 1)


def ridge_design(data, h, blind, context_mean, context_scale):
    c = (data["context"] - context_mean) / context_scale
    parts = [c, data["z"]]
    if not blind:
        parts.append(data["actions"][:, : h + 1].reshape(len(c), -1))
    return np.concatenate(parts, 1).astype(np.float64)


def fit_ridge_world_model(train, validation, penalties=(1e4, 3e4, 1e5), blind=False):
    """Per-horizon ridge with one penalty chosen by masked validation MSE."""
    context_mean = train["context"].mean(0)
    context_scale = train["context"].std(0) + 1e-6
    horizon = train["mask"].shape[1]
    stats = []
    for h in range(horizon):
        m = train["mask"][:, h]
        x = ridge_design(train, h, blind, context_mean, context_scale)[m]
        y = (train["targets"][:, h] - train["z"])[m].astype(np.float64)
        mean, target = x.mean(0), y.mean(0)
        xc = x - mean
        stats.append((xc.T @ xc, xc.T @ (y - target), mean, target))
    table, best = {}, None
    for penalty in penalties:
        weights = [np.linalg.solve(g + penalty * np.eye(len(g)), b) for g, b, _, _ in stats]
        model = RidgeWorldModel(
            weights,
            [s[3] for s in stats],
            [s[2] for s in stats],
            context_mean,
            context_scale,
            blind,
        )
        v = validation
        errors = rollout_errors(model, v["context"], v["z"], v["actions"], v["targets"])
        table[str(penalty)] = float((errors * v["mask"]).sum() / v["mask"].sum())
        if best is None or table[str(penalty)] < best[0]:
            best = (table[str(penalty)], penalty, model)
    return best[2], {"penalty": best[1], "validation": best[0], "validation_by_penalty": table}


def sequence_bootstrap(values, groups, rng, repeats=2000):
    """95% interval of the mean of per-group means (resampling groups)."""
    groups = np.asarray(groups)
    names = np.unique(groups)
    means = np.array([values[groups == g].mean() for g in names])
    draws = rng.integers(0, len(means), (repeats, len(means)))
    boot = means[draws].mean(1)
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def controllability(model, data, groups, sequences, rng, draws=5, blind_model=None, no_input=None):
    """Horizon-wise MSE: copy-last, true, shuffled-same-group, optional no-input and blind.

    data: context, z, actions, targets, mask. Only fully valid chunks are shuffle sources.
    `sequences` labels frames for the sequence-level bootstrap of shuffled minus true.
    """
    mask = data["mask"]
    true = rollout_errors(model, data["context"], data["z"], data["actions"], data["targets"])
    copy = ((data["targets"] - data["z"][:, None]) ** 2).mean(-1)
    shuffled = []
    for _ in range(draws):
        source = shuffled_starts(groups, mask.all(1), rng)
        shuffled.append(
            rollout_errors(
                model, data["context"], data["z"], data["actions"][source], data["targets"]
            )
        )
    shuffled = np.mean(shuffled, 0)
    extra = {}
    if no_input is not None:
        idle = np.broadcast_to(no_input, data["actions"].shape).astype(np.float32)
        extra["no_input"] = rollout_errors(model, data["context"], data["z"], idle, data["targets"])
    if blind_model is not None:
        extra["action_blind_model"] = rollout_errors(
            blind_model, data["context"], data["z"], np.zeros_like(data["actions"]), data["targets"]
        )
    report = {}
    for h in range(mask.shape[1]):
        m = mask[:, h]
        if not m.any():
            continue
        row = {
            "examples": int(m.sum()),
            "copy_last": float(copy[m, h].mean()),
            "true_actions": float(true[m, h].mean()),
            "shuffled_actions": float(shuffled[m, h].mean()),
        }
        for name, errors in extra.items():
            row[name] = float(errors[m, h].mean())
        row["relative_gain_vs_shuffled"] = (row["shuffled_actions"] - row["true_actions"]) / row[
            "shuffled_actions"
        ]
        row["shuffled_minus_true_ci95"] = sequence_bootstrap(
            shuffled[m, h] - true[m, h], np.asarray(sequences)[m], rng
        )
        report[f"h{h + 1}"] = row
    return report
