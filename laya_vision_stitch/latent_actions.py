"""Small LAPA/Genie-style latent action model over frozen P2P image tokens; offline only.

The encoder sees standardized image tokens (z_t, z_t+1) and emits a few discrete latent
tokens (one small VQ codebook per token, straight-through gradients, commitment loss,
dead-code replacement). The decoder predicts z_t+1 as a residual on z_t; latent dropout
trains it to also predict without latents, so "zeroed latents" is a fitted action-free
reference. Latents summarize frame change without labels; they are not actions. Probes,
NMI and live-trial parsing are evaluation tools. Nothing here dispatches input or enters
the deployed runtime.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from .flow_action_head import BINARY, DIM
from .p2p_adaptation import KEYS_WITH_TAB

KEYS = frozenset(k for k in KEYS_WITH_TAB if k is not None)
MOVEMENT = ("w", "a", "s", "d")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_split(cache, split):
    """Cached arrays plus compact rows; rows follow array order (sequence, then step)."""
    arrays = dict(np.load(Path(cache) / f"{split}.npz"))
    rows = read_jsonl(Path(cache) / f"{split}.jsonl")
    if len(rows) != len(arrays["images"]):
        raise ValueError(f"{split}: rows and arrays disagree")
    return arrays, rows


def offset_index(sequence_index, steps, index, offset):
    """Frame index+offset and whether it is the same sequence exactly `offset` steps away."""
    sequence_index, steps, index = map(np.asarray, (sequence_index, steps, index))
    target = index + offset
    inside = (target >= 0) & (target < len(steps))
    target = np.clip(target, 0, len(steps) - 1)
    same = (sequence_index[target] == sequence_index[index]) & (
        steps[target] == steps[index] + offset
    )
    return target, inside & same


def successors(sequence_index, steps, gap=1):
    """Indices t whose frame t+gap exists in the same sequence."""
    if gap < 1:
        raise ValueError("Gap must be positive")
    index = np.arange(len(steps))
    _, valid = offset_index(sequence_index, steps, index, gap)
    return index[valid]


class Standardizer:
    """Per-dimension standardization; fit on training rows only."""

    def __init__(self, mean, scale):
        self.mean = np.asarray(mean, np.float32)
        self.scale = np.asarray(scale, np.float32)
        if self.mean.shape != self.scale.shape or np.any(self.scale <= 0):
            raise ValueError("Invalid standardization statistics")

    @classmethod
    def fit(cls, x):
        x = np.asarray(x, np.float64)
        if x.ndim != 2 or len(x) < 2:
            raise ValueError("Need a [N,D] training matrix with N >= 2")
        return cls(x.mean(0), x.std(0) + 1e-6)

    def __call__(self, x):
        return ((np.asarray(x, np.float32) - self.mean) / self.scale).astype(np.float32)

    def save(self, path):
        np.savez(path, mean=self.mean, scale=self.scale)

    @classmethod
    def load(cls, path):
        a = np.load(path)
        return cls(a["mean"], a["scale"])


def mlp(inputs, width, outputs, depth=2):
    layers = [nn.Linear(inputs, width), nn.GELU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.GELU()]
    return nn.Sequential(*layers, nn.Linear(width, outputs))


class VectorQuantizer(nn.Module):
    """Independent codebooks per latent token, [tokens, codes, dim]; Euclidean nearest code."""

    def __init__(self, tokens=4, codes=8, dim=16):
        super().__init__()
        self.tokens, self.codes, self.dim = tokens, codes, dim
        self.codebook = mx.random.normal((tokens, codes, dim))

    def indices(self, z):
        d = ((z[:, :, None] - self.codebook[None]) ** 2).sum(-1)
        return mx.argmin(d, -1)

    def lookup(self, index):
        onehot = (index[..., None] == mx.arange(self.codes)).astype(self.codebook.dtype)
        return mx.einsum("btc,tcd->btd", onehot, self.codebook)

    def __call__(self, z):
        if z.ndim != 3 or z.shape[1:] != (self.tokens, self.dim):
            raise ValueError("Expected [B,tokens,dim] encoder outputs")
        index = mx.stop_gradient(self.indices(z))
        q = self.lookup(index)
        codebook_loss = ((mx.stop_gradient(z) - q) ** 2).mean()
        commitment = ((z - mx.stop_gradient(q)) ** 2).mean()
        return z + mx.stop_gradient(q - z), index, codebook_loss, commitment


def replace_dead_codes(quantizer, usage, encoded, rng, threshold=0.02, noise=0.01):
    """Move codes whose running usage fraction is below `threshold` onto random encodings.

    usage [tokens,codes] running frequencies (updated in place for replaced codes);
    encoded [B,tokens,dim] current pre-quantization outputs. Returns the replaced count.
    """
    dead = np.argwhere(usage < threshold)
    if not len(dead):
        return 0
    book = np.array(quantizer.codebook)
    encoded = np.asarray(encoded)
    for t, c in dead:
        pick = encoded[rng.integers(len(encoded)), t]
        book[t, c] = pick + noise * rng.standard_normal(pick.shape)
        usage[t, c] = 1.0 / quantizer.codes
    quantizer.codebook = mx.array(book.astype(np.float32))
    return len(dead)


class LatentActionModel(nn.Module):
    """Encoder (z_t, z_t+1) -> quantized latents; decoder (z_t, latents) -> z_t+1 - z_t.

    `context` limits what the decoder sees of z_t: -1 all of it, 0 nothing (the change
    must come from the latents), k > 0 a learned k-D projection. Narrow context prevents
    the decoder from memorizing clip appearance on small data.
    """

    def __init__(self, features=1024, tokens=4, codes=8, dim=16, width=512, context=-1):
        super().__init__()
        if not 16 <= dim <= 32 or tokens < 1 or codes < 2:
            raise ValueError("Use 16-32D codes, >=1 token and >=2 codes")
        if context < -1 or context > features:
            raise ValueError("Decoder context must be -1, 0 or a projection width")
        self.features, self.tokens, self.codes, self.dim = features, tokens, codes, dim
        self.context = context
        self.encoder = nn.Sequential(
            nn.LayerNorm(3 * features), mlp(3 * features, width, tokens * dim)
        )
        self.quantizer = VectorQuantizer(tokens, codes, dim)
        if context > 0:
            self.projection = nn.Linear(features, context)
        seen = features if context == -1 else context
        self.decoder = mlp(seen + tokens * dim, width, features)
        last = self.decoder.layers[-1]
        last.weight, last.bias = mx.zeros_like(last.weight), mx.zeros_like(last.bias)

    def encode(self, z0, z1):
        x = mx.concatenate([z0, z1, z1 - z0], -1)
        return self.encoder(x).reshape(-1, self.tokens, self.dim)

    def decode(self, z0, latents):
        parts = [latents.reshape(z0.shape[0], -1)]
        if self.context == -1:
            parts.insert(0, z0)
        elif self.context > 0:
            parts.insert(0, self.projection(z0))
        return z0 + self.decoder(mx.concatenate(parts, -1))

    def __call__(self, z0, z1, keep=None):
        encoded = self.encode(z0, z1)
        q, index, codebook_loss, commitment = self.quantizer(encoded)
        if keep is not None:
            q = q * keep[:, None, None]
        return self.decode(z0, q), index, codebook_loss, commitment, encoded


def lam_loss(model, z0, z1, keep, beta=0.25):
    """Next-token MSE + codebook loss + beta * commitment; aux = (mse, codes, encodings)."""
    pred, index, codebook_loss, commitment, encoded = model(z0, z1, keep)
    mse = ((pred - z1) ** 2).mean()
    return mse + codebook_loss + beta * commitment, (mse, index, encoded)


def batched(fn, *arrays, size=1024):
    """Apply fn to aligned NumPy arrays in chunks and concatenate NumPy outputs."""
    outs = []
    for start in range(0, len(arrays[0]), size):
        out = fn(*[mx.array(a[start : start + size]) for a in arrays])
        out = out if isinstance(out, tuple) else (out,)
        mx.eval(*out)
        outs.append([np.asarray(o) for o in out])
    merged = tuple(np.concatenate(parts) for parts in zip(*outs, strict=True))
    return merged if len(merged) > 1 else merged[0]


def lam_codes(model, z0, z1):
    """Integer codes [N,tokens] for standardized pairs."""
    return batched(lambda a, b: model.quantizer.indices(model.encode(a, b)), z0, z1).astype(
        np.int64
    )


def lam_errors(model, z0, z1):
    """Per-pair MSE for copy-last, decoder with zeroed latents, and the full model."""

    def run(a, b):
        full = model(a, b)[0]
        zero = model.decode(a, mx.zeros((a.shape[0], model.tokens, model.dim)))
        return (
            ((a - b) ** 2).mean(-1),
            ((zero - b) ** 2).mean(-1),
            ((full - b) ** 2).mean(-1),
        )

    copy, zero, full = batched(run, z0, z1)
    return {"copy_last": copy, "zeroed_latents": zero, "full": full}


def code_embeddings(model, codes):
    """Quantized embeddings [N,tokens*dim] for integer codes."""
    return batched(lambda c: model.quantizer.lookup(c).reshape(c.shape[0], -1), codes)


def combine_codes(codes, size):
    """[N,tokens] -> one integer per code combination."""
    codes = np.asarray(codes, np.int64)
    return (codes * size ** np.arange(codes.shape[1])).sum(1)


def fit(
    model, loss, batches, validate, steps, learning_rate, every=250, weight_decay=1e-4, on_step=None
):
    """AdamW training; keeps the parameters with the lowest `validate()` (step 0 included).

    `batches(step)` returns loss arguments; `loss` may return (value, aux) and `on_step`
    receives (step, aux). Only validation data may drive selection.
    """
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    value_grad = nn.value_and_grad(model, loss)
    model.eval()
    best, best_step = validate(), 0
    best_state = tree_flatten(model.parameters())
    history = [{"step": 0, "validation": best}]
    for step in range(1, steps + 1):
        model.train()
        value, grads = value_grad(model, *batches(step))
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, value)
        if on_step is not None:
            on_step(step, value)
        if step % every == 0 or step == steps:
            model.eval()
            v = validate()
            train = value[0] if isinstance(value, tuple) else value
            history.append({"step": step, "train_loss": float(train), "validation": v})
            if v < best:
                best, best_step, best_state = v, step, tree_flatten(model.parameters())
    model.load_weights(best_state)
    model.eval()
    return {"selected_step": best_step, "validation": best, "history": history}


def train_lam(
    model,
    train,
    validation,
    steps=3000,
    batch=256,
    learning_rate=3e-4,
    seed=0,
    latent_dropout=0.1,
    beta=0.25,
    replace_every=100,
    every=250,
    weight_decay=1e-4,
):
    """Fit on standardized (z0, z1) pairs; validation full-model MSE selects the checkpoint."""
    rng = np.random.default_rng(seed)
    z0, z1 = train
    usage = np.full((model.tokens, model.codes), 1.0 / model.codes)
    replaced = [0]

    def batches(step):
        idx = rng.integers(0, len(z0), batch)
        keep = (rng.random(batch) >= latent_dropout).astype(np.float32)
        return mx.array(z0[idx]), mx.array(z1[idx]), mx.array(keep), beta

    def on_step(step, value):
        _, (_, index, encoded) = value
        index = np.asarray(index)
        counts = np.stack(
            [np.bincount(index[:, t], minlength=model.codes) for t in range(model.tokens)]
        )
        usage[:] = 0.98 * usage + 0.02 * counts / len(index)
        if step % replace_every == 0 and step <= 0.8 * steps:
            replaced[0] += replace_dead_codes(model.quantizer, usage, encoded, rng)

    def validate():
        return float(np.mean(lam_errors(model, *validation)["full"]))

    report = fit(
        model, lam_loss, batches, validate, steps, learning_rate, every, weight_decay, on_step
    )
    report["replaced_codes"] = replaced[0]
    return report


def entropy(labels):
    _, counts = np.unique(np.asarray(labels), return_counts=True, axis=0)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def mutual_information(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b) or not len(a):
        raise ValueError("Labels must be nonempty and aligned")
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    joint = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(joint, (ai, bi), 1)
    joint /= joint.sum()
    pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (pa @ pb)[nz])).sum())


def normalized_mutual_information(a, b):
    """MI / arithmetic mean of entropies (sklearn's default); 1 when both are constant."""
    ha, hb = entropy(a), entropy(b)
    if ha == 0 and hb == 0:
        return 1.0
    return max(0.0, mutual_information(a, b)) / ((ha + hb) / 2)


def chance_nmi(a, b, rng, repeats=20):
    """Mean NMI after permuting `a`: the small-sample bias of many-valued codes."""
    a = np.asarray(a)
    return float(
        np.mean([normalized_mutual_information(rng.permutation(a), b) for _ in range(repeats)])
    )


def button_classes(buttons):
    """Button sets -> integer class per distinct set."""
    names = {s: i for i, s in enumerate(sorted({tuple(sorted(b)) for b in buttons}))}
    return np.array([names[tuple(sorted(b))] for b in buttons], np.int64)


def mouse_direction(mouse):
    """0 = no motion, 1..8 = 45 degree sectors starting at +x (screen coordinates)."""
    mouse = np.asarray(mouse, float).reshape(-1, 2)
    angle = np.arctan2(mouse[:, 1], mouse[:, 0])
    sector = np.floor(((angle + np.pi / 8) % (2 * np.pi)) / (np.pi / 4)).astype(np.int64) % 8
    return np.where(np.any(mouse != 0, 1), 1 + sector, 0)


def action_facts(buttons, mouse, previous, held=("w", "mouse_left"), groups=None):
    """Binary facts for target frames -> {name: (labels, valid)}.

    `previous` holds the previous frame's buttons or None when unknown (onset invalid).
    `groups` maps a name to controls where any held counts as positive.
    """
    mouse = np.asarray(mouse, float).reshape(-1, 2)
    n = len(buttons)
    if len(mouse) != n or len(previous) != n:
        raise ValueError("Facts need aligned buttons, mouse and previous buttons")
    everywhere = np.ones(n, bool)
    facts = {
        "mouse_motion": (np.any(mouse != 0, 1), everywhere),
        "mouse_x_positive": (mouse[:, 0] > 0, mouse[:, 0] != 0),
    }
    for control in held:
        facts[f"{control}_held"] = (np.array([control in b for b in buttons]), everywhere)
    for name, controls in (groups or {}).items():
        facts[name] = (np.array([bool(set(b) & set(controls)) for b in buttons]), everywhere)
    onset = [
        p is not None and bool((set(b) - set(p)) & KEYS)
        for b, p in zip(buttons, previous, strict=True)
    ]
    facts["key_onset"] = (np.array(onset), np.array([p is not None for p in previous]))
    return facts


def lagged_facts(buttons, mouse, sequence_index, steps, index, lag, **kwargs):
    """Facts of the action at frame index-lag for pairs starting at `index`."""
    target, valid = offset_index(sequence_index, steps, index, -lag)
    prior, prior_valid = offset_index(sequence_index, steps, index, -lag - 1)
    facts = action_facts(
        [buttons[t] for t in target],
        np.asarray(mouse)[target],
        [buttons[p] if ok else None for p, ok in zip(prior, prior_valid, strict=True)],
        **kwargs,
    )
    return {k: (y, v & valid) for k, (y, v) in facts.items()}, target, valid


class LogisticProbe:
    """L2-regularized logistic regression (Newton steps) on train-standardized features."""

    def __init__(self, penalty=1e-2, iterations=50):
        if penalty <= 0:
            raise ValueError("Penalty must be positive")
        self.penalty, self.iterations = penalty, iterations

    def _design(self, x):
        x = (np.asarray(x, np.float64) - self.mean) / self.scale
        return np.concatenate([x, np.ones((len(x), 1))], 1)

    def fit(self, x, y):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        if x.ndim != 2 or len(x) != len(y) or not len(y):
            raise ValueError("Probe needs aligned nonempty [N,D] features and labels")
        self.mean, self.scale = x.mean(0), x.std(0) + 1e-6
        design = self._design(x)
        self.constant = None if 0 < y.mean() < 1 else float(y.mean())
        w = np.zeros(design.shape[1])
        if self.constant is not None:
            self.weights = w
            return self
        ridge = np.full(len(w), self.penalty)
        ridge[-1] = 1e-3 * self.penalty
        for _ in range(self.iterations):
            p = 1 / (1 + np.exp(-np.clip(design @ w, -30, 30)))
            grad = design.T @ (p - y) / len(y) + ridge * w
            hess = (design.T * (p * (1 - p))) @ design / len(y) + np.diag(ridge)
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.abs(step).max() < 1e-7:
                break
        self.weights = w
        return self

    def probability(self, x):
        if self.constant is not None:
            return np.full(len(x), self.constant)
        return 1 / (1 + np.exp(-np.clip(self._design(x) @ self.weights, -30, 30)))


def binary_scores(y, probability):
    """Accuracy, balanced accuracy and eval-split majority-class accuracy."""
    y = np.asarray(y, bool)
    pred = np.asarray(probability) >= 0.5
    rates = [np.mean(pred[y == c] == c) for c in (True, False) if np.any(y == c)]
    positive = float(y.mean()) if len(y) else 0.0
    return {
        "examples": int(len(y)),
        "positive_rate": positive,
        "accuracy": float(np.mean(pred == y)) if len(y) else None,
        "balanced_accuracy": float(np.mean(rates)) if rates else None,
        "majority_accuracy": max(positive, 1 - positive) if len(y) else None,
    }


PENALTIES = (1e-3, 1e-2, 1e-1, 1.0)


def select_probe(train, validation, penalties=PENALTIES):
    """Pick the penalty by validation balanced accuracy; train/validation are (x, y)."""
    best = None
    for penalty in penalties:
        probe = LogisticProbe(penalty).fit(*train)
        score = binary_scores(validation[1], probe.probability(validation[0]))["balanced_accuracy"]
        score = -1.0 if score is None else score
        if best is None or score > best[0]:
            best = (score, penalty, probe)
    return best[2], best[1], best[0]


class ActionHead(nn.Module):
    """Small MLP from a frozen feature to 24-D action vectors, plus a LAM-code output."""

    def __init__(self, inputs=1024, width=256, tokens=4, codes=8, dropout=0.1):
        super().__init__()
        self.tokens, self.codes = tokens, codes
        self.trunk = nn.Sequential(
            nn.LayerNorm(inputs),
            nn.Linear(inputs, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.action = nn.Linear(width, DIM)
        self.latent = nn.Linear(width, tokens * codes)

    def __call__(self, x):
        return self.action(self.trunk(x))

    def code_logits(self, x):
        return self.latent(self.trunk(x)).reshape(-1, self.tokens, self.codes)


def action_loss(head, x, target, mask):
    """BCE on binary controls + MSE on tanh mouse, masked by label completeness."""
    out = head(x)
    binary = nn.losses.binary_cross_entropy(
        out[:, :BINARY], (target[:, :BINARY] > 0).astype(mx.float32), reduction="none"
    ).mean(-1)
    mouse = ((mx.tanh(out[:, BINARY:]) - target[:, BINARY:]) ** 2).mean(-1)
    return ((binary + mouse) * mask).sum() / mx.maximum(mask.sum(), 1)


def code_loss(head, x, codes):
    return nn.losses.cross_entropy(head.code_logits(x), codes, reduction="mean")


def predict_vectors(head, x):
    def run(a):
        out = head(a)
        return mx.concatenate(
            [mx.where(out[:, :BINARY] > 0, 1.0, -1.0), mx.tanh(out[:, BINARY:])], -1
        )

    return batched(run, x)


def read_trial(directory):
    """Live-trial frames with the controls actually dispatched after each frame.

    The runner logs `bounded_action` after dispatch, including playfield mouse clipping;
    frames whose action was not applied get an empty action. Saved frames are already the
    logged 1280x720 viewport crop. Steps must be consecutive and timestamps increasing.
    """
    directory = Path(directory)
    events = read_jsonl(directory / "events.jsonl")
    if not events:
        raise ValueError(f"{directory} has no events")
    rows = []
    for k, e in enumerate(events):
        if e["step"] != k:
            raise ValueError(f"{directory}: steps are not consecutive at {k}")
        path = directory / e["image"]
        if not path.exists():
            raise ValueError(f"Missing frame {path}")
        action = e["bounded_action"] if e["applied"] else {"buttons": [], "mouse_delta": [0.0, 0.0]}
        delta = [float(v) for v in action["mouse_delta"]]
        if len(delta) != 2 or not np.isfinite(delta).all():
            raise ValueError(f"{directory}: invalid mouse delta at {k}")
        rows.append(
            {
                "trial": directory.name,
                "step": k,
                "elapsed_s": float(e["elapsed_s"]),
                "image": str(path),
                "applied": bool(e["applied"]),
                "action": {"buttons": sorted(action["buttons"]), "mouse_delta": delta},
            }
        )
    elapsed = np.array([r["elapsed_s"] for r in rows])
    if np.any(np.diff(elapsed) <= 0):
        raise ValueError(f"{directory}: timestamps must increase")
    return rows
