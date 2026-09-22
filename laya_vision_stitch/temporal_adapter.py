"""Goal-conditioned moment tokens and learned temporal action prediction."""

import mlx.core as mx
import mlx.nn as nn

from .temporal_memory import MemoryBlock


class TemporalActionAdapter(nn.Module):
    def __init__(self, visual_width, language_width, config):
        super().__init__()
        d = config.temporal_width
        self.visual_norm = nn.LayerNorm(visual_width)
        self.language_norm = nn.LayerNorm(language_width)
        self.visual = nn.Linear(visual_width, d)
        self.language = nn.Linear(language_width, d)
        self.spatial = mx.random.normal((16, d)) * 0.02
        self.queries = mx.random.normal((4, d)) * 0.02
        self.read = nn.MultiHeadAttention(d, 4)
        self.combine = nn.Linear(4 * d, d)
        self.history = nn.Sequential(
            nn.Linear(len(config.buttons) + 7, d), nn.GELU(), nn.Linear(d, d)
        )
        self.blocks = [MemoryBlock(d, config.temporal_adapter, 32), MemoryBlock(d, "attention", 8)]
        self.norm = nn.LayerNorm(d)
        self.buttons = nn.Linear(d, len(config.buttons))
        self.camera = nn.Linear(d, 2 * len(config.mouse_bins))
        self.action_embedding = nn.Linear(len(config.buttons) + 2, d)
        # Fixed target: spatial frozen vision summaries, not a jointly collapsed encoder.
        self.dynamics = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 64))
        self.bin_count = len(config.mouse_bins)

    def tokens(self, visual, language):
        # [batch, time, 16, visual_width], [batch, time, 2, language_width]
        batch, length = visual.shape[:2]
        v = self.visual(self.visual_norm(visual)) + self.spatial
        language_tokens = self.language(self.language_norm(language))
        kv = mx.concatenate([v, language_tokens], -2).reshape(batch * length, 18, -1)
        q = mx.broadcast_to(self.queries, (batch * length, *self.queries.shape))
        return self.combine(self.read(q, kv, kv).reshape(batch, length, -1))

    def __call__(self, visual, language, previous, state=None):
        x = self.tokens(visual, language) + self.history(previous)
        states = []
        for i, block in enumerate(self.blocks):
            x, s = block(x, None if state is None else state[i])
            states.append(s)
        x = self.norm(x)
        return {
            "buttons": self.buttons(x),
            "camera": self.camera(x).reshape(*x.shape[:2], 2, self.bin_count),
            "hidden": x,
        }, states

    def future_delta(self, hidden, action):
        # Called only by training/evaluation of the auxiliary objective.
        return self.dynamics(mx.concatenate([hidden, self.action_embedding(action)], -1))


def spatial_pool(patches, coordinates):
    """Preserve coarse spatial structure while bounding adapter token count."""
    import numpy as np

    xy = np.asarray(coordinates[:, :2])
    index = np.clip(((xy + 1) * 2).astype(int), 0, 3)
    indices = index[:, 1] * 4 + index[:, 0]
    matrix = np.array([indices == i for i in range(16)], dtype=np.float32)
    matrix /= np.maximum(matrix.sum(-1, keepdims=True), 1)
    return mx.array(matrix) @ patches


def dynamics_target(pooled):
    """64 deterministic spatial/channel-group means from frozen features."""
    # The Qwen width 2560 divides evenly; no random or learned target projection.
    if pooled.shape[-1] % 4:
        raise ValueError("Visual width must divide into four feature groups")
    normalized = pooled * mx.rsqrt(mx.mean(pooled * pooled, axis=-1, keepdims=True) + 1e-5)
    return normalized.reshape(*pooled.shape[:-1], 4, -1).mean(-1).reshape(*pooled.shape[:-2], 64)
