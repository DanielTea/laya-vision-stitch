"""Slow-path planner: a stronger frozen encoder and the goal produce intent residuals.

The planner may run asynchronously at a lower rate; the fast P2P controller consumes its
latest (possibly stale) output. Outputs are zero-initialized residuals on the fast
policy's goal and image tokens, so initialization preserves pretrained behavior.
"""

import math

import mlx.core as mx
import mlx.nn as nn


class SlowPlanner(nn.Module):
    def __init__(self, feature_width=768, width=256, queries=4, goal_width=768):
        super().__init__()
        self.width, self.queries = width, queries
        self.feature_norm = nn.LayerNorm(feature_width, affine=False)
        self.goal_norm = nn.LayerNorm(goal_width, affine=False)
        self.key = nn.Linear(feature_width, width)
        self.value = nn.Linear(feature_width, width)
        self.query = nn.Linear(goal_width, queries * width)
        self.query_position = mx.random.normal((1, queries, width)) * 0.02
        self.mix = nn.Sequential(
            nn.Linear(queries * width, 2 * width), nn.SiLU(), nn.Linear(2 * width, 2 * width)
        )
        self.language = nn.Linear(2 * width, 1024)
        self.image = nn.Linear(2 * width, 1024)
        for layer in (self.language, self.image):
            layer.weight = mx.zeros_like(layer.weight)
            layer.bias = mx.zeros_like(layer.bias)

    def attention(self, features, goal):
        f = self.feature_norm(features.astype(mx.float32))
        q = self.query(self.goal_norm(goal)).reshape(-1, self.queries, self.width)
        q = q + self.query_position
        weights = mx.softmax(q @ self.key(f).transpose(0, 2, 1) / math.sqrt(self.width), -1)
        return weights, (weights @ self.value(f)).reshape(-1, self.queries * self.width)

    def __call__(self, features, goal):
        """features [B,P,C] patches, goal [B,768] -> (language, image) residuals [B,1024]."""
        _, pooled = self.attention(features, goal)
        h = nn.silu(self.mix(pooled))
        return self.language(h), self.image(h)
