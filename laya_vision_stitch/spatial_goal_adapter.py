"""Goal-conditioned attention over frozen spatial features, with offline auxiliary heads."""

import math

import mlx.core as mx
import mlx.nn as nn


class SpatialGoalAdapter(nn.Module):
    def __init__(self, width=64, queries=4):
        super().__init__()
        if width < 1 or queries < 1:
            raise ValueError("Positive spatial width and query count required")
        self.width, self.queries = width, queries
        self.image_norm = nn.LayerNorm(112, affine=False)
        self.goal_norm = nn.LayerNorm(768, affine=False)
        self.patch = nn.Linear(112, width)
        self.position = mx.random.normal((1, 144, width)) * 0.01
        self.query = nn.Linear(768, queries * width)
        self.query_position = mx.random.normal((1, queries, width)) * 0.01
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(queries * width, 1024, bias=False)
        self.output.weight = mx.zeros_like(self.output.weight)
        # Auxiliary prediction heads are training/evaluation only; __call__ skips them.
        self.grounding_head = nn.Linear(queries * width, 2)
        self.distillation_head = nn.Linear(width, 768)

    def features(self, spatial, goal):
        if spatial.ndim != 4 or spatial.shape[1:] != (12, 12, 112):
            raise ValueError("Expected frozen 12x12x112 spatial features")
        if goal.shape != (spatial.shape[0], 768):
            raise ValueError("One 768D goal per image is required")
        patches = self.patch(self.image_norm(spatial.reshape(-1, 144, 112))) + self.position
        query = (
            self.query(self.goal_norm(goal)).reshape(-1, self.queries, self.width)
            + self.query_position
        )
        attention = mx.softmax(
            query @ self.key(patches).transpose(0, 2, 1) / math.sqrt(self.width), axis=-1
        )
        pooled = (attention @ self.value(patches)).reshape(-1, self.queries * self.width)
        return patches, pooled, attention

    def __call__(self, spatial, goal, image):
        _, pooled, _ = self.features(spatial, goal)
        return image + self.output(pooled)


def install_spatial_adapter(model, width=64, queries=4):
    if hasattr(model.policy, "spatial_adapter") or hasattr(model.policy, "visual_adapter"):
        raise ValueError("Start with an unadapted visual path")
    model.policy.spatial_adapter = SpatialGoalAdapter(width, queries)
