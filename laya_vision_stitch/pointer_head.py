"""Absolute click-position head over a frozen visual feature grid (optionally with context).

The policy decides whether a button is pressed; this head decides where on screen the
press lands. It predicts a distribution over grid cells plus a within-cell offset from
pixels (and, optionally, the policy's memory); it has no cursor or game-state input.
Games with a locked crosshair teach it to press at the center; pointer games teach it
to target objects. It adds no gameplay rule. The default grid is the P2P 12x12x112 map;
`grid=(14, 24), features=768` fits C-RADIOv3-B patches at 384x224.
"""

import mlx.core as mx
import mlx.nn as nn

GRID = 12


class PointerHead(nn.Module):
    def __init__(self, width=192, context_width=1024, grid=(GRID, GRID), features=112, depth=2):
        super().__init__()
        self.rows, self.cols = grid
        self.features, self.use_context = features, context_width > 0
        cells = self.rows * self.cols
        self.patch = nn.Sequential(nn.LayerNorm(features), nn.Linear(features, width))
        self.position = mx.random.normal((1, cells, width)) * 0.02
        if self.use_context:
            self.condition = nn.Sequential(
                nn.LayerNorm(context_width), nn.Linear(context_width, 2 * width)
            )
        self.mix = nn.TransformerEncoder(depth, width, 4, mlp_dims=2 * width)
        self.score = nn.Linear(width, 1)
        self.offset = nn.Linear(width, 2)

    def __call__(self, spatial, context=None):
        """spatial [B,rows,cols,features] -> (cell logits [B,cells], offsets [B,cells,2])."""
        b = spatial.shape[0]
        cells = self.rows * self.cols
        h = self.patch(spatial.reshape(b, cells, self.features).astype(mx.float32)) + self.position
        if self.use_context:
            scale, shift = mx.split(self.condition(context.astype(mx.float32))[:, None], 2, -1)
            h = h * (1 + scale) + shift
        h = self.mix(h, None)
        return self.score(h)[..., 0], mx.sigmoid(self.offset(h))

    def targets(self, xy):
        """Normalized [B,2] (x, y) -> cell index [B] and within-cell offset [B,2]."""
        scale = mx.array([self.cols, self.rows], mx.float32)
        cell = mx.clip(mx.floor(xy * scale), 0, scale - 1)
        index = (cell[:, 1] * self.cols + cell[:, 0]).astype(mx.int32)
        return index, xy * scale - cell

    def loss(self, spatial, context, xy):
        logits, offsets = self(spatial, context)
        index, offset = self.targets(xy)
        ce = nn.losses.cross_entropy(logits, index, reduction="mean")
        chosen = mx.take_along_axis(offsets, index[:, None, None], axis=1)[:, 0]
        return ce + 2.0 * mx.abs(chosen - offset).mean()

    def predict(self, spatial, context=None):
        """Most probable cell plus its offset, as normalized (x, y)."""
        logits, offsets = self(spatial, context)
        index = mx.argmax(logits, -1)
        chosen = mx.take_along_axis(offsets, index[:, None, None], axis=1)[:, 0]
        cell = mx.stack([index % self.cols, index // self.cols], -1).astype(mx.float32)
        return (cell + chosen) / mx.array([self.cols, self.rows], mx.float32)
