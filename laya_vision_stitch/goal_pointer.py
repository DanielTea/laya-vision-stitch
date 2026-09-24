"""Goal-conditioned pointing: where on screen is the object the goal refers to?

A goal token (the bridged Laya goal vector) attends jointly with frozen C-RADIOv3-B patch
tokens. The head predicts a distribution over the 14x24 patch grid and whether any goal
object is visible. It is trained from offline teacher points across many games; at
inference it uses only pixels and the goal text. No game-specific rule is involved.
"""

import mlx.core as mx
import mlx.nn as nn

ROWS, COLS = 14, 24


class GoalPointer(nn.Module):
    def __init__(self, width=256, depth=2, features=768, goal_width=768):
        super().__init__()
        self.patch = nn.Sequential(nn.LayerNorm(features), nn.Linear(features, width))
        self.position = mx.random.normal((1, ROWS * COLS, width)) * 0.02
        self.goal = nn.Sequential(nn.LayerNorm(goal_width), nn.Linear(goal_width, width))
        self.mix = nn.TransformerEncoder(depth, width, 4, mlp_dims=2 * width)
        self.cell = nn.Linear(width, 1)
        self.present = nn.Linear(width, 1)

    def __call__(self, patches, goal):
        """patches [B,14,24,768], goal [B,768] -> (cell logits [B,336], presence logit [B])."""
        b = patches.shape[0]
        h = self.patch(patches.reshape(b, ROWS * COLS, -1).astype(mx.float32)) + self.position
        g = self.goal(goal.astype(mx.float32))[:, None]
        # FiLM-free conditioning: the goal token is part of the sequence and every patch
        # attends to it, so the same image can be scored differently for different goals.
        x = self.mix(mx.concatenate([g, h + g], 1), None)
        return self.cell(x[:, 1:])[..., 0], self.present(x[:, 0])[:, 0]

    @staticmethod
    def cell_targets(points_list):
        """Lists of normalized (x, y) points -> multi-hot [B,336] over the patch grid."""
        import numpy as np

        target = np.zeros((len(points_list), ROWS * COLS), np.float32)
        for i, points in enumerate(points_list):
            for x, y in points or []:
                c, r = min(COLS - 1, int(x * COLS)), min(ROWS - 1, int(y * ROWS))
                target[i, r * COLS + c] = 1
        return target

    def loss(self, patches, goal, cells, present):
        logits, presence = self(patches, goal)
        log_p = logits - mx.logsumexp(logits, -1, keepdims=True)
        mass = cells.sum(-1)
        # Put probability mass on any correct cell (log of summed probability).
        hit = mx.logsumexp(mx.where(cells > 0, log_p, -1e9), -1)
        cell_loss = -(mx.where(mass > 0, hit, 0.0)).sum() / mx.maximum((mass > 0).sum(), 1)
        presence_loss = nn.losses.binary_cross_entropy(
            presence, present, with_logits=True, reduction="mean"
        )
        return cell_loss + presence_loss

    def predict(self, patches, goal):
        """(x, y) at the most likely cell center and the presence probability."""
        logits, presence = self(patches, goal)
        index = mx.argmax(logits, -1)
        xy = mx.stack(
            [(index % COLS).astype(mx.float32) + 0.5, (index // COLS).astype(mx.float32) + 0.5], -1
        )
        return xy / mx.array([COLS, ROWS], mx.float32), mx.sigmoid(presence)
