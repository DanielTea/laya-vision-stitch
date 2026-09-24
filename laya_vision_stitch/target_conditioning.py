"""Condition the controller on a screen point to act on (from a planner at runtime).

Training labels are hindsight: the point is where the player next pressed a mouse button
within a short horizon (or last pressed, shortly before), taken from recorded input logs.
In games where the mouse turns the camera the OS cursor is not a screen location, so the
point there is the crosshair (screen center). Which kind a game is follows from the data:
how strongly mouse motion predicts image change. No game names or rules are involved.

The encoded point is added to the policy's constant "thinking" token. The output layer
starts at zero, so an untrained encoder or a missing point leaves the policy unchanged.
"""

import bisect
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

MOUSE_LOOK_CORRELATION = 0.2


class TargetEncoder(nn.Module):
    def __init__(self, width=1024, hidden=256, frequencies=6):
        super().__init__()
        self.frequencies = int(frequencies)
        self.hidden = nn.Linear(4 * self.frequencies + 3, hidden)
        self.out = nn.Linear(hidden, width)
        self.out.weight = mx.zeros_like(self.out.weight)
        self.out.bias = mx.zeros_like(self.out.bias)

    def __call__(self, xy):
        """xy [N, 2] normalized screen points; NaN rows mean no target -> zero vectors."""
        xy = xy.astype(mx.float32)
        present = mx.logical_not(mx.any(mx.isnan(xy), -1, keepdims=True))
        centered = (mx.where(present, xy, 0.5) - 0.5) * 2
        scales = (2.0 ** mx.arange(self.frequencies)) * math.pi
        angles = (centered[:, :, None] * scales).reshape(xy.shape[0], -1)
        features = mx.concatenate(
            [centered, mx.sin(angles), mx.cos(angles), mx.ones((xy.shape[0], 1))], -1
        )
        return self.out(nn.gelu(self.hidden(features))) * present


def install_target_encoder(model, **config):
    if hasattr(model.policy, "target_encoder"):
        raise ValueError("Target encoder already installed")
    model.policy.target_encoder = TargetEncoder(**config)


def hindsight_targets(times, presses, ahead=2.0, behind=1.0):
    """Per frame time: the next press within `ahead` s, else the last within `behind` s.

    presses: sorted [(t, x, y, button)] with x, y already normalized. Returns target_xy
    [N, 2] (NaN when none), dt [N] (press time minus frame time) and button codes [N].
    """
    stamps = [p[0] for p in presses]
    xy = np.full((len(times), 2), np.nan, np.float32)
    dt = np.full(len(times), np.nan, np.float32)
    button = np.zeros(len(times), np.int8)
    codes = {"left": 1, "right": 2, "middle": 3}
    for i, t in enumerate(times):
        j = bisect.bisect_left(stamps, t)
        chosen = None
        if j < len(presses) and presses[j][0] - t < ahead:
            chosen = presses[j]
        elif j > 0 and t - presses[j - 1][0] <= behind:
            chosen = presses[j - 1]
        if chosen is not None:
            xy[i] = chosen[1:3]
            dt[i] = chosen[0] - t
            button[i] = codes.get(chosen[3], 0)
    return xy, dt, button


def mouse_look_correlation(images, mouse, sequence, steps):
    """Correlation of mouse motion with the change to the next frame's image token.

    images [N, D] frozen image tokens, mouse [N] motion magnitude of the action at each
    frame. Consecutive frames of the same sequence only. NaN if motion never varies.
    """
    order = np.lexsort((steps, sequence))
    a, b = order[:-1], order[1:]
    same = sequence[a] == sequence[b]
    a, b = a[same], b[same]
    change = np.linalg.norm(images[b].astype(np.float32) - images[a].astype(np.float32), axis=1)
    motion = mouse[a].astype(np.float32)
    if motion.std() == 0 or change.std() == 0:
        return float("nan")
    return float(np.corrcoef(motion, change)[0, 1])


MOVES = {
    "w": (0, -1),
    "up": (0, -1),
    "s": (0, 1),
    "down": (0, 1),
    "a": (-1, 0),
    "left": (-1, 0),
    "d": (1, 0),
    "right": (1, 0),
}


def _cosines(vectors, directions):
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(directions, axis=1)
    keep = norms > 0
    return (vectors[keep] * directions[keep]).sum(1) / norms[keep]


def toward_target(buttons, mouse, target_xy, cursor_xy=None, far=0.1):
    """Does behavior point at the target? buttons: sets per frame; mouse [N, 2] deltas.

    move_cos: cosine between the held movement keys (screen directions) and the direction
    from the screen center to the target, over frames that move and whose target is at
    least `far` from the center. mouse_cos: cosine between the mouse delta and the
    direction from the cursor to the target. press_rate: frames holding a mouse button.
    """
    target_xy = np.asarray(target_xy, np.float32)
    has = np.isfinite(target_xy).all(1)
    move = np.array(
        [np.sum([MOVES[b] for b in bs if b in MOVES] or [(0, 0)], 0) for bs in buttons], np.float32
    )
    move = move.reshape(-1, 2)
    away = has & (np.linalg.norm(np.nan_to_num(target_xy - 0.5), axis=1) >= far)
    result = {
        "frames": int(has.sum()),
        "press_rate": float(
            np.mean(
                [bool({"mouse_left", "mouse_right"} & set(buttons[i])) for i in np.flatnonzero(has)]
            )
        )
        if has.any()
        else None,
        "move_rate": float(np.mean(np.abs(move[has]).sum(1) > 0)) if has.any() else None,
    }
    cos = _cosines(move[away], target_xy[away] - 0.5)
    result["move_cos"] = float(cos.mean()) if len(cos) else None
    result["move_frames"] = int(len(cos))
    if cursor_xy is not None:
        cursor_xy = np.asarray(cursor_xy, np.float32)
        near = has & np.isfinite(cursor_xy).all(1)
        offset = np.nan_to_num(target_xy - cursor_xy)
        near &= np.linalg.norm(offset, axis=1) >= far / 2
        cos = _cosines(np.asarray(mouse, np.float32)[near], offset[near])
        result["mouse_cos"] = float(cos.mean()) if len(cos) else None
        result["mouse_frames"] = int(len(cos))
    return result
