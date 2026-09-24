"""Follow a planner target in newer frames by matching frozen RADIO patch features.

The template is the 3x3 patches around the target in the frame the planner saw (center
weighted double). Newer frames are searched within a window around the last position; the
best weighted cosine match moves the target, and a weak match marks it lost. The match is
refined below one patch by fitting a parabola through the neighboring scores. Because
planner answers arrive seconds late, the first search after an answer uses a wider window
(`catch_up` through buffered frames matched it on recorded trials at higher cost). General
feature matching, no game-specific cue.
"""

import mlx.core as mx
import numpy as np

WIDTH, HEIGHT = 1024, 576  # 64 x 36 patches of 16 pixels
OFFSETS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
WEIGHTS = np.array([2.0 if o == (0, 0) else 1.0 for o in OFFSETS]) / 10


def patch_features(encoder, image, size=(WIDTH, HEIGHT)):
    from PIL import Image

    width, height = size
    pixels = (
        np.asarray(image.convert("RGB").resize((width, height), Image.BICUBIC), np.float32) / 255
    )
    patches, _ = encoder(mx.array(pixels[None]).astype(encoder.input_dtype))
    grid = np.asarray(patches.astype(mx.float32)).reshape(height // 16, width // 16, -1)
    return grid / (np.linalg.norm(grid, axis=-1, keepdims=True) + 1e-6)


def _parabola(left, center, right):
    """Offset in [-0.5, 0.5] of the peak of a parabola through three samples."""
    if not (np.isfinite(left) and np.isfinite(right)):
        return 0.0
    curvature = left - 2 * center + right
    if curvature >= 0:
        return 0.0
    return float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))


def template_scores(grid, template):
    """Weighted cosine of each 3x3 neighborhood of `grid` with the template [9, D]."""
    rows, cols = grid.shape[:2]
    score = np.zeros((rows, cols), np.float32)
    for (dr, dc), weight, patch in zip(OFFSETS, WEIGHTS, template, strict=True):
        shifted = np.pad(grid @ patch, 1, mode="edge")
        score += weight * shifted[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
    return score


class FeatureTracker:
    # On synthetic shifts of recorded Hordes frames (evaluate_tracker.py) true matches score
    # 0.88 median and wrong ones in the search window 0.76; over seconds of real motion true
    # matches fall to about 0.77, so matches below 0.75 count as lost. mode="mean" is the
    # previous 768-pixel single-descriptor matcher, kept for comparison.
    def __init__(
        self,
        encoder,
        window=0.2,
        threshold=0.75,
        refine=True,
        size=(WIDTH, HEIGHT),
        mode="template",
        update=0.0,
    ):
        if mode not in ("template", "mean"):
            raise ValueError("Unknown tracker mode")
        self.encoder, self.window, self.threshold = encoder, window, threshold
        self.refine, self.size, self.mode, self.update = refine, size, mode, float(update)
        self.template, self.xy, self.similarity = None, None, None

    def set_target(self, image, xy):
        grid = patch_features(self.encoder, image, self.size)
        rows, cols = grid.shape[:2]
        r, c = min(rows - 1, int(xy[1] * rows)), min(cols - 1, int(xy[0] * cols))
        self.template = np.stack(
            [
                grid[int(np.clip(r + dr, 0, rows - 1)), int(np.clip(c + dc, 0, cols - 1))]
                for dr, dc in OFFSETS
            ]
        )
        if self.mode == "mean":
            mean = self.template.mean(0)
            self.template = mean / (np.linalg.norm(mean) + 1e-6)
        self.xy, self.similarity = [float(xy[0]), float(xy[1])], 1.0

    def track(self, image, window=None):
        """Return (xy, similarity) in the new frame, or (None, similarity) when lost.

        `window` widens the search for one call, e.g. when a late planner answer arrives.
        """
        if self.template is None:
            return None, None
        grid = patch_features(self.encoder, image, self.size)
        rows, cols = grid.shape[:2]
        score = (
            grid @ self.template if self.mode == "mean" else template_scores(grid, self.template)
        )
        ys, xs = (np.arange(rows)[:, None] + 0.5) / rows, (np.arange(cols)[None, :] + 0.5) / cols
        window = self.window if window is None else window
        near = (np.abs(xs - self.xy[0]) <= window) & (np.abs(ys - self.xy[1]) <= window)
        score = np.where(near, score, -np.inf)
        r, c = np.unravel_index(int(np.argmax(score)), score.shape)
        self.similarity = float(score[r, c])
        if self.similarity < self.threshold:
            return None, self.similarity
        dx = dy = 0.0
        if self.refine:
            dx = _parabola(
                score[r, c - 1] if c > 0 else -np.inf,
                score[r, c],
                score[r, c + 1] if c + 1 < cols else -np.inf,
            )
            dy = _parabola(
                score[r - 1, c] if r > 0 else -np.inf,
                score[r, c],
                score[r + 1, c] if r + 1 < rows else -np.inf,
            )
        self.xy = [float((c + 0.5 + dx) / cols), float((r + 0.5 + dy) / rows)]
        if self.update and self.mode == "template":
            # Follow gradual appearance change: blend in the matched neighborhood.
            block = np.stack(
                [
                    grid[int(np.clip(r + dr, 0, rows - 1)), int(np.clip(c + dc, 0, cols - 1))]
                    for dr, dc in OFFSETS
                ]
            )
            blended = (1 - self.update) * self.template + self.update * block
            self.template = blended / (np.linalg.norm(blended, axis=-1, keepdims=True) + 1e-6)
        return list(self.xy), self.similarity

    def catch_up(self, frames):
        """Track through intermediate frames (oldest first); stops early if lost."""
        xy, similarity = list(self.xy), self.similarity
        for frame in frames:
            xy, similarity = self.track(frame)
            if xy is None:
                break
        return xy, similarity
