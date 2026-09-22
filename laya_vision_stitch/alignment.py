"""Dual ridge regression; the connector is fitted, both neural models stay frozen."""

import numpy as np


class RidgeConnector:
    def fit(self, x, y, alpha):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
            raise ValueError("Expected aligned matrices with at least two rows")
        if alpha <= 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Finite inputs and positive regularization required")
        self.mean = x.mean(0)
        self.scale = x.std(0)
        self.scale[self.scale < 1e-6] = 1
        self.anchors = (x - self.mean) / self.scale
        self.offset = y.mean(0)
        self.alpha = float(alpha)
        kernel = self.anchors @ self.anchors.T / x.shape[1]
        self.dual = np.linalg.solve(kernel + alpha * np.eye(len(x)), y - self.offset)
        return self

    def predict(self, x):
        x = np.asarray(x, np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("Incompatible or non-finite visual features")
        kernel = ((x - self.mean) / self.scale) @ self.anchors.T / x.shape[1]
        return (self.offset + kernel @ self.dual).astype(np.float32)

    def save(self, path):
        np.savez_compressed(
            path,
            mean=self.mean,
            scale=self.scale,
            anchors=self.anchors,
            offset=self.offset,
            dual=self.dual,
            alpha=self.alpha,
        )

    @classmethod
    def load(cls, path):
        result = cls()
        with np.load(path, allow_pickle=False) as saved:
            for key in ("mean", "scale", "anchors", "offset", "dual", "alpha"):
                setattr(result, key, saved[key])
        return result


def agreement(expected, predicted):
    expected, predicted = np.asarray(expected), np.asarray(predicted)
    classes = np.unique(expected)
    per_class = {str(c): float((predicted[expected == c] == c).mean()) for c in classes}
    return {
        "agreement": float((expected == predicted).mean()),
        "balanced_agreement": float(np.mean(list(per_class.values()))),
        "per_class": per_class,
        "teacher_counts": {str(c): int((expected == c).sum()) for c in classes},
    }
