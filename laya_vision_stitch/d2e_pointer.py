"""Absolute cursor and click labels from D2E input logs (supervision for a pointer head).

Positions are normalized to the recorded game window, so they apply at any resolution.
Camera-locked games click at the crosshair (window center) and pointer games click on
objects; both are kept, so a model learns when pointing matters instead of a game rule.
"""

import bisect
import json
from pathlib import Path

import numpy as np


def load_pointer(path):
    """Return window rect, cursor track [(t, x, y)] and click presses [(t, x, y, button)]."""
    from mcap.reader import make_reader

    rect, track, presses = None, [], []
    with Path(path).open("rb") as f:
        for _, channel, message in make_reader(f).iter_messages(log_time_order=True):
            topic = channel.topic
            if topic not in ("window", "mouse", "mouse/state"):
                continue
            t, data = message.log_time / 1e9, json.loads(message.data)
            if topic == "window":
                if rect is None:
                    rect = data["rect"]
            elif topic == "mouse/state":
                track.append((t, data["x"], data["y"]))
            elif data["event_type"] in ("move", "click"):
                track.append((t, data["x"], data["y"]))
                if data["event_type"] == "click" and data.get("pressed"):
                    presses.append((t, data["x"], data["y"], data["button"]))
    if rect is None or not track:
        raise ValueError("Recording lacks window or cursor information")
    return rect, track, presses


def load_presses(path):
    """Window rect and click presses [(t, x, y, button)] only; skips cursor motion."""
    from mcap.reader import make_reader

    rect, presses = None, []
    with Path(path).open("rb") as f:
        for _, channel, message in make_reader(f).iter_messages(
            topics=["window", "mouse"], log_time_order=True
        ):
            if channel.topic == "window":
                if rect is None:
                    rect = json.loads(message.data)["rect"]
            elif b"click" in message.data:
                data = json.loads(message.data)
                if data["event_type"] == "click" and data.get("pressed"):
                    presses.append((message.log_time / 1e9, data["x"], data["y"], data["button"]))
    if rect is None:
        raise ValueError("Recording lacks window information")
    return rect, presses


class PointerTimeline:
    def __init__(self, rect, track, presses):
        x0, y0, x1, y1 = rect
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Invalid window rectangle")
        self.rect = (x0, y0, x1 - x0, y1 - y0)
        self.track, self.times = track, [t for t, *_ in track]
        self.presses, self.press_times = presses, [t for t, *_ in presses]

    def normalize(self, x, y):
        x0, y0, w, h = self.rect
        return [float(np.clip((x - x0) / w, 0, 1)), float(np.clip((y - y0) / h, 0, 1))]

    def label(self, start, duration):
        """Cursor at `start` and the first button press in [start, start + duration)."""
        i = bisect.bisect_right(self.times, start) - 1
        cursor = self.normalize(*self.track[i][1:]) if i >= 0 else None
        lo = bisect.bisect_left(self.press_times, start)
        hi = bisect.bisect_left(self.press_times, start + duration)
        press = None
        if hi > lo:
            _, x, y, button = self.presses[lo]
            press = {"button": button, "xy": self.normalize(x, y)}
        return {"cursor_xy": cursor, "press": press}

    @classmethod
    def from_mcap(cls, path):
        return cls(*load_pointer(path))
