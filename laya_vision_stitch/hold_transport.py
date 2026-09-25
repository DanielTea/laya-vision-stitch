"""Stateful input transport: holds, drags, cursor clutching and mouse-wheel notches.

The pulse transports in `temporal_live` press every control for 50 ms per step and release
it, which turns "hold a button and drag" (camera rotation in MMOs, map dragging and box
selection in strategy games) into click spam. Here a control stays down until the model
stops outputting it:

- keys and the three mouse buttons send down/up events only when their state changes;
- mouse motion while a button is held is a drag event for that button, with the model's
  full delta in the event's delta fields (pointer-locked cameras read those);
- when a drag would carry the cursor out of the allowed area, the buttons are released,
  the cursor re-centered and the buttons pressed again, like lifting a physical mouse;
- `scroll_up` / `scroll_down` post one wheel notch each step they are output;
- a watchdog releases everything if no new action arrives within `watchdog` seconds.

No game-specific rule is involved.
"""

import threading
import time

MOUSE = {"mouse_left": 0, "mouse_right": 1, "mouse_middle": 2}
WHEEL = {"scroll_up": 1, "scroll_down": -1}


class HoldTransport:
    CENTRAL, POINTER = (0.36, 0.72, 0.22, 0.63), (0.03, 0.97, 0.08, 0.97)

    def __init__(self, target, crop, keys, pointer=False, permit=None, watchdog=0.25):
        self.target, self.crop, self.keys = target, crop, keys
        self.pointer, self.permit, self.watchdog = pointer, permit, float(watchdog)
        self.bounds = self.POINTER if pointer else self.CENTRAL
        b = target.original["kCGWindowBounds"]
        self.origin = (b["X"] + crop[0], b["Y"] + crop[1])
        self.center = (self.origin[0] + crop[2] / 2, self.origin[1] + crop[3] / 2)
        self.point = self.center
        self.lock = threading.Lock()
        self.keys_down, self.buttons_down = set(), set()
        self.timer = None
        self.clutches = 0

    # --- low-level posting (overridden in tests) -------------------------------------------
    def post_key(self, key, down):
        self.target.event(key, down)

    def post_mouse(self, kind, button, point, delta=(0, 0)):
        """kind: move, down, up or drag; button: 0 left, 1 right, 2 middle."""
        Q = self.target.Q
        types = {
            ("move", 0): Q.kCGEventMouseMoved,
            ("down", 0): Q.kCGEventLeftMouseDown,
            ("up", 0): Q.kCGEventLeftMouseUp,
            ("drag", 0): Q.kCGEventLeftMouseDragged,
            ("down", 1): Q.kCGEventRightMouseDown,
            ("up", 1): Q.kCGEventRightMouseUp,
            ("drag", 1): Q.kCGEventRightMouseDragged,
            ("down", 2): Q.kCGEventOtherMouseDown,
            ("up", 2): Q.kCGEventOtherMouseUp,
            ("drag", 2): Q.kCGEventOtherMouseDragged,
        }
        event_type = types[(kind, 0 if kind == "move" else button)]
        event = Q.CGEventCreateMouseEvent(None, event_type, point, button)
        if event is None:
            raise RuntimeError("Could not construct a mouse event")
        Q.CGEventSetIntegerValueField(event, Q.kCGMouseEventDeltaX, int(delta[0]))
        Q.CGEventSetIntegerValueField(event, Q.kCGMouseEventDeltaY, int(delta[1]))
        if button == 2:
            Q.CGEventSetIntegerValueField(event, Q.kCGMouseEventButtonNumber, 2)
        Q.CGEventPost(Q.kCGHIDEventTap, event)

    def post_wheel(self, notches, point):
        Q = self.target.Q
        event = Q.CGEventCreateScrollWheelEvent(None, Q.kCGScrollEventUnitLine, 1, int(notches))
        if event is None:
            raise RuntimeError("Could not construct a scroll event")
        Q.CGEventSetLocation(event, point)
        Q.CGEventPost(Q.kCGHIDEventTap, event)

    # --- guards ----------------------------------------------------------------------------
    def check(self, deadline):
        if self.permit is not None:
            reason = self.permit()
            if reason:
                raise RuntimeError(reason)
        else:
            if not self.target.focused() or not self.target.unchanged():
                raise RuntimeError("Game window lost focus or moved")
            if self.target.text_input_focused():
                raise RuntimeError("Text input focused")
        if time.perf_counter() >= deadline:
            raise RuntimeError("Trial deadline reached")

    def inside(self, x, y):
        x0, x1, y0, y1 = self.bounds
        ox, oy, w, h = self.origin[0], self.origin[1], self.crop[2], self.crop[3]
        return ox + x0 * w <= x <= ox + x1 * w and oy + y0 * h <= y <= oy + y1 * h

    def clamp(self, x, y):
        x0, x1, y0, y1 = self.bounds
        ox, oy, w, h = self.origin[0], self.origin[1], self.crop[2], self.crop[3]
        return (min(max(x, ox + x0 * w), ox + x1 * w), min(max(y, oy + y0 * h), oy + y1 * h))

    # --- state changes ---------------------------------------------------------------------
    def _release_all(self):
        for key in sorted(self.keys_down):
            self.post_key(key, False)
        for name in sorted(self.buttons_down, key=MOUSE.get):
            self.post_mouse("up", MOUSE[name], self.point)
        self.keys_down, self.buttons_down = set(), set()

    def release(self):
        with self.lock:
            self._release_all()

    def _arm_watchdog(self):
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(self.watchdog, self.release)
        self.timer.daemon = True
        self.timer.start()

    def settle(self):
        """Nothing to wait for: controls persist between steps."""

    def finish(self):
        if self.timer:
            self.timer.cancel()
            self.timer = None
        self.release()

    def apply(self, action, deadline):
        start = time.perf_counter()
        with self.lock:
            try:
                self.check(deadline)
                first = None

                def posted():
                    nonlocal first
                    if first is None:
                        first = time.perf_counter()

                wanted = set(action["buttons"])
                keys = {k for k in wanted if k in self.keys}
                buttons = {b for b in wanted if b in MOUSE}
                if action.get("planner_click"):
                    # A planner click is always a fresh left press at its target.
                    if "mouse_left" in self.buttons_down:
                        self.post_mouse("up", 0, self.point)
                        self.buttons_down.discard("mouse_left")
                    buttons.add("mouse_left")
                onset = buttons - self.buttons_down
                if self.pointer and onset and action.get("pointer_xy") is not None:
                    px, py = action["pointer_xy"]
                    self.point = self.clamp(
                        self.origin[0] + px * self.crop[2], self.origin[1] + py * self.crop[3]
                    )
                    self.post_mouse("move", 0, self.point)
                    posted()
                    action["pointer_applied"] = [
                        (self.point[0] - self.origin[0]) / self.crop[2],
                        (self.point[1] - self.origin[1]) / self.crop[3],
                    ]
                for key in sorted(self.keys_down - keys):
                    self.post_key(key, False)
                for key in sorted(keys - self.keys_down):
                    self.post_key(key, True)
                    posted()
                for name in sorted(self.buttons_down - buttons, key=MOUSE.get):
                    self.post_mouse("up", MOUSE[name], self.point)
                for name in sorted(onset, key=MOUSE.get):
                    self.post_mouse("down", MOUSE[name], self.point)
                    posted()
                self.keys_down, self.buttons_down = keys, buttons
                dx, dy = (round(v * 512) for v in action["mouse_delta"])
                if dx or dy:
                    target = (self.point[0] + dx, self.point[1] + dy)
                    if self.buttons_down and not self.inside(*target):
                        # Clutch: lift, re-center and press again, then continue the drag.
                        for name in sorted(self.buttons_down, key=MOUSE.get):
                            self.post_mouse("up", MOUSE[name], self.point)
                        self.point = self.center
                        self.post_mouse("move", 0, self.point)
                        for name in sorted(self.buttons_down, key=MOUSE.get):
                            self.post_mouse("down", MOUSE[name], self.point)
                        self.clutches += 1
                        target = (self.point[0] + dx, self.point[1] + dy)
                    nx, ny = self.clamp(*target)
                    held = sorted(self.buttons_down, key=MOUSE.get)
                    kind, button = ("drag", MOUSE[held[0]]) if held else ("move", 0)
                    self.post_mouse(kind, button, (nx, ny), (dx, dy))
                    posted()
                    self.point = (nx, ny)
                for name, notches in WHEEL.items():
                    if name in wanted:
                        self.post_wheel(notches, self.point)
                        posted()
                action["held"] = sorted(self.keys_down | self.buttons_down)
                self._arm_watchdog()
                return {
                    "dispatch_start": start,
                    "first_event_posted": first,
                    "pulse_wait_ms": 0.0,
                    "dispatch_guard_ms": 0.0,
                }
            except BaseException:
                self._release_all()
                raise
