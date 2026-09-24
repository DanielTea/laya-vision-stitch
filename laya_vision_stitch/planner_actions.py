"""High-level actions the slow planner takes on its target; the fast controller does the rest.

Once Molmo has picked a target, the runtime selects it with a click, approaches it and uses
the skill Molmo pointed to on the skill bar. These rely on common control conventions
rather than learned behavior, stated here so they are easy to audit:

- the controlled character stays near the screen center (third-person or top-down camera);
- W/A/S/D move up/left/down/right on screen;
- clicking a skill-bar button uses that skill on the selected target; skill bars sit at
  the bottom of the screen, and the leftmost skill is the primary one.

No game names, key bindings or screen layouts are hard-coded; the skill button's position
comes from Molmo on the current screen.
"""

import math
import time

MOVEMENT = frozenset({"w", "a", "s", "d", "up", "down", "left", "right"})
# Planner clicks stay inside this screen area (x0, x1, y0, y1), clear of window edges and
# the top menu band where browser games keep settings and logout buttons.
SAFE_AREA = (0.03, 0.97, 0.1, 0.97)
# Two skill answers must agree within this distance before the button is used.
SKILL_AGREEMENT = 0.04
# A key is held when the direction has a component beyond sin(22.5 deg): eight sectors.
SECTOR = math.sin(math.radians(22.5))


def direction_keys(dx, dy):
    """W/A/S/D for the screen direction (dx, dy), y pointing down; empty at zero."""
    norm = math.hypot(dx, dy)
    if norm == 0:
        return []
    ux, uy = dx / norm, dy / norm
    keys = []
    if uy < -SECTOR:
        keys.append("w")
    if uy > SECTOR:
        keys.append("s")
    if ux < -SECTOR:
        keys.append("a")
    if ux > SECTOR:
        keys.append("d")
    return keys


def in_safe_area(xy):
    x0, x1, y0, y1 = SAFE_AREA
    return x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1


class TargetActions:
    """Select, approach and act on a tracked target.

    near: distance from the avatar (screen fraction) at which the character stops moving
    and uses the skill. reselect: seconds between repeated selection clicks while
    approaching (a missed click would otherwise go unnoticed). act_every: seconds between
    skill clicks.
    """

    def __init__(
        self, avatar=(0.5, 0.5), near=0.12, reselect=4.0, act_every=0.8, clock=time.perf_counter
    ):
        self.avatar, self.near, self.reselect, self.act_every = (
            avatar,
            float(near),
            float(reselect),
            float(act_every),
        )
        self.clock = clock
        self.skill_xy, self.skill_candidate = None, None
        self.reset()

    def skill_answer(self, xy):
        """Molmo's skill-bar point; returns True once two answers in the safe area agree."""
        if xy is None or not in_safe_area(xy):
            self.skill_candidate = None
            return False
        previous, self.skill_candidate = self.skill_candidate, list(xy)
        if previous is not None and math.dist(previous, xy) <= SKILL_AGREEMENT:
            self.skill_xy = [(a + b) / 2 for a, b in zip(previous, xy, strict=True)]
            return True
        return False

    def reset(self):
        self.last_select = self.last_act = None

    def selected(self):
        """Record the planner's selection click on a newly acquired target."""
        self.last_select = self.clock()

    def step(self, target_xy):
        """Planner action for this step: {'hold': keys} and/or {'click': {...}}; {} if none."""
        if target_xy is None or self.last_select is None:
            return {}
        now = self.clock()
        dx, dy = target_xy[0] - self.avatar[0], target_xy[1] - self.avatar[1]
        distance = math.hypot(dx, dy)
        if distance > self.near:
            if now - self.last_select >= self.reselect and in_safe_area(target_xy):
                self.last_select = now
                return {
                    "hold": direction_keys(dx, dy),
                    "click": {"xy": list(target_xy), "kind": "select"},
                }
            return {"hold": direction_keys(dx, dy)}
        if self.skill_xy is not None and (
            self.last_act is None or now - self.last_act >= self.act_every
        ):
            self.last_act = now
            return {"hold": [], "click": {"xy": list(self.skill_xy), "kind": "skill"}}
        return {"hold": []}
