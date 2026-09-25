"""Streaming stitched model with explicit feedback of actually dispatched controls.

Defaults reproduce the original runtime. Options are explicit and transport-level only:

- Goal vectors are cached per goal string (at most 16). Laya and the bridge are frozen,
  so a cached vector equals a recomputed one; `cache_goal=False` recomputes every frame.
- `max_gap_seconds` (default 0.1) is the largest screenshot gap that keeps memory.
  Larger values keep memory across slow frames; the released policy was trained at a
  fixed cadence and receives no elapsed time unless a time-gap adapter is installed.
- If the policy has a `time_gap_adapter` attribute (dt seconds [B] -> residual [B, 1024]),
  the time since the previous frame in memory is added to the image token before
  `policy.prefix`. The first frame of a memory episode uses the nominal 50 ms step, as
  in training. Without the attribute nothing is added.
- `compile=True` compiles vision+prefix and action decoding, and the policy context only
  for the steady-state full 200-frame cache. The growing cache changes shape every frame;
  it runs eagerly instead of tracing 200 graphs. Graphs are traced and parity-checked
  against the eager path at construction; any failure falls back to eager execution
  and is reported in `compile_status`.
- `commit(action)` feeds back the dispatched action before the next screenshot, moving
  the memory update off the screenshot-to-dispatch path. Memory contents are unchanged.
"""

import secrets
import time
from collections import OrderedDict

import mlx.core as mx
import numpy as np

from .laya_p2p import LayaP2PRuntime
from .p2p_adaptation import encode_action
from .p2p_pretrained_policy import KEY_NAMES, STEP_TOKENS, physical_action, policy_mask
from .p2p_pretrained_vision import preprocess

MEMORY_TOKENS = 200 * STEP_TOKENS
NOMINAL_GAP_SECONDS = 0.05
GOAL_CACHE_SIZE = 16


def encode_frame(policy, pixels, goal, dt=None, target=None):
    """Vision, optional elapsed-time residual on the image token, then the policy prefix.

    `target` [1, 2] is a screen point to act on (NaN: none) for a target-encoder checkpoint.
    """
    spatial, image = policy.vision(pixels)
    if dt is not None:
        residual = policy.time_gap_adapter(dt)
        if residual.shape != image.shape:
            raise ValueError("Time-gap adapter must return one 1024D residual per frame")
        image = image + residual.astype(image.dtype)
    if target is None:
        return policy.prefix(image, goal, spatial)
    return policy.prefix(image, goal, spatial, target)


def full_cache_context(policy, prefix, actions, caches, offset):
    """`OpenP2PPolicy.context` for a full 200-frame cache, with an array RoPE offset.

    The released mask depends only on positions relative to the retained cache, so one
    compiled graph serves every later frame. Parity with the eager method is checked
    before use; this function is not used for partially filled caches.
    """
    if len(caches) != len(policy.policy.layers) or any(
        k.shape[2] != MEMORY_TOKENS for k, _ in caches
    ):
        raise ValueError("Compiled context requires a full 200-frame cache")
    caches = [(k[:, :, STEP_TOKENS:], v[:, :, STEP_TOKENS:]) for k, v in caches]
    previous = MEMORY_TOKENS - STEP_TOKENS
    target = (
        mx.zeros((prefix.shape[0], 8, 1024), prefix.dtype)
        if actions is None
        else policy.action_embeddings(actions) + policy.action_position
    )
    x = mx.concatenate([prefix, target], 1)
    q, k = mx.arange(previous, MEMORY_TOKENS), mx.arange(MEMORY_TOKENS)
    output, updated = policy.policy(x, offset, caches, policy_mask(q, k))
    return output[:, 3:4], updated


def _relative_error(reference, value):
    reference, value = reference.astype(mx.float32), value.astype(mx.float32)
    return mx.abs(reference - value).max().item() / max(1.0, mx.abs(reference).max().item())


class LayaP2PStream:
    def __init__(
        self,
        runtime,
        *,
        max_gap_seconds=0.1,
        temperature=1.0,
        cache_goal=True,
        compile=False,
        planner=None,
        track_every=8,
        max_target_age=30.0,
        plan_interval=4.0,
        avatar_radius=0.08,
        avoid_avatar=None,
        acquire_window=0.3,
        act=False,
        skill_refresh=60.0,
    ):
        if not np.isfinite(max_gap_seconds) or not 0 < max_gap_seconds <= 10:
            raise ValueError("Maximum memory gap must be in (0, 10] seconds")
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError("Temperature must be finite and nonnegative")
        self.runtime = runtime
        self.max_gap_seconds, self.temperature = float(max_gap_seconds), float(temperature)
        self.cache_goal, self.goals = bool(cache_goal), OrderedDict()
        policy = runtime.model.policy
        self.time_gap = callable(getattr(policy, "time_gap_adapter", None))
        self.target_input = hasattr(policy, "target_encoder")
        if runtime.metadata.get("time_gap_adapter") and not self.time_gap:
            raise ValueError("Checkpoint declares a time-gap adapter that is not installed")
        self.compiled, self.compile_status, self.compile_parity = None, "disabled", None
        # Optional slow planner (e.g. molmo_planner.AsyncPlanner) and RADIO target tracker.
        self.planner, self.track_every, self.max_target_age = (
            planner,
            int(track_every),
            float(max_target_age),
        )
        self.tracker, self.target_time, self.steps = None, None, 0
        self.plan_interval, self.avatar_radius, self.last_submit = (
            float(plan_interval),
            float(avatar_radius),
            None,
        )
        # Clicks predicted on the avatar move off it; on by default with a planner.
        self.avoid_avatar = planner is not None if avoid_avatar is None else bool(avoid_avatar)
        self.acquire_window = float(acquire_window)
        # Optional planner actions on the target: select, approach, use the skill Molmo found.
        self.actions, self.skill_refresh, self.skill_time = None, float(skill_refresh), None
        if act:
            if planner is None:
                raise ValueError("Planner actions need a planner")
            from .planner_actions import TargetActions

            self.actions = TargetActions(avatar=(0.5, 0.5))
        if planner is not None:
            from .target_tracker import FeatureTracker

            encoder = getattr(policy, "pointer_encoder", None)
            if encoder is None:
                raise ValueError("Planner tracking needs a checkpoint with a RADIO pointer encoder")
            self.tracker = FeatureTracker(encoder)
        self.last_tokens = self.last_logits = None
        if compile:
            self._compile()
        self.reset()

    @classmethod
    def load(cls, bundle, **options):
        return cls(LayaP2PRuntime.load(bundle), **options)

    def reset(self):
        self.caches, self.pending, self.position, self.committed = None, None, 0, False
        self.session, self.goal, self.timestamp = None, None, None

    def predict(self, row, *, session_id, timestamp_seconds, reset=False):
        if not isinstance(session_id, str) or not session_id or not np.isfinite(timestamp_seconds):
            raise ValueError("A session ID and finite timestamp are required")
        if len(row["frames"]) != 1 or row["frames"][0]["age_seconds"] != 0:
            raise ValueError("One current screenshot is required")
        if reset:
            self.reset()
        if (
            self.timestamp is not None
            and session_id == self.session
            and timestamp_seconds <= self.timestamp
        ):
            raise ValueError("Screenshot timestamps must increase")
        previous = row.get("previous_actions", [])
        if len(previous) > 1:
            raise ValueError("Provide only the immediately preceding applied action")
        if previous and self.committed:
            raise ValueError("The preceding action was already committed")
        gap = None if self.timestamp is None else timestamp_seconds - self.timestamp
        if (
            session_id != self.session
            or row["goal"] != self.goal
            or (gap is not None and gap > self.max_gap_seconds)
            or not (previous or self.committed)
        ):
            # Unknown/unsent actions cannot be stored as if our proposal was executed.
            self.caches, self.pending, self.position, self.committed = None, None, 0, False
        did_reset = self.pending is None and not self.committed
        start = time.perf_counter()
        try:
            if self.pending is not None:
                self._commit(previous[0])
            self.committed = False
            goal = self._goal_vector(row["goal"])
            dt = NOMINAL_GAP_SECONDS if did_reset else gap
            pixels = mx.array(preprocess(row["frames"][0]["image"]))
            # The planner target (if any) is updated first so the policy acts on this frame's.
            plan = self._plan(row["frames"][0]["image"], row["goal"])
            target = self._target()
            prefix = self._encode(pixels, goal, dt, target)
            context = self._propose(prefix)
            tokens, logits = self._decode(context)
            mx.eval(tokens, logits, prefix, self.caches)
            pointer = self._pointer(pixels, context, tokens, row["frames"][0]["image"])
            if not all(np.isfinite(np.asarray(logit.astype(mx.float32))).all() for logit in logits):
                raise FloatingPointError("Nonfinite model logits")
        except BaseException:
            self.reset()
            raise
        action = physical_action(
            tokens.tolist()[0], key_names=self.runtime.metadata.get("key_names", KEY_NAMES)
        )
        self.last_tokens, self.last_logits = tokens, logits
        self.pending = prefix
        self.session, self.goal, self.timestamp = session_id, row["goal"], timestamp_seconds
        result = {
            **action,
            "image_to_outputs_ms": 1000 * (time.perf_counter() - start),
            "state_reset": did_reset,
            "input_events_sent": 0,
            "deployment_eligible": False,
            "supervised_outputs": ["buttons", "relative_mouse"],
        }
        if self.time_gap:
            result["time_gap_seconds"] = dt
        if self.target_input:
            result["target_input"] = (
                None if np.isnan(np.asarray(target)).any() else np.asarray(target)[0].tolist()
            )
        if plan:
            result["planner"] = plan
            if plan.get("click"):
                result["planner_click"] = {**plan["click"], "kind": "select"}
            elif plan.get("action_click"):
                result["planner_click"] = {**plan["action_click"], "button": "mouse_left"}
            if plan.get("hold") is not None:
                result["planner_hold"] = plan["hold"]
        if pointer is not None:
            result["pointer_source"] = pointer[1]
            if pointer[0] is not None:
                result["pointer_xy"] = pointer[0]
                result["supervised_outputs"] = [*result["supervised_outputs"], "absolute_pointer"]
        return result

    def _target(self):
        """Tracked planner target for the target encoder, NaN when there is none."""
        live = self.tracker is not None and self.target_time is not None
        return mx.array([self.tracker.xy if live else [np.nan, np.nan]], mx.float32)

    def _pointer(self, pixels, context, tokens, image):
        """Absolute click position when a mouse button is proposed and a head is installed.

        Features are computed only on those steps, leaving compiled paths unchanged. A
        RADIO pointer reads the full screenshot at 384x224; otherwise the P2P grid is used.
        """
        policy = self.runtime.model.policy
        head = getattr(policy, "pointer_head", None)
        if not bool(mx.any(tokens[:, 4:6] != 0).item()):
            return None
        if self.tracker is not None and self.target_time is not None:
            if time.perf_counter() - self.target_time <= self.max_target_age:
                xy, _ = self.tracker.track(image)
                if xy is not None:
                    return xy, "planner"
                self.target_time = None  # lost: the next step asks the planner again
        if head is None:
            return None
        xy, source = self._pointer_head(pixels, context, image)
        if self.avoid_avatar and (xy[0] - 0.5) ** 2 + (xy[1] - 0.5) ** 2 <= self.avatar_radius**2:
            # Same camera assumption as the planner: the center is the avatar. The press moves
            # just off it (the cursor may already rest at the center, so withholding the
            # move would still click the avatar).
            from .planner_actions import clear_of_avatar

            return clear_of_avatar(xy), "avatar_zone_shifted"
        return xy, source

    def _pointer_head(self, pixels, context, image):
        policy = self.runtime.model.policy
        head = policy.pointer_head
        encoder = getattr(policy, "pointer_encoder", None)
        if encoder is not None:
            from PIL import Image

            frame = (
                np.asarray(image.convert("RGB").resize((384, 224), Image.BICUBIC), np.float32) / 255
            )
            patches, _ = encoder(mx.array(frame[None]).astype(encoder.input_dtype))
            xy = head.predict(patches.reshape(1, 14, 24, -1))
        else:
            spatial, _ = policy.vision(pixels)
            xy = head.predict(spatial, context.reshape(1, -1))
        mx.eval(xy)
        return [float(v) for v in np.asarray(xy)[0]], "pointer_head"

    def _plan(self, image, goal):
        """Collect finished planner answers, keep the target tracked, submit when idle."""
        if self.planner is None:
            return None
        info = {}
        result = self.planner.poll()
        if result is not None and result.get("points") is None and not result.get("error"):
            # Skill-only answer: the target being followed is unaffected.
            info = {"planner_seconds": result["planner_seconds"]}
            if self.actions is not None and "skill" in result:
                info["skill"] = result["skill"]
                if self.actions.skill_answer(result["skill"]):
                    self.skill_time = time.perf_counter()
                    info["skill_confirmed"] = self.actions.skill_xy
            result = None
        elif result is not None:
            info = {k: result[k] for k in ("phrase", "points", "planner_seconds", "error")}
            # A third-person camera keeps the controlled character at the screen center, so
            # candidates there are treated as the avatar, never as a click target.
            candidates = [
                q
                for q in result["points"]
                if (q[0] - 0.5) ** 2 + (q[1] - 0.5) ** 2 > self.avatar_radius**2
            ]
            if candidates and result["image"] is not None:
                # The nearest remaining candidate to the camera focus is the nearest target.
                xy = min(candidates, key=lambda q: (q[0] - 0.5) ** 2 + (q[1] - 0.5) ** 2)
                self.tracker.set_target(result["image"], xy)
                self.target_time = time.perf_counter()
                info["target"] = xy
                # The answer is seconds old: follow the target onto the current frame with a
                # wider search, then issue the planner's high-level action, one click on it.
                current, similarity = self.tracker.track(image, window=self.acquire_window)
                if current is not None:
                    from .planner_actions import in_safe_area

                    if in_safe_area(current):
                        info["click"] = {
                            "xy": current,
                            "button": "mouse_left",
                            "similarity": similarity,
                        }
                        if self.actions is not None:
                            self.actions.selected()
                else:
                    self.target_time = None
            else:
                self.target_time = None
        elif (
            self.target_time is not None
            and time.perf_counter() - self.target_time > self.max_target_age
        ):
            self.target_time = None  # stale target: request a fresh plan
            info = {"expired": True}
        elif self.target_time is not None and self.steps % self.track_every == 0:
            xy, similarity = self.tracker.track(image)
            info = {"tracked": xy, "similarity": similarity}
            if xy is None:
                self.target_time = None
        self.steps += 1
        if self.actions is not None:
            if self.target_time is None:
                self.actions.reset()
            elif not info.get("click"):
                act = self.actions.step(self.tracker.xy)
                if act:
                    info["hold"] = act["hold"]
                    if act.get("click"):
                        info["action_click"] = act["click"]
        # Plan only when no target is being followed, and not more often than plan_interval:
        # Molmo shares the GPU with the fast controller.
        now = time.perf_counter()
        due = self.last_submit is None or now - self.last_submit >= self.plan_interval
        if self.target_time is None and due and self.planner.submit(image, goal) is not None:
            self.last_submit = now
            info["submitted"] = True
        elif self.target_time is not None and self.actions is not None:
            # While a target is followed the planner is idle: use it to find the skill button.
            stale = self.skill_time is None or now - self.skill_time >= self.skill_refresh
            if stale and self.planner.submit(image, goal, skill=True, point=False) is not None:
                info["submitted_skill"] = True
        return info

    def commit(self, action):
        """Feed back the dispatched action now instead of with the next screenshot.

        Memory equals passing it as `previous_actions` to the next `predict`, which must
        then omit `previous_actions`. A new session, goal or over-long gap still resets.
        """
        if self.pending is None:
            raise ValueError("No proposal is awaiting feedback")
        try:
            self._commit(action)
            mx.eval(self.caches)
        except BaseException:
            self.reset()
            raise
        self.committed = True

    def _commit(self, action):
        key_names = self.runtime.metadata.get("key_names", KEY_NAMES)
        actual = encode_action(action, key_names)
        if any(k >= len(key_names) for k in actual[:4]):
            raise ValueError("Applied action is outside this checkpoint's vocabulary")
        actions = mx.array([actual], mx.int32)
        if self._steady():
            self.caches = self.compiled["commit"](
                self.pending, actions, self.caches, mx.array(self.position)
            )
        else:
            _, self.caches = self.runtime.model.policy.context(
                self.pending, actions, self.caches, self.position
            )
        self.position += STEP_TOKENS
        self.pending = None

    def _goal_vector(self, goal):
        if self.cache_goal and goal in self.goals:
            self.goals.move_to_end(goal)
            return self.goals[goal]
        model = self.runtime.model
        vector = model.bridge(model.goal_features(*self.runtime.prepare_goal(goal)))
        if self.cache_goal:
            mx.eval(vector)
            self.goals[goal] = vector
            if len(self.goals) > GOAL_CACHE_SIZE:
                self.goals.popitem(last=False)
        return vector

    def _encode(self, pixels, goal, dt, target=None):
        dt = mx.array([dt], mx.float32) if self.time_gap else None
        target = target if self.target_input else None
        if self.compiled is not None:
            extra = ([] if dt is None else [dt]) + ([] if target is None else [target])
            return self.compiled["encode"](pixels, goal, *extra)
        return encode_frame(self.runtime.model.policy, pixels, goal, dt, target)

    def _propose(self, prefix):
        if self._steady():
            return self.compiled["propose"](prefix, self.caches, mx.array(self.position))
        context, _ = self.runtime.model.policy.context(
            prefix, caches=self.caches, position=self.position
        )
        return context

    def _decode(self, context):
        if self.compiled is not None:
            return self.compiled["decode"](context)
        return self.runtime.model.policy.decode(context, temperature=self.temperature)

    def _steady(self):
        return (
            self.compiled is not None
            and self.caches is not None
            and self.caches[0][0].shape[2] == MEMORY_TOKENS
        )

    def _compile(self):
        policy, temperature = self.runtime.model.policy, self.temperature
        state = [mx.random.state]
        try:
            if self.time_gap and self.target_input:
                encode = mx.compile(lambda px, goal, dt, tg: encode_frame(policy, px, goal, dt, tg))
            elif self.time_gap:
                encode = mx.compile(lambda px, goal, dt: encode_frame(policy, px, goal, dt))
            elif self.target_input:
                encode = mx.compile(lambda px, goal, tg: encode_frame(policy, px, goal, None, tg))
            else:
                encode = mx.compile(lambda px, goal: encode_frame(policy, px, goal))
            if temperature > 0:
                decode = mx.compile(
                    lambda c: policy.decode(c, temperature=temperature),
                    inputs=state,
                    outputs=state,
                )
            else:
                decode = mx.compile(lambda c: policy.decode(c, temperature=0.0))
            compiled = {
                "encode": encode,
                "decode": decode,
                "commit": mx.compile(lambda p, a, c, o: full_cache_context(policy, p, a, c, o)[1]),
                "propose": mx.compile(lambda p, c, o: full_cache_context(policy, p, None, c, o)[0]),
            }
            self.compile_parity = self._check_compiled(policy, compiled)
        except Exception as error:  # Any tracing or parity failure keeps the eager path.
            self.compiled = None
            self.compile_status = f"fallback: {type(error).__name__}: {error}"
            return
        self.compiled, self.compile_status = compiled, "compiled"

    def _check_compiled(self, policy, compiled):
        """Trace every graph at its live shape and compare with eager outputs."""
        rng = np.random.default_rng(0)
        pixels = mx.array(rng.random((1, 192, 192, 3), dtype=np.float32))
        goal = mx.array(rng.standard_normal((1, 768), dtype=np.float32))
        dt = [mx.array([3 * NOMINAL_GAP_SECONDS], mx.float32)] if self.time_gap else []
        target = [mx.array([[0.3, 0.6]], mx.float32)] if self.target_input else []
        prefix = encode_frame(policy, pixels, goal, *(dt or [None]), *target)
        errors = {"encode": _relative_error(prefix, compiled["encode"](pixels, goal, *dt, *target))}
        heads = policy.policy.layers[0].attention.heads
        shape = (1, heads, MEMORY_TOKENS, 1024 // heads)
        caches = [
            tuple(
                mx.array(rng.standard_normal(shape, dtype=np.float32)).astype(prefix.dtype)
                for _ in range(2)
            )
            for _ in policy.policy.layers
        ]
        position, actions = MEMORY_TOKENS + 5 * STEP_TOKENS, mx.zeros((1, 8), mx.int32)
        _, eager_caches = policy.context(prefix, actions, caches, position)
        traced = compiled["commit"](prefix, actions, caches, mx.array(position))
        errors["commit"] = max(
            _relative_error(a, b)
            for eager, other in zip(eager_caches, traced, strict=True)
            for a, b in zip(eager, other, strict=True)
        )
        context, _ = policy.context(prefix, caches=caches, position=position)
        errors["propose"] = _relative_error(
            context, compiled["propose"](prefix, caches, mx.array(position))
        )
        seed = secrets.randbits(31)
        mx.random.seed(seed)
        eager_tokens, eager_logits = policy.decode(context, temperature=self.temperature)
        mx.random.seed(seed)
        tokens, logits = compiled["decode"](context)
        errors["decode"] = max(
            _relative_error(a, b) for a, b in zip(eager_logits, logits, strict=True)
        )
        if not mx.array_equal(eager_tokens, tokens).item():
            raise ValueError("Compiled decoding changed the selected tokens")
        # Do not leave a known RNG state behind after the seeded comparison.
        mx.random.seed(secrets.randbits(31))
        tolerance = 1e-4 if prefix.dtype == mx.float32 else 2e-2
        if max(errors.values()) > tolerance:
            raise ValueError(f"Compiled outputs differ from eager outputs: {errors}")
        return errors
