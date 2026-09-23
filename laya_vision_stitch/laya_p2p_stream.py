"""Streaming stitched model with explicit feedback of actually dispatched controls."""

import time

import mlx.core as mx
import numpy as np

from .laya_p2p import LayaP2PRuntime
from .p2p_adaptation import encode_action
from .p2p_pretrained_policy import KEY_NAMES, physical_action
from .p2p_pretrained_vision import preprocess


class LayaP2PStream:
    def __init__(self, runtime):
        self.runtime = runtime
        self.reset()

    @classmethod
    def load(cls, bundle):
        return cls(LayaP2PRuntime.load(bundle))

    def reset(self):
        self.caches, self.pending, self.position = None, None, 0
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
        if (
            session_id != self.session
            or row["goal"] != self.goal
            or (self.timestamp is not None and timestamp_seconds - self.timestamp > 0.1)
            or not previous
        ):
            # Unknown/unsent actions cannot be stored as if our proposal was executed.
            self.caches, self.pending, self.position = None, None, 0
        did_reset = self.pending is None
        start = time.perf_counter()
        policy = self.runtime.model.policy
        if self.pending is not None:
            actual = encode_action(previous[0])
            key_names = self.runtime.metadata.get("key_names", KEY_NAMES)
            if any(k >= len(key_names) for k in actual[:4]):
                raise ValueError("Applied action is outside this checkpoint's vocabulary")
            _, self.caches = policy.context(
                self.pending, mx.array([actual], mx.int32), self.caches, self.position
            )
            self.position += 12
        goal = self.runtime.model.bridge(
            self.runtime.model.goal_features(*self.runtime.prepare_goal(row["goal"]))
        )
        spatial, image = policy.vision(mx.array(preprocess(row["frames"][0]["image"])))
        prefix = policy.prefix(image, goal, spatial)
        context, _ = policy.context(prefix, caches=self.caches, position=self.position)
        tokens, logits = policy.decode(context, temperature=1.0)
        mx.eval(tokens, logits, prefix, self.caches)
        if not all(np.isfinite(np.asarray(logit.astype(mx.float32))).all() for logit in logits):
            self.reset()
            raise FloatingPointError("Nonfinite model logits")
        action = physical_action(
            tokens.tolist()[0], key_names=self.runtime.metadata.get("key_names", KEY_NAMES)
        )
        self.pending = prefix
        self.session, self.goal, self.timestamp = session_id, row["goal"], timestamp_seconds
        return {
            **action,
            "image_to_outputs_ms": 1000 * (time.perf_counter() - start),
            "state_reset": did_reset,
            "input_events_sent": 0,
            "deployment_eligible": False,
            "supervised_outputs": ["buttons", "relative_mouse"],
        }
