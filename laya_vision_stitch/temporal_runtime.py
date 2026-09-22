"""One-checkpoint streaming policy. Returns proposed controls; never posts events."""

import argparse
import json
import time

import mlx.core as mx
import numpy as np

from .temporal_adapter import spatial_pool
from .trainable_model import TrainableRuntime
from .visual_action_adapter import encode_action_history


def frame_inputs(runtime, row):
    """Only current pixels and goal enter frozen Qwen/Laya. No labels or future frames."""
    if len(row["frames"]) != 1 or row["frames"][0]["age_seconds"] != 0:
        raise ValueError("Streaming input requires exactly one current screenshot")
    patches, coords = runtime.features({"frames": row["frames"]})
    prompt = {k: row[k] for k in ("goal", "controls", "choices") if k in row}
    prepared = runtime.prepare({**prompt, "previous_actions": []})
    context, _ = runtime.module.action_context(patches, coords, *prepared)
    # CLS and mean visual-slot context retain goal conditioning without long token caches.
    start = prepared[2]
    language = mx.stack(
        [
            context[:, 0],
            context[:, start : start + runtime.module.policy_config.visual_slots].mean(1),
        ],
        1,
    )[0]
    return spatial_pool(patches, coords), language.astype(mx.float32)


def decode(output, config):
    buttons = np.asarray(output["buttons"])
    camera = np.asarray(output["camera"]).argmax(-1)
    return [
        {
            "buttons": [b for b, x in zip(config.buttons, v, strict=True) if x >= 0],
            "mouse_delta": [config.mouse_bins[int(c)] for c in m],
            "duration_seconds": 0.05,
        }
        for v, m in zip(
            buttons.reshape(-1, len(config.buttons)), camera.reshape(-1, 2), strict=True
        )
    ]


class TemporalRuntime:
    """A runtime owns one session. Reset on episode/goal change or a dropped-frame gap.

    State is detached after each call. Separate sessions require separate instances.
    Previous controls must be the actually applied action, never the next target.
    """

    def __init__(self, runtime):
        if runtime.module.policy_config.temporal_adapter == "none":
            raise ValueError("Not a temporal checkpoint")
        self.runtime = runtime
        self.reset()

    @classmethod
    def load(cls, bundle):
        return cls(TrainableRuntime.load(bundle))

    def reset(self):
        self.state = None
        self.session = None
        self.prompt_context = None
        self.timestamp = None

    def predict(self, row, *, session_id, timestamp_seconds, reset=False):
        if not isinstance(session_id, str) or not session_id or not np.isfinite(timestamp_seconds):
            raise ValueError("A session ID and finite timestamp are required")
        if reset:
            self.reset()
        if (
            self.timestamp is not None
            and timestamp_seconds <= self.timestamp
            and session_id == self.session
        ):
            raise ValueError("Screenshots must arrive in increasing timestamp order")
        prompt_context = (
            row["goal"],
            row.get("controls", ""),
            json.dumps(row.get("choices"), sort_keys=True),
        )
        if (
            session_id != self.session
            or prompt_context != self.prompt_context
            or (self.timestamp is not None and timestamp_seconds - self.timestamp > 0.1)
        ):
            self.state = None
        did_reset = self.state is None
        started = time.perf_counter()
        visual, language = frame_inputs(self.runtime, row)
        history = encode_action_history(row, self.runtime.module.policy_config)
        output, state = self.runtime.module.temporal_actions(
            visual[None, None], language[None, None], history[:, None], self.state
        )
        mx.eval(output, state)
        prediction = decode(output, self.runtime.module.policy_config)[0]
        # mlx.utils.tree_map also handles nested recurrent and attention states.
        from mlx.utils import tree_map

        self.state = tree_map(mx.stop_gradient, state)
        self.session, self.prompt_context, self.timestamp = (
            session_id,
            prompt_context,
            timestamp_seconds,
        )
        return {
            **prediction,
            "image_to_outputs_ms": (time.perf_counter() - started) * 1000,
            "state_reset": did_reset,
            "input_events_sent": 0,
            "deployment_eligible": False,
            "supervised_outputs": ["buttons", "relative_mouse"],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument(
        "--requests",
        required=True,
        help="JSONL with current image, goal, session_id, timestamp_seconds",
    )
    args = parser.parse_args()
    runtime = TemporalRuntime.load(args.bundle)
    with open(args.requests) as source:
        for line in source:
            row = json.loads(line)
            print(
                json.dumps(
                    runtime.predict(
                        row,
                        session_id=row["session_id"],
                        timestamp_seconds=row["timestamp_seconds"],
                        reset=row.get("reset", False),
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
