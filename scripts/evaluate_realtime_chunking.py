"""Offline simulation of asynchronous chunk execution: naive switching versus RTC.

Every `stride` steps a new chunk is requested from the context at that time; it arrives
`delay` steps later, while the previous chunk keeps executing. Real-time chunking
inpaints the new chunk to agree with actions that execute during the delay. Contexts are
teacher-forced from recorded history, so this measures chunk consistency and agreement,
not closed-loop gameplay.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

from laya_vision_stitch.flow_action_head import (
    BINARY,
    ChunkHead,
    realtime_sample,
    sample,
    vectors_to_actions,
)
from laya_vision_stitch.sequence_metrics import evaluate_actions, token_buttons, token_mouse


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def simulate(head, contexts, delay, stride, method, key):
    """Executed action vectors for one sequence of cached contexts."""
    n, horizon = len(contexts), head.horizon
    executed = np.zeros((n, head.output.weight.shape[0]), np.float32)
    active, origin, switches = None, 0, []
    for t in range(n):
        if t % stride == 0:
            key, sub = mx.random.split(key)
            c = mx.array(contexts[t : t + 1])
            if active is None or method == "naive" or delay == 0:
                chunk = np.asarray(sample(head, c, key=sub))[0]
            else:
                committed = np.zeros_like(active)
                shift = t - origin
                committed[: horizon - shift] = active[shift:]
                chunk = np.asarray(
                    realtime_sample(
                        head, c, mx.array(committed)[None], delay, key=sub, overlap=horizon - shift
                    )
                )[0]
            pending = (t + (delay if active is not None else 0), t, chunk)
        if pending is not None and t >= pending[0]:
            _, origin, active = pending
            pending = None
            switches.append(t)
        executed[t] = active[t - origin]
    return executed, switches


def smoothness(vectors, switches, sequences):
    buttons = vectors[:, :BINARY] > 0
    changes = np.any(buttons[1:] != buttons[:-1], 1)
    same_sequence = sequences[1:] == sequences[:-1]
    at_switch = np.zeros(len(vectors) - 1, bool)
    for s in switches:
        if 0 < s < len(vectors):
            at_switch[s - 1] = True
    mouse_jump = np.abs(np.diff(vectors[:, BINARY:], axis=0)).sum(1)
    flicker = np.any(buttons[2:] & ~buttons[1:-1] & buttons[:-2], 1)
    valid2 = same_sequence[1:] & same_sequence[:-1]
    return {
        "button_change_rate_at_switch": float(changes[at_switch & same_sequence].mean())
        if (at_switch & same_sequence).any()
        else None,
        "button_change_rate_elsewhere": float(changes[~at_switch & same_sequence].mean()),
        "mouse_jump_at_switch": float(mouse_jump[at_switch & same_sequence].mean())
        if (at_switch & same_sequence).any()
        else None,
        "mouse_jump_elsewhere": float(mouse_jump[~at_switch & same_sequence].mean()),
        "key_flicker_rate": float(flicker[valid2].mean()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head", type=Path, required=True, help="Trained flow chunk-head directory")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["validation", "test", "fresh_test"])
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--delays", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--repeats", type=int, default=2)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.head / "args.json").read_text())
    if config["kind"] != "flow":
        raise ValueError("Real-time chunking needs a flow-matching head")
    head = ChunkHead(horizon=config["horizon"], flow=True)
    head.load_weights(str(args.head / "head.safetensors"))
    report = {"head": str(args.head), "stride": args.stride, "splits": {}}
    for split in args.splits:
        if not (args.cache / f"{split}.npz").exists():
            continue
        rows = read(args.cache / f"{split}.jsonl")
        a = np.load(args.cache / f"{split}.npz")
        groups = defaultdict(list)
        for i, s in enumerate(a["sequence_index"]):
            groups[int(s)].append(i)
        truth, true_mouse = token_buttons(a["tokens"]), token_mouse(a["tokens"])
        result = {}
        for delay in [0, *args.delays]:
            for method in ["naive", "rtc"] if delay else ["no_delay"]:
                runs = []
                for r in range(args.repeats):
                    key = mx.random.key(1000 * delay + r)
                    executed = np.zeros((len(rows), head.output.weight.shape[0]), np.float32)
                    switches = []
                    for idx in groups.values():
                        idx = sorted(idx, key=lambda i: a["steps"][i])
                        key, sub = mx.random.split(key)
                        vec, sw = simulate(
                            head, a["contexts"][idx], delay, args.stride, method, sub
                        )
                        executed[idx] = vec
                        switches.extend(idx[s] for s in sw)
                    buttons, mouse = vectors_to_actions(executed)
                    metrics = evaluate_actions(rows, buttons, truth, mouse, true_mouse)
                    runs.append(
                        {
                            "macro": metrics["macro"],
                            "smoothness": smoothness(executed, switches, a["sequence_index"]),
                        }
                    )
                name = f"delay_{delay}_{method}"
                result[name] = runs
                summary = {
                    k: round(float(np.mean([x["macro"][k] for x in runs])), 4)
                    for k in ["button_f1", "onset_f1"]
                }
                smooth = {
                    k: round(
                        float(
                            np.mean(
                                [x["smoothness"][k] for x in runs if x["smoothness"][k] is not None]
                            )
                        ),
                        4,
                    )
                    for k in [
                        "button_change_rate_at_switch",
                        "mouse_jump_at_switch",
                        "key_flicker_rate",
                    ]
                }
                print(json.dumps({split: {name: {**summary, **smooth}}}), flush=True)
        report["splits"][split] = result
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
