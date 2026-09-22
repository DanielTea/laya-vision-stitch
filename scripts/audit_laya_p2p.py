"""Paired screenshot/goal ablations against recorded controls, without live inputs."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from laya_vision_stitch.laya_p2p import LayaP2PRuntime
from laya_vision_stitch.p2p_pretrained_policy import KEY_NAMES, MOUSE_NAMES
from laya_vision_stitch.p2p_pretrained_vision import preprocess


def encode_buttons(buttons):
    """Reject unsupported labels rather than quietly labelling them as idle."""
    unsupported = set(buttons) - set(KEY_NAMES) - set(MOUSE_NAMES)
    if unsupported:
        raise ValueError(f"Unsupported buttons: {sorted(unsupported)}")
    keys = sorted({KEY_NAMES.index(x) for x in buttons if x in KEY_NAMES})
    mouse = sorted({MOUSE_NAMES.index(x) for x in buttons if x in MOUSE_NAMES})
    if len(keys) > 4 or len(mouse) > 2:
        raise ValueError("Too many simultaneous buttons")
    return keys + [0] * (4 - len(keys)) + mouse + [0] * (2 - len(mouse)) + [11, 8]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-game", type=int, default=40)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = LayaP2PRuntime.load(args.bundle)
    grouped, excluded = defaultdict(list), Counter()
    for line in args.manifest.read_text().splitlines():
        row = json.loads(line)
        try:
            row["tokens"] = encode_buttons(row["action"]["buttons"])
        except ValueError as exc:
            excluded[str(exc)] += 1
            continue
        grouped[row["game"]].append(row)
    rng = np.random.default_rng(20260922)
    report = {
        "scope": "Offline single-frame control likelihood, teacher forcing within each action. No temporal history; no live success claims. Hordes labels are weak controller demonstrations. Public P2P games may overlap parent pretraining.",
        "bundle": str(args.bundle),
        "manifest": str(args.manifest),
        "input_events_sent": 0,
        "excluded": dict(excluded),
        "games": {},
    }
    output = []
    for game, pool in sorted(grouped.items()):
        rows = [
            pool[i] for i in rng.choice(len(pool), min(args.per_game, len(pool)), replace=False)
        ]
        images = [Image.open(r["frames"][-1]["image"]).convert("RGB") for r in rows]
        # Shift the permutation to avoid pairing any image with itself.
        shifted = np.roll(np.arange(len(rows)), 1)
        measurements = defaultdict(list)
        for i, (row, image) in enumerate(zip(rows, images, strict=True)):
            target = mx.array([row["tokens"]], mx.int32)
            goal = runtime.model.bridge(
                runtime.model.goal_features(*runtime.prepare_goal(row["goal"]))
            )
            opposite = runtime.model.bridge(
                runtime.model.goal_features(*runtime.prepare_goal("Stand still and do nothing."))
            )
            conditions = {
                "actual": (image, goal),
                "shuffled_image": (images[shifted[i]], goal),
                "blank_image": (Image.new("RGB", image.size), goal),
                "wait_goal": (image, opposite),
            }
            record = {
                "id": row["id"],
                "game": game,
                "goal": row["goal"],
                "buttons": row["action"]["buttons"],
                "conditions": {},
            }
            probabilities = {}
            for condition, (pixels, text) in conditions.items():
                _, embedding = runtime.model.policy.vision(mx.array(preprocess(pixels)))
                prefix = runtime.model.policy.prefix(embedding, text)
                context, _ = runtime.model.policy.context(prefix)
                _, logits = runtime.model.policy.decode(context, forced=target)
                mx.eval(logits)
                probs = [np.asarray(mx.softmax(logit))[0] for logit in logits[:6]]
                probabilities[condition] = probs
                nll = [-float(np.log(max(probs[j][row["tokens"][j]], 1e-30))) for j in range(6)]
                active = [n for j, n in enumerate(nll) if row["tokens"][j] != 0]
                result = {
                    "button_nll": float(np.mean(nll)),
                    "active_button_nll": float(np.mean(active)) if active else None,
                    "first_button_accuracy": int(probs[0].argmax() == row["tokens"][0]),
                }
                record["conditions"][condition] = result
                measurements[condition].append(result)
            for condition in conditions:
                tv = float(
                    np.mean(
                        [
                            np.abs(a - b).sum() / 2
                            for a, b in zip(
                                probabilities["actual"], probabilities[condition], strict=True
                            )
                        ]
                    )
                )
                record["conditions"][condition]["probability_total_variation"] = tv
            output.append(record)
        result = {}
        for condition, records in measurements.items():
            result[condition] = {
                key: float(np.mean([r[key] for r in records if r[key] is not None]))
                for key in records[0]
                if any(r[key] is not None for r in records)
            }
        report["games"][game] = {
            "examples": len(rows),
            "active_examples": sum(bool(r["action"]["buttons"]) for r in rows),
            "conditions": result,
        }
        print(json.dumps({game: report["games"][game]}), flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output / "rows.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in output)
    )


if __name__ == "__main__":
    main()
