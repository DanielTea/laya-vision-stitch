"""Measure planner-target tracking precision on recorded frames; sends no input.

Synthetic test: each frame with Molmo points is shifted and scaled by a known amount, so
the true new position of every point is known. Real test: each planner answer's target is
followed from the frame Molmo saw to the frame where the answer arrived (in one jump, or
through frames sampled every `--catch-up-step` seconds) and back again. With no ground
truth on live frames, the forward-backward drift measures consistency.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from laya_vision_stitch.laya_p2p import LayaP2PRuntime
from laya_vision_stitch.target_tracker import FeatureTracker


def answers(trial):
    """(source image path, answer image path, points, target, frames between) per answer.

    `trial` is a live trial directory, or `replay_dir@recorded_trial_dir` for a replay.
    """
    trial, _, recorded = trial.partition("@")
    trial = Path(trial)
    if (trial / "events.jsonl").exists():
        records = [
            json.loads(line)
            for line in (trial / "events.jsonl").read_text().splitlines()
            if line.strip()
        ]
        records = [
            {
                "t": r["elapsed_s"],
                "image": r.get("image"),
                "planner": r["proposal"].get("planner") or {},
            }
            for r in records
        ]
        root = trial
    else:
        replay = [
            json.loads(line)
            for line in (trial / "replay.jsonl").read_text().splitlines()
            if line.strip()
        ]
        records = [
            {"t": r["recorded_s"], "image": r["image"], "planner": r["planner"]} for r in replay
        ]
        root = Path(recorded)
    records = [r for r in records if r["image"]]
    found = []
    for r in records:
        p = r["planner"]
        if not p.get("points") or not p.get("target"):
            continue
        start = r["t"] - p["planner_seconds"]
        source = min(
            (q for q in records if q["t"] <= start + 0.05), key=lambda q: abs(q["t"] - start)
        )
        between = [q for q in records if source["t"] < q["t"] < r["t"]]
        after = [q for q in records if q["t"] > r["t"]]
        found.append(
            {
                "source": root / source["image"],
                "answer": root / r["image"],
                "points": p["points"],
                "target": p["target"],
                "between": [(q["t"], root / q["image"]) for q in between],
                "after": [(q["t"], root / q["image"]) for q in after],
                "t": r["t"],
                "t0": source["t"],
            }
        )
    return found


def transform(image, shift, scale):
    w, h = image.size
    cx, cy = w / 2, h / 2
    # Output pixel u maps back to source (u - c - d) / s + c.
    a, e = 1 / scale, 1 / scale
    c = cx - (cx + shift[0] * w) / scale
    f = cy - (cy + shift[1] * h) / scale
    return image.transform((w, h), Image.AFFINE, (a, 0, c, 0, e, f), Image.BICUBIC)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--trials", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--catch-up-step", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=5)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    encoder = LayaP2PRuntime.load(args.bundle).model.policy.pointer_encoder
    variants = {
        "previous_768_mean": dict(mode="mean", refine=False, size=(768, 432), threshold=0.78),
        "768_template": dict(size=(768, 432)),
        "1024_mean_refined": dict(mode="mean", size=(1024, 576), threshold=0.78),
        "1024_template": dict(),
    }
    found = [a for trial in args.trials for a in answers(trial)]
    rng = np.random.default_rng(args.seed)
    synthetic = {name: [] for name in variants}
    for a in found:
        image = Image.open(a["source"]).convert("RGB")
        w, h = image.size
        for point in a["points"]:
            shift, scale = rng.uniform(-0.06, 0.06, 2), rng.uniform(0.95, 1.05)
            moved = transform(image, shift, scale)
            truth = (np.array(point) - 0.5) * scale + 0.5 + shift
            if not ((truth > 0.02) & (truth < 0.98)).all():
                continue
            for name, options in variants.items():
                tracker = FeatureTracker(encoder, **options)
                tracker.set_target(image, point)
                xy, _ = tracker.track(moved)
                error = (
                    None if xy is None else float(np.linalg.norm((np.array(xy) - truth) * [w, h]))
                )
                synthetic[name].append(error)
    report = {"answers": len(found), "synthetic": {}}
    for name, errors in synthetic.items():
        kept = np.array([e for e in errors if e is not None])
        report["synthetic"][name] = {
            "points": len(errors),
            "lost": int(sum(e is None for e in errors)),
            "median_px": float(np.median(kept)),
            "p90_px": float(np.percentile(kept, 90)),
            "within_16px": float(np.mean(kept <= 16)),
        }
    # Real frames: follow each planner answer from the frame Molmo saw to the frame where
    # the answer arrived, then back again; the distance from the start is the drift.
    real_variants = {
        "previous_jump": (variants["previous_768_mean"], None),
        "template_jump_w0.3": (dict(window=0.3, threshold=0.7), None),
        "template_chain": (dict(threshold=0.7), args.catch_up_step),
        "template_chain_update": (dict(threshold=0.7, update=0.3), args.catch_up_step),
    }
    images = {}

    def image(path):
        if path not in images:
            images[path] = Image.open(path).convert("RGB")
        return images[path]

    report["real"] = {}
    for name, (options, step) in real_variants.items():
        errors, lost = [], 0
        for a in found:
            frames, last = [], a["t0"]
            if step:
                for t, path in a["between"]:
                    if t - last >= step:
                        frames.append(path)
                        last = t
            forward = FeatureTracker(encoder, **options)
            forward.set_target(image(a["source"]), a["target"])
            xy, _ = forward.catch_up([image(f) for f in [*frames, a["answer"]]])
            back_xy = None
            if xy is not None:
                back = FeatureTracker(encoder, **options)
                back.set_target(image(a["answer"]), xy)
                back_xy, _ = back.catch_up([image(f) for f in [*frames[::-1], a["source"]]])
            if back_xy is None:
                lost += 1
                continue
            w, h = image(a["source"]).size
            errors.append(float(np.linalg.norm((np.array(back_xy) - a["target"]) * [w, h])))
        report["real"][name] = {
            "answers": len(found),
            "lost": lost,
            "forward_backward_median_px": float(np.median(errors)) if errors else None,
            "within_24px": float(np.mean(np.array(errors) <= 24)) if errors else None,
            "errors_px": [round(e, 1) for e in errors],
        }
    print(json.dumps(report["real"], indent=1))
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["synthetic"], indent=1))


if __name__ == "__main__":
    main()
