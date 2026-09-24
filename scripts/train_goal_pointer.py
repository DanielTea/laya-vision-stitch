"""Train and test the goal-conditioned pointer on Molmo teacher points.

Training frames come from training games only; 10% of their recordings (episode hash)
select the checkpoint. Held-out games and Hordes frames are reported only. Goal phrasings:
all but one phrasing per category train; the last is an unseen-wording test. Metrics:
presence accuracy and hit@r (prediction within r of any teacher point, frames with points).
Controls: window center, the same model with the goal removed, and a wrong-category goal.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from PIL import Image

from laya_vision_stitch.goal_pointer import GoalPointer
from laya_vision_stitch.laya_p2p import LayaP2PRuntime
from laya_vision_stitch.radio_vision import RadioVision

CATEGORIES = {
    "enemies": ["attack the enemy", "fight the monster", "defeat the nearby monsters"],
    "doors": ["go through the door", "open the door"],
    "characters": ["talk to the character", "walk to the person"],
    "animals": ["go to the animal", "approach the animal"],
    "vehicles": ["get in the vehicle", "walk to the car"],
    "containers": ["open the chest", "loot the container"],
}
LIVE_GOAL = "Defeat nearby monsters. Avoid attacking players. Retreat when health is low."


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def hits(pred, points, radius):
    if not points:
        return None
    return float(min(np.hypot(pred[0] - x, pred[1] - y) for x, y in points) <= radius)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument(
        "--features", type=Path, required=True, help="press-feature cache with features.npy"
    )
    p.add_argument(
        "--bundle", type=Path, required=True, help="Stitched bundle providing Laya and the bridge"
    )
    p.add_argument("--radio", type=Path, default=Path("artifacts/radio-source"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260923)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    runtime = LayaP2PRuntime.load(args.bundle)
    cache = {}

    def goal(text):
        if text not in cache:
            v = runtime.model.bridge(runtime.model.goal_features(*runtime.prepare_goal(text)))
            cache[text] = np.asarray(v.astype(mx.float32))[0]
        return cache[text]

    labels = [r for r in read(args.labels) if r["points"] is not None]
    rows = read(args.features / "rows.jsonl")
    index = {r["image"]: i for i, r in enumerate(rows)}
    episode = {r["image"]: r["episode"] for r in rows}
    features = np.load(args.features / "features.npy", mmap_mode="r")
    radio = RadioVision.from_source(args.radio, mx.float16)
    extra = {}
    for r in labels:
        if r["image"] not in index and r["image"] not in extra:
            im = Image.open(r["image"]).convert("RGB").resize((384, 224), Image.BICUBIC)
            patches, _ = radio(mx.array(np.asarray(im, np.float32)[None] / 255).astype(mx.float16))
            extra[r["image"]] = np.asarray(patches.astype(mx.float16)).reshape(14, 24, 768)

    def patches(r):
        return extra[r["image"]] if r["image"] in extra else np.asarray(features[index[r["image"]]])

    def is_validation(r):
        return (
            int(hashlib.sha256(episode.get(r["image"], r["image"]).encode()).hexdigest(), 16) % 10
            == 0
        )

    train = [r for r in labels if r["split"] == "train" and not is_validation(r)]
    val = [r for r in labels if r["split"] == "train" and is_validation(r)]
    held = [r for r in labels if r["split"] == "heldout"]
    hordes = [r for r in labels if r["split"] == "hordes_eval"]
    model = GoalPointer()
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=3e-4, weight_decay=0.01)
    value_grad = mx.value_and_grad(lambda m, x, g, c, pr: m.loss(x, g, c, pr))

    def batch_arrays(items, texts):
        x = mx.array(np.stack([patches(r) for r in items]))
        g = mx.array(np.stack([goal(t) for t in texts]))
        c = mx.array(GoalPointer.cell_targets([r["points"] for r in items]))
        pr = mx.array([float(bool(r["points"])) for r in items])
        return x, g, c, pr

    def evaluate(items, text_for):
        out = defaultdict(list)
        for s in range(0, len(items), 64):
            chunk = items[s : s + 64]
            texts = [text_for(r) for r in chunk]
            x = mx.array(np.stack([patches(r) for r in chunk]))
            g = mx.array(np.stack([goal(t) if t else np.zeros(768, np.float32) for t in texts]))
            xy, present = model.predict(x, g)
            mx.eval(xy, present)
            for r, q, pv in zip(chunk, np.asarray(xy), np.asarray(present), strict=True):
                truth = bool(r["points"])
                out["presence_correct"].append(float((pv > 0.5) == truth))
                for radius in (0.06, 0.1):
                    h = hits(q, r["points"], radius)
                    if h is not None:
                        out[f"hit@{radius}"].append(h)
                        out[f"center_hit@{radius}"].append(hits((0.5, 0.5), r["points"], radius))
        return {k: float(np.mean(v)) for k, v in out.items()} | {
            "frames": len(items),
            "with_points": sum(bool(r["points"]) for r in items),
        }

    def train_text(r):
        return str(rng.choice(CATEGORIES[r["category"]][:-1]))

    def eval_text(r, unseen=False):
        return CATEGORIES[r["category"]][-1 if unseen else 0]

    def wrong_text(r):
        others = sorted(c for c in CATEGORIES if c != r["category"])
        return CATEGORIES[
            others[int(hashlib.md5(r["image"].encode()).hexdigest(), 16) % len(others)]
        ][0]

    best, best_step, history = -1.0, 0, []
    for step in range(1, args.steps + 1):
        items = [train[i] for i in rng.integers(0, len(train), args.batch)]
        loss, grads = value_grad(model, *batch_arrays(items, [train_text(r) for r in items]))
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        if step % 250 == 0 or step == args.steps:
            v = evaluate(val, eval_text)
            score = v.get("hit@0.1", 0) + v["presence_correct"]
            history.append({"step": step, "loss": float(loss), **v})
            print(json.dumps(history[-1]), flush=True)
            if score > best:
                best, best_step = score, step
                model.save_weights(str(args.output / "goal_pointer.safetensors"))
    model.load_weights(str(args.output / "goal_pointer.safetensors"))
    report = {
        "selected_step": best_step,
        "history": history,
        "sizes": {
            "train": len(train),
            "validation": len(val),
            "heldout": len(held),
            "hordes": len(hordes),
        },
    }
    for name, items in [("validation", val), ("heldout_games", held), ("hordes_eval", hordes)]:
        if not items:
            continue
        report[name] = {
            "goal": evaluate(items, eval_text),
            "unseen_wording": evaluate(items, lambda r: eval_text(r, True)),
            "no_goal": evaluate(items, lambda r: ""),
            "wrong_goal": evaluate(items, wrong_text),
        }
        if name == "hordes_eval":
            report[name]["live_trial_goal"] = evaluate(items, lambda r: LIVE_GOAL)
        print(
            json.dumps(
                {
                    name: {
                        k: {m: round(x, 3) for m, x in v.items() if isinstance(x, float)}
                        for k, v in report[name].items()
                    }
                }
            ),
            flush=True,
        )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "config.json").write_text(
        json.dumps(
            {
                "goal_pointer": {"width": 256, "depth": 2},
                "encoder": "C-RADIOv3-B 384x224",
                "teacher": "Molmo-7B-D-0924-4bit (offline labels only)",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
