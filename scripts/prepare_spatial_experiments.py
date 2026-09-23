"""Cache single screenshots for spatial, grounding, distillation and goal experiments."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from laya_vision_stitch.laya_p2p import LayaP2PRuntime
from laya_vision_stitch.p2p_adaptation import encode_action
from laya_vision_stitch.p2p_pretrained_vision import preprocess
from scripts.train_p2p_visual_adapter import digest

QUESTION = "Is a large game menu, inventory or map overlay open? Exclude ordinary HUD and in-world objects."
TEMPLATES = {
    "train": [
        "Hold W if a large menu overlay is {state}; otherwise release every key.",
        "Use W only when the game menu is {state}. Stop pressing keys in the other case.",
    ],
    "validation": ["When the menu is {state}, press W; when it is {opposite}, release all keys."],
    "test": ["Leave every key released unless the large game menu is {state}; then hold W."],
}


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = LayaP2PRuntime.load(args.bundle)
    review = json.loads(Path("annotations/menu-grounding-001.json").read_text())
    fresh = json.loads(Path("annotations/fresh-games-menu-001.json").read_text())
    metadata = {
        "source_weights_sha256": digest(args.bundle / "model.safetensors"),
        "scope": "Only the current screenshot. No previous controls, sequences, captions or answer labels enter neural inputs.",
        "teacher_model": "google/siglip2-base-patch16-224",
        "teacher_revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    }
    goal_cache = {}
    images_cache = {}
    seen = {}
    seen_episodes = {}

    def goal(text):
        if text not in goal_cache:
            value = runtime.model.bridge(runtime.model.goal_features(*runtime.prepare_goal(text)))
            goal_cache[text] = np.asarray(value.astype(mx.float32))[0]
        return goal_cache[text]

    for split in ["train", "validation", "test"]:
        rows = []
        for row in read(args.data / f"{split}.jsonl"):
            try:
                tokens = encode_action(row["action"])
            except ValueError:
                continue
            rows.append(
                {
                    **row,
                    "frames": [row["frames"][-1]],
                    "previous_actions": [],
                    "kind": "gameplay",
                    "tokens": tokens,
                    "ground_label": -1,
                }
            )
        source_path = (
            Path("artifacts/fresh-games-001/test.jsonl")
            if split == "test"
            else Path(f"artifacts/learning-gate-reproduced-001/{split}.jsonl")
        )
        labels = {
            r["id"]: r for r in (fresh["labels"] if split == "test" else review["splits"][split])
        }
        for source in read(source_path):
            label = labels[source["id"]]
            frame = dict(source["frames"][-1])
            if not Path(frame["image"]).is_absolute():
                frame["image"] = str((source_path.parent / frame["image"]).resolve())
            if digest(Path(frame["image"])) != label["image_sha256"]:
                raise ValueError("Reviewed image changed")
            frame["sha256"] = label["image_sha256"]
            row = {k: source[k] for k in ["id", "game", "episode", "provenance"]}
            row.update(
                frames=[frame],
                previous_actions=[],
                controls="",
                source_id=source["id"],
                ground_label=int(label["large_menu_open"]),
            )
            rows.append(
                {
                    **row,
                    "id": row["id"] + "-ground",
                    "kind": "grounding",
                    "goal": QUESTION,
                    "tokens": encode_action({"buttons": [], "mouse_delta": [0, 0]}),
                }
            )
            for t, template in enumerate(TEMPLATES[split]):
                for state in ["open", "closed"]:
                    want = (state == "open") == label["large_menu_open"]
                    rows.append(
                        {
                            **row,
                            "id": row["id"] + f"-goal-{t}-{state}",
                            "kind": "goal",
                            "pair": row["id"] + f"-pair-{t}",
                            "goal": template.format(
                                state=state, opposite="closed" if state == "open" else "open"
                            ),
                            "tokens": encode_action(
                                {"buttons": ["w"] if want else [], "mouse_delta": [0, 0]}
                            ),
                            "goal_source": "Explicit synthetic conditional control, not demonstrated intent",
                        }
                    )
        spatial, image_tokens, goals, labels, tokens = [], [], [], [], []
        for r in rows:
            episode = (r["game"], r["episode"])
            if episode in seen_episodes and seen_episodes[episode] != split:
                raise ValueError("Cross-split episode leakage")
            seen_episodes[episode] = split
            path = Path(r["frames"][0]["image"])
            sha = digest(path)
            if sha != r["frames"][0]["sha256"]:
                raise ValueError("Manifest image changed")
            if sha in seen and seen[sha] != split:
                raise ValueError("Cross-split image leakage")
            seen[sha] = split
            if sha not in images_cache:
                with Image.open(path) as im:
                    s, i = runtime.model.policy.vision(mx.array(preprocess(im)))
                images_cache[sha] = (
                    np.asarray(s.astype(mx.float32))[0],
                    np.asarray(i.astype(mx.float32))[0],
                )
            s, i = images_cache[sha]
            spatial.append(s)
            image_tokens.append(i)
            goals.append(goal(r["goal"]))
            labels.append(r["ground_label"])
            tokens.append(r["tokens"])
        np.savez(
            args.output / f"{split}.npz",
            spatial=np.stack(spatial),
            images=np.stack(image_tokens),
            goals=np.stack(goals),
            labels=np.array(labels, np.int32),
            tokens=np.array(tokens, np.int32),
        )
        (args.output / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(
            json.dumps(
                {
                    "split": split,
                    "examples": len(rows),
                    "unique_images": len(set(r["frames"][0]["sha256"] for r in rows)),
                }
            ),
            flush=True,
        )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
