"""Hindsight goal captions for gameplay clips from a local VLM (offline training labels).

Each 1.6 s window gets a short goal phrase describing what the player did, from its first
and last frames plus a summary of the recorded inputs. Captions become goal text for
goal-conditioned training; the VLM never runs at inference. Frames are the stored 192x192
policy inputs, so captions are coarse by design.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

PROMPT = (
    "These are the first and last frames of a 1.6-second clip of someone playing {game}. "
    "Recorded inputs: {inputs}. In 3 to 8 words, state what the player is trying to do, "
    "as an instruction: one verb (such as attack, move toward, open, pick up, aim at, "
    "drive, build, flee from) and an object that is actually visible in these frames. "
    "Do not mention keys or mouse buttons. Answer with the instruction only."
)


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def summarize_inputs(rows):
    held = Counter(b for r in rows for b in r["action"]["buttons"])
    motion = np.abs(np.array([r["action"]["mouse_delta"] for r in rows])).sum(0) * 512
    parts = [f"{k} held {v}/{len(rows)} steps" for k, v in held.most_common(4)]
    presses = sum(bool(r.get("pointer") and r["pointer"].get("press")) for r in rows)
    if presses:
        parts.append(f"{presses} mouse clicks")
    if motion.sum() > 50:
        parts.append(
            f"mouse moved about {int(motion[0])} px horizontally and {int(motion[1])} px vertically"
        )
    return "; ".join(parts) if parts else "no input"


def clean(text):
    text = text.strip().split("\n")[0].strip().strip('."').strip()
    text = re.sub(r"^(instruction|answer)\s*:\s*", "", text, flags=re.I)
    words = text.split()
    return " ".join(words[:10]) if 2 <= len(words) <= 12 else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data", nargs="+", type=Path, required=True, help="Sequence manifest directories"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-per-game", type=int, default=120)
    p.add_argument("--eval-per-game", type=int, default=40)
    p.add_argument("--seed", type=int, default=20260923)
    args = p.parse_args()
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    from laya_vision_stitch.backends import QwenVision

    qwen = QwenVision(width=768)
    rng = np.random.default_rng(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = (
        {json.loads(x)["key"] for x in args.output.read_text().splitlines()}
        if args.output.exists()
        else set()
    )
    with args.output.open("a") as log:
        for data in args.data:
            for manifest in sorted(data.glob("*.jsonl")):
                split = manifest.stem
                cap = args.train_per_game if split == "train" else args.eval_per_game
                windows = defaultdict(list)
                for r in read(manifest):
                    windows[r["sequence"]].append(r)
                by_game = defaultdict(list)
                for name, rows in windows.items():
                    by_game[rows[0]["game"]].append(name)
                for game, names in by_game.items():
                    for name in rng.permutation(sorted(names))[:cap]:
                        key = f"{data.name}:{split}:{name}"
                        if key in done:
                            continue
                        rows = sorted(windows[name], key=lambda r: r["sequence_step"])
                        frames = [
                            Image.open(rows[i]["frames"][0]["image"]).convert("RGB")
                            for i in (0, -1)
                        ]
                        tile = Image.new("RGB", (768, 384))
                        for k, im in enumerate(frames):
                            tile.paste(im.resize((384, 384), Image.BICUBIC), (384 * k, 0))
                        prompt = PROMPT.format(
                            game=game.replace("_", " "), inputs=summarize_inputs(rows)
                        )
                        formatted = apply_chat_template(
                            qwen.processor,
                            qwen.model.config,
                            prompt,
                            num_images=1,
                            enable_thinking=False,
                        )
                        text = generate(
                            qwen.model,
                            qwen.processor,
                            formatted,
                            image=[tile],
                            max_tokens=24,
                            temperature=0,
                            verbose=False,
                        ).text
                        record = {
                            "key": key,
                            "data": str(data),
                            "split": split,
                            "sequence": name,
                            "game": game,
                            "caption": clean(text),
                            "raw": text,
                        }
                        log.write(json.dumps(record) + "\n")
                        log.flush()


if __name__ == "__main__":
    main()
