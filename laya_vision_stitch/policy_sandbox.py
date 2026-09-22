"""Closed-loop target approach sandbox driven only by screenshot model outputs.

Environment coordinates are used for rendering/physics/scoring, never as model
inputs. This is a simple synthetic environment, not a claim of real-game play.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .scaling_data import COLORS, SHAPES, render
from .trainable_model import TrainableRuntime


def run(bundle, output, episodes=40, mode="buttons", seed=9401):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    runtime = TrainableRuntime.load(bundle)
    rng = np.random.default_rng(seed)
    results, timings, film = [], [], []
    for episode in range(episodes):
        colors = rng.choice(list(COLORS), 2, replace=False).tolist()
        shapes = rng.choice(SHAPES, 2).tolist()
        world = [float(rng.uniform(-0.32, -0.2)), float(rng.uniform(0.2, 0.32))]
        target = int(rng.integers(2))
        player, previous = 0.0, []
        records = []
        success = False
        style = str(rng.choice(["plain", "grid", "dark", "panel"]))
        for step in range(10):
            path = output / f"episode-{episode:03d}-step-{step:02d}.png"
            objects = [
                (c, s, 0.5 + x - player, 0.5, 0.08)
                for c, s, x in zip(colors, shapes, world, strict=True)
            ]
            # The world scrolls around the centered player; only pixels reveal relative positions.
            render(path, objects, style, np.random.default_rng(seed + episode))
            with Image.open(path) as source:
                image = source.convert("RGB")
            draw = ImageDraw.Draw(image)
            x = image.width // 2
            draw.line(
                (x, image.height * 0.78, x, image.height * 0.93), fill=(120, 120, 120), width=3
            )
            image.save(path)
            row = {
                "frames": [{"image": str(path), "age_seconds": 0}],
                "goal": f"Move toward the {colors[target]} {shapes[target]}. Which direction should you move?",
                "controls": "A moves left. D moves right.",
                "previous_actions": previous,
                "choices": {"left": "Move left.", "right": "Move right."},
            }
            prediction = runtime.predict(row)
            timings.append(prediction["image_to_outputs_ms"])
            if mode == "buttons":
                buttons = prediction["buttons"]
                direction = int("d" in buttons) - int("a" in buttons)
            else:
                choice = max(
                    prediction["choice_probabilities"], key=prediction["choice_probabilities"].get
                )
                direction = -1 if choice == "left" else 1
                buttons = ["a" if direction < 0 else "d"]
            player += direction * 0.07
            success = abs(player - world[target]) <= 0.065
            records.append(
                {
                    "step": step,
                    "image": path.name,
                    "buttons": buttons,
                    "direction": direction,
                    "success": success,
                    "prediction": prediction,
                }
            )
            previous = [{"buttons": buttons, "duration_seconds": 0.1}]
            if episode < 8:
                preview = image.resize((320, 320))
                label = ImageDraw.Draw(preview)
                label.rectangle((0, 0, 320, 35), fill="white")
                label.text(
                    (8, 4), f"Goal: {colors[target]} {shapes[target]} | {mode}", fill="black"
                )
                label.text(
                    (8, 18),
                    f"Episode {episode + 1}, step {step + 1}: {'reached' if success else buttons}",
                    fill="black",
                )
                film.append(preview)
            if success:
                break
        results.append(
            {
                "episode": episode,
                "target_color": colors[target],
                "target_shape": shapes[target],
                "initial_target_side": "left" if world[target] < 0 else "right",
                "style": style,
                "success": success,
                "steps": len(records),
                "trace": records,
            }
        )
        if (episode + 1) % 10 == 0:
            print(
                f"{episode + 1}/{episodes}: {sum(r['success'] for r in results)} reached",
                flush=True,
            )
    if film:
        film[0].save(
            output / "preview.gif", save_all=True, append_images=film[1:], duration=240, loop=0
        )
    report = {
        "mode": mode,
        "episodes": episodes,
        "successes": sum(r["success"] for r in results),
        "success_rate": sum(r["success"] for r in results) / episodes,
        "best_constant_direction_success_rate": max(
            sum(r["initial_target_side"] == "left" for r in results),
            sum(r["initial_target_side"] == "right" for r in results),
        )
        / episodes,
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "warm_inference_p50_ms": float(np.median(timings[1:])),
        "warm_inference_p95_ms": float(np.percentile(timings[1:], 95)),
        "seed": seed,
        "results": results,
        "os_input_events_sent": 0,
        "note": "Synthetic one-dimensional target approach; not real-game or combat validation.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--mode", choices=("buttons", "choices"), default="buttons")
    parser.add_argument("--seed", type=int, default=9401)
    args = parser.parse_args()
    if args.episodes < 2:
        parser.error("At least two episodes required")
    run(args.bundle, args.output, args.episodes, args.mode, args.seed)


if __name__ == "__main__":
    main()
