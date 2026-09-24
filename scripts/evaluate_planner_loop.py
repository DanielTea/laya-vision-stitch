"""Replay a recorded Hordes trial in real time through controller + Molmo planner + tracker.

Frames are fed at their recorded times; the controller's own outputs become its feedback.
No input is sent anywhere. Reports controller latency with the planner running, planner
update times, target availability, and renders frames with planner and tracked targets.
`--recorded-planner` replays an earlier run's Molmo answers at their recorded times, so
different bundles receive identical targets; actions while a target is live are scored
for movement toward it (open loop: the recorded frames do not react to the actions).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from laya_vision_stitch.laya_p2p_stream import LayaP2PStream
from laya_vision_stitch.molmo_planner import AsyncPlanner
from laya_vision_stitch.target_conditioning import toward_target


class RecordedPlanner:
    """Planner stand-in that returns an earlier run's answers when replay time reaches them."""

    def __init__(self, replay, frames):
        records = [
            json.loads(line)
            for line in (replay / "replay.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self.answers = []
        for r in records:
            p = r["planner"]
            if "points" in p:
                start = r["recorded_s"] - p["planner_seconds"]
                source = min(
                    frames, key=lambda f: abs(f[0] - start) if f[0] <= start + 0.05 else np.inf
                )
                self.answers.append({**p, "t": r["recorded_s"], "source": source[1]})
        self.now = 0.0

    def submit(self, image, goal, skill=False, point=True):
        return None

    def poll(self):
        if not self.answers or self.now < self.answers[0]["t"]:
            return None
        a = self.answers.pop(0)
        return {
            "frame_id": -1,
            "phrase": a.get("phrase"),
            "points": a["points"],
            "raw": "",
            "error": None,
            "planner_seconds": a["planner_seconds"],
            "image": Image.open(a["source"]).convert("RGB"),
        }

    def close(self):
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--trial", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--goal",
        default="Defeat nearby monsters. Avoid attacking players. Retreat when health is low.",
    )
    p.add_argument(
        "--no-planner", action="store_true", help="Latency reference without the planner"
    )
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument(
        "--recorded-planner", type=Path, help="Earlier replay output whose planner answers to reuse"
    )
    p.add_argument("--seed", type=int, default=0, help="Sampling seed for the controller")
    args = p.parse_args()
    import mlx.core as mx

    mx.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "args.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2) + "\n"
    )
    events = [
        json.loads(line)
        for line in (args.trial / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    events = [e for e in events if e.get("image") and e["elapsed_s"] <= args.seconds]
    if args.recorded_planner:
        planner = RecordedPlanner(
            args.recorded_planner, [(e["elapsed_s"], args.trial / e["image"]) for e in events]
        )
    else:
        planner = None if args.no_planner else AsyncPlanner()
    stream = LayaP2PStream.load(args.bundle, compile=True, planner=planner)
    for e in events[:3]:  # warm-up on the first frames, then restart memory
        stream.predict(
            {
                "frames": [
                    {"image": Image.open(args.trial / e["image"]).convert("RGB"), "age_seconds": 0}
                ],
                "goal": args.goal,
                "previous_actions": [],
            },
            session_id="warm",
            timestamp_seconds=e["elapsed_s"],
        )
    stream.reset()
    log, previous, start = [], [], time.perf_counter()
    for e in events:
        wait = e["elapsed_s"] - (time.perf_counter() - start)
        if wait > 0:
            time.sleep(wait)
        image = Image.open(args.trial / e["image"]).convert("RGB")
        if isinstance(planner, RecordedPlanner):
            planner.now = e["elapsed_s"]
        row = {
            "frames": [{"image": image, "age_seconds": 0}],
            "goal": args.goal,
            "previous_actions": previous,
        }
        out = stream.predict(
            row, session_id="replay", timestamp_seconds=time.perf_counter() - start
        )
        previous = [{"buttons": out["buttons"], "mouse_delta": out["mouse_delta"]}]
        record = {
            "image": e["image"],
            "recorded_s": e["elapsed_s"],
            "wall_s": time.perf_counter() - start,
            "model_ms": out["image_to_outputs_ms"],
            "buttons": out["buttons"],
            "pointer_xy": out.get("pointer_xy"),
            "pointer_source": out.get("pointer_source"),
            "planner_click": out.get("planner_click"),
            "planner": {k: v for k, v in (out.get("planner") or {}).items() if k != "raw"},
            "target_live": stream.target_time is not None,
            "target_input": out.get("target_input"),
            "target_xy": None if stream.target_time is None else list(stream.tracker.xy),
        }
        log.append(record)
    if planner is not None:
        planner.close()
    (args.output / "replay.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
    ms = np.array([r["model_ms"] for r in log])
    answers = [r["planner"] for r in log if "points" in r["planner"]]
    summary = {
        "frames": len(log),
        "model_ms_p50": float(np.median(ms)),
        "model_ms_p95": float(np.percentile(ms, 95)),
        "planner_answers": len(answers),
        "planner_answers_with_points": sum(bool(a["points"]) for a in answers),
        "planner_seconds_p50": float(np.median([a["planner_seconds"] for a in answers]))
        if answers
        else None,
        "target_available_fraction": float(np.mean([r["target_live"] for r in log])),
        "clicks": sum(r["pointer_xy"] is not None for r in log),
        "clicks_from_planner": sum(r["pointer_source"] == "planner" for r in log),
        "planner_clicks": sum(r["planner_click"] is not None for r in log),
        "phrase": next((a["phrase"] for a in answers if a.get("phrase")), None),
    }
    # Behavior while a target is live: movement toward it, clicks and ability keys.
    live = [r for r in log if r["target_xy"] is not None]
    summary["with_target"] = {
        **toward_target(
            [set(r["buttons"]) for r in live],
            np.zeros((len(live), 2)),
            np.array([r["target_xy"] for r in live]).reshape(-1, 2),
        ),
        "ability_key_steps": sum(bool({"1", "2", "3", "4"} & set(r["buttons"])) for r in live),
    }
    summary["ability_key_steps"] = sum(bool({"1", "2", "3", "4"} & set(r["buttons"])) for r in log)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    # Contact sheet: every planner answer with points, drawn on the frame it was computed for.
    shown = [r for r in log if r["planner_click"]][:16]
    if shown:
        sheet = Image.new("RGB", (4 * 480, ((len(shown) + 3) // 4) * 270))
        for i, r in enumerate(shown):
            im = Image.open(args.trial / r["image"]).convert("RGB")
            d = ImageDraw.Draw(im)
            for x, y in r["planner"].get("points", []):
                d.ellipse(
                    (x * 1280 - 10, y * 720 - 10, x * 1280 + 10, y * 720 + 10),
                    outline=(255, 0, 255),
                    width=4,
                )
            click = (r["planner_click"] or {}).get("xy")
            for xy, color in [(r["planner"].get("target"), (0, 255, 255)), (click, (255, 255, 0))]:
                if xy:
                    d.rectangle(
                        (xy[0] * 1280 - 16, xy[1] * 720 - 16, xy[0] * 1280 + 16, xy[1] * 720 + 16),
                        outline=color,
                        width=5,
                    )
            d.text(
                (8, 60),
                f"t={r['recorded_s']:.1f}s {r['pointer_source'] or 'plan'}",
                fill=(255, 255, 255),
            )
            sheet.paste(im.resize((480, 270)), ((i % 4) * 480, (i // 4) * 270))
        sheet.save(args.output / "targets.jpg", quality=85)


if __name__ == "__main__":
    main()
