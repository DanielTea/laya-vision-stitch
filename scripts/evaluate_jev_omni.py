"""Evaluate the Jev-Omni decision classifier (MLX port) as a slow game-state reader.

1. Parity with the repository's reference probabilities (`unified/verification_unified.json`).
2. Latency for a text question and a 1280x720 screenshot question.
3. Hordes: "Is a monster selected as the target?" on live-trial frames. Ground truth is
   the target panel's health-bar color (red: monster, green: own character), used only
   for scoring; the model sees the screenshot and the question.
4. Held-out D2E games: at a recorded click, which of nine screen regions is clicked?
   Compared with the most common region and with the RADIO click head.

Sends no game input.
"""

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from laya_vision_stitch.jev_omni import REPO, REVISION, JevOmni

REGIONS = [f"{v} {h}" for v in ("top", "middle", "bottom") for h in ("left", "center", "right")]
HELD_OUT = {"Core_Keeper", "Raft", "Rainbow_Six", "Satisfactory", "Stardew_Valley"}
TRIALS = ["hordes-general-live-001", "hordes-planner-live-001"]  # same 1280x720 crop


def region(xy):
    col = min(2, int(xy[0] * 3))
    row = min(2, int(xy[1] * 3))
    return row * 3 + col


def panel_label(frame):
    """Target panel health bar in the [9, 87, 1280, 720] Hordes crop."""
    bar = np.asarray(frame.convert("RGB"))[648:668, 660:880].astype(int)
    red = ((bar[..., 0] > 170) & (bar[..., 1] < 110) & (bar[..., 2] < 110)).mean()
    green = ((bar[..., 1] > 160) & (bar[..., 0] < 170) & (bar[..., 2] < 120)).mean()
    if red > 0.08:
        return "monster"
    return "self" if green > 0.13 else "none"


def auc(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels, bool)
    pos, neg = scores[labels], scores[~labels]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bits", type=int, default=None, help="Quantize the text decoder (e.g. 8)")
    p.add_argument("--hordes-per-class", type=int, default=40)
    p.add_argument("--clicks-per-game", type=int, default=20)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(REPO, revision=REVISION, allow_patterns=["unified/*", "head.pt"]))
    start = time.perf_counter()
    jev = JevOmni(path, bits=args.bits)
    report = {"revision": REVISION, "bits": args.bits, "load_seconds": time.perf_counter() - start}

    # 1. Parity with the reference probabilities.
    ref = json.loads((path / "unified" / "verification_unified.json").read_text())
    diffs = []
    for case, expected in zip(ref["cases"], ref["reference"], strict=True):
        got = jev.predict(case["state"], case["question"], case["options"])["probabilities"]
        diffs.append(max(abs(got[k] - v) for k, v in expected.items()))
    report["parity"] = {"cases": len(diffs), "max_abs_diff": max(diffs), "per_case": diffs}
    print(json.dumps({"parity": report["parity"]}), flush=True)

    # 2. Latency (after one warm-up of each kind).
    frame = Image.open(args.artifacts / TRIALS[1] / "frames" / "00300.jpg").convert("RGB")
    timings = {"text": [], "image": []}
    for kind in ["text", "image"] * 6:
        t = time.perf_counter()
        jev.predict(
            "It is raining heavily." if kind == "text" else "A screenshot of a video game.",
            "Should the player take an umbrella?"
            if kind == "text"
            else "Is the player's health low?",
            ["Yes", "No"],
            image=frame if kind == "image" else None,
        )
        timings[kind].append(time.perf_counter() - t)
    report["latency_ms"] = {k: 1000 * float(np.median(v[1:])) for k, v in timings.items()}
    print(json.dumps({"latency_ms": report["latency_ms"]}), flush=True)

    # 3. Hordes target selection.
    pool = {"monster": [], "other": []}
    for trial in TRIALS:
        events = [
            json.loads(x)
            for x in (args.artifacts / trial / "events.jsonl").read_text().splitlines()
            if x.strip()
        ]
        for e in events[::5]:
            if e.get("image"):
                image_path = args.artifacts / trial / e["image"]
                label = panel_label(Image.open(image_path))
                pool["monster" if label == "monster" else "other"].append((image_path, label))
    rows = []
    for cls in pool:
        for image_path, label in rng.sample(pool[cls], min(args.hordes_per_class, len(pool[cls]))):
            out = jev.predict(
                "A screenshot of a video game.",
                "Is an enemy or monster currently selected as the player's target?",
                ["Yes", "No"],
                image=Image.open(image_path),
            )
            rows.append(
                {"image": str(image_path), "label": label, "p_yes": out["probabilities"]["Yes"]}
            )
    truth = [r["label"] == "monster" for r in rows]
    pred = [r["p_yes"] > 0.5 for r in rows]
    report["hordes_selected"] = {
        "counts": dict(Counter(r["label"] for r in rows)),
        "accuracy": float(np.mean(np.equal(truth, pred))),
        "balanced_accuracy": float(
            np.mean(
                [
                    np.mean([p for t, p in zip(truth, pred, strict=True) if t == c] == np.array(c))
                    for c in (True, False)
                ]
            )
        ),
        "auc": auc([r["p_yes"] for r in rows], truth),
        "self_selected_said_yes": float(
            np.mean([r["p_yes"] > 0.5 for r in rows if r["label"] == "self"])
        )
        if any(r["label"] == "self" for r in rows)
        else None,
        "rows": rows,
    }
    print(
        json.dumps({k: v for k, v in report["hordes_selected"].items() if k != "rows"}), flush=True
    )

    # 4. Click regions on held-out D2E games, against the majority region and the RADIO head.
    presses = []
    for f in sorted(args.artifacts.glob("d2e-presses-*/presses.jsonl")):
        presses += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    by_game = {}
    for r in presses:
        if r["game"] in HELD_OUT and Path(r["image"]).exists():
            by_game.setdefault(r["game"], []).append(r)
    sample = [
        r
        for game in sorted(by_game)
        for r in rng.sample(by_game[game], min(args.clicks_per_game, len(by_game[game])))
    ]
    from laya_vision_stitch.laya_p2p import LayaP2PRuntime

    policy = LayaP2PRuntime.load(args.artifacts / "laya-p2p-general-001").model.policy
    import mlx.core as mx

    click_rows = []
    for r in sample:
        image = Image.open(r["image"]).convert("RGB")
        out = jev.predict(
            "A screenshot of a video game, taken as the player clicks the mouse.",
            "In which region of the screen does the player click?",
            REGIONS,
            image=image,
        )
        pixels = np.asarray(image.resize((384, 224), Image.BICUBIC), np.float32) / 255
        patches, _ = policy.pointer_encoder(
            mx.array(pixels[None]).astype(policy.pointer_encoder.input_dtype)
        )
        head_xy = np.asarray(policy.pointer_head.predict(patches.reshape(1, 14, 24, -1)))[0]
        click_rows.append(
            {
                "game": r["game"],
                "truth": region(r["xy"]),
                "jev": out["prediction_index"],
                "jev_p_truth": out["probabilities"][REGIONS[region(r["xy"])]],
                "radio_head": region(head_xy),
            }
        )
    majority = Counter(x["truth"] for x in click_rows).most_common(1)[0][0]
    per_game = {}
    for game in sorted({x["game"] for x in click_rows}):
        g = [x for x in click_rows if x["game"] == game]
        per_game[game] = {
            "n": len(g),
            "jev": float(np.mean([x["jev"] == x["truth"] for x in g])),
            "radio_head": float(np.mean([x["radio_head"] == x["truth"] for x in g])),
            "majority_region": float(np.mean([majority == x["truth"] for x in g])),
        }
    report["d2e_click_region"] = {
        "n": len(click_rows),
        "majority_region": REGIONS[majority],
        "jev": float(np.mean([x["jev"] == x["truth"] for x in click_rows])),
        "radio_head": float(np.mean([x["radio_head"] == x["truth"] for x in click_rows])),
        "majority": float(np.mean([x["truth"] == majority for x in click_rows])),
        "jev_mean_p_truth": float(np.mean([x["jev_p_truth"] for x in click_rows])),
        "chance": 1 / 9,
        "per_game": per_game,
        "rows": click_rows,
    }
    print(
        json.dumps({k: v for k, v in report["d2e_click_region"].items() if k != "rows"}), flush=True
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
