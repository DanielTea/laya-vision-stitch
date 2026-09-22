"""Import recorded controls, with whole-run holdouts; no policy-state inputs."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .p2p_data import CONTROLS

GOAL = "Defeat nearby monsters. Avoid attacking players. Retreat when health is low."
HORDES_CONTROLS = (
    CONTROLS + " Hordes.io: WASD moves; Tab selects; 1 attacks; right mouse drag turns the camera."
)


def label(record):
    if record.get("input_applied"):
        if record.get("status") == "applied" and record.get("key"):
            key = record["key"].lower()
            if key not in {"w", "a", "s", "d", "tab", "1", "2", "3", "4"}:
                return None
            return {"buttons": [key], "mouse_delta": [0, 0], "duration_seconds": 0.05}
        if record.get("status") == "camera_applied" and record.get("camera_delta"):
            return {
                "buttons": ["mouse_right"],
                "mouse_delta": [v / 512 for v in record["camera_delta"]],
                "duration_seconds": 0.05,
            }
    elif record.get("status") == "wait":
        return {"buttons": [], "mouse_delta": [0, 0], "duration_seconds": 0.05}
    return None


def collect(root):
    groups = []
    for log in sorted(root.glob("*-fast-live/events.jsonl")):
        rows = []
        for line in log.read_text().splitlines():
            record = json.loads(line)
            image = log.parent / record.get("image", "missing")
            if not image.is_file() or "elapsed_s" not in record:
                continue
            state = record.get("state", {})
            if not state.get("hud") or not state.get("ocr_fresh"):
                continue
            action = label(record)
            if action is None:
                continue
            rows.append(
                {
                    "id": f"{log.parent.name}-{record['step']}",
                    "game": "Hordes.io",
                    "episode": log.parent.name,
                    "goal": GOAL,
                    "controls": HORDES_CONTROLS,
                    "frames": [
                        {
                            "image": str(image.resolve()),
                            "age_seconds": 0,
                            "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                        }
                    ],
                    "previous_actions": [],
                    "action": action,
                    "timestamp_seconds": record["elapsed_s"],
                    "provenance": {
                        "type": "recorded_controller_demonstration",
                        "log": str(log.resolve()),
                        "step": record["step"],
                        "verified_optimal": False,
                        "duration": "Control identity/delta retained; pulse standardized to 50 ms",
                    },
                }
            )
        if len(rows) >= 12:
            groups.append(rows)
    if len(groups) < 6:
        raise ValueError("Need six complete demonstration runs")
    return groups


def attack_outcomes(rows, horizon=1.5):
    """Audit weak supervision; outcome correlation is explicitly not causation."""
    logs, count, health_drop, out_of_range, complete = {}, 0, 0, 0, 0
    for row in rows:
        if row["game"] != "Hordes.io" or row["action"]["buttons"] != ["1"]:
            continue
        source = row["provenance"]
        path = source["log"]
        if path not in logs:
            logs[path] = [json.loads(line) for line in Path(path).read_text().splitlines()]
        sequence = logs[path]
        original = next(record for record in sequence if record["step"] == source["step"])
        later = [
            r for r in sequence if 0 < r.get("elapsed_s", 0) - original["elapsed_s"] <= horizon
        ]
        name, hp = original["state"].get("target"), original["state"].get("target_health")
        count += 1
        complete += any(r.get("elapsed_s", 0) >= original["elapsed_s"] + horizon for r in sequence)
        health_drop += any(
            r.get("state", {}).get("target") == name
            and hp is not None
            and r.get("state", {}).get("target_health") is not None
            and r["state"]["target_health"] < hp - 0.02
            for r in later
        )
        out_of_range += any(r.get("state", {}).get("out_of_range") for r in later)
    return {
        "attack_labels": count,
        "complete_followup_windows": complete,
        "same_named_target_health_drop": health_drop,
        "out_of_range_observed": out_of_range,
        "horizon_seconds": horizon,
        "scope": "Observations only. Other players or same-named targets can explain damage; delayed outcomes may be missed. Not successful-attack labels or a runtime policy.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--p2p", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    groups = collect(args.runs)
    splits = {
        "train": [r for g in groups[:-4] for r in g],
        "validation": [r for g in groups[-4:-2] for r in g],
        "test": [r for g in groups[-2:] for r in g],
    }
    # Keep a compact, game-balanced replay of the original multi-game data.
    rng = np.random.default_rng(271)
    for split in splits:
        rows = [json.loads(line) for line in (args.p2p / f"{split}.jsonl").read_text().splitlines()]
        by_game = {}
        for row in rows:
            by_game.setdefault(row["game"], []).append(row)
        for candidates in by_game.values():
            for index in rng.choice(
                len(candidates), min(len(candidates), 64 if split == "train" else 24), replace=False
            ):
                row = candidates[int(index)]
                splits[split].append(
                    {
                        **{
                            k: row[k]
                            for k in ("id", "game", "episode", "goal", "controls", "action")
                        },
                        "frames": [
                            {**f, "image": str((args.p2p / f["image"]).resolve())}
                            for f in row["frames"]
                        ],
                        "previous_actions": [],
                        "provenance": {"type": "p2p_human_control", "source": row["source"]},
                    }
                )
    # Never silently retain identical screenshots on different sides of the split.
    owners, filtered, removed = {}, {}, Counter()
    for split in ("test", "validation", "train"):
        filtered[split] = []
        for row in splits[split]:
            frame = row["frames"][0]
            digest = hashlib.sha256(Path(frame["image"]).read_bytes()).hexdigest()
            frame["sha256"] = digest
            if digest in owners:
                removed[split] += 1
                continue
            owners[digest] = split
            filtered[split].append(row)
        (args.output / f"{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in filtered[split])
        )
    audit = {
        "scope": "Weak recorded Hordes controller demonstrations plus human multi-game replay; no game state in inputs",
        "pulse_limitation": "Original 50–80 ms key holds and multi-event camera drags are normalized to one 50 ms target, not dense human 20 FPS labels",
        "rows": {s: len(r) for s, r in filtered.items()},
        "games": {s: dict(Counter(r["game"] for r in rows)) for s, rows in filtered.items()},
        "runs": {
            s: sorted({r["episode"] for r in rows if r["game"] == "Hordes.io"})
            for s, rows in filtered.items()
        },
        "hordes_buttons": {
            s: dict(
                Counter(b for r in rows if r["game"] == "Hordes.io" for b in r["action"]["buttons"])
            )
            for s, rows in filtered.items()
        },
        "duplicate_images_removed": dict(removed),
        "attack_outcome_audit": {s: attack_outcomes(rows) for s, rows in filtered.items()},
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
