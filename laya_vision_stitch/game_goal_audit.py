"""Prepare reviewed withheld-game probes, then evaluate a gated fixed checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

from .d2e_data import BUTTONS
from .decoder_probe import prepare
from .goal_curriculum import DEVELOPMENT, TRAIN, expand
from .policy_data import check_separation, read_manifest
from .robust_decoder import score
from .trainable_model import PolicyConfig, TrainableRuntime


def prepare_manifest(source, annotations):
    rows = read_manifest(source / "test.jsonl", PolicyConfig(buttons=BUTTONS))
    review = json.loads(annotations.read_text())
    labels = {r["id"]: r for r in review["labels"]}
    seen = set()
    unique = []
    excluded = []
    for row in rows:
        digest = row["frames"][-1]["sha256"]
        if digest in seen:
            excluded.append(row["id"])
        else:
            seen.add(digest)
            unique.append(row)
    templates = [("seen-0", TRAIN[0])] + [(f"dev-{i}", s) for i, s in enumerate(DEVELOPMENT)]
    expanded = expand(unique, labels, templates)
    with (source / "goal-test.jsonl").open("x") as f:
        f.writelines(json.dumps(r) + "\n" for r in expanded)
    protocol = {
        "frames": len(unique),
        "examples": len(expanded),
        "duplicate_current_frames_removed": excluded,
        "game_gate": {"balanced_accuracy_min": 0.85, "both_goals_correct_min": 0.80},
        "scope": "Games withheld from this project training; unknown foundation pretraining overlap. Synthetic menu-conditioned actions, not gameplay.",
        "sampling": "Uniform timeline; strongly correlated within each recording",
        "annotations_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
    }
    (source / "goal-protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


def wording_passed(results):
    return (
        isinstance(results, dict)
        and bool(results.get("by_template"))
        and all(
            m["button_exact_match"] >= 0.85
            and m["balanced_button_exact_match"] >= 0.80
            and m["both_goals_correct"] >= 0.75
            for m in results["by_template"].values()
        )
    )


def evaluate(args):
    previous = json.loads((args.bundle.parent / "report.json").read_text())
    if not (
        previous["train_gate_passed"]
        and previous["validation_gate_passed"]
        and wording_passed(previous["sealed_wording"])
    ):
        raise ValueError(
            "Training, development and reserved-wording gates must pass before fresh-game scoring"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = TrainableRuntime.load(args.bundle)
    config = runtime.module.policy_config
    rows = read_manifest(args.source / "goal-test.jsonl", config)
    training = read_manifest(args.training_manifest, config)
    check_separation(training, rows, holdout_games=True)
    protocol = json.loads((args.source / "goal-protocol.json").read_text())
    report = {
        "source_checkpoint": str(args.bundle),
        "protocol": protocol,
        "games": {},
        "live_inputs_sent": 0,
    }
    for game in sorted({r["game"] for r in rows}):
        selected = [r for r in rows if r["game"] == game]
        report["games"][game] = {
            variant: score(runtime, selected, *prepare(runtime, selected, args.cache, variant))
            for variant in ("actual", "shuffled_vision", "zero_vision")
        }
    report["passed"] = all(
        m["actual"]["overall"]["balanced_button_exact_match"]
        >= protocol["game_gate"]["balanced_accuracy_min"]
        and m["actual"]["overall"]["both_goals_correct"]
        >= protocol["game_gate"]["both_goals_correct_min"]
        for m in report["games"].values()
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("artifacts/fresh-games-001"))
    p.add_argument(
        "--annotations", type=Path, default=Path("annotations/fresh-games-menu-001.json")
    )
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--bundle", type=Path)
    p.add_argument(
        "--training-manifest", type=Path, default=Path("artifacts/goal-curriculum-002/train.jsonl")
    )
    p.add_argument("--cache", type=Path, default=Path("artifacts/vision-feature-cache"))
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.prepare_only:
        print(prepare_manifest(args.source, args.annotations))
    elif args.bundle is None or args.output is None:
        p.error("--bundle and --output are required for evaluation")
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
