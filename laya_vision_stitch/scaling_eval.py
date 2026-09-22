"""Held-out compositions/styles, counterfactual goals and visual controls."""

import argparse
import json
from pathlib import Path

import numpy as np

from .policy_data import check_separation, read_manifest
from .policy_training import cache_examples, evaluate
from .scaling_data import HOLDOUT
from .trainable_model import TrainableRuntime


class ShuffledImages:
    def __init__(self, examples):
        self.examples = examples
        # Shift whole scenes, keeping opposite-goal pairs together.
        self.offset = 2

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        row, inputs = self.examples[index]
        other = self.examples[(index + self.offset) % len(self.examples)][1]
        return row, (*other[:2], *inputs[2:])


def summarize(result):
    return {k: v for k, v in result.items() if k != "predictions"}


def run(bundle, data, output, cache=None, training_manifest=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    runtime = TrainableRuntime.load(bundle)
    train = read_manifest(
        training_manifest or Path(data) / "train.jsonl", runtime.module.policy_config
    )
    result = {"bundle": str(bundle), "splits": {}, "cross_game_capability_established": False}
    for name in ("validation", "composition_test", "style_test"):
        rows = read_manifest(Path(data) / f"{name}.jsonl", runtime.module.policy_config)
        check_separation(train, rows)
        examples = cache_examples(runtime, rows, cache)
        actual = evaluate(runtime, examples)
        controls = {
            "zero_visual": evaluate(runtime, examples, zero_visual=True),
            "shuffled_images": evaluate(runtime, ShuffledImages(examples)),
        }
        reversed_examples = []
        for row, inputs in examples:
            reverse = dict(row, choices=dict(reversed(list(row["choices"].items()))))
            reversed_examples.append((reverse, (*inputs[:2], *runtime.prepare(reverse))))
        controls["reversed_options"] = evaluate(runtime, reversed_examples)
        # A scene-level bootstrap retains correlated opposite-goal questions.
        correct = np.array(
            [p["choice"] == r["answer"] for p, r in zip(actual["predictions"], rows, strict=True)]
        )
        novel = np.array(
            [tuple(r["scene"]["objects"][r["scene"]["target"]][:2]) in HOLDOUT for r in rows]
        )
        paired = correct.reshape(-1, 2).mean(axis=1)
        rng = np.random.default_rng(81)
        bootstrap = paired[rng.integers(len(paired), size=(5000, len(paired)))].mean(axis=1)
        result["splits"][name] = {
            "actual": summarize(actual),
            "novel_target_count": int(novel.sum()),
            "novel_target_accuracy": float(correct[novel].mean()) if novel.any() else None,
            "accuracy_scene_bootstrap_95": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
            **{k: summarize(v) for k, v in controls.items()},
        }
        print(json.dumps({"split": name, **result["splits"][name]}), flush=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument(
        "--training-manifest", type=Path, help="Full actual training manifest for leakage checks"
    )
    args = parser.parse_args()
    run(args.bundle, args.data, args.output, args.feature_cache, args.training_manifest)


if __name__ == "__main__":
    main()
