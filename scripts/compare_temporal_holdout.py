"""Compare frozen checkpoints on newly held-out gameplay without training."""

import argparse
import json
from pathlib import Path

import mlx.core as mx

from laya_vision_stitch.policy_training import fingerprint
from laya_vision_stitch.temporal_training import cache_sequences, evaluate, read_sequences
from laya_vision_stitch.trainable_model import TrainableRuntime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, default=Path("artifacts/p2p-pilot-004/bundle"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--bundles", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=Path("artifacts/temporal-feature-cache"))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    parent = TrainableRuntime.load(a.parent)
    frozen = {
        k: fingerprint(getattr(parent.module, k))
        for k in ("vision", "laya", "connector", "actions")
    }
    _, groups = read_sequences(a.data, parent.module.policy_config)
    sequences = groups["fresh_test"]
    data = cache_sequences(parent, {"fresh_test": sequences}, a.cache, frozen)["fresh_test"]
    report = {
        "scope": "Same fresh held-out recordings for old and new frozen checkpoints; no training or checkpoint selection",
        "models": {},
    }
    for path in a.bundles:
        runtime = TrainableRuntime.load(path)
        if frozen != {k: fingerprint(getattr(runtime.module, k)) for k in frozen}:
            raise ValueError("Different frozen parent")
        result = evaluate(
            runtime.module.temporal_actions,
            sequences,
            data,
            runtime.module.policy_config,
            {"delta_scale": mx.ones(64)},
            ["actual", "shuffled_observations", "reset_each_frame", "self_fed_controls"],
        )
        # This comparison concerns controls; avoid presenting a different auxiliary scale.
        result["actual"].pop("future_delta_standardized_mse")
        result["actual"].pop("zero_delta_standardized_mse")
        report["models"][str(path)] = result
        (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: {m: v[m] for m in ["button_micro_f1", "transition_exact", "moving_mouse_mae_px"]}
                for k, r in report["models"].items()
                for v in [r["actual"]]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
