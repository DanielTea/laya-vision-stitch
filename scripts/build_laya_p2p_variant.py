"""Stitch an existing fitted Laya goal bridge onto another converted Open-P2P policy.

The bridge targets the dataset's 768D Gemma text space, which every released Open-P2P
checkpoint projects with its own text layer, so the bridge transfers without refitting.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx

from laya_vision_stitch.laya_p2p import LayaP2PRuntime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", type=Path, required=True, help="Converted open-p2p-mlx-1 bundle")
    p.add_argument("--bridge-bundle", type=Path, required=True, help="Fitted laya-p2p-1 bundle")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source = json.loads((args.bridge_bundle / "config.json").read_text())
    if not source.get("trained_goal_bridge"):
        raise ValueError("Bridge bundle has no fitted goal bridge")
    runtime = LayaP2PRuntime.build(args.policy)
    weights = mx.load(str(args.bridge_bundle / "model.safetensors"))
    bridge = [(k.removeprefix("bridge."), v) for k, v in weights.items() if k.startswith("bridge.")]
    if {k for k, _ in bridge} != {"mean", "scale", "projection.weight", "projection.bias"}:
        raise ValueError("Unexpected bridge parameters")
    runtime.model.bridge.load_weights(bridge, strict=True)
    mx.eval(runtime.model.parameters())
    runtime.metadata.update(
        {
            "trained_goal_bridge": True,
            "bridge_source": str(args.bridge_bundle),
            "bridge_regularization": source.get("bridge_regularization"),
            "goal_training_examples": source.get("goal_training_examples"),
            "preprocessing": source.get("preprocessing"),
        }
    )
    runtime.save(args.output)
    print(
        json.dumps({"output": str(args.output), "depth": len(runtime.model.policy.policy.layers)})
    )


if __name__ == "__main__":
    main()
