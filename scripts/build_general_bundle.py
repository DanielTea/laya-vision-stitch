"""Assemble one checkpoint: stitched policy with trained LoRA, plus the general pointer head.

The pointer head and its frozen C-RADIOv3-B encoder are stored in the same weights file,
so inference still loads a single bundle. Nothing game-specific is added.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx

from laya_vision_stitch.laya_p2p import LayaP2PRuntime
from laya_vision_stitch.p2p_adaptation import install_control_adapter, install_policy_lora
from laya_vision_stitch.pointer_head import PointerHead
from laya_vision_stitch.radio_vision import RadioVision
from laya_vision_stitch.target_conditioning import install_target_encoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base", type=Path, required=True, help="Stitched bundle the LoRA was trained on"
    )
    p.add_argument("--policy", type=Path, required=True, help="train_policy_lora_sequences output")
    p.add_argument("--pointer", type=Path, required=True, help="train_pointer_radio output")
    p.add_argument("--radio", type=Path, default=Path("artifacts/radio-source"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    runtime = LayaP2PRuntime.load(args.base)
    trained = json.loads((args.policy / "config.json").read_text())
    install_control_adapter(runtime.model, **trained["control_adapter"])
    if trained.get("policy_lora"):
        install_policy_lora(runtime.model, **trained["policy_lora"])
    if "target_encoder" in trained:
        install_target_encoder(runtime.model, **trained["target_encoder"])
    runtime.model.load_weights(str(args.policy / "model.safetensors"), strict=True)
    pointer = json.loads((args.pointer / "config.json").read_text())
    head = PointerHead(**pointer["pointer_head"])
    head.load_weights(str(args.pointer / "pointer_head.safetensors"), strict=True)
    runtime.model.policy.pointer_head = head
    runtime.model.policy.pointer_encoder = RadioVision.from_source(args.radio, mx.float16)
    mx.eval(runtime.model.parameters())
    policy_report = json.loads((args.policy / "report.json").read_text())
    pointer_report = json.loads((args.pointer / "report.json").read_text())
    runtime.metadata.update(
        {
            **{
                k: trained[k]
                for k in ["control_adapter", "policy_lora", "target_encoder", "key_names"]
                if k in trained
            },
            "pointer_head": pointer["pointer_head"],
            "pointer_encoder": "radio",
            "pointer_encoder_source": "nvidia/C-RADIOv3-B 44653a0482cf460bb4f12595fc3cc3dfecc403d1 (FP16)",
            "training": {
                "policy": str(args.policy),
                "policy_selected_step": policy_report["selected_step"],
                "pointer": str(args.pointer),
                "pointer_selected_step": pointer_report["selected_step"],
                "data": "D2E-480p games (24 training, 5 held out), public P2P replay; no Hordes data",
            },
            "deployment_eligible": False,
        }
    )
    runtime.save(args.output)
    reloaded = LayaP2PRuntime.load(args.output)
    ok = all(
        mx.array_equal(a, b).item()
        for a, b in zip(
            [v for _, v in sorted(_flat(runtime.model.parameters()))],
            [v for _, v in sorted(_flat(reloaded.model.parameters()))],
            strict=True,
        )
    )
    print(json.dumps({"output": str(args.output), "reload_identical": bool(ok)}))


def _flat(tree):
    from mlx.utils import tree_flatten

    return tree_flatten(tree)


if __name__ == "__main__":
    main()
