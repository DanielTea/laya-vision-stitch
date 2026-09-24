"""Official-reference parity and fresh-frame timing for the pretrained policy port."""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from PIL import Image

from laya_vision_stitch.p2p_pretrained_policy import OpenP2PPolicy, physical_action
from laya_vision_stitch.p2p_pretrained_vision import P2P_ID, P2P_REVISION, preprocess


def comparison(actual, expected):
    actual = np.asarray(actual.astype(mx.float32))
    expected = expected.detach().float().numpy()
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise FloatingPointError("Nonfinite reference comparison")
    difference = actual - expected
    metrics = {
        "max_abs_error": float(np.abs(difference).max()),
        "relative_l2_error": float(
            np.linalg.norm(difference) / max(np.linalg.norm(expected), 1e-8)
        ),
    }
    if metrics["relative_l2_error"] > 1e-4 or metrics["max_abs_error"] > 0.005:
        raise RuntimeError(f"FP32 policy parity failed: {metrics}")
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=240)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.upstream.resolve()))
    import torch
    from elefant.policy_model.action_decoder import ActionDecoder, ActionDecoderConfig
    from elefant.policy_model.kv_cache import RollingStepKVCache
    from elefant.policy_model.policy_transformer import _img_policy_causal_mask
    from elefant.policy_model.transformer import Transformer
    from torch.nn.attention import flex_attention as fa
    from torchvision.models import efficientnet_b0

    torch.set_num_threads(4)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)[
        "state_dict"
    ]
    state = {k: v.float() for k, v in state.items()}
    model = OpenP2PPolicy.from_state({k: v.float().numpy() for k, v in state.items()})
    depth = len(model.policy.layers)

    def subset(prefix):
        return {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}

    vision = efficientnet_b0(weights=None).features[:6].eval()
    vision.load_state_dict(subset("image_tokenizer.efficientnet_preprocess."), strict=True)
    projection = torch.nn.Sequential(
        torch.nn.Linear(112 * 12 * 12, 1024), torch.nn.LayerNorm(1024)
    ).eval()
    projection.load_state_dict(subset("image_tokenizer.mlp."), strict=True)
    policy = Transformer(
        SimpleNamespace(
            embed_dim=1024,
            n_q_head=16,
            n_kv_head=16,
            n_transformer_layers=depth,
            dropout=0,
            n_kv_sink_tokens=0,
        )
    )
    policy.max_seq_len = 4096
    policy.construct_transformer_layers()
    policy.load_state_dict(subset("bc_transformer._transformer."), strict=True)
    policy.eval()
    cache_config = RollingStepKVCache(1, 12, 200, 16, 64, torch.device("cpu"), torch.float32)
    for layer in policy.transformer_layers:
        layer.self_attention.kv_cache = cache_config
    decoder = (
        ActionDecoder(
            ActionDecoderConfig(
                embed_dim=1024,
                input_action_token_dim=1024,
                n_transformer_layers=3,
                n_q_head=8,
                n_kv_head=8,
                n_action_tokens=9,
            )
        )
        .float()
        .eval()
    )
    decoder.load_state_dict(subset("bc_transformer.action_decoder."), strict=True)
    ref_cache = [cache_config.init_state() for _ in range(depth)]
    native_cache = None
    frames = sorted(args.frames.glob("*.jpg"))
    if len(frames) < 3:
        raise ValueError("Need at least three real screenshots")
    report = {
        "model": P2P_ID,
        "revision": P2P_REVISION,
        "parity": [],
        "input_events_sent": 0,
        "scope": "Pretrained Open-P2P conversion only. No Laya yet; no live gameplay validation. Fresh predictions exclude screen capture and event dispatch. Uses the byte-checked upstream Hamming interpolation port.",
    }
    embedding_names = [
        "key_action_embedding",
        "mouse_button_embedding",
        "mouse_delta_x_embedding",
        "mouse_delta_y_embedding",
    ]
    output_names = [
        "keyboard_out_logits",
        "mouse_button_out_logits",
        "mouse_delta_x_out_logits",
        "mouse_delta_y_out_logits",
    ]

    def ref_embeddings(tokens):
        return torch.stack(
            [
                torch.nn.functional.embedding(
                    tokens[:, i], state[embedding_names[model.action_type(i)] + ".weight"].float()
                )
                for i in range(8)
            ],
            1,
        )

    def logits_at(h, i):
        root = output_names[model.action_type(i)]
        return torch.nn.functional.linear(h, state[root + ".weight"], state[root + ".bias"])

    with torch.inference_mode():
        for step, frame in enumerate(frames[:3]):
            pixels = preprocess(Image.open(frame))
            forced = np.array([[11 if step % 2 == 0 else 6, 0, 0, 0, 1, 0, 12, 8]], np.int32)
            tokens, logits, next_cache, context = model.step(
                mx.array(pixels), caches=native_cache, position=step * 12, forced=mx.array(forced)
            )
            mx.eval(tokens, logits, next_cache, context)
            im = projection(vision(torch.from_numpy(pixels).permute(0, 3, 1, 2)).flatten(1))[
                :, None
            ]
            prefix = torch.cat(
                [
                    state["bc_transformer.text_embedding_for_no_text_input"]
                    + state["bc_transformer.text_pos_tokens"],
                    im + state["bc_transformer.img_pos_tokens"],
                    state["bc_transformer.thinking_pos_tokens"],
                    state["bc_transformer.action_out_token"],
                ],
                1,
            ).float()
            base_mask = _img_policy_causal_mask(
                0, 1, 1, 9, history_len=200, n_text_tokens=1, n_kv_sink_tokens=0
            )
            offset = step * 12

            def mask_fn(b, h, q, k):
                return base_mask(b, h, q + offset, k)

            mask = fa.create_block_mask(
                mask_fn, B=None, H=None, Q_LEN=12, KV_LEN=offset + 12, BLOCK_SIZE=128, device="cpu"
            )
            for layer in policy.transformer_layers:
                layer.self_attention.block_mask = mask
            x = torch.cat([prefix, torch.zeros(1, 8, 1024)], 1)
            result, *_ = policy(
                x, input_pos=torch.arange(offset, offset + 12), kv_cache_state=ref_cache
            )
            ref_context = result[:, 3:4]
            context_parity = comparison(context, ref_context)
            embeddings = ref_embeddings(torch.from_numpy(forced).long())
            decoded = decoder(ref_context, embeddings[:, None])
            comparisons = [comparison(logits[i], logits_at(decoded[:, 0, i], i)) for i in range(8)]
            item = {
                "frame": frame.name,
                "context": context_parity,
                "logits": comparisons,
                "argmax_disagreements": sum(
                    int(
                        np.argmax(np.asarray(logits[i]))
                        != int(logits_at(decoded[:, 0, i], i).argmax())
                    )
                    for i in range(8)
                ),
            }
            x = torch.cat([prefix, embeddings + state["bc_transformer.action_pos_tokens"]], 1)
            _, ref_cache, *_ = policy(
                x, input_pos=torch.arange(offset, offset + 12), kv_cache_state=ref_cache
            )
            item["cache_max_abs_error"] = max(
                comparison(a, b)["max_abs_error"]
                for ours, ref in zip(next_cache, ref_cache, strict=True)
                for a, b in zip(ours, ref, strict=True)
            )
            report["parity"].append(item)
            native_cache = next_cache
            print(
                json.dumps(
                    {
                        "parity_frame": step,
                        "context": item["context"],
                        "cache_max_abs_error": item["cache_max_abs_error"],
                    }
                ),
                flush=True,
            )
    model.save_weights(str(args.output / "model.safetensors"))
    (args.output / "config.json").write_text(
        json.dumps(
            {
                "format": "open-p2p-mlx-1",
                "model": P2P_ID,
                "revision": P2P_REVISION,
                "precision": "float32",
                "laya_stitched": False,
                "depth": depth,
                "checkpoint": str(args.checkpoint),
            },
            indent=2,
        )
        + "\n"
    )
    del vision, projection, decoder, policy, ref_cache
    state = {}
    native_cache, times, actions = None, [], []
    for step in range(args.steps):
        image = Image.open(frames[step % len(frames)]).convert("RGB")
        start = time.perf_counter()
        output = model.step(mx.array(preprocess(image)), caches=native_cache, position=step * 12)
        mx.eval(output)
        action = physical_action(output[0])
        times.append((time.perf_counter() - start) * 1000)
        native_cache = output[2]
        actions.append({"step": step, "frame": frames[step % len(frames)].name, **action})
        if step % 50 == 0:
            print(
                json.dumps({"step": step, "latency_ms": times[-1], "buttons": action["buttons"]}),
                flush=True,
            )
    report["timing"] = {}
    for name, values in (("growing_cache", times[5:200]), ("full_200_frame_cache", times[200:])):
        if values:
            report["timing"][name] = {
                "samples": len(values),
                "p50_ms": float(np.median(values)),
                "p95_ms": float(np.percentile(values, 95)),
            }
    report["predicted_buttons"] = dict(Counter(b for a in actions for b in a["buttons"]))
    report["nonzero_mouse_steps"] = sum(any(a["mouse_delta"]) for a in actions)
    (args.output / "predictions.jsonl").write_text("".join(json.dumps(a) + "\n" for a in actions))
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({"timing": report["timing"], "predicted_buttons": report["predicted_buttons"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
