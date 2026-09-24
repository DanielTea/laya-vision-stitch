"""Decoding comparisons on cached teacher-forced contexts: greedy, sampling and goal CFG.

Classifier-free guidance uses the released policy's own no-goal embedding as the
unconditional branch, so it needs no retraining. Results are offline imitation metrics.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from laya_vision_stitch.laya_p2p import LayaP2PRuntime
from laya_vision_stitch.sequence_metrics import (
    baselines,
    evaluate_actions,
    token_buttons,
    token_mouse,
)
from laya_vision_stitch.sequence_policy import guided_decode

GENERIC = "Continue the current activity."


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def decode_all(policy, contexts, unconditional, scale, temperature, seed):
    key = mx.random.key(seed)
    out = []
    for start in range(0, len(contexts), 256):
        key, sub = mx.random.split(key)
        c = mx.array(contexts[start : start + 256])
        u = None if unconditional is None else mx.array(unconditional[start : start + 256])
        tokens, _ = guided_decode(policy, c, u, scale, temperature, sub)
        mx.eval(tokens)
        out.append(np.asarray(tokens))
    return np.concatenate(out)


def nll(policy, contexts, tokens):
    total = []
    for start in range(0, len(contexts), 256):
        c = mx.array(contexts[start : start + 256])[:, None]
        t = mx.array(tokens[start : start + 256])
        logits = policy.teacher_logits(c, t)
        loss = sum(
            nn.losses.cross_entropy(logit.astype(mx.float32), t[:, j], reduction="none")
            for j, logit in enumerate(logits)
        )
        mx.eval(loss)
        total.append(np.asarray(loss))
    return np.concatenate(total)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["validation", "test", "fresh_test"])
    p.add_argument("--scales", nargs="+", type=float, default=[2.0, 4.0])
    p.add_argument("--samples", type=int, default=3)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    policy = LayaP2PRuntime.load(args.bundle).model.policy
    report = {}
    for split in args.splits:
        if not (args.cache / f"{split}.npz").exists():
            continue
        rows = read(args.cache / f"{split}.jsonl")
        a = np.load(args.cache / f"{split}.npz")
        truth, true_mouse = token_buttons(a["tokens"]), token_mouse(a["tokens"])
        result = {"frames": len(rows), **baselines(rows, truth, true_mouse)}

        def score(tokens):
            return evaluate_actions(
                rows, token_buttons(tokens), truth, token_mouse(tokens), true_mouse
            )

        c, u = a["contexts"], a["unconditional_contexts"]
        greedy = decode_all(policy, c, None, 1.0, 0.0, 0)
        result["greedy"] = score(greedy)
        result["greedy_no_goal"] = score(decode_all(policy, u, None, 1.0, 0.0, 0))
        sampled = [decode_all(policy, c, None, 1.0, 1.0, s) for s in range(args.samples)]
        result["sampled_t1"] = [score(s) for s in sampled]
        for scale in args.scales:
            result[f"cfg_{scale:g}_greedy"] = score(decode_all(policy, c, u, scale, 0.0, 0))
            result[f"cfg_{scale:g}_sampled"] = [
                score(decode_all(policy, c, u, scale, 1.0, s)) for s in range(args.samples)
            ]
        # Goal sensitivity on specifically instructed frames only.
        instructed = np.array([r["goal"] != GENERIC for r in rows])
        cond_nll, uncond_nll = nll(policy, c, a["tokens"]), nll(policy, u, a["tokens"])
        greedy_u = decode_all(policy, u, None, 1.0, 0.0, 0)
        result["goal_audit"] = {
            "instructed_frames": int(instructed.sum()),
            "mean_nll_with_goal": float(cond_nll[instructed].mean()) if instructed.any() else None,
            "mean_nll_without_goal": float(uncond_nll[instructed].mean())
            if instructed.any()
            else None,
            "greedy_changed_fraction": float(
                np.mean(np.any(greedy[instructed] != greedy_u[instructed], 1))
            )
            if instructed.any()
            else None,
            "all_frames_nll_with_goal": float(cond_nll.mean()),
            "all_frames_nll_without_goal": float(uncond_nll.mean()),
        }
        for scale in args.scales:
            guided = decode_all(policy, c, u, scale, 0.0, 0)
            result["goal_audit"][f"cfg_{scale:g}_greedy_changed_fraction"] = (
                float(np.mean(np.any(guided[instructed] != greedy_u[instructed], 1)))
                if instructed.any()
                else None
            )
        report[split] = result
        print(
            json.dumps(
                {
                    split: {
                        k: (v["macro"] if isinstance(v, dict) and "macro" in v else None)
                        for k, v in result.items()
                        if isinstance(v, dict)
                    }
                }
            ),
            flush=True,
        )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
