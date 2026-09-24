"""Fine-tune the pretrained action decoder with LoRA on cached sequence contexts.

The temporal policy stays frozen, so its teacher-forced contexts are cached once. The
decoder learns the new games' control conventions while a KL term keeps the released
decoder's distribution on public P2P replay. Validation NLL selects the checkpoint.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from laya_vision_stitch.laya_p2p import LayaP2PRuntime
from laya_vision_stitch.p2p_adaptation import install_control_adapter
from laya_vision_stitch.sequence_metrics import (
    baselines,
    evaluate_actions,
    token_buttons,
    token_mouse,
)
from laya_vision_stitch.sequence_policy import guided_decode


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load(cache, split):
    a = dict(np.load(cache / f"{split}.npz"))
    return read(cache / f"{split}.jsonl"), a


def nll(policy, contexts, tokens, reduce=True):
    logits = policy.teacher_logits(contexts[:, None], tokens)
    loss = sum(
        nn.losses.cross_entropy(logit.astype(mx.float32), tokens[:, j], reduction="none")
        for j, logit in enumerate(logits)
    )
    return loss.mean() if reduce else loss


def split_nll(policy, contexts, tokens):
    out = []
    for s in range(0, len(contexts), 512):
        loss = nll(policy, mx.array(contexts[s : s + 512]), mx.array(tokens[s : s + 512]), False)
        mx.eval(loss)
        out.append(np.asarray(loss))
    return float(np.concatenate(out).mean())


def decode(policy, contexts, temperature, seed):
    key, out = mx.random.key(seed), []
    for s in range(0, len(contexts), 512):
        key, sub = mx.random.split(key)
        tokens, _ = guided_decode(
            policy, mx.array(contexts[s : s + 512]), temperature=temperature, key=sub
        )
        mx.eval(tokens)
        out.append(np.asarray(tokens))
    return np.concatenate(out)


def evaluate(policy, cache, split, samples):
    rows, a = load(cache, split)
    truth, mouse = token_buttons(a["tokens"]), token_mouse(a["tokens"])

    def score(tokens):
        return evaluate_actions(rows, token_buttons(tokens), truth, token_mouse(tokens), mouse)

    result = {
        "frames": len(rows),
        "mean_nll": split_nll(policy, a["contexts"], a["tokens"]),
        "greedy": score(decode(policy, a["contexts"], 0.0, 0)),
        "sampled": [score(decode(policy, a["contexts"], 1.0, s)) for s in range(samples)],
        **baselines(rows, truth, mouse),
    }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--train", nargs="+", required=True, help="cache:split pairs used for training")
    p.add_argument("--replay", nargs="*", default=[], help="cache:split pairs for KL preservation")
    p.add_argument("--validation", required=True, help="cache:split used to select the checkpoint")
    p.add_argument("--evaluate", nargs="+", required=True, help="cache:split pairs to report")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--replay-kl", type=float, default=1.0)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260923)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "args.json").write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2
        )
    )
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    runtime = LayaP2PRuntime.load(args.bundle)
    model = runtime.model
    policy = model.policy

    def pairs(items):
        for item in items:
            cache, split = item.rsplit(":", 1)
            yield Path(cache), split

    reference = {}
    for cache, split in [*pairs(args.evaluate)]:
        reference[f"{cache.name}:{split}"] = evaluate(policy, cache, split, args.samples)
        print(
            json.dumps(
                {
                    "pretrained": f"{cache.name}:{split}",
                    "nll": reference[f"{cache.name}:{split}"]["mean_nll"],
                }
            ),
            flush=True,
        )

    def gather(items):
        c, t = [], []
        for cache, split in pairs(items):
            _, a = load(cache, split)
            keep = a["label_complete"]
            c.append(a["contexts"][keep])
            t.append(a["tokens"][keep])
        return (np.concatenate(c), np.concatenate(t)) if c else (None, None)

    train_c, train_t = gather(args.train)
    replay_c, replay_t = gather(args.replay)
    replay_logits = None
    if replay_c is not None:
        replay_logits = []
        for s in range(0, len(replay_c), 512):
            logits = policy.teacher_logits(
                mx.array(replay_c[s : s + 512])[:, None], mx.array(replay_t[s : s + 512])
            )
            mx.eval(logits)
            replay_logits.append([np.asarray(x.astype(mx.float32)) for x in logits])
        replay_logits = [np.concatenate([r[j] for r in replay_logits]) for j in range(8)]
    ((vcache, vsplit),) = pairs([args.validation])
    _, va = load(vcache, vsplit)
    install_control_adapter(model, rank=args.rank)
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))

    def loss_fn(m, c, t, rc, rt, rl):
        loss = nll(m.policy, c, t)
        if rc is not None:
            logits = m.policy.teacher_logits(rc[:, None], rt)
            kl = 0.0
            for j, logit in enumerate(logits):
                ref = rl[j]
                lr = ref - mx.logsumexp(ref, -1, keepdims=True)
                lp = logit.astype(mx.float32)[..., : ref.shape[-1]]
                lp = lp - mx.logsumexp(logit.astype(mx.float32), -1, keepdims=True)
                kl = kl + (mx.exp(lr) * (lr - lp)).sum(-1).mean()
            loss = loss + args.replay_kl * kl / 8
        return loss

    value_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.01)
    best = split_nll(policy, va["contexts"], va["tokens"])
    best_step, history = 0, [{"step": 0, "validation_nll": best}]
    model.save_weights(str(args.output / "model.safetensors"))
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(train_c), args.batch)
        extra = (None, None, None)
        if replay_c is not None:
            r = rng.integers(0, len(replay_c), args.batch // 2)
            extra = (
                mx.array(replay_c[r]),
                mx.array(replay_t[r]),
                [mx.array(x[r]) for x in replay_logits],
            )
        loss, grads = value_grad(model, mx.array(train_c[idx]), mx.array(train_t[idx]), *extra)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        if step % 250 == 0 or step == args.steps:
            v = split_nll(policy, va["contexts"], va["tokens"])
            history.append({"step": step, "train_loss": float(loss), "validation_nll": v})
            print(json.dumps(history[-1]), flush=True)
            if v < best:
                best, best_step = v, step
                model.save_weights(str(args.output / "model.safetensors"))
    model.load_weights(str(args.output / "model.safetensors"))
    report = {
        "trainable_parameters": trainable,
        "selected_step": best_step,
        "history": history,
        "pretrained": reference,
        "adapted": {
            f"{cache.name}:{split}": evaluate(policy, cache, split, args.samples)
            for cache, split in pairs(args.evaluate)
        },
    }
    for name in report["adapted"]:
        before, after = report["pretrained"][name], report["adapted"][name]
        print(
            json.dumps(
                {
                    name: {
                        "nll": [round(before["mean_nll"], 4), round(after["mean_nll"], 4)],
                        "greedy_f1": [
                            round(before["greedy"]["macro"]["button_f1"], 4),
                            round(after["greedy"]["macro"]["button_f1"], 4),
                        ],
                        "greedy_onset": [
                            round(before["greedy"]["macro"]["onset_f1"], 4),
                            round(after["greedy"]["macro"]["onset_f1"], 4),
                        ],
                        "sampled_onset": [
                            round(
                                float(np.mean([x["macro"]["onset_f1"] for x in before["sampled"]])),
                                4,
                            ),
                            round(
                                float(np.mean([x["macro"]["onset_f1"] for x in after["sampled"]])),
                                4,
                            ),
                        ],
                        "repeat_f1": round(after["repeat_previous"]["macro"]["button_f1"], 4),
                    }
                }
            ),
            flush=True,
        )
    config = {
        **runtime.metadata,
        "control_adapter": {"rank": args.rank},
        "deployment_eligible": False,
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
