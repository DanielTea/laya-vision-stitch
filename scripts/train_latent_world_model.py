"""Train action-conditioned latent world models and measure action controllability.

Variants: `ridge` (closed-form per-horizon linear model on context, image token and
actions), `mlp_action_only` (residual rollout driven by actions alone) and
`mlp_full_content` (rollout from the full context and image token). Each has an
action-blind twin trained on the same rows.
p2p: fit on P2P train; evaluate validation/test/fresh_test with true, shuffled (random
     other frame of the same game), no-input and action-blind inputs.
hordes: fit on live-trial frames with dispatched actions (contexts from
     `train_latent_actions.py encode-hordes`); evaluate on a held-out trial.
Validation MSE selects every checkpoint and penalty; test splits and held-out trials never
do. Nothing here sends keyboard or mouse input or runs a policy-improvement loop.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from laya_vision_stitch.flow_action_head import BINARY, CONTROLS, DIM, tokens_to_vectors
from laya_vision_stitch.latent_actions import MOVEMENT, Standardizer, load_split, read_jsonl
from laya_vision_stitch.latent_world_model import (
    LatentWorldModel,
    RidgeWorldModel,
    action_index,
    controllability,
    fit_ridge_world_model,
    future_index,
    train_world_model,
)

EVAL_SPLITS = ("validation", "test", "fresh_test")
VARIANTS = ("ridge", "mlp_action_only", "mlp_full_content")
NO_INPUT = np.concatenate([-np.ones(BINARY), np.zeros(DIM - BINARY)]).astype(np.float32)
MOVE_COLUMNS = [CONTROLS.index(k) for k in MOVEMENT]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def plain(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list | tuple):
        return [plain(v) for v in value]
    return value


def save_args(args):
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "args.json", {k: plain(v) for k, v in vars(args).items() if k != "func"}
    )


def chunks(z, contexts, vectors, sequence_index, steps, horizon, complete=None, starts=None):
    """Rollout inputs/targets; chunks stop at sequence ends and incomplete action labels."""
    index, mask = future_index(sequence_index, steps, horizon)
    acts = action_index(index)
    if complete is not None:
        mask &= np.cumprod(complete[acts], 1).astype(bool)
    if starts is not None:
        mask &= starts[:, None]
    keep = mask[:, 0]
    return {
        "context": contexts[keep].astype(np.float32),
        "z": z[keep],
        "actions": vectors[acts[keep]],
        "targets": z[index[keep]],
        "mask": mask[keep],
        "rows": np.flatnonzero(keep),
    }


def save_ridge(model, path):
    arrays = {"context_mean": model.context_mean, "context_scale": model.context_scale}
    for h, (w, b, m) in enumerate(zip(model.weights, model.biases, model.means, strict=True)):
        arrays |= {f"weight_{h}": w, f"bias_{h}": b, f"mean_{h}": m}
    np.savez(path, blind=model.blind, **{k: np.asarray(v) for k, v in arrays.items()})


def load_ridge(path):
    a = np.load(path)
    horizon = len([k for k in a.files if k.startswith("weight_")])
    return RidgeWorldModel(
        [a[f"weight_{h}"] for h in range(horizon)],
        [a[f"bias_{h}"] for h in range(horizon)],
        [a[f"mean_{h}"] for h in range(horizon)],
        a["context_mean"],
        a["context_scale"],
        bool(a["blind"]),
    )


def fit_variant(args, variant, train, validation):
    """Action-conditioned and action-blind models on the same rows and budget."""
    models, reports = {}, {}
    for name, blind in (("action_conditioned", False), ("action_blind", True)):
        if variant == "ridge":
            models[name], reports[name] = fit_ridge_world_model(
                train, validation, tuple(args.penalties), blind
            )
            continue
        mx.random.seed(args.seed)
        content = 0 if variant == "mlp_action_only" else -1
        model = LatentWorldModel(width=args.width, content=content)
        mx.eval(model.parameters())
        reports[name] = train_world_model(
            model,
            train,
            validation,
            steps=args.steps,
            batch=args.batch,
            learning_rate=args.learning_rate,
            seed=args.seed,
            every=args.every,
            blind=blind,
            weight_decay=args.weight_decay,
        )
        reports[name]["parameters"] = sum(v.size for _, v in tree_flatten(model.parameters()))
        models[name] = model
    return models, reports


def save_variant(models, variant, prefix):
    for name, model in models.items():
        if variant == "ridge":
            save_ridge(model, f"{prefix}_{name}.npz")
        else:
            model.save_weights(f"{prefix}_{name}.safetensors")


def change_by_activity(data, active):
    """One-step copy-last MSE on frames where the action fact holds versus not."""
    copy = ((data["targets"][:, 0] - data["z"]) ** 2).mean(-1)
    if not active.any() or active.all():
        return None
    return {
        "active_frames": int(active.sum()),
        "active": float(copy[active].mean()),
        "inactive": float(copy[~active].mean()),
        "ratio": float(copy[active].mean() / copy[~active].mean()),
    }


def evaluate(args, models, data, rng):
    return controllability(
        models["action_conditioned"],
        data,
        data["games"],
        data["sequences"],
        rng,
        draws=args.draws,
        blind_model=models["action_blind"],
        no_input=NO_INPUT,
    )


def p2p(args):
    save_args(args)
    rng = np.random.default_rng(args.seed)
    data = {s: load_split(args.cache, s) for s in ("train", *EVAL_SPLITS)}
    std = Standardizer.fit(data["train"][0]["images"])
    std.save(args.output / "standardizer.npz")
    sets = {}
    for split, (a, rows) in data.items():
        sets[split] = chunks(
            std(a["images"]),
            a["contexts"],
            tokens_to_vectors(a["tokens"]),
            a["sequence_index"],
            a["steps"],
            args.horizon,
            complete=a["label_complete"],
        )
        keep = sets[split]["rows"]
        sets[split]["games"] = np.array([rows[i]["game"] for i in keep])
        sets[split]["sequences"] = a["sequence_index"][keep]
    write_json(
        args.output / "model_config.json",
        {"width": args.width, "horizon": args.horizon, "variants": list(args.variants)},
    )
    report = {
        "examples": {s: int(len(v["z"])) for s, v in sets.items()},
        "change_by_activity": {},
        "variants": {},
    }
    for split in EVAL_SPLITS:
        v = sets[split]
        report["change_by_activity"][split] = {
            "mouse_motion_at_t": change_by_activity(v, np.any(v["actions"][:, 0, BINARY:] != 0, 1)),
            "any_control_at_t": change_by_activity(v, np.any(v["actions"][:, 0, :BINARY] > 0, 1)),
        }
    for variant in args.variants:
        models, training = fit_variant(args, variant, sets["train"], sets["validation"])
        save_variant(models, variant, str(args.output / variant))
        result = {"training": training, "splits": {}}
        for split in EVAL_SPLITS:
            result["splits"][split] = evaluate(args, models, sets[split], rng)
        report["variants"][variant] = result
        summary = {
            s: {h: round(r["relative_gain_vs_shuffled"], 5) for h, r in v.items()}
            for s, v in result["splits"].items()
        }
        print(json.dumps({variant: summary}), flush=True)
    write_json(args.output / "report.json", report)


def load_trial(cache, name):
    a = dict(np.load(Path(cache) / f"{name}.npz"))
    if len(read_jsonl(Path(cache) / f"{name}.jsonl")) != len(a["images"]):
        raise ValueError(f"{name}: rows and arrays disagree")
    return a


def previous_activity(vectors, frames):
    """Movement key / mouse motion in the action dispatched one frame before each start."""
    prior = vectors[np.clip(frames - 1, 0, None)]
    known = frames >= 1
    return {
        "movement_key_at_t-1": known & np.any(prior[:, MOVE_COLUMNS] > 0, 1),
        "mouse_motion_at_t-1": known & np.any(prior[:, BINARY:] != 0, 1),
    }


def hordes(args):
    save_args(args)
    rng = np.random.default_rng(args.seed)
    std = Standardizer.load(args.init / "standardizer.npz")
    trials = {t: load_trial(args.hordes_cache, t) for t in args.trials}
    report = {
        "folds": {},
        "context_note": "32-frame teacher-forced windows with dispatched actions",
    }
    for test in args.test_trials:
        parts = {"fit": [], "validation": [], "test": []}
        for name, a in trials.items():
            n = len(a["images"])
            vectors = tokens_to_vectors(a["tokens"])
            inputs = (
                std(a["images"]),
                a["contexts"],
                vectors,
                np.zeros(n, np.int64),
                a["frame"],
                args.horizon,
            )
            if name == test:
                selected = {"test": np.ones(n, bool)}
            else:
                cut = int(round(n * (1 - args.validation_fraction)))
                frame = np.arange(n)
                # Fit chunks end before the validation block starts (no shared targets).
                selected = {"fit": frame < cut - args.horizon, "validation": frame >= cut}
            for part, starts in selected.items():
                c = chunks(*inputs, starts=starts)
                c["games"] = np.full(len(c["z"]), name)
                c["sequences"] = np.array([f"{name}/{w}" for w in a["sequence_index"][c["rows"]]])
                c |= previous_activity(vectors, a["frame"][c["rows"]])
                parts[part].append(c)
        merged = {
            part: {k: np.concatenate([c[k] for c in cs]) for k in cs[0]}
            for part, cs in parts.items()
        }
        test_data = merged["test"]
        fold = {
            "test_trial": test,
            "train_trials": [t for t in args.trials if t != test],
            "examples": {k: int(len(v["z"])) for k, v in merged.items()},
            "change_by_activity": {
                k: change_by_activity(test_data, test_data[k])
                for k in ("movement_key_at_t-1", "mouse_motion_at_t-1")
            },
            "variants": {},
        }
        zero_shot = {
            name: load_ridge(args.init / f"ridge_{name}.npz")
            for name in ("action_conditioned", "action_blind")
        }
        fold["variants"]["p2p_ridge_zero_shot"] = {
            "training": None,
            "evaluation": {
                p: evaluate(args, zero_shot, merged[p], rng) for p in ("validation", "test")
            },
        }
        for variant in args.variants:
            models, training = fit_variant(args, variant, merged["fit"], merged["validation"])
            save_variant(models, variant, str(args.output / f"{variant}_holdout_{test}"))
            fold["variants"][variant] = {
                "training": training,
                "evaluation": {
                    p: evaluate(args, models, merged[p], rng) for p in ("validation", "test")
                },
            }
        report["folds"][test] = fold
        summary = {
            v: {
                h: round(r["relative_gain_vs_shuffled"], 5)
                for h, r in x["evaluation"]["test"].items()
            }
            for v, x in fold["variants"].items()
        }
        print(json.dumps({"holdout": test, "relative_gain_vs_shuffled": summary}), flush=True)
    write_json(args.output / "report.json", report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(required=True)
    for name, func in (("p2p", p2p), ("hordes", hordes)):
        s = sub.add_parser(name)
        s.add_argument("--output", type=Path, required=True)
        s.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
        s.add_argument("--horizon", type=int, default=4)
        s.add_argument("--width", type=int, default=512)
        s.add_argument("--batch", type=int, default=256)
        s.add_argument("--draws", type=int, default=5)
        s.add_argument("--weight-decay", type=float, default=1e-4)
        s.add_argument("--penalties", nargs="+", type=float, default=[1e3, 1e4, 3e4, 1e5])
        s.add_argument("--seed", type=int, default=20260923)
        s.set_defaults(func=func)
        if name == "p2p":
            s.add_argument("--cache", type=Path, required=True)
            s.add_argument("--steps", type=int, default=3000)
            s.add_argument("--learning-rate", type=float, default=3e-4)
            s.add_argument("--every", type=int, default=250)
        else:
            s.add_argument("--init", type=Path, required=True, help="P2P world-model directory")
            s.add_argument("--hordes-cache", type=Path, required=True)
            s.add_argument("--trials", nargs="+", required=True)
            s.add_argument("--test-trials", nargs="+", required=True)
            s.add_argument("--validation-fraction", type=float, default=0.2)
            s.add_argument("--steps", type=int, default=1500)
            s.add_argument("--learning-rate", type=float, default=1e-4)
            s.add_argument("--every", type=int, default=100)
    args = p.parse_args()
    if not 1 <= args.horizon <= 8:
        raise ValueError("Horizon must be 1-8 frames")
    args.func(args)


if __name__ == "__main__":
    main()
