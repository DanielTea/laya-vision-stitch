"""Latent action model experiments on cached frozen P2P features and Hordes live trials.

p2p: fit the LAM on train consecutive pairs; report next-token MSE, code/action NMI and
     linear probes on validation/test/fresh_test (validation selects checkpoint, action
     lag and probe penalty).
lapa: scarce-label utility test; action heads from scratch vs pretrained on LAM codes.
encode-hordes: frozen vision tokens and 32-frame teacher-forced contexts for live trials,
     using the actions the runner reports as dispatched.
hordes: P2P LAM and a Hordes-fine-tuned LAM probed on a held-out trial.
Nothing here sends keyboard or mouse input; test splits never select weights.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from laya_vision_stitch.flow_action_head import tokens_to_vectors, vectors_to_actions
from laya_vision_stitch.latent_actions import (
    MOVEMENT,
    ActionHead,
    LatentActionModel,
    Standardizer,
    action_loss,
    binary_scores,
    button_classes,
    chance_nmi,
    code_embeddings,
    code_loss,
    combine_codes,
    fit,
    lagged_facts,
    lam_codes,
    lam_errors,
    load_split,
    mouse_direction,
    normalized_mutual_information,
    predict_vectors,
    read_jsonl,
    read_trial,
    select_probe,
    successors,
    train_lam,
)
from laya_vision_stitch.sequence_metrics import (
    baselines,
    evaluate_actions,
    token_buttons,
    token_mouse,
)

EVAL_SPLITS = ("validation", "test", "fresh_test")
P2P_FACTS = {"held": ("w", "mouse_left")}
HORDES_FACTS = {"held": ("w", "a", "s", "d", "mouse_right"), "groups": {"movement_key": MOVEMENT}}
LAGS = (0, 1, 2)


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


def load_lam(directory):
    directory = Path(directory)
    config = json.loads((directory / "lam_config.json").read_text())
    model = LatentActionModel(**config)
    model.load_weights(str(directory / "lam.safetensors"))
    model.eval()
    mx.eval(model.parameters())
    return model, config, Standardizer.load(directory / "standardizer.npz")


def perplexity(codes, size):
    out = []
    for t in range(codes.shape[1]):
        p = np.bincount(codes[:, t], minlength=size) / len(codes)
        p = p[p > 0]
        out.append(float(np.exp(-(p * np.log(p)).sum())))
    return out


class PCA:
    """Top principal directions of training features (reference probe input)."""

    def __init__(self, x, components):
        self.mean = x.mean(0)
        _, _, vt = np.linalg.svd(x - self.mean, full_matrices=False)
        self.basis = vt[:components].T

    def __call__(self, x):
        return (x - self.mean) @ self.basis


def mse_summary(errors):
    out = {k: float(np.mean(v)) for k, v in errors.items()}
    out["full_over_copy"] = out["full"] / out["copy_last"]
    out["zeroed_over_copy"] = out["zeroed_latents"] / out["copy_last"]
    return out


def probe_suite(features, facts, fit_part, select_part, eval_parts):
    """features: {name: {part: [M,D]}} for pairs; facts: {part: {fact: (y, valid)}}.

    Penalty is chosen on `select_part`; returns scores per eval part, feature and fact.
    """
    report = defaultdict(dict)
    penalties = {}
    for fact in facts[fit_part]:
        for name, parts in features.items():
            y, valid = facts[fit_part][fact]
            vy, vvalid = facts[select_part][fact]
            if not valid.any() or not vvalid.any():
                continue
            probe, penalty, _ = select_probe(
                (parts[fit_part][valid], y[valid]), (parts[select_part][vvalid], vy[vvalid])
            )
            penalties[f"{fact}/{name}"] = penalty
            for part in eval_parts:
                ey, evalid = facts[part][fact]
                if not evalid.any():
                    continue
                scores = binary_scores(ey[evalid], probe.probability(parts[part][evalid]))
                report[part].setdefault(fact, {})[name] = scores
    return dict(report), penalties


def select_lag(delta, facts_by_lag, fit_part, select_part):
    """Mean validation balanced accuracy of the delta-PCA probe per lag; highest wins."""
    table = {}
    for lag, facts in facts_by_lag.items():
        scores, _ = probe_suite({"delta": delta}, facts, fit_part, select_part, [select_part])
        values = [v["delta"]["balanced_accuracy"] for v in scores.get(select_part, {}).values()]
        table[lag] = float(np.mean(values)) if values else -1.0
    return max(table, key=table.get), table


def nmi_report(codes, buttons, mouse, valid, size, rng):
    combo = combine_codes(codes[valid], size)
    out = {}
    for name, labels in (
        ("button_set", button_classes([b for b, v in zip(buttons, valid, strict=True) if v])),
        ("mouse_direction", mouse_direction(np.asarray(mouse)[valid])),
    ):
        out[name] = {
            "nmi": normalized_mutual_information(combo, labels),
            "permuted_code_nmi": chance_nmi(combo, labels, rng),
            "examples": int(valid.sum()),
        }
    return out


def p2p(args):
    save_args(args)
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = {s: load_split(args.cache, s) for s in ("train", *EVAL_SPLITS)}
    std = Standardizer.fit(data["train"][0]["images"])
    std.save(args.output / "standardizer.npz")
    parts = {}
    for split, (a, rows) in data.items():
        z = std(a["images"])
        i = successors(a["sequence_index"], a["steps"])
        parts[split] = {"index": i, "z0": z[i], "z1": z[i + 1], "a": a, "rows": rows}
    config = {
        "tokens": args.tokens,
        "codes": args.codes,
        "dim": args.dim,
        "width": args.width,
        "context": args.context,
    }
    write_json(args.output / "lam_config.json", config)
    model = LatentActionModel(**config)
    mx.eval(model.parameters())
    tr, va = parts["train"], parts["validation"]
    training = train_lam(
        model,
        (tr["z0"], tr["z1"]),
        (va["z0"], va["z1"]),
        steps=args.steps,
        batch=args.batch,
        learning_rate=args.learning_rate,
        seed=args.seed,
        latent_dropout=args.latent_dropout,
        weight_decay=args.weight_decay,
    )
    model.save_weights(str(args.output / "lam.safetensors"))
    report = {
        "config": config,
        "parameters": sum(v.size for _, v in tree_flatten(model.parameters())),
        "training": training,
        "pairs": {s: int(len(p["index"])) for s, p in parts.items()},
        "next_token_mse": {},
        "codebook": {},
    }
    saved = {}
    for split, p in parts.items():
        p["codes"] = lam_codes(model, p["z0"], p["z1"])
        p["embed"] = code_embeddings(model, p["codes"])
        saved[f"{split}_codes"], saved[f"{split}_index"] = p["codes"], p["index"]
        report["codebook"][split] = {
            "perplexity_per_token": perplexity(p["codes"], args.codes),
            "unique_combinations": int(len(np.unique(combine_codes(p["codes"], args.codes)))),
        }
        if split != "train":
            report["next_token_mse"][split] = mse_summary(lam_errors(model, p["z0"], p["z1"]))
    np.savez(args.output / "codes.npz", **saved)
    print(json.dumps({"next_token_mse": report["next_token_mse"]}), flush=True)

    pca = PCA(tr["z1"] - tr["z0"], args.pca)
    for p in parts.values():
        p["delta"] = pca(p["z1"] - p["z0"])
        p["buttons"] = token_buttons(p["a"]["tokens"])
        p["mouse"] = token_mouse(p["a"]["tokens"])

    def facts_at(lag):
        out, targets = {}, {}
        for split, p in parts.items():
            a = p["a"]
            f, target, valid = lagged_facts(
                p["buttons"],
                p["mouse"],
                a["sequence_index"],
                a["steps"],
                p["index"],
                lag,
                **P2P_FACTS,
            )
            complete = a["label_complete"][target]
            out[split] = {k: (y, v & valid & complete) for k, (y, v) in f.items()}
            targets[split] = (target, valid & complete)
        return out, targets

    by_lag = {lag: facts_at(lag) for lag in LAGS}
    delta = {s: p["delta"] for s, p in parts.items()}
    lag, lag_table = select_lag(delta, {k: v[0] for k, v in by_lag.items()}, "train", "validation")
    facts, targets = by_lag[lag]
    features = {"latent": {s: p["embed"] for s, p in parts.items()}, "delta_pca": delta}
    probes, penalties = probe_suite(features, facts, "train", "validation", list(EVAL_SPLITS))
    report.update(
        {
            "lag_selection": {"validation_balanced_accuracy": lag_table, "selected_lag": lag},
            "probes": probes,
            "probe_penalties": penalties,
            "nmi": {},
        }
    )
    for split in EVAL_SPLITS:
        p = parts[split]
        target, valid = targets[split]
        report["nmi"][split] = nmi_report(
            p["codes"],
            [p["buttons"][t] for t in target],
            p["mouse"][target],
            valid,
            args.codes,
            rng,
        )
    write_json(args.output / "report.json", report)
    print(json.dumps({"selected_lag": lag, "nmi": report["nmi"], "probes": probes}, indent=1))


def stratified_sequences(rows, sequence_index, fraction, rng):
    """About `fraction` of each game's sequences (at least one); returns a frame mask."""
    by_game = defaultdict(set)
    for r, s in zip(rows, sequence_index, strict=True):
        by_game[r["game"]].add(int(s))
    chosen = set()
    for game in sorted(by_game):
        seqs = sorted(by_game[game])
        k = max(1, round(fraction * len(seqs)))
        chosen |= set(rng.choice(seqs, k, replace=False).tolist())
    return np.isin(sequence_index, sorted(chosen))


def action_report(head, x, a, rows):
    buttons, mouse = vectors_to_actions(predict_vectors(head, x))
    report = evaluate_actions(
        rows, buttons, token_buttons(a["tokens"]), mouse, token_mouse(a["tokens"])
    )
    return report["macro"]


def lapa(args):
    save_args(args)
    lam, config, std = load_lam(args.lam)
    data = {s: load_split(args.cache, s) for s in ("train", *EVAL_SPLITS)}
    codes, vectors = {}, {}
    for split, (a, _) in data.items():
        z = std(a["images"])
        i = successors(a["sequence_index"], a["steps"])
        c = np.full((len(z), config["tokens"]), -1, np.int64)
        c[i] = lam_codes(lam, z[i], z[i + 1])
        codes[split], vectors[split] = c, tokens_to_vectors(a["tokens"])
    report = {
        "baselines": {},
        "runs": [],
        "lam": {"config": config, "directory": str(args.lam)},
        "input_note": "contexts are teacher-forced with recorded previous actions",
    }
    for split in EVAL_SPLITS:
        a, rows = data[split]
        b = baselines(rows, token_buttons(a["tokens"]), token_mouse(a["tokens"]))
        report["baselines"][split] = {k: v["macro"] for k, v in b.items()}
    for source in args.inputs:
        x = {s: data[s][0][source].astype(np.float32) for s in data}
        a_tr, rows_tr = data["train"]
        a_va = data["validation"][0]
        has_code = codes["train"][:, 0] >= 0
        val_code = codes["validation"][:, 0] >= 0
        mask_tr = a_tr["label_complete"].astype(np.float32)
        mask_va = a_va["label_complete"].astype(np.float32)

        def validate_actions(head):
            return lambda: float(
                action_loss(
                    head,
                    mx.array(x["validation"]),
                    mx.array(vectors["validation"]),
                    mx.array(mask_va),
                )
            )

        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            mx.random.seed(seed)
            pre = ActionHead(tokens=config["tokens"], codes=config["codes"])
            pool = np.flatnonzero(has_code)

            def code_batches(step, pre_rng=rng, pool=pool):
                idx = pre_rng.choice(pool, args.batch)
                return mx.array(x["train"][idx]), mx.array(codes["train"][idx])

            def validate_codes(pre=pre):
                v = mx.array(x["validation"][val_code]), mx.array(codes["validation"][val_code])
                return float(code_loss(pre, *v))

            pre_report = fit(
                pre,
                code_loss,
                code_batches,
                validate_codes,
                args.pretrain_steps,
                args.learning_rate,
                every=100,
            )
            pretrained = tree_flatten(pre.parameters())
            for fraction in args.fractions:
                subset = (
                    np.ones(len(mask_tr), bool)
                    if fraction >= 1
                    else stratified_sequences(
                        rows_tr, a_tr["sequence_index"], fraction, np.random.default_rng(seed)
                    )
                )
                labeled = np.flatnonzero(subset & (mask_tr > 0))
                for method in ("scratch", "lam_pretrained"):
                    mx.random.seed(seed + 1)
                    head = ActionHead(tokens=config["tokens"], codes=config["codes"])
                    if method == "lam_pretrained":
                        head.load_weights(pretrained)
                    batch_rng = np.random.default_rng(seed + 2)

                    def batches(step, head=head, batch_rng=batch_rng):
                        idx = batch_rng.choice(labeled, args.batch)
                        return (
                            mx.array(x["train"][idx]),
                            mx.array(vectors["train"][idx]),
                            mx.array(mask_tr[idx]),
                        )

                    training = fit(
                        head,
                        action_loss,
                        batches,
                        validate_actions(head),
                        args.finetune_steps,
                        args.learning_rate,
                        every=100,
                    )
                    run = {
                        "input": source,
                        "seed": seed,
                        "label_fraction": fraction,
                        "labeled_frames": int(len(labeled)),
                        "method": method,
                        "selected_step": training["selected_step"],
                        "validation_loss": training["validation"],
                        "pretrain_selected_step": pre_report["selected_step"]
                        if method != "scratch"
                        else None,
                        "pretrain_validation_code_ce": pre_report["validation"]
                        if method != "scratch"
                        else None,
                        "splits": {s: action_report(head, x[s], *data[s]) for s in EVAL_SPLITS},
                    }
                    report["runs"].append(run)
                    print(
                        json.dumps(
                            {
                                k: run[k]
                                for k in (
                                    "input",
                                    "seed",
                                    "label_fraction",
                                    "method",
                                    "selected_step",
                                )
                            }
                            | {
                                s: {m: round(v[m], 4) for m in ("button_f1", "onset_f1")}
                                for s, v in run["splits"].items()
                            }
                        ),
                        flush=True,
                    )
    summary = defaultdict(dict)
    groups = defaultdict(list)
    for run in report["runs"]:
        groups[(run["input"], run["label_fraction"], run["method"])].append(run)
    for (source, fraction, method), runs in groups.items():
        key = f"{source}/labels_{fraction:g}/{method}"
        for split in EVAL_SPLITS:
            summary[key][split] = {
                m: {
                    "mean": float(np.mean([r["splits"][split][m] for r in runs])),
                    "std": float(np.std([r["splits"][split][m] for r in runs])),
                }
                for m in ("button_f1", "onset_f1", "exact_accuracy", "idle_false_positive_rate")
            }
    report["summary"] = dict(summary)
    write_json(args.output / "report.json", report)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def encode_hordes(args):
    from PIL import Image

    from laya_vision_stitch.laya_p2p import LayaP2PRuntime
    from laya_vision_stitch.p2p_adaptation import encode_action
    from laya_vision_stitch.p2p_pretrained_vision import preprocess
    from laya_vision_stitch.sequence_policy import sequence_contexts

    save_args(args)
    runtime = LayaP2PRuntime.load(args.bundle)
    model, policy = runtime.model, runtime.model.policy
    metadata = {
        "bundle": str(args.bundle),
        "source_weights_sha256": digest(args.bundle / "model.safetensors"),
        "window": args.window,
        "trials": {},
        "scope": "Frozen vision tokens; contexts are teacher-forced with dispatched actions in "
        "non-overlapping windows (memory restarts each window). Frames are the logged viewport.",
    }
    owners = {}
    for trial in args.trials:
        rows = read_trial(trial)
        goal_text = json.loads((trial / "config.json").read_text())["goal"]
        goal = np.asarray(
            model.bridge(model.goal_features(*runtime.prepare_goal(goal_text))).astype(mx.float32)
        )[0]
        n = len(rows)
        images = np.zeros((n, 1024), np.float32)
        hashes = []
        for start in range(0, n, 32):
            batch = rows[start : start + 32]
            pixels = []
            for r in batch:
                hashes.append(digest(r["image"]))
                with Image.open(r["image"]) as im:
                    if im.size != (1280, 720):
                        raise ValueError(f"{r['image']} is not the logged 1280x720 viewport")
                    pixels.append(preprocess(im)[0])
            _, token = policy.vision(mx.array(np.stack(pixels)))
            mx.eval(token)
            images[start : start + len(batch)] = np.asarray(token)
        for h in hashes:
            if owners.setdefault(h, trial.name) != trial.name:
                raise ValueError("Frame shared between trials")
        tokens = np.array([encode_action(r["action"]) for r in rows], np.int32)
        frame = np.arange(n)
        window, step = frame // args.window, frame % args.window
        contexts = np.zeros((n, 1024), np.float32)
        for w in np.unique(window):
            sel = np.flatnonzero(window == w)
            c = sequence_contexts(
                policy,
                mx.array(images[sel][None]),
                mx.array(np.broadcast_to(goal, (1, len(sel), 768)).copy()),
                mx.array(tokens[sel][None]),
            )
            mx.eval(c)
            contexts[sel] = np.asarray(c.astype(mx.float32))[0]
        mouse = np.array([r["action"]["mouse_delta"] for r in rows], np.float64) * 512
        np.savez(
            args.output / f"{trial.name}.npz",
            images=images,
            contexts=contexts,
            tokens=tokens,
            frame=frame,
            sequence_index=window.astype(np.int32),
            steps=step.astype(np.int32),
            elapsed=np.array([r["elapsed_s"] for r in rows]),
            mouse_px=mouse,
            applied=np.array([r["applied"] for r in rows]),
        )
        (args.output / f"{trial.name}.jsonl").write_text(
            "".join(
                json.dumps({**r, "image_sha256": h, "window": int(w), "window_step": int(s)}) + "\n"
                for r, h, w, s in zip(rows, hashes, window, step, strict=True)
            )
        )
        metadata["trials"][trial.name] = {
            "frames": n,
            "goal": goal_text,
            "median_frame_interval_s": float(np.median(np.diff([r["elapsed_s"] for r in rows]))),
            "events_sha256": digest(trial / "events.jsonl"),
        }
        print(json.dumps({trial.name: metadata["trials"][trial.name]}), flush=True)
    write_json(args.output / "metadata.json", metadata)


def load_trial(cache, name):
    a = dict(np.load(Path(cache) / f"{name}.npz"))
    rows = read_jsonl(Path(cache) / f"{name}.jsonl")
    if len(rows) != len(a["images"]):
        raise ValueError(f"{name}: rows and arrays disagree")
    a["buttons"] = [frozenset(r["action"]["buttons"]) for r in rows]
    return a


def time_split(n, fraction, margin):
    """Start frames before cut-margin fit; starts from cut on validate (time-separated)."""
    cut = int(round(n * (1 - fraction)))
    return np.arange(n) < cut - margin, np.arange(n) >= cut


def hordes(args):
    save_args(args)
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    p2p_lam, config, std = load_lam(args.lam)
    trials = {name: load_trial(args.hordes_cache, name) for name in args.trials}
    report = {"folds": {}, "config": {k: plain(v) for k, v in vars(args).items() if k != "func"}}
    for test in args.test_trials:
        fold = {"test_trial": test, "train_trials": [t for t in args.trials if t != test]}
        parts = defaultdict(lambda: defaultdict(list))
        for name, a in trials.items():
            n = len(a["images"])
            seq, steps = np.zeros(n, np.int64), a["frame"]
            z = std(a["images"])
            i = successors(seq, steps)
            if name == test:
                selections = {"test": np.ones(len(i), bool)}
            else:
                fit_mask, val_mask = time_split(n, args.validation_fraction, 1)
                selections = {"fit": fit_mask[i], "validation": val_mask[i]}
            for part, keep in selections.items():
                j = i[keep]
                p = parts[part]
                p["z0"].append(z[j])
                p["z1"].append(z[j + 1])
                for lag in LAGS:
                    f, target, valid = lagged_facts(
                        a["buttons"], a["mouse_px"], seq, steps, j, lag, **HORDES_FACTS
                    )
                    p[f"facts_{lag}"].append(f)
                    p[f"buttons_{lag}"] += [a["buttons"][t] for t in target]
                    p[f"mouse_{lag}"].append(a["mouse_px"][target])
                    p[f"valid_{lag}"].append(valid)
        merged = {}
        for part, p in parts.items():
            m = {"z0": np.concatenate(p["z0"]), "z1": np.concatenate(p["z1"])}
            for lag in LAGS:
                facts = p[f"facts_{lag}"]
                m[f"facts_{lag}"] = {
                    k: (
                        np.concatenate([f[k][0] for f in facts]),
                        np.concatenate([f[k][1] for f in facts]),
                    )
                    for k in facts[0]
                }
                m[f"buttons_{lag}"] = p[f"buttons_{lag}"]
                m[f"mouse_{lag}"] = np.concatenate(p[f"mouse_{lag}"])
                m[f"valid_{lag}"] = np.concatenate(p[f"valid_{lag}"])
            merged[part] = m
        fold["pairs"] = {k: int(len(v["z0"])) for k, v in merged.items()}
        tuned = LatentActionModel(**config)
        tuned.load_weights(str(Path(args.lam) / "lam.safetensors"))
        mx.eval(tuned.parameters())
        fit_part, val_part = merged["fit"], merged["validation"]
        fold["finetune"] = train_lam(
            tuned,
            (fit_part["z0"], fit_part["z1"]),
            (val_part["z0"], val_part["z1"]),
            steps=args.steps,
            batch=args.batch,
            learning_rate=args.learning_rate,
            seed=args.seed,
            every=100,
        )
        tuned.save_weights(str(args.output / f"lam_finetuned_holdout_{test}.safetensors"))
        lams = {"p2p_lam": p2p_lam, "hordes_finetuned_lam": tuned}
        pca = PCA(fit_part["z1"] - fit_part["z0"], args.pca)
        delta = {k: pca(v["z1"] - v["z0"]) for k, v in merged.items()}
        lag, lag_table = select_lag(
            delta,
            {g: {k: v[f"facts_{g}"] for k, v in merged.items()} for g in LAGS},
            "fit",
            "validation",
        )
        fold["lag_selection"] = {"validation_balanced_accuracy": lag_table, "selected_lag": lag}
        features = {"delta_pca": delta}
        fold["next_token_mse"], fold["nmi"], fold["codebook"] = {}, {}, {}
        for lam_name, lam in lams.items():
            codes = {k: lam_codes(lam, v["z0"], v["z1"]) for k, v in merged.items()}
            features[lam_name] = {k: code_embeddings(lam, c) for k, c in codes.items()}
            fold["next_token_mse"][lam_name] = {
                k: mse_summary(lam_errors(lam, merged[k]["z0"], merged[k]["z1"]))
                for k in ("validation", "test")
            }
            t = merged["test"]
            fold["nmi"][lam_name] = nmi_report(
                codes["test"],
                t[f"buttons_{lag}"],
                t[f"mouse_{lag}"],
                t[f"valid_{lag}"],
                config["codes"],
                rng,
            )
            fold["codebook"][lam_name] = {
                "test_perplexity_per_token": perplexity(codes["test"], config["codes"]),
                "test_unique_combinations": int(
                    len(np.unique(combine_codes(codes["test"], config["codes"])))
                ),
            }
        facts = {k: v[f"facts_{lag}"] for k, v in merged.items()}
        fold["probes"], fold["probe_penalties"] = probe_suite(
            features, facts, "fit", "validation", ["validation", "test"]
        )
        report["folds"][test] = fold
        print(
            json.dumps(
                {
                    "holdout": test,
                    "lag": lag,
                    "mse": fold["next_token_mse"],
                    "test_probes": fold["probes"].get("test"),
                },
                indent=1,
            ),
            flush=True,
        )
    write_json(args.output / "report.json", report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(required=True)
    a = sub.add_parser("p2p")
    a.add_argument("--cache", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--tokens", type=int, default=4)
    a.add_argument("--codes", type=int, default=8)
    a.add_argument("--dim", type=int, default=16)
    a.add_argument("--width", type=int, default=512)
    a.add_argument(
        "--context", type=int, default=0, help="Decoder view of z_t: -1 all, 0 none, k projection"
    )
    a.add_argument("--weight-decay", type=float, default=1e-4)
    a.add_argument("--steps", type=int, default=3000)
    a.add_argument("--batch", type=int, default=256)
    a.add_argument("--learning-rate", type=float, default=3e-4)
    a.add_argument("--latent-dropout", type=float, default=0.1)
    a.add_argument("--pca", type=int, default=256)
    a.add_argument("--seed", type=int, default=20260923)
    a.set_defaults(func=p2p)
    b = sub.add_parser("lapa")
    b.add_argument("--cache", type=Path, required=True)
    b.add_argument("--lam", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument(
        "--inputs", nargs="+", default=["contexts", "images"], choices=["contexts", "images"]
    )
    b.add_argument("--fractions", nargs="+", type=float, default=[0.1, 1.0])
    b.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    b.add_argument("--pretrain-steps", type=int, default=2000)
    b.add_argument("--finetune-steps", type=int, default=1500)
    b.add_argument("--batch", type=int, default=128)
    b.add_argument("--learning-rate", type=float, default=3e-4)
    b.set_defaults(func=lapa)
    c = sub.add_parser("encode-hordes")
    c.add_argument("--bundle", type=Path, required=True)
    c.add_argument("--trials", type=Path, nargs="+", required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--window", type=int, default=32)
    c.set_defaults(func=encode_hordes)
    d = sub.add_parser("hordes")
    d.add_argument("--lam", type=Path, required=True)
    d.add_argument("--hordes-cache", type=Path, required=True)
    d.add_argument("--output", type=Path, required=True)
    d.add_argument("--trials", nargs="+", required=True)
    d.add_argument("--test-trials", nargs="+", required=True)
    d.add_argument("--validation-fraction", type=float, default=0.2)
    d.add_argument("--steps", type=int, default=1500)
    d.add_argument("--batch", type=int, default=256)
    d.add_argument("--learning-rate", type=float, default=1e-4)
    d.add_argument("--pca", type=int, default=128)
    d.add_argument("--seed", type=int, default=20260923)
    d.set_defaults(func=hordes)
    args = p.parse_args()
    if getattr(args, "window", 1) < 1 or getattr(args, "window", 1) > 200:
        raise ValueError("Context windows must hold 1-200 frames")
    args.func(args)


if __name__ == "__main__":
    main()
