"""Learn a small Laya-to-policy goal bridge from public precomputed text features."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import mlx.core as mx
import numpy as np

from laya_vision_stitch.laya_p2p import LayaP2PRuntime
from laya_vision_stitch.p2p_data import read_annotation
from laya_vision_stitch.policy_training import fingerprint

HELDOUT = {"roblox/a-dusty-trip", "roblox/be-a-snake", "roblox/evade"}


def collect(root):
    seen, texts, conflicts = set(), {}, set()
    for path in sorted(root.glob("p2p-*/dataset/*/annotation.proto")):
        annotation = read_annotation(path)
        episode = annotation.metadata.id
        if episode in seen:
            continue
        seen.add(episode)
        game = "/".join([annotation.metadata.env.env, annotation.metadata.env.env_subtype]).rstrip(
            "/"
        )
        for frame in annotation.frame_annotations:
            for entry in frame.frame_text_annotation:
                goal = entry.instruction.strip()
                version = entry.frame_text_annotator.version
                if not goal or version not in entry.text_embedding_dict:
                    continue
                available = entry.text_embedding_dict[version].text_embeddings
                if "gemma" not in available:
                    continue
                embedding = np.array(available["gemma"].values, np.float32)
                if embedding.shape != (768,) or not np.isfinite(embedding).all():
                    continue
                if goal in texts:
                    if not np.allclose(texts[goal]["embedding"], embedding, rtol=1e-5, atol=1e-5):
                        conflicts.add(goal)
                    texts[goal]["games"].add(game)
                else:
                    texts[goal] = {
                        "goal": goal,
                        "embedding": embedding,
                        "games": {game},
                        "source": str(path),
                        "episode": episode,
                    }
    rows = []
    for goal, row in sorted(texts.items()):
        if goal in conflicts:
            continue
        digest = hashlib.sha256(goal.encode()).hexdigest()
        split = (
            "test"
            if row["games"] & HELDOUT
            else "validation"
            if int(digest[:8], 16) % 5 == 0
            else "train"
        )
        rows.append({**row, "games": sorted(row["games"]), "split": split, "goal_sha256": digest})
    return rows, {
        "recordings": len(seen),
        "conflicting_instructions_removed": len(conflicts),
        "unique_instructions": len(rows),
        "splits": dict(Counter(r["split"] for r in rows)),
        "test_games": sorted(HELDOUT),
        "teacher": "Released annotation Gemma raw pooled features; no online teacher or target lookup at inference",
    }


def metrics(prediction, target, mean):
    error = float(np.mean((prediction - target) ** 2))
    baseline = float(np.mean((mean - target) ** 2))

    def normalize(x):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

    retrieval = normalize(prediction - mean) @ normalize(target - mean).T
    return {
        "examples": len(target),
        "mse": error,
        "training_mean_mse": baseline,
        "r2_vs_training_mean": 1 - error / baseline,
        "cosine": float((normalize(prediction) * normalize(target)).sum(-1).mean()),
        "centered_cosine": float(
            (normalize(prediction - mean) * normalize(target - mean)).sum(-1).mean()
        ),
        "paired_retrieval_top1": float((retrieval.argmax(-1) == np.arange(len(target))).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, audit = collect(args.artifacts)
    if min(audit["splits"].values()) < 20:
        raise ValueError("Insufficient disjoint goal supervision")
    print(json.dumps(audit), flush=True)
    runtime = LayaP2PRuntime.build(args.policy)
    parents = {name: fingerprint(getattr(runtime.model, name)) for name in ("laya", "policy")}
    vectors = []
    for i, row in enumerate(rows):
        feature = runtime.model.goal_features(*runtime.prepare_goal(row["goal"]))
        mx.eval(feature)
        vectors.append(np.asarray(feature)[0])
        if (i + 1) % 100 == 0:
            print(f"Encoded {i + 1}/{len(rows)} goals", flush=True)
    x, y = np.stack(vectors), np.stack([r["embedding"] for r in rows])
    np.savez(args.output / "features.npz", laya=x, gemma=y)
    (args.output / "goals.jsonl").write_text(
        "".join(json.dumps({k: v for k, v in r.items() if k != "embedding"}) + "\n" for r in rows)
    )
    indices = {
        s: np.array([i for i, r in enumerate(rows) if r["split"] == s])
        for s in ("train", "validation", "test")
    }
    mean, scale = x[indices["train"]].mean(0), np.maximum(x[indices["train"]].std(0), 1e-3)
    target_mean = y[indices["train"]].mean(0)
    normalized = (x - mean) / scale
    train_x, train_y = (
        normalized[indices["train"]].astype(np.float64),
        (y[indices["train"]] - target_mean).astype(np.float64),
    )
    # Dual ridge is cheaper when the number of unique goals is below feature width.
    kernel = train_x @ train_x.T
    identity = np.eye(len(train_x))
    best, candidates = None, []
    for regularization in (0.01, 0.1, 1, 10, 100):
        coefficient = np.linalg.solve(kernel + regularization * len(train_x) * identity, train_y)
        weight = (train_x.T @ coefficient).astype(np.float32)
        result = metrics(
            normalized[indices["validation"]] @ weight + target_mean,
            y[indices["validation"]],
            target_mean,
        )
        candidates.append({"regularization": regularization, "validation": result})
        if best is None or result["mse"] < best[0]:
            best = result["mse"], weight, regularization
    _, weight, selected = best
    bridge = runtime.model.bridge
    bridge.mean, bridge.scale = mx.array(mean), mx.array(scale)
    bridge.projection.weight, bridge.projection.bias = mx.array(weight.T), mx.array(target_mean)
    runtime.metadata.update(
        trained_goal_bridge=True,
        bridge_regularization=selected,
        goal_training_examples=len(train_x),
        preprocessing="fast_image_resize 5.1.4 Hamming interpolation, uint8 rounding each axis",
    )
    report = {
        "audit": audit,
        "parent_fingerprints": parents,
        "selected_regularization": selected,
        "candidates": candidates,
        "results": {
            s: metrics(normalized[idx] @ weight + target_mean, y[idx], target_mean)
            for s, idx in indices.items()
        },
        "scope": "Goal-feature alignment only. Does not establish correct game actions, Hordes transfer or live latency.",
    }
    report["parents_unchanged"] = parents == {
        name: fingerprint(getattr(runtime.model, name)) for name in parents
    }
    if not report["parents_unchanged"]:
        raise RuntimeError("Frozen pretrained parent changed")
    runtime.save(args.output / "bundle")
    loaded = LayaP2PRuntime.load(args.output / "bundle")
    sample = rows[indices["test"][0]]["goal"]
    before = runtime.model.bridge(runtime.model.goal_features(*runtime.prepare_goal(sample)))
    after = loaded.model.bridge(loaded.model.goal_features(*loaded.prepare_goal(sample)))
    report["reload_max_abs_error"] = float(mx.abs(before - after).max())
    if report["reload_max_abs_error"] > 1e-5:
        raise RuntimeError("Goal bridge export differs")
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "selected_regularization": selected,
                "results": report["results"],
                "parents_unchanged": report["parents_unchanged"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
