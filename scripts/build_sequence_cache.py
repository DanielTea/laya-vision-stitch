"""Cache frozen per-frame features and teacher-forced policy contexts for sequence manifests.

Caching is a training/evaluation optimization; deployed inference still encodes fresh images.
Rows must form consecutive sequences (`sequence`, `sequence_step`) with one current frame.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from laya_vision_stitch.laya_p2p import LayaP2PRuntime
from laya_vision_stitch.p2p_adaptation import encode_action
from laya_vision_stitch.p2p_pretrained_policy import KEY_NAMES, MOUSE_NAMES
from laya_vision_stitch.p2p_pretrained_vision import preprocess
from laya_vision_stitch.sequence_policy import sequence_contexts


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sequences(rows):
    grouped = defaultdict(list)
    for i, r in enumerate(rows):
        grouped[r["sequence"]].append(i)
    result = []
    for name, idx in grouped.items():
        idx.sort(key=lambda i: rows[i]["sequence_step"])
        steps = [rows[i]["sequence_step"] for i in idx]
        if steps != list(range(len(idx))):
            raise ValueError(f"Sequence {name} is not consecutive from step zero")
        if len({rows[i]["game"] for i in idx}) != 1:
            raise ValueError(f"Sequence {name} mixes games")
        result.append(idx)
    return result


def supported(action):
    """Drop controls outside the released vocabulary (no Tab); the frame is flagged incomplete."""
    known = (set(KEY_NAMES) | set(MOUSE_NAMES)) - {None}
    buttons = [b for b in action["buttons"] if b in known]
    return {**action, "buttons": buttons}, len(buttons) == len(action["buttons"])


def resolve(image, root):
    path = Path(image)
    return path if path.is_absolute() else root / path


def check_separation(output, data, splits):
    """No image may appear in two splits, except near-blank frames (e.g. black loading
    screens, value spread <= 8 levels), which carry no content; their count is returned."""
    owners, paths, shared = {}, {}, set()
    for split in splits:
        manifest = {r["id"]: r["frames"][0]["image"] for r in read(data / f"{split}.jsonl")}
        for r in read(output / f"{split}.jsonl"):
            first = owners.setdefault(r["image_sha256"], split)
            paths.setdefault(r["image_sha256"], resolve(manifest[r["id"]], data))
            if first != split:
                shared.add(r["image_sha256"])
    for digest_value in shared:
        with Image.open(paths[digest_value]) as im:
            pixels = np.asarray(im)
        if int(pixels.max()) - int(pixels.min()) > 8:
            raise ValueError("Cross-split duplicate image")
    return len(shared)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "validation", "test", "fresh_test"])
    p.add_argument("--spatial", action="store_true", help="Also store 12x12x112 FP16 grids")
    p.add_argument(
        "--spatial-press",
        action="store_true",
        help="Store 12x12x112 grids only for frames with a pointer press (pointer-head training)",
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = LayaP2PRuntime.load(args.bundle)
    model = runtime.model
    policy = model.policy
    goal_cache = {}

    def goal_vector(text):
        if text not in goal_cache:
            vector = model.bridge(model.goal_features(*runtime.prepare_goal(text)))
            goal_cache[text] = np.asarray(vector.astype(mx.float32))[0]
        return goal_cache[text]

    metadata = {
        "bundle": str(args.bundle),
        "source_weights_sha256": digest(args.bundle / "model.safetensors"),
        "data": str(args.data),
        "splits": {},
        "scope": "Frozen features; contexts use recorded previous actions (teacher forcing)",
    }
    for split in args.splits:
        manifest = args.data / f"{split}.jsonl"
        if not manifest.exists():
            continue
        rows = read(manifest)
        groups = sequences(rows)
        order = [i for idx in groups for i in idx]
        rows = [rows[i] for i in order]
        n = len(rows)
        images = np.zeros((n, 1024), np.float32)
        spatial = np.zeros((n, 12, 12, 112), np.float16) if args.spatial else None
        has_fovea = "fovea" in rows[0]["frames"][0]
        fovea = np.zeros((n, 1024), np.float32) if has_fovea else None
        goals = np.zeros((n, 768), np.float32)
        tokens = np.zeros((n, 8), np.int32)
        complete = np.ones(n, bool)
        has_pointer = any(r.get("pointer") for r in rows)
        cursor = np.full((n, 2), np.nan, np.float32)
        press_xy = np.full((n, 2), np.nan, np.float32)
        press_button = np.zeros(n, np.int8)
        buttons = {"left": 1, "right": 2, "middle": 3}
        for i, r in enumerate(rows):
            pointer = r.get("pointer") or {}
            if pointer.get("cursor_xy"):
                cursor[i] = pointer["cursor_xy"]
            if pointer.get("press"):
                press_xy[i] = pointer["press"]["xy"]
                press_button[i] = buttons.get(pointer["press"]["button"], 0)
        press_rows = np.flatnonzero(press_button > 0)
        press_spatial = (
            np.zeros((len(press_rows), 12, 12, 112), np.float16) if args.spatial_press else None
        )
        press_slot = {int(i): k for k, i in enumerate(press_rows)}
        hashes = []
        for start in range(0, n, 32):
            batch = rows[start : start + 32]
            pixels = []
            for r in batch:
                path = resolve(r["frames"][0]["image"], args.data)
                hashes.append(digest(path))
                with Image.open(path) as im:
                    pixels.append(preprocess(im)[0])
            grid, token = policy.vision(mx.array(np.stack(pixels)))
            mx.eval(grid, token)
            images[start : start + len(batch)] = np.asarray(token)
            if spatial is not None:
                spatial[start : start + len(batch)] = np.asarray(grid.astype(mx.float16))
            if press_spatial is not None:
                grid16 = np.asarray(grid.astype(mx.float16))
                for j in range(len(batch)):
                    if start + j in press_slot:
                        press_spatial[press_slot[start + j]] = grid16[j]
            if fovea is not None:
                crops = []
                for r in batch:
                    with Image.open(resolve(r["frames"][0]["fovea"], args.data)) as im:
                        crops.append(preprocess(im)[0])
                _, crop_token = policy.vision(mx.array(np.stack(crops)))
                mx.eval(crop_token)
                fovea[start : start + len(batch)] = np.asarray(crop_token)
            for j, r in enumerate(batch):
                goals[start + j] = goal_vector(r["goal"])
                action, complete[start + j] = supported(r["action"])
                tokens[start + j] = encode_action(action)
        contexts = np.zeros((n, 1024), np.float32)
        unconditional = np.zeros((n, 1024), np.float32)
        offset = 0
        seq_index = np.zeros(n, np.int32)
        lengths = sorted({len(idx) for idx in groups})
        spans = []
        for k, idx in enumerate(groups):
            spans.append((offset, len(idx)))
            seq_index[offset : offset + len(idx)] = k
            offset += len(idx)
        for length in lengths:
            same = [s for s in spans if s[1] == length]
            for b in range(0, len(same), 8):
                chunk = same[b : b + 8]
                sel = np.concatenate([np.arange(o, o + length) for o, _ in chunk])
                im = mx.array(images[sel].reshape(len(chunk), length, 1024))
                tk = mx.array(tokens[sel].reshape(len(chunk), length, 8))
                gl = mx.array(goals[sel].reshape(len(chunk), length, 768))
                c = sequence_contexts(policy, im, gl, tk)
                u = sequence_contexts(policy, im, mx.zeros_like(gl), tk)
                mx.eval(c, u)
                contexts[sel] = np.asarray(c.astype(mx.float32)).reshape(-1, 1024)
                unconditional[sel] = np.asarray(u.astype(mx.float32)).reshape(-1, 1024)
        arrays = {
            "images": images,
            "goals": goals,
            "tokens": tokens,
            "contexts": contexts,
            "unconditional_contexts": unconditional,
            "sequence_index": seq_index,
            "label_complete": complete,
            "steps": np.array([r["sequence_step"] for r in rows], np.int32),
            "timestamps": np.array([r.get("timestamp_seconds", 0.0) for r in rows], np.float64),
        }
        if spatial is not None:
            arrays["spatial"] = spatial
        if fovea is not None:
            arrays["fovea_images"] = fovea
        if has_pointer:
            arrays.update(cursor_xy=cursor, press_xy=press_xy, press_button=press_button)
        if press_spatial is not None:
            arrays.update(press_rows=press_rows.astype(np.int32), press_spatial=press_spatial)
        np.savez(args.output / f"{split}.npz", **arrays)
        compact = [
            {
                "id": r["id"],
                "game": r["game"],
                "sequence": r["sequence"],
                "step": r["sequence_step"],
                "goal": r["goal"],
                "image_sha256": h,
                "episode": r.get("episode"),
            }
            for r, h in zip(rows, hashes, strict=True)
        ]
        (args.output / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in compact))
        metadata["splits"][split] = {
            "frames": n,
            "sequences": len(groups),
            "games": sorted({r["game"] for r in rows}),
            "manifest_sha256": digest(manifest),
            "incomplete_labels": int((~complete).sum()),
        }
        print(json.dumps({split: metadata["splits"][split]}), flush=True)
    metadata["uniform_images_shared_across_splits"] = check_separation(
        args.output, args.data, metadata["splits"]
    )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
