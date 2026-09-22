"""Frozen CLIP + paired reference attention + Laya, in one inference module.

The bank is reference memory, not fitted weights. Retrieved descriptions remain
separate sequences: their meanings are not averaged into invalid token mixtures.
"""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .clip_backend import FrozenCLIP, config_from_json, pixels
from .lexical_stitch import LAYA_MLX_ID, LAYA_MLX_REVISION, normalized
from .vendor.clip_model import ClipVisionModel

LAYA_MODELS = {
    "english": ("aac6fef/laya-mlx", "20aed815fc6acde75733882e7ec0e3f28aeb9717"),
    "multilingual": (LAYA_MLX_ID, LAYA_MLX_REVISION),
}


def read_references(manifest):
    manifest = Path(manifest)
    rows = json.loads(manifest.read_text())
    seen_ids, seen_images = set(), set()
    for row in rows:
        if not isinstance(row.get("caption"), str) or not row["caption"].strip():
            raise ValueError("Every reference needs a nonempty caption")
        image = (manifest.parent / row["image"]).resolve()
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if row["id"] in seen_ids or digest in seen_images:
            raise ValueError("Reference IDs and image files must be unique")
        seen_ids.add(row["id"])
        seen_images.add(digest)
        row["image"], row["sha256"] = str(image), digest
    if len(rows) < 4:
        raise ValueError("At least four paired references are required")
    return rows


def load_laya(kind):
    import laya_mlx
    from huggingface_hub import snapshot_download

    model_id, revision = LAYA_MODELS[kind]
    return laya_mlx.load(snapshot_download(model_id, revision=revision, local_files_only=True))


class ReferenceStitch(nn.Module):
    def __init__(
        self,
        vision,
        projection,
        laya,
        image_keys,
        text_keys,
        token_ids,
        token_mask,
        top_k=4,
        temperature=0.02,
    ):
        super().__init__()
        if not 1 <= top_k <= len(image_keys) or temperature <= 0:
            raise ValueError("Invalid fixed retrieval settings")
        if (
            image_keys.shape != text_keys.shape
            or token_ids.shape != token_mask.shape
            or len(token_ids) != len(image_keys)
        ):
            raise ValueError("Reference memory shapes disagree")
        self.vision, self.projection, self.laya = vision, projection, laya
        self.image_keys, self.text_keys = image_keys, text_keys
        self.reference_ids, self.reference_mask = token_ids, token_mask
        self.top_k, self.temperature = top_k, temperature
        self.eval()
        self.freeze()

    def __call__(self, image, prefix, marker_pos, marker_mask, scale=1.0, key_source="image"):
        query = normalized(self.projection(self.vision(image).pooler_output))[0]
        if key_source not in ("image", "text"):
            raise ValueError("key_source must be image or text")
        keys = self.image_keys if key_source == "image" else self.text_keys
        similarities = keys @ query
        indices = mx.argsort(similarities)[-self.top_k :][::-1]
        weights = mx.softmax(similarities[indices] / self.temperature)
        prefix = mx.broadcast_to(prefix[None], (self.top_k, len(prefix)))
        ids = mx.concatenate([prefix, self.reference_ids[indices]], axis=1)
        mask = mx.concatenate(
            [mx.ones(prefix.shape, dtype=mx.bool_), self.reference_mask[indices]], axis=1
        )
        logits, _ = self.laya(
            ids,
            mask,
            mx.broadcast_to(marker_pos[None], (self.top_k, len(marker_pos))),
            mx.broadcast_to(marker_mask[None], (self.top_k, len(marker_mask))),
            mx.zeros((self.top_k,), dtype=mx.int32),
        )
        reference_probabilities = mx.softmax(logits / scale, axis=-1)
        mixture = (weights[:, None] * reference_probabilities).sum(0)
        return mixture, indices, weights, reference_probabilities


class ReferenceRuntime:
    def __init__(self, module, agent, references, metadata):
        self.module, self.agent = module, agent
        self.references, self.metadata = references, metadata

    @classmethod
    def build(cls, manifest, laya_kind="english"):
        rows = read_references(manifest)
        clip = FrozenCLIP()
        agent = load_laya(laya_kind)
        image_keys = mx.stack([clip.image(row["image"]) for row in rows])
        text_keys = mx.stack([clip.text(row["caption"]) for row in rows])
        # The stored values are exact caption token IDs, never synthetic latent averages.
        ids = [agent.tok(row["caption"])["input_ids"] + [agent.tok.sep_token_id] for row in rows]
        length = max(map(len, ids))
        if length > 128:
            raise ValueError("Reference descriptions must fit within 128 Laya tokens")
        tokens = np.full((len(ids), length), agent.tok.pad_token_id, np.int32)
        valid = np.zeros(tokens.shape, bool)
        for i, item in enumerate(ids):
            tokens[i, : len(item)], valid[i, : len(item)] = item, True
        module = ReferenceStitch(
            clip.model.vision_model,
            clip.model.visual_projection,
            agent.model,
            image_keys,
            text_keys,
            mx.array(tokens),
            mx.array(valid),
        )
        mx.eval(module.parameters())
        model_id, revision = LAYA_MODELS[laya_kind]
        metadata = {
            "format": 1,
            "method": "paired-reference attention with frozen Laya decision mixture",
            "clip": clip.metadata,
            "laya_model": model_id,
            "laya_revision": revision,
            "encoder_config": agent.encoder_cfg,
            "agent_config": agent.cfg,
            "temperature": agent.temperature,
            "temperature_by_options": agent.temperature_by_options,
            "top_k": 4,
            "reference_temperature": 0.02,
            "reference_token_length": length,
            "reference_count": len(rows),
            "new_training": False,
            "connector_fitting": False,
            "generated_tokens": 0,
            "pretrained_backbones_updated": False,
        }
        return cls(module, agent, rows, metadata)

    def prepare(self, question, choices):
        from laya_mlx.common import temp_bucket

        if not isinstance(choices, dict) or not 2 <= len(choices) <= 32:
            raise ValueError("Need 2..32 named choices")
        item = self.agent.prepare(
            "", {"decision": {"type": "choice", "instructions": question, "criteria": choices}}
        )[0][0]
        prefix = item["ids"][:-1]
        if len(prefix) + self.metadata["reference_token_length"] > self.agent.cfg["max_len"]:
            raise ValueError("Question and reference memory exceed Laya's context")
        scale = self.metadata["temperature_by_options"].get(
            temp_bucket(0, len(choices)), self.metadata["temperature"][0]
        )
        return (
            mx.array(prefix),
            mx.array(item["markers"]),
            mx.ones((len(item["markers"]),), dtype=mx.bool_),
            scale,
        )

    def decide(self, path, question, choices, key_source="image"):
        started = time.perf_counter()
        prepared = self.prepare(question, choices)
        probs, indices, weights, individual = self.module(pixels(path), *prepared, key_source)
        mx.eval(probs, indices, weights, individual)
        elapsed = (time.perf_counter() - started) * 1000
        probabilities = np.asarray(probs)
        if not np.isfinite(probabilities).all():
            raise ValueError("Non-finite decision mixture")
        labels = list(choices)
        picked = np.asarray(indices).tolist()
        return {
            "choice": labels[int(probabilities.argmax())],
            "scores": dict(zip(labels, probabilities.tolist(), strict=True)),
            "nearest_laya_choice": labels[int(np.asarray(individual[0]).argmax())],
            "reference_ids": [self.references[i]["id"] for i in picked],
            "reference_indices": picked,
            "reference_weights": np.asarray(weights).tolist(),
            "image_to_scores_ms": elapsed,
            "input_events_sent": 0,
        }

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.module.save_weights(str(directory / "model.safetensors"))
        (directory / "config.json").write_text(json.dumps(self.metadata, indent=2) + "\n")
        # Raw images and source paths are unnecessary for inference and are not bundled.
        references = [{k: v for k, v in r.items() if k != "image"} for r in self.references]
        (directory / "references.json").write_text(json.dumps(references, indent=2) + "\n")
        shutil.copytree(self.agent.model_dir / "tokenizer", directory / "tokenizer")

    @classmethod
    def load(cls, directory):
        from laya_mlx.agent import Agent
        from laya_mlx.model import DecisionModel, EncoderConfig
        from laya_mlx.tokenizer import Tokenizer

        directory = Path(directory)
        metadata = json.loads((directory / "config.json").read_text())
        if metadata["format"] != 1:
            raise ValueError("Unsupported reference-stitch format")
        config = config_from_json(metadata["clip"]["config"])
        laya = DecisionModel(
            EncoderConfig.from_dict(metadata["encoder_config"]), metadata["agent_config"]
        )
        count, length = metadata["reference_count"], metadata["reference_token_length"]
        module = ReferenceStitch(
            ClipVisionModel(config.vision_config),
            nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False),
            laya,
            mx.zeros((count, config.projection_dim)),
            mx.zeros((count, config.projection_dim)),
            mx.zeros((count, length), dtype=mx.int32),
            mx.zeros((count, length), dtype=mx.bool_),
            metadata["top_k"],
            metadata["reference_temperature"],
        )
        module.load_weights(str(directory / "model.safetensors"), strict=True)
        mx.eval(module.parameters())
        agent = Agent.__new__(Agent)
        agent.cfg, agent.model_dir = metadata["agent_config"], directory
        agent.tok, agent._prefix_cache = Tokenizer(directory / "tokenizer"), None
        agent.model = laya
        return cls(module, agent, json.loads((directory / "references.json").read_text()), metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--references", type=Path)
    source.add_argument("--bundle", type=Path)
    parser.add_argument("--laya", choices=LAYA_MODELS, default="english")
    parser.add_argument("--save", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--question")
    parser.add_argument("--choices", type=Path)
    args = parser.parse_args()
    if not args.save and not args.image:
        parser.error("Specify --save or an inference --image")
    if args.image and (not args.question or not args.choices):
        parser.error("Inference needs --question and --choices")
    runtime = (
        ReferenceRuntime.load(args.bundle)
        if args.bundle
        else ReferenceRuntime.build(args.references, args.laya)
    )
    if args.save:
        runtime.save(args.save)
    if args.image:
        print(
            json.dumps(
                runtime.decide(args.image, args.question, json.loads(args.choices.read_text())),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
