# SPDX-License-Identifier: Apache-2.0
# Laya embedding-forward adaptation follows laya-mlx 0.2.0; see third_party/.
"""Experimental frozen lexical attention: Qwen vision -> shared words -> Laya.

No regression, optimizer, action labels, captions, or Qwen decoder forward pass.
The bridge is a hypothesis, not an established multimodal alignment method.
"""

import argparse
import json
import re
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from .backends import QWEN_ID, QWEN_REVISION, QwenVision

LAYA_MLX_ID = "aac6fef/laya-multilingual-mlx"
LAYA_MLX_REVISION = "f2b4faf51023039425946074e2cf1361d2db11d5"


def normalized(x):
    x = x.astype(mx.float32)
    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


class LexicalBridge(nn.Module):
    def __init__(self, keys, values, top_k=8, temperature=0.05):
        super().__init__()
        if keys.ndim != 2 or values.ndim != 2 or len(keys) != len(values):
            raise ValueError("Expected aligned word embedding matrices")
        if not 1 <= top_k <= len(keys) or temperature <= 0:
            raise ValueError("Invalid fixed attention settings")
        self.anchor_keys = normalized(keys)
        self.anchor_values = values.astype(mx.float32)
        self.top_k, self.temperature = top_k, temperature
        self.freeze()

    def __call__(self, visual):
        similarities = normalized(visual) @ self.anchor_keys.T
        indices = mx.argpartition(similarities, similarities.shape[-1] - self.top_k, axis=-1)
        indices = indices[:, -self.top_k :]
        scores = mx.take_along_axis(similarities, indices, axis=-1)
        weights = mx.softmax(scores / self.temperature, axis=-1)
        embedded = (self.anchor_values[indices] * weights[..., None]).sum(axis=-2)
        return embedded, indices, weights


class FrozenStitch(nn.Module):
    """One MLX module containing the visual tower, fixed bridge, and Laya."""

    def __init__(self, vision, laya, bridge, grid_size=4):
        super().__init__()
        self.vision, self.laya, self.bridge = vision, laya, bridge
        self.grid_size = grid_size
        self.eval()
        self.freeze()

    def visual_tokens(self, pixels, grid):
        visual, _ = self.vision(pixels.astype(self.vision.patch_embed.proj.weight.dtype), grid)
        t, height, width = map(int, grid[0])
        merge = self.vision.spatial_merge_size
        h, w = height // merge, width // merge
        if t != 1 or min(h, w) < self.grid_size:
            raise ValueError("Need one image and enough spatial patches")
        visual = visual.reshape(h, w, -1)
        n = self.grid_size
        return mx.stack(
            [
                visual[r * h // n : (r + 1) * h // n, c * w // n : (c + 1) * w // n].mean(
                    axis=(0, 1)
                )
                for r in range(n)
                for c in range(n)
            ]
        )

    def score_embeddings(self, embeddings, attention_mask, marker_pos, marker_mask, qtype):
        from laya_mlx.model import attention_masks

        encoder = self.laya.encoder
        dtype = encoder.embeddings.tok_embeddings.weight.dtype
        h = encoder.embeddings.norm(embeddings.astype(dtype))
        masks = attention_masks(attention_mask, encoder.config.local_attention)
        for layer in encoder.layers:
            h = layer(h, masks[layer.attention_type])
        h = encoder.final_norm(h) + self.laya.type_emb(qtype)[:, None, :]
        h = self.laya.head(h, attention_mask[:, None, None, :].astype(mx.bool_))
        markers = h[mx.arange(h.shape[0])[:, None], mx.maximum(marker_pos, 0)]
        logits = self.laya.scorer(markers).squeeze(-1).astype(mx.float32)
        return mx.where(marker_mask, logits, -1e4)

    def __call__(self, pixels, grid, batch, start):
        visual = self.visual_tokens(pixels, grid)
        state, indices, weights = self.bridge(visual)
        original = self.laya.encoder.embeddings.tok_embeddings(batch["input_ids"])
        dtype = original.dtype
        combined = mx.concatenate(
            [
                original[:, :start],
                state[None].astype(dtype),
                original[:, start + len(state) :],
            ],
            axis=1,
        )
        scores = self.score_embeddings(
            combined, **{k: v for k, v in batch.items() if k != "input_ids"}
        )
        return scores, indices, weights


def shared_words(qwen_tokenizer, laya_tokenizer):
    """Deterministic vocabulary intersection, with no game-specific word list."""
    found = {}
    for token_id in range(len(qwen_tokenizer)):
        decoded = qwen_tokenizer.decode([token_id])
        word = decoded.strip()
        if not re.fullmatch(r"[A-Za-z]{2,24}", word):
            continue
        ids = laya_tokenizer(word, add_special_tokens=False)["input_ids"]
        if not 1 <= len(ids) <= 4:
            continue
        # Prefer a word with a leading space over a word continuation token.
        rank = (not decoded.startswith(" "), token_id)
        if word not in found or rank < found[word][0]:
            found[word] = (rank, token_id, ids)
    return [(word, found[word][1], found[word][2]) for word in sorted(found)]


class StitchRuntime:
    def __init__(self, module, agent, image_processor, words, metadata, width=512):
        if not 128 <= width <= 1024:
            raise ValueError("width must be in 128..1024")
        self.module, self.agent, self.image_processor = module, agent, image_processor
        self.words, self.metadata, self.width = words, metadata, width

    @classmethod
    def build(cls, width=512):
        import laya_mlx
        from huggingface_hub import snapshot_download

        path = snapshot_download(LAYA_MLX_ID, revision=LAYA_MLX_REVISION, local_files_only=True)
        agent = laya_mlx.load(path)
        qwen = QwenVision(width)
        vocabulary = shared_words(qwen.processor.tokenizer, agent.tok)
        ids = mx.array([x[1] for x in vocabulary])
        q_embedding = qwen.model.language_model.model.embed_tokens
        keys = q_embedding(ids).astype(mx.float32)
        # Averaging subword embeddings is a fixed construction, not a learned alignment.
        l_embedding = agent.model.encoder.embeddings.tok_embeddings
        max_len = max(len(x[2]) for x in vocabulary)
        padded = np.zeros((len(vocabulary), max_len), dtype=np.int32)
        mask = np.zeros_like(padded, dtype=np.float32)
        for i, (_, _, pieces) in enumerate(vocabulary):
            padded[i, : len(pieces)] = pieces
            mask[i, : len(pieces)] = 1
        values = (l_embedding(mx.array(padded)).astype(mx.float32) * mx.array(mask)[..., None]).sum(
            1
        )
        values = values / mx.array(mask.sum(1))[:, None]
        bridge = LexicalBridge(keys, values)
        module = FrozenStitch(qwen.model.vision_tower, agent.model, bridge)
        mx.eval(module.parameters())
        metadata = {
            "method": "fixed shared-vocabulary attention, experimental",
            "qwen_model": QWEN_ID,
            "qwen_revision": QWEN_REVISION,
            "laya_model": LAYA_MLX_ID,
            "laya_revision": LAYA_MLX_REVISION,
            "word_count": len(vocabulary),
            "grid_size": 4,
            "top_k": 8,
            "temperature": 0.05,
            "vision_config": asdict(module.vision.config),
            "encoder_config": agent.encoder_cfg,
            "agent_config": agent.cfg,
            "new_training": False,
            "connector_fitting": False,
            "qwen_decoder_layers_executed": 0,
            "generated_tokens": 0,
            "cross_game_capability_established": False,
            "bundle_format": 1,
        }
        return cls(
            module,
            agent,
            qwen.processor.image_processor,
            [x[0] for x in vocabulary],
            metadata,
            width,
        )

    def prepare(self, instructions, choices):
        from laya_mlx.agent import collate_items

        if not instructions.strip() or not 2 <= len(choices) <= 32:
            raise ValueError("Need instructions and 2..32 choices")
        question = {
            "decision": {"type": "choice", "instructions": instructions, "criteria": choices}
        }
        item = self.agent.prepare("", question)[0][0]
        prefix = item["ids"][:-1]
        prefix += self.agent.tok("Image patches left to right, top to bottom:")["input_ids"]
        start = len(prefix)
        count = self.module.grid_size**2
        item["ids"] = prefix + [self.agent.tok.pad_token_id] * count + [self.agent.tok.sep_token_id]
        if len(item["ids"]) > self.agent.cfg["max_len"]:
            raise ValueError("Question and visual slots exceed the context limit")
        batch = collate_items([item], self.agent.tok.pad_token_id)
        return {k: mx.array(v) for k, v in batch.items()}, start

    def pixels(self, path):
        with Image.open(path) as image:
            image = image.convert("RGB")
        if image.width > self.width:
            image = image.resize((self.width, round(image.height * self.width / image.width)))
        inputs = self.image_processor(images=[image], return_tensors="np")
        return mx.array(inputs["pixel_values"]), mx.array(inputs["image_grid_thw"])

    def decide(self, path, instructions, choices):
        started = time.perf_counter()
        batch, start = self.prepare(instructions, choices)
        pixels, grid = self.pixels(path)
        scores, indices, weights = self.module(pixels, grid, batch, start)
        mx.eval(scores, indices, weights)
        elapsed = (time.perf_counter() - started) * 1000
        scores = np.asarray(scores[0])[: len(choices)]
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite decision scores")
        best = np.asarray(indices)[np.arange(len(indices)), np.asarray(weights).argmax(-1)]
        return {
            "choice": list(choices)[int(scores.argmax())],
            "scores": dict(zip(choices, scores.tolist(), strict=True)),
            "image_to_scores_ms": elapsed,
            "diagnostic_top_words": [self.words[i] for i in best],
            "input_events_sent": 0,
        }

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.module.save_weights(str(directory / "model.safetensors"))
        (directory / "config.json").write_text(json.dumps(self.metadata, indent=2) + "\n")
        (directory / "words.json").write_text(json.dumps(self.words) + "\n")
        shutil.copytree(self.agent.model_dir / "tokenizer", directory / "tokenizer")
        self.image_processor.save_pretrained(directory / "image_processor")

    @classmethod
    def load(cls, directory, width=512):
        from laya_mlx.agent import Agent
        from laya_mlx.model import DecisionModel, EncoderConfig
        from laya_mlx.tokenizer import Tokenizer
        from mlx_vlm.models.qwen3_5.config import VisionConfig
        from mlx_vlm.models.qwen3_5.vision import VisionModel
        from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor

        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        if config.get("bundle_format", 1) != 1:
            raise ValueError("Unsupported stitched checkpoint format")
        words = json.loads((directory / "words.json").read_text())
        vision = VisionModel(VisionConfig.from_dict(config["vision_config"]))
        laya = DecisionModel(
            EncoderConfig.from_dict(config["encoder_config"]), config["agent_config"]
        )
        bridge = LexicalBridge(
            mx.zeros((len(words), config["vision_config"]["out_hidden_size"])),
            mx.zeros((len(words), config["encoder_config"]["hidden_size"])),
            config["top_k"],
            config["temperature"],
        )
        module = FrozenStitch(vision, laya, bridge, config["grid_size"])
        module.load_weights(str(directory / "model.safetensors"), strict=True)
        mx.eval(module.parameters())
        agent = Agent.__new__(Agent)
        agent.cfg, agent.encoder_cfg, agent.model = (
            config["agent_config"],
            config["encoder_config"],
            laya,
        )
        agent._prefix_cache = None
        agent.tok, agent.model_dir = Tokenizer(directory / "tokenizer"), directory
        processor = Qwen3VLImageProcessor.from_pretrained(
            directory / "image_processor", local_files_only=True
        )
        return cls(module, agent, processor, words, config, width)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, help="Load a self-contained stitched checkpoint")
    parser.add_argument(
        "--save", type=Path, help="Save one stitched weights file and configuration"
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--choices", required=True, type=Path, help="JSON object of label: description"
    )
    args = parser.parse_args()
    runtime = StitchRuntime.load(args.bundle) if args.bundle else StitchRuntime.build()
    if args.save:
        runtime.save(args.save)
    print(
        json.dumps(
            runtime.decide(args.image, args.question, json.loads(args.choices.read_text())),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
