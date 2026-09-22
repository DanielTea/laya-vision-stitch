# SPDX-License-Identifier: Apache-2.0
# Embedding-forward adaptation follows laya-mlx; see third_party/laya-mlx.NOTICE.
"""Differentiable Qwen vision -> goal-conditioned connector -> frozen Laya.

No reference bank, caption generation or game-specific policy. The new action
heads require demonstrations; random initialization is not a playable agent.
"""

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from PIL import Image

from .backends import QWEN_ID, QWEN_REVISION, QwenVision
from .reference_stitch import LAYA_MODELS, load_laya


@dataclass
class PolicyConfig:
    visual_slots: int = 16
    connector_width: int = 128
    heads: int = 4
    max_frames: int = 4
    image_width: int = 320
    lora_rank: int = 0
    lora_layers: int = 2
    connector_type: str = "queries"
    buttons: tuple = ("w", "a", "s", "d", "space", "mouse_left", "mouse_right")
    durations: tuple = (0.05, 0.1, 0.2, 0.4)

    def __post_init__(self):
        if not 1 <= self.visual_slots <= 64 or not 1 <= self.max_frames <= 8:
            raise ValueError("Invalid visual slot/frame count")
        if self.heads < 1 or self.connector_width < 8 or self.connector_width % self.heads:
            raise ValueError("Connector width must be divisible by head count")
        if not 128 <= self.image_width <= 1024:
            raise ValueError("image_width must be in 128..1024")
        if not 0 <= self.lora_rank <= 64 or self.lora_layers < 1:
            raise ValueError("Invalid LoRA rank/layer count")
        if self.connector_type not in ("queries", "spatial", "aligned"):
            raise ValueError("Unknown connector type")
        if (
            self.connector_type == "spatial"
            and int(self.visual_slots**0.5) ** 2 != self.visual_slots
        ):
            raise ValueError("Spatial connector requires a square number of slots")
        if not self.buttons or len(set(self.buttons)) != len(self.buttons):
            raise ValueError("Provide unique button names")
        if any(not isinstance(b, str) or not b.strip() for b in self.buttons):
            raise ValueError("Button names must be nonempty strings")
        if not self.durations or any(not np.isfinite(x) or x <= 0 for x in self.durations):
            raise ValueError("Durations must be finite positive seconds")
        if list(self.durations) != sorted(set(self.durations)):
            raise ValueError("Durations must be distinct and ascending")


class LoRALinear(nn.Module):
    """Frozen linear plus a small rank-r update. Base weights are never changed."""

    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.base.freeze()
        out_width, in_width = base.weight.shape
        self.lora_a = mx.random.normal((rank, in_width)) * in_width**-0.5
        self.lora_b = mx.zeros((out_width, rank))

    def __call__(self, x):
        original = self.base(x)
        delta = (x.astype(mx.float32) @ self.lora_a.T) @ self.lora_b.T
        return original + delta.astype(original.dtype)


class GoalConnector(nn.Module):
    def __init__(self, visual_width, language_width, config):
        super().__init__()
        d = config.connector_width
        self.visual_norm = nn.LayerNorm(visual_width)
        self.visual = nn.Linear(visual_width, d)
        self.text = nn.Linear(language_width, d)
        self.coordinates = nn.Linear(4, d)  # x, y, seconds before now, frame order
        self.queries = mx.random.normal((config.visual_slots, d)) * 0.02
        self.goal_attention = nn.MultiHeadAttention(d, config.heads)
        self.image_attention = nn.MultiHeadAttention(d, config.heads)
        self.norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        self.output = nn.Linear(d, language_width)

    def __call__(self, patches, coordinates, goal_embeddings):
        visual = self.visual(self.visual_norm(patches)) + self.coordinates(coordinates)
        text = self.text(goal_embeddings.astype(mx.float32))
        q = self.queries[None] + self.goal_attention(self.queries[None], text, text)
        q = q + self.image_attention(q, visual[None], visual[None])
        q = q + self.ffn(self.norm(q))
        return self.output(q)


class SpatialConnector(nn.Module):
    """Preserve a regular spatial lattice instead of initially uniform queries.

    This is generic image tokenization, with no object/color/action rules. Frame
    time remains a feature; slots aggregate all supplied frames at each location.
    """

    def __init__(self, visual_width, language_width, config):
        super().__init__()
        self.grid_size = int(config.visual_slots**0.5)
        self.norm = nn.LayerNorm(visual_width)
        self.project = nn.Sequential(
            nn.Linear(visual_width + 4, config.connector_width),
            nn.GELU(),
            nn.Linear(config.connector_width, language_width),
        )
        self.goal = nn.Linear(language_width, language_width, bias=False)

    def __call__(self, patches, coordinates, goal_embeddings):
        axis = mx.linspace(-1, 1, self.grid_size)
        yy, xx = mx.meshgrid(axis, axis, indexing="ij")
        centers = mx.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
        distances = mx.sum((centers[:, None] - coordinates[None, :, :2]) ** 2, axis=-1)
        weights = mx.softmax(-distances * self.grid_size**2, axis=-1)
        features = mx.concatenate([self.norm(patches), coordinates], axis=-1)
        pooled = self.project(weights @ features)[None]
        goal = self.goal(goal_embeddings.astype(mx.float32).mean(axis=1))[:, None]
        return pooled * (1 + mx.tanh(goal))


class AlignedConnector(nn.Module):
    """Learn a residual around text-manifold soft tokens; no decoding at inference.

    Anchors are trainable parameters initialized from training descriptions only.
    They are not a memory of reference images or runtime target descriptions.
    """

    def __init__(self, visual_width, language_width, config):
        super().__init__()
        self.slots, self.width = config.visual_slots, language_width
        d = config.connector_width
        self.norm = nn.LayerNorm(visual_width)
        self.visual = nn.Linear(visual_width + 4, d)
        self.hidden = nn.Linear(16 * d, d)
        self.output = nn.Linear(d, self.slots * language_width)
        self.output.weight = mx.zeros_like(self.output.weight)
        self.output.bias = mx.zeros_like(self.output.bias)
        self.anchors = mx.zeros((self.slots, language_width))

    def __call__(self, patches, coordinates, goal_embeddings):
        axis = mx.linspace(-1, 1, 4)
        yy, xx = mx.meshgrid(axis, axis, indexing="ij")
        centers = mx.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
        weights = mx.softmax(
            -16 * mx.sum((centers[:, None] - coordinates[None, :, :2]) ** 2, axis=-1), axis=-1
        )
        local = nn.gelu(self.visual(mx.concatenate([self.norm(patches), coordinates], axis=-1)))
        state = nn.gelu(self.hidden((weights @ local).reshape(1, -1)))
        return self.anchors[None] + self.output(state).reshape(1, self.slots, self.width)


class ActionHeads(nn.Module):
    def __init__(self, width, config):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.buttons = nn.Linear(width, len(config.buttons))
        self.mouse = nn.Linear(width, 2)
        self.pointer = nn.Linear(width, 2)
        self.pointer_active = nn.Linear(width, 1)
        self.duration = nn.Linear(width, len(config.durations))

    def __call__(self, state):
        state = self.norm(state.astype(mx.float32))
        return {
            "buttons": self.buttons(state),
            "mouse": mx.tanh(self.mouse(state)),
            "pointer": mx.sigmoid(self.pointer(state)),
            "pointer_active": self.pointer_active(state).squeeze(-1),
            "duration": self.duration(state),
        }


class TrainableStitch(nn.Module):
    def __init__(self, vision, laya, config):
        super().__init__()
        self.vision, self.laya = vision, laya
        self.policy_config = config
        width = laya.encoder.config.hidden_size
        connector_class = {
            "spatial": SpatialConnector,
            "queries": GoalConnector,
            "aligned": AlignedConnector,
        }[config.connector_type]
        self.connector = connector_class(vision.config.out_hidden_size, width, config)
        self.actions = ActionHeads(width, config)
        self.vision.freeze()
        self.laya.freeze()
        if config.lora_rank:
            if config.lora_layers > len(self.laya.encoder.layers):
                raise ValueError("LoRA layer count exceeds Laya encoder depth")
            for layer in self.laya.encoder.layers[-config.lora_layers :]:
                layer.attn.Wqkv = LoRALinear(layer.attn.Wqkv, config.lora_rank)
                layer.attn.Wo = LoRALinear(layer.attn.Wo, config.lora_rank)
        self.eval()

    def encode_frame(self, pixels, grid, age, frame_index):
        visual, _ = self.vision(pixels.astype(self.vision.patch_embed.proj.weight.dtype), grid)
        t, h, w = map(int, np.asarray(grid)[0])
        m = self.vision.spatial_merge_size
        if t != 1:
            raise ValueError("Frames must be individual images")
        h, w = h // m, w // m
        yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
        coordinates = np.stack(
            [xx.ravel(), yy.ravel(), np.full(h * w, age), np.full(h * w, frame_index)], axis=-1
        )
        return mx.stop_gradient(visual.astype(mx.float32)), mx.array(coordinates, mx.float32)

    def from_features(self, patches, coordinates, batch, goal_ids, start):
        encoder = self.laya.encoder
        goal = encoder.embeddings.tok_embeddings(goal_ids)
        state = self.connector(patches, coordinates, goal)
        return self.from_state(state, batch, start)

    def from_state(self, state, batch, start):
        """Shared Laya path; explicit state input also enables text-oracle audits."""
        from laya_mlx.model import attention_masks

        encoder = self.laya.encoder
        original = encoder.embeddings.tok_embeddings(batch["input_ids"])
        state = state.astype(original.dtype)
        h = mx.concatenate(
            [original[:, :start], state, original[:, start + self.policy_config.visual_slots :]],
            axis=1,
        )
        h = encoder.embeddings.norm(h)
        masks = attention_masks(batch["attention_mask"], encoder.config.local_attention)
        for layer in encoder.layers:
            h = layer(h, masks[layer.attention_type])
        h = encoder.final_norm(h) + self.laya.type_emb(batch["qtype"])[:, None, :]
        h = self.laya.head(h, batch["attention_mask"][:, None, None, :].astype(mx.bool_))
        markers = h[mx.arange(h.shape[0])[:, None], batch["marker_pos"]]
        choices = self.laya.scorer(markers).squeeze(-1).astype(mx.float32)
        choices = mx.where(batch["marker_mask"], choices, -1e4)
        return {
            "choices": choices,
            "visual_state": state.astype(mx.float32),
            **self.actions(h[:, 0]),
        }

    def __call__(self, frames, batch, goal_ids, start):
        encoded = [self.encode_frame(*frame) for frame in frames]
        return self.from_features(
            mx.concatenate([x[0] for x in encoded]),
            mx.concatenate([x[1] for x in encoded]),
            batch,
            goal_ids,
            start,
        )


def context_text(row):
    return (
        f"Goal: {row['goal']}\nControls: {row.get('controls', '')}\n"
        f"Previous actions: {json.dumps(row.get('previous_actions', []), ensure_ascii=False)}"
    )


class TrainableRuntime:
    def __init__(self, module, agent, processor, metadata):
        self.module, self.agent, self.processor, self.metadata = module, agent, processor, metadata

    def add_lora(self, rank, layers):
        config = self.module.policy_config
        if config.lora_rank:
            raise ValueError("Checkpoint already contains LoRA")
        if not 1 <= rank <= 64 or not 1 <= layers <= len(self.module.laya.encoder.layers):
            raise ValueError("Invalid LoRA dimensions")
        for layer in self.module.laya.encoder.layers[-layers:]:
            layer.attn.Wqkv = LoRALinear(layer.attn.Wqkv, rank)
            layer.attn.Wo = LoRALinear(layer.attn.Wo, rank)
        config.lora_rank, config.lora_layers = rank, layers
        self.metadata["policy_config"] = asdict(config)
        self.metadata["laya_lora_enabled"] = True

    @classmethod
    def build(cls, config=None):
        config = config or PolicyConfig()
        agent = load_laya("english")
        qwen = QwenVision(config.image_width)
        module = TrainableStitch(qwen.model.vision_tower, agent.model, config)
        metadata = {
            "format": "trainable-stitch-1",
            "qwen_model": QWEN_ID,
            "qwen_revision": QWEN_REVISION,
            "laya_model": LAYA_MODELS["english"][0],
            "laya_revision": LAYA_MODELS["english"][1],
            "vision_config": asdict(module.vision.config),
            "encoder_config": agent.encoder_cfg,
            "agent_config": agent.cfg,
            "policy_config": asdict(config),
            "training_steps": 0,
            "action_training_examples": 0,
            "pretrained_backbones_updated": False,
            "laya_lora_enabled": bool(config.lora_rank),
            "cross_game_capability_established": False,
        }
        mx.eval(module.parameters())
        return cls(module, agent, qwen.processor.image_processor, metadata)

    def prepare(self, row):
        from laya_mlx.agent import collate_items
        from laya_mlx.common import render_options

        choices = row.get("choices") or {"act": "Take the next action.", "wait": "Wait."}
        context = context_text(row)
        # Keep Laya's sequence grammar but reject overflow instead of silently
        # truncating goals/options using upstream head_max_len and 48-token caps.
        tok = self.agent.tok
        clean_context = context.replace(tok.mask_token, " ")
        prefix = [tok.cls_token_id]
        prefix += tok(f"choice question: {clean_context}", add_special_tokens=False)["input_ids"]
        prefix += [tok.sep_token_id]
        markers = []
        for option in render_options({"t": "choice", "crit": choices}):
            markers.append(len(prefix))
            prefix += [tok.mask_token_id]
            prefix += tok(" " + option.replace(tok.mask_token, " "), add_special_tokens=False)[
                "input_ids"
            ]
        prefix += [tok.sep_token_id]
        item = {"markers": markers, "qtype": 0}
        start = len(prefix)
        item["ids"] = (
            prefix
            + [self.agent.tok.pad_token_id] * self.module.policy_config.visual_slots
            + [self.agent.tok.sep_token_id]
        )
        if len(item["ids"]) > self.agent.cfg["max_len"]:
            raise ValueError("Prompt and visual tokens exceed Laya context; no silent truncation")
        batch = {
            k: mx.array(v) for k, v in collate_items([item], self.agent.tok.pad_token_id).items()
        }
        goal_ids = mx.array([self.agent.tok(context)["input_ids"]])
        return batch, goal_ids, start

    def frames(self, row):
        frames = row["frames"]
        if not 1 <= len(frames) <= self.module.policy_config.max_frames:
            raise ValueError("Frame history exceeds configured limit")
        result = []
        for index, frame in enumerate(frames):
            if isinstance(frame["image"], Image.Image):
                image = frame["image"].convert("RGB")
            else:
                with Image.open(frame["image"]) as source:
                    image = source.convert("RGB")
            width = self.module.policy_config.image_width
            if image.width > width:
                image = image.resize((width, max(1, round(image.height * width / image.width))))
            data = self.processor(images=[image], return_tensors="np")
            result.append(
                (
                    mx.array(data["pixel_values"]),
                    mx.array(data["image_grid_thw"]),
                    frame["age_seconds"],
                    index / max(1, len(frames) - 1),
                )
            )
        return result

    def features(self, row):
        frames = [self.module.encode_frame(*frame) for frame in self.frames(row)]
        result = mx.concatenate([f[0] for f in frames]), mx.concatenate([f[1] for f in frames])
        mx.eval(result)
        return result

    def predict(self, row):
        started = time.perf_counter()
        output = self.module(self.frames(row), *self.prepare(row))
        mx.eval(output)
        elapsed = (time.perf_counter() - started) * 1000
        config = self.module.policy_config
        button_p = np.asarray(mx.sigmoid(output["buttons"][0]))
        result = {
            "button_probabilities": dict(zip(config.buttons, button_p.tolist(), strict=True)),
            "buttons": [b for b, p in zip(config.buttons, button_p, strict=True) if p >= 0.5],
            "mouse_delta_normalized": np.asarray(output["mouse"][0]).tolist(),
            "pointer_xy_normalized": np.asarray(output["pointer"][0]).tolist(),
            "pointer_active_probability": float(mx.sigmoid(output["pointer_active"][0]).item()),
            "duration_seconds": config.durations[int(mx.argmax(output["duration"][0]).item())],
            "image_to_outputs_ms": elapsed,
            "action_heads_trained": self.metadata["action_training_examples"] > 0,
            "input_events_sent": 0,
        }
        if row.get("choices"):
            probs = mx.softmax(output["choices"][0])
            result["choice_probabilities"] = dict(
                zip(row["choices"], np.asarray(probs).tolist(), strict=True)
            )
        return result

    def parameter_counts(self):
        return {
            "total": sum(v.size for _, v in tree_flatten(self.module.parameters())),
            "trainable": sum(v.size for _, v in tree_flatten(self.module.trainable_parameters())),
            "trainable_roots": sorted(
                {k.split(".")[0] for k, _ in tree_flatten(self.module.trainable_parameters())}
            ),
        }

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.module.save_weights(str(directory / "model.safetensors"))
        (directory / "config.json").write_text(json.dumps(self.metadata, indent=2) + "\n")
        shutil.copytree(self.agent.model_dir / "tokenizer", directory / "tokenizer")
        self.processor.save_pretrained(directory / "image_processor")

    @classmethod
    def load(cls, directory):
        from laya_mlx.agent import Agent
        from laya_mlx.model import DecisionModel, EncoderConfig
        from laya_mlx.tokenizer import Tokenizer
        from mlx_vlm.models.qwen3_5.config import VisionConfig
        from mlx_vlm.models.qwen3_5.vision import VisionModel
        from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor

        directory = Path(directory)
        metadata = json.loads((directory / "config.json").read_text())
        if metadata["format"] != "trainable-stitch-1":
            raise ValueError("Unsupported checkpoint")
        laya = DecisionModel(
            EncoderConfig.from_dict(metadata["encoder_config"]), metadata["agent_config"]
        )
        module = TrainableStitch(
            VisionModel(VisionConfig.from_dict(metadata["vision_config"])),
            laya,
            PolicyConfig(**metadata["policy_config"]),
        )
        module.load_weights(str(directory / "model.safetensors"), strict=True)
        mx.eval(module.parameters())
        agent = Agent.__new__(Agent)
        agent.model, agent.cfg, agent.encoder_cfg = (
            laya,
            metadata["agent_config"],
            metadata["encoder_config"],
        )
        agent.tok, agent.model_dir, agent._prefix_cache = (
            Tokenizer(directory / "tokenizer"),
            directory,
            None,
        )
        processor = Qwen3VLImageProcessor.from_pretrained(
            directory / "image_processor", local_files_only=True
        )
        return cls(module, agent, processor, metadata)
