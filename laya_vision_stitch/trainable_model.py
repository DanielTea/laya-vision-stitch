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
    action_chunk_size: int = 1
    normalize_action_context: bool = False
    action_context_source: str = "decision"
    visual_action_adapter: bool = False
    numeric_action_history: bool = False
    temporal_adapter: str = "none"
    temporal_width: int = 128
    visual_fusion_layers: int = 0
    mouse_bins: tuple = (
        -1.0,
        -0.5,
        -0.25,
        -0.125,
        -0.0625,
        -0.03125,
        -0.015625,
        -0.0078125,
        0.0,
        0.0078125,
        0.015625,
        0.03125,
        0.0625,
        0.125,
        0.25,
        0.5,
        1.0,
    )
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
        if self.connector_type not in ("queries", "spatial", "aligned", "temporal"):
            raise ValueError("Unknown connector type")
        if not 1 <= self.action_chunk_size <= 8:
            raise ValueError("Invalid action chunk size")
        if self.action_context_source not in ("decision", "encoder"):
            raise ValueError("Unknown action context source")
        if self.visual_action_adapter and self.action_chunk_size < 2:
            raise ValueError("Visual action adapter requires chunk outputs")
        if self.temporal_adapter not in ("none", "mamba3", "attention"):
            raise ValueError("Unknown temporal adapter")
        if self.temporal_width < 32 or self.temporal_width % 32:
            raise ValueError("Temporal width must be a multiple of 32")
        if not isinstance(self.visual_fusion_layers, int) or self.visual_fusion_layers < 0:
            raise ValueError("Invalid visual fusion layer count")
        if self.temporal_adapter != "none" and self.visual_action_adapter:
            raise ValueError("Choose one action adapter")
        if self.numeric_action_history and not self.visual_action_adapter:
            raise ValueError("Numeric history requires the visual action adapter")
        if len(self.mouse_bins) < 3 or list(self.mouse_bins) != sorted(set(self.mouse_bins)):
            raise ValueError("Mouse bins must be distinct and ascending")
        if any(not np.isfinite(x) or not -1 <= x <= 1 for x in self.mouse_bins):
            raise ValueError("Mouse bins must be finite in [-1, 1]")
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


class TemporalConnector(nn.Module):
    """Keep spatial tokens separate by frame before learned temporal attention.

    An initially zero residual preserves the source connector on conversion.
    Everything is learned from pixels/text; no game-state features are inputs.
    """

    def __init__(self, visual_width, language_width, config):
        super().__init__()
        self.base = AlignedConnector(visual_width, language_width, config)
        d = config.connector_width
        self.norm = nn.LayerNorm(visual_width)
        self.visual = nn.Linear(visual_width + 4, d)
        self.temporal_norm = nn.LayerNorm(d)
        self.temporal = nn.MultiHeadAttention(d, config.heads)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.queries = mx.random.normal((config.visual_slots, d)) * 0.02
        self.goal = nn.Linear(language_width, d, bias=False)
        self.read = nn.MultiHeadAttention(d, config.heads)
        self.output = nn.Linear(d, language_width)
        self.output.weight = mx.zeros_like(self.output.weight)
        self.output.bias = mx.zeros_like(self.output.bias)

    def __call__(self, patches, coordinates, goal_embeddings):
        axis = mx.linspace(-1, 1, 4)
        yy, xx = mx.meshgrid(axis, axis, indexing="ij")
        centers = mx.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
        # Coordinates are input metadata, never trainable tensors. Preserve all
        # observed frame identities rather than averaging them on one lattice.
        orders = mx.array(np.unique(np.asarray(coordinates[:, 3])))
        distance = mx.sum((centers[:, None] - coordinates[None, :, :2]) ** 2, axis=-1)
        mask = orders[:, None, None] == coordinates[None, None, :, 3]
        weights = mx.softmax(mx.where(mask, -16 * distance[None], -1e9), axis=-1)
        values = self.visual(mx.concatenate([self.norm(patches), coordinates], axis=-1))
        tokens = (weights @ values).reshape(1, -1, values.shape[-1])
        normalized = self.temporal_norm(tokens)
        tokens = tokens + self.temporal(normalized, normalized, normalized)
        tokens = tokens + self.ffn(self.temporal_norm(tokens))
        queries = (
            self.queries[None] + self.goal(goal_embeddings.astype(mx.float32).mean(axis=1))[:, None]
        )
        residual = self.output(self.read(queries, tokens, tokens))
        return self.base(patches, coordinates, goal_embeddings) + residual


class ActionHeads(nn.Module):
    def __init__(self, width, config):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.buttons = nn.Linear(width, len(config.buttons))
        self.mouse = nn.Linear(width, 2)
        self.pointer = nn.Linear(width, 2)
        self.pointer_active = nn.Linear(width, 1)
        self.duration = nn.Linear(width, len(config.durations))
        self.chunk_size, self.button_count = config.action_chunk_size, len(config.buttons)
        self.bin_count = len(config.mouse_bins)
        if config.normalize_action_context:
            self.input_norm = nn.LayerNorm(width)
        if self.chunk_size > 1:
            d = config.connector_width
            self.action_queries = mx.random.normal((self.chunk_size, d)) * 0.02
            self.context = nn.Linear(width, d)
            self.attention = nn.MultiHeadAttention(d, config.heads)
            self.readout = nn.Linear(d, width)
            self.readout.weight = mx.zeros_like(self.readout.weight)
            self.readout.bias = mx.zeros_like(self.readout.bias)
            self.chunk_mouse = nn.Linear(width, 2 * self.bin_count)

    def __call__(self, state, tokens=None):
        if hasattr(self, "input_norm"):
            state = self.input_norm(state.astype(mx.float32))
            if tokens is not None:
                tokens = self.input_norm(tokens.astype(mx.float32))
        if self.chunk_size > 1:
            context = self.context(tokens.astype(mx.float32))
            queries = self.action_queries[None] + self.context(state.astype(mx.float32))[:, None]
            states = state[:, None] + self.readout(self.attention(queries, context, context))
            states = self.norm(states.astype(mx.float32))
            state = states[:, 0]
        else:
            state = self.norm(state.astype(mx.float32))
        result = {
            "buttons": self.buttons(state),
            "mouse": mx.tanh(self.mouse(state)),
            "pointer": mx.sigmoid(self.pointer(state)),
            "pointer_active": self.pointer_active(state).squeeze(-1),
            "duration": self.duration(state),
        }
        if self.chunk_size > 1:
            result["chunk_buttons"] = self.buttons(states)
            result["chunk_mouse"] = self.chunk_mouse(states).reshape(
                -1, self.chunk_size, 2, self.bin_count
            )
        return result


class GatedVisualFusion(nn.Module):
    """Learned dense visual cross-attention inside the pretrained Laya stack.

    A zero-initialized gate preserves the original layer exactly. All visual
    selection is learned; inputs contain no game-state estimates or action rules.
    """

    def __init__(self, visual_width, language_width, width=128):
        super().__init__()
        self.language_norm = nn.LayerNorm(language_width)
        self.visual_norm = nn.LayerNorm(visual_width)
        self.query = nn.Linear(language_width, width)
        self.visual = nn.Linear(visual_width + 4, width)
        self.attention = nn.MultiHeadAttention(width, 4)
        self.output = nn.Linear(width, language_width)
        self.gate = mx.array(0.0)

    def __call__(self, hidden, patches, coordinates):
        queries = self.query(self.language_norm(hidden.astype(mx.float32)))
        visual = self.visual(
            mx.concatenate([self.visual_norm(patches.astype(mx.float32)), coordinates], -1)
        )[None]
        delta = self.output(self.attention(queries, visual, visual))
        return hidden + (mx.tanh(self.gate) * delta).astype(hidden.dtype)


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
            "temporal": TemporalConnector,
        }[config.connector_type]
        self.connector = connector_class(vision.config.out_hidden_size, width, config)
        if config.visual_fusion_layers > len(laya.encoder.layers):
            raise ValueError("Visual fusion depth exceeds Laya depth")
        self.actions = ActionHeads(width, config)
        if config.visual_action_adapter:
            from .visual_action_adapter import VisualActionAdapter

            self.visual_actions = VisualActionAdapter(vision.config.out_hidden_size, width, config)
        self.vision.freeze()
        self.laya.freeze()
        if config.lora_rank:
            if config.lora_layers > len(self.laya.encoder.layers):
                raise ValueError("LoRA layer count exceeds Laya encoder depth")
            for layer in self.laya.encoder.layers[-config.lora_layers :]:
                layer.attn.Wqkv = LoRALinear(layer.attn.Wqkv, config.lora_rank)
                layer.attn.Wo = LoRALinear(layer.attn.Wo, config.lora_rank)
        if config.visual_fusion_layers:
            self.connector.fusion = [
                GatedVisualFusion(vision.config.out_hidden_size, width)
                for _ in range(config.visual_fusion_layers)
            ]
        if config.temporal_adapter != "none":
            from .temporal_adapter import TemporalActionAdapter

            self.temporal_actions = TemporalActionAdapter(
                vision.config.out_hidden_size, width, config
            )
            self.freeze()
            self.temporal_actions.unfreeze()
        self.eval()

    def encode_frame(self, pixels, grid, age, frame_index):
        dtype = (
            self.vision.input_dtype
            if hasattr(self.vision, "input_dtype")
            else self.vision.patch_embed.proj.weight.dtype
        )
        visual, _ = self.vision(pixels.astype(dtype), grid)
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
        return self.from_state(state, batch, start, patches, coordinates)

    def from_state(self, state, batch, start, patches=None, coordinates=None):
        """Shared Laya path; explicit state input also enables text-oracle audits."""
        if getattr(self.policy_config, "temporal_adapter", "none") != "none":
            raise ValueError(
                "Temporal checkpoints require TemporalRuntime and explicit session state"
            )
        h, choices = self.encode_state(state, batch, start, patches, coordinates)
        actions = self.actions(h[:, 0], h)
        if self.policy_config.visual_action_adapter:
            if patches is None or coordinates is None:
                raise ValueError("Visual action adapter requires raw visual features")
            actions = self.visual_actions.fuse(
                actions, patches, coordinates, h, batch.get("numeric_action_history")
            )
        return {
            "choices": choices,
            "visual_state": state.astype(mx.float32),
            **actions,
        }

    def action_context(self, patches, coordinates, batch, goal_ids, start):
        """Frozen-context caching for decoder-only experiments, without labels."""
        goal = self.laya.encoder.embeddings.tok_embeddings(goal_ids)
        state = self.connector(patches, coordinates, goal)
        return self.encode_state(state, batch, start, patches, coordinates)

    def encode_state(self, state, batch, start, patches=None, coordinates=None):
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
        fusion_start = len(encoder.layers) - self.policy_config.visual_fusion_layers
        if self.policy_config.visual_fusion_layers and (patches is None or coordinates is None):
            raise ValueError("Deep visual fusion requires actual visual features")
        for index, layer in enumerate(encoder.layers):
            if index >= fusion_start:
                h = self.connector.fusion[index - fusion_start](h, patches, coordinates)
            h = layer(h, masks[layer.attention_type])
        h = encoder.final_norm(h) + self.laya.type_emb(batch["qtype"])[:, None, :]
        encoder_context = h
        h = self.laya.head(h, batch["attention_mask"][:, None, None, :].astype(mx.bool_))
        markers = h[mx.arange(h.shape[0])[:, None], batch["marker_pos"]]
        choices = self.laya.scorer(markers).squeeze(-1).astype(mx.float32)
        choices = mx.where(batch["marker_mask"], choices, -1e4)
        return (
            encoder_context if self.policy_config.action_context_source == "encoder" else h
        ), choices

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


def decode_action_chunk(output, config):
    """Parallel neural outputs only; mouse modes avoid averaging opposite turns."""
    if "chunk_buttons" in output:
        probabilities = np.asarray(mx.sigmoid(output["chunk_buttons"][0]))
        distribution = np.asarray(mx.softmax(output["chunk_mouse"][0], axis=-1))
        movement = np.asarray(config.mouse_bins)[distribution.argmax(axis=-1)]
    else:
        probabilities = np.asarray(mx.sigmoid(output["buttons"]))
        movement = np.asarray(output["mouse"])
        distribution = None
    return [
        {
            "buttons": [
                b for b, p in zip(config.buttons, probabilities[i], strict=True) if p >= 0.5
            ],
            "mouse_delta": movement[i].tolist(),
            **(
                {"mouse_bin_probabilities": distribution[i].tolist()}
                if distribution is not None
                else {}
            ),
        }
        for i in range(len(probabilities))
    ]


def apply_supervision_scope(result, metadata):
    """Do not expose unsupervised chunk/pointer heads as usable pilot actions."""
    if not (metadata.get("visual_adapter_experiment") or metadata.get("first_step_control_only")):
        return result
    result = dict(result)
    result.pop("pointer_xy_normalized", None)
    result.pop("pointer_active_probability", None)
    result["action_chunk"] = result["action_chunk"][:1]
    result["chunk_step_seconds"] = 0.05
    # The dataset supplies this fixed control interval, not a duration target.
    result["duration_seconds"] = 0.05
    result["duration_source"] = "fixed_training_interval"
    result["supervised_outputs"] = ["first_step_buttons", "first_step_relative_mouse"]
    result["deployment_eligible"] = False
    return result


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

    def normalize_action_inputs(self):
        """Opt-in migration: normalize Laya activations before action attention."""
        config = self.module.policy_config
        if not config.normalize_action_context:
            self.module.actions.input_norm = nn.LayerNorm(
                self.module.laya.encoder.config.hidden_size
            )
            config.normalize_action_context = True
            self.metadata["policy_config"] = asdict(config)

    def expand_buttons(self, names):
        """Extend action vocabulary while preserving all existing learned logits."""
        config = self.module.policy_config
        names = tuple(names)
        if config.action_chunk_size > 1 and names != tuple(config.buttons):
            raise ValueError("Expand buttons before adding the chunk decoder")
        if len(set(names)) != len(names) or not set(config.buttons) <= set(names):
            raise ValueError("New vocabulary must contain each old button exactly once")
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("Button names must be nonempty strings")
        old = self.module.actions.buttons
        new = nn.Linear(old.weight.shape[1], len(names))
        # Rare additional buttons start inactive; existing rows remain exact.
        new.bias = mx.full((len(names),), -3.0)
        for index, name in enumerate(config.buttons):
            target = names.index(name)
            new.weight[target] = old.weight[index]
            new.bias[target] = old.bias[index]
        self.module.actions.buttons = new
        config.buttons = names
        self.metadata["policy_config"] = asdict(config)

    def upgrade_temporal(self):
        config = self.module.policy_config
        if config.connector_type != "aligned" or config.action_chunk_size != 1:
            raise ValueError("Temporal conversion requires an aligned, single-action checkpoint")
        width = self.module.laya.encoder.config.hidden_size
        connector = TemporalConnector(self.module.vision.config.out_hidden_size, width, config)
        connector.base = self.module.connector
        self.module.connector = connector
        config.connector_type, config.action_chunk_size = "temporal", 4
        old = self.module.actions
        new = ActionHeads(width, config)
        for name in ("norm", "buttons", "mouse", "pointer", "pointer_active", "duration"):
            setattr(new, name, getattr(old, name))
        self.module.actions = new
        self.metadata["policy_config"] = asdict(config)

    @classmethod
    def build(cls, config=None, *, radio_source=None):
        config = config or PolicyConfig()
        agent = load_laya("english")
        if radio_source is not None:
            from .radio_vision import RADIO_ID, RADIO_REVISION, RadioProcessor, RadioVision

            vision = RadioVision.from_source(radio_source, dtype=mx.float16)
            processor = RadioProcessor()
            source_metadata = {
                "vision_type": "radio_v3",
                "vision_model": RADIO_ID,
                "vision_revision": RADIO_REVISION,
                "vision_precision": "float16",
            }
        else:
            qwen = QwenVision(config.image_width)
            vision, processor = qwen.model.vision_tower, qwen.processor.image_processor
            source_metadata = {"qwen_model": QWEN_ID, "qwen_revision": QWEN_REVISION}
        module = TrainableStitch(vision, agent.model, config)
        metadata = {
            "format": "trainable-stitch-1",
            **source_metadata,
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
        return cls(module, agent, processor, metadata)

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
        if self.module.policy_config.numeric_action_history:
            from .visual_action_adapter import encode_action_history

            batch["numeric_action_history"] = encode_action_history(row, self.module.policy_config)
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
        if self.module.policy_config.temporal_adapter != "none":
            raise ValueError(
                "Temporal checkpoints require TemporalRuntime and explicit session state"
            )
        started = time.perf_counter()
        output = self.module(self.frames(row), *self.prepare(row))
        mx.eval(output)
        elapsed = (time.perf_counter() - started) * 1000
        config = self.module.policy_config
        chunk = decode_action_chunk(output, config)
        button_p = np.asarray(mx.sigmoid(output["buttons"][0]))
        result = {
            "button_probabilities": dict(zip(config.buttons, button_p.tolist(), strict=True)),
            "buttons": [b for b, p in zip(config.buttons, button_p, strict=True) if p >= 0.5],
            "mouse_delta_normalized": chunk[0]["mouse_delta"],
            "pointer_xy_normalized": np.asarray(output["pointer"][0]).tolist(),
            "pointer_active_probability": float(mx.sigmoid(output["pointer_active"][0]).item()),
            "duration_seconds": config.durations[int(mx.argmax(output["duration"][0]).item())],
            "image_to_outputs_ms": elapsed,
            "action_heads_trained": self.metadata["action_training_examples"] > 0,
            "input_events_sent": 0,
        }
        if config.action_chunk_size > 1:
            result["action_chunk"] = chunk
            result["chunk_step_seconds"] = 0.1
        if row.get("choices"):
            probs = mx.softmax(output["choices"][0])
            result["choice_probabilities"] = dict(
                zip(row["choices"], np.asarray(probs).tolist(), strict=True)
            )
        return apply_supervision_scope(result, self.metadata)

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
        if metadata.get("vision_type", "qwen3_5") == "radio_v3":
            from .radio_vision import RadioConfig, RadioProcessor, RadioVision

            vision = RadioVision(RadioConfig(**metadata["vision_config"]))
            processor = RadioProcessor()
            saved = json.loads(
                (directory / "image_processor" / "preprocessor_config.json").read_text()
            )
            if saved != processor.to_dict():
                raise ValueError("Unsupported RADIO preprocessing")
        elif metadata.get("vision_type", "qwen3_5") == "qwen3_5":
            vision = VisionModel(VisionConfig.from_dict(metadata["vision_config"]))
            processor = Qwen3VLImageProcessor.from_pretrained(
                directory / "image_processor", local_files_only=True
            )
        else:
            raise ValueError("Unknown vision architecture")
        module = TrainableStitch(
            vision,
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
        return cls(module, agent, processor, metadata)
