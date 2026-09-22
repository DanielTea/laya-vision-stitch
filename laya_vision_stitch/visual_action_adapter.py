"""Small visual residuals conditioned on Laya, with independent control branches."""

from dataclasses import asdict, replace

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def encode_action_history(row, config):
    """Numeric controls plus availability flags; targets are never read here."""
    values = np.zeros(len(config.buttons) + 7, dtype=np.float32)
    previous = row.get("previous_actions", [])
    if previous:
        action = previous[-1]
        values[0] = 1
        if "buttons" in action:
            buttons = action["buttons"]
            if not isinstance(buttons, list) or set(buttons) - set(config.buttons):
                raise ValueError("Invalid previous action buttons")
            values[1] = 1
            values[3 : 3 + len(config.buttons)] = [b in buttons for b in config.buttons]
        if "mouse_delta" in action:
            delta = np.asarray(action["mouse_delta"], dtype=np.float32)
            if delta.shape != (2,) or not np.isfinite(delta).all() or (np.abs(delta) > 1).any():
                raise ValueError("Invalid previous mouse delta")
            values[2] = 1
            values[-4:-2] = delta
            values[-2:] = np.sign(delta) * np.log1p(512 * np.abs(delta)) / np.log(513)
    return mx.array(values[None])


class VisualControlBranch(nn.Module):
    def __init__(self, visual_width, language_width, output_width, config):
        super().__init__()
        d = config.connector_width
        self.visual_norm = nn.LayerNorm(visual_width)
        self.language_norm = nn.LayerNorm(language_width)
        self.visual = nn.Linear(visual_width + 4, d)
        self.language = nn.Linear(language_width, d)
        self.numeric_action_history = config.numeric_action_history
        if self.numeric_action_history:
            self.history = nn.Sequential(
                nn.Linear(len(config.buttons) + 7, d), nn.GELU(), nn.Linear(d, d)
            )
        self.queries = mx.random.normal((config.action_chunk_size, d)) * 0.02
        self.goal_attention = nn.MultiHeadAttention(d, config.heads)
        self.visual_attention = nn.MultiHeadAttention(d, config.heads)
        self.norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.output = nn.Linear(d, output_width)
        # Adding this branch is initially an exact identity conversion.
        self.output.weight = mx.zeros_like(self.output.weight)
        self.output.bias = mx.zeros_like(self.output.bias)

    def __call__(self, patches, coordinates, context, history=None):
        visual = self.visual(mx.concatenate([self.visual_norm(patches), coordinates], -1))[None]
        language = self.language(self.language_norm(context.astype(mx.float32)))
        q = self.queries[None]
        if self.numeric_action_history:
            if history is None:
                raise ValueError(
                    "Numeric action history input is required, including missing flags"
                )
            q = q + self.history(history)[:, None, :]
        q = q + self.goal_attention(q, language, language)
        q = q + self.visual_attention(self.norm(q), visual, visual)
        q = q + self.ffn(self.norm(q))
        return self.output(self.norm(q))


class VisualActionAdapter(nn.Module):
    def __init__(self, visual_width, language_width, config):
        super().__init__()
        if config.action_chunk_size < 2:
            raise ValueError("Visual action residual requires categorical chunk outputs")
        self.buttons = VisualControlBranch(
            visual_width, language_width, len(config.buttons), config
        )
        self.camera = VisualControlBranch(
            visual_width, language_width, 2 * len(config.mouse_bins), config
        )
        self.bin_count = len(config.mouse_bins)

    def __call__(self, patches, coordinates, context, history=None):
        return {
            "buttons": self.buttons(patches, coordinates, context, history),
            "camera": self.camera(patches, coordinates, context, history).reshape(
                1, -1, 2, self.bin_count
            ),
        }

    def fuse(self, original, patches, coordinates, context, history=None):
        residual = self(patches, coordinates, context, history)
        buttons = original["chunk_buttons"] + residual["buttons"]
        return {
            **original,
            "buttons": buttons[:, 0],
            "chunk_buttons": buttons,
            "chunk_mouse": original["chunk_mouse"] + residual["camera"],
        }


def attach(runtime, numeric_history=False):
    model = runtime.module
    if model.policy_config.visual_action_adapter:
        raise ValueError("Checkpoint already has a visual action adapter")
    config = replace(
        model.policy_config, visual_action_adapter=True, numeric_action_history=numeric_history
    )
    model.visual_actions = VisualActionAdapter(
        model.vision.config.out_hidden_size, model.laya.encoder.config.hidden_size, config
    )
    model.policy_config = config
    runtime.metadata["policy_config"] = asdict(config)
    model.freeze()
    model.visual_actions.unfreeze()
