"""Native MLX inference for the pinned C-RADIOv3-B backbone.

Original implementation of its public ViT architecture and weight mapping.
Weights retain NVIDIA's model license; no teacher adaptors are loaded. Validate
against NVIDIA's reference before using converted weights in an experiment.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

RADIO_ID = "nvidia/C-RADIOv3-B"
RADIO_REVISION = "44653a0482cf460bb4f12595fc3cc3dfecc403d1"


@dataclass
class RadioConfig:
    out_hidden_size: int = 768
    depth: int = 12
    heads: int = 12
    prefix_tokens: int = 8
    position_side: int = 128
    patch_size: int = 16


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)

    def __call__(self, x):
        b, n, d = x.shape
        q, k, v = mx.split(self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads), 3, axis=2)
        q, k, v = [a.squeeze(2).transpose(0, 2, 1, 3) for a in (q, k, v)]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=(d // self.heads) ** -0.5)
        return self.proj(y.transpose(0, 2, 1, 3).reshape(b, n, d))


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(width, 4 * width), nn.Linear(4 * width, width)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.out_hidden_size
        self.norm1, self.norm2 = nn.LayerNorm(d, eps=1e-6), nn.LayerNorm(d, eps=1e-6)
        self.attn, self.mlp = Attention(d, config.heads), MLP(d)
        self.scale1, self.scale2 = mx.ones(d), mx.ones(d)

    def __call__(self, x):
        x = x + self.scale1 * self.attn(self.norm1(x))
        return x + self.scale2 * self.mlp(self.norm2(x))


class RadioVision(nn.Module):
    spatial_merge_size = 1

    def __init__(self, config=None):
        super().__init__()
        self.config = config or RadioConfig()
        c = self.config
        self.embedder = nn.Linear(3 * c.patch_size**2, c.out_hidden_size, bias=False)
        self.tokens = mx.zeros((c.prefix_tokens, c.out_hidden_size))
        self.positions = mx.zeros((1, c.position_side**2, c.out_hidden_size))
        self.mean = mx.zeros(3)
        self.std = mx.ones(3)
        self.blocks = [Block(c) for _ in range(c.depth)]

    @property
    def input_dtype(self):
        return self.embedder.weight.dtype

    def position_encoding(self, height, width):
        # Reference: bilinear align_corners=True to a max(H,W) square,
        # followed by a top-left rectangular crop. No learned values are changed.
        size, source = max(height, width), self.config.position_side
        coord = mx.arange(size, dtype=mx.float32) * ((source - 1) / max(size - 1, 1))
        lo = mx.floor(coord).astype(mx.int32)
        hi = mx.minimum(lo + 1, source - 1)
        fraction = coord - lo
        ly, lx = lo[:height, None], lo[None, :width]
        hy, hx = hi[:height, None], hi[None, :width]
        wy, wx = fraction[:height, None, None], fraction[None, :width, None]
        p = self.positions[0]
        top = (
            p[ly * source + lx].astype(mx.float32) * (1 - wx)
            + p[ly * source + hx].astype(mx.float32) * wx
        )
        bottom = (
            p[hy * source + lx].astype(mx.float32) * (1 - wx)
            + p[hy * source + hx].astype(mx.float32) * wx
        )
        return (
            (top * (1 - wy) + bottom * wy).reshape(1, height * width, -1).astype(self.input_dtype)
        )

    def __call__(self, pixels, grid=None):
        b, height, width, channels = pixels.shape
        patch = self.config.patch_size
        if channels != 3 or height % patch or width % patch:
            raise ValueError("RADIO expects NHWC RGB images divisible by its patch size")
        h, w = height // patch, width // patch
        x = (pixels - self.mean) / self.std
        x = x.reshape(b, h, patch, w, patch, 3).transpose(0, 1, 3, 5, 2, 4).reshape(b, h * w, -1)
        x = self.embedder(x) + self.position_encoding(h, w)
        x = mx.concatenate([mx.broadcast_to(self.tokens[None], (b, *self.tokens.shape)), x], axis=1)
        for block in self.blocks:
            x = block(x)
        return x[:, self.config.prefix_tokens :].reshape(-1, self.config.out_hidden_size), x[
            :, :3
        ].reshape(b, -1)

    @classmethod
    def from_source(cls, directory, dtype=mx.float32):
        weights = mx.load(str(Path(directory) / "model.safetensors"))
        config = json.loads((Path(directory) / "config.json").read_text())
        if config["version"] != "c-radio_v3-b" or config["feature_normalizer_config"] is not None:
            raise ValueError("Only the pinned C-RADIOv3-B architecture is supported")
        rename = {
            "radio_model.input_conditioner.norm_mean": "mean",
            "radio_model.input_conditioner.norm_std": "std",
            "radio_model.model.patch_generator.cls_token.token": "tokens",
            "radio_model.model.patch_generator.embedder.weight": "embedder.weight",
            "radio_model.model.patch_generator.pos_embed": "positions",
        }
        mapped = []
        for k, value in weights.items():
            if k in {"radio_model.model.reg_token", "radio_model.summary_idxs"}:
                continue  # Unused by the backbone's dense feature path.
            key = rename.get(k, k.removeprefix("radio_model.model."))
            key = key.replace("ls1.grandma", "scale1").replace("ls2.grandma", "scale2")
            if key in {"mean", "std"}:
                value = value.reshape(3)
            mapped.append((key, value.astype(dtype)))
        model = cls()
        model.load_weights(mapped, strict=True)
        model.freeze()
        model.eval()
        mx.eval(model.parameters())
        return model


class RadioProcessor:
    def to_dict(self):
        return {"type": "radio_rgb_bicubic_patch16", "version": 1}

    def __call__(self, images, return_tensors="np"):
        if len(images) != 1:
            raise ValueError("Single-image processor")
        image = images[0].convert("RGB")
        width, height = [max(16, round(v / 16) * 16) for v in image.size]
        image = image.resize((width, height), Image.Resampling.BICUBIC)
        return {
            "pixel_values": np.asarray(image, dtype=np.float32)[None] / 255,
            "image_grid_thw": np.array([[1, height // 16, width // 16]], dtype=np.int32),
        }

    def save_pretrained(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "preprocessor_config.json").write_text(json.dumps(self.to_dict()) + "\n")
