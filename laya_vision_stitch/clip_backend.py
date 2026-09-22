"""Frozen CLIP inference using pinned Apple MLX example code and converted weights."""

import json
from dataclasses import asdict, fields

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps

from .lexical_stitch import normalized
from .vendor.clip_model import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig
from .vendor.clip_tokenizer import CLIPTokenizer

CLIP_ID = "mlx-community/clip-vit-base-patch32"
CLIP_REVISION = "b0d393a1f061c5bdcbaa4bfba8682091f04d0c0d"


def legacy_weights(weights):
    """Rename the historical MLX export; do not change any tensor values."""
    result = {}
    for name, value in weights.items():
        name = name.replace(".layers.", ".encoder.layers.")
        for old, new in (
            (".attention.", ".self_attn."),
            (".key_proj.", ".k_proj."),
            (".query_proj.", ".q_proj."),
            (".value_proj.", ".v_proj."),
            (".linear1.", ".mlp.fc1."),
            (".linear2.", ".mlp.fc2."),
            (".ln1.", ".layer_norm1."),
            (".ln2.", ".layer_norm2."),
            ("vision_model.pre_layernorm.", "vision_model.pre_layrnorm."),
        ):
            name = name.replace(old, new)
        for prefix in ("text_model", "vision_model"):
            for field in (
                "token_embedding",
                "position_embedding",
                "patch_embedding",
                "class_embedding",
            ):
                if name.startswith(f"{prefix}.{field}"):
                    name = name.replace(f"{prefix}.{field}", f"{prefix}.embeddings.{field}")
        if name.endswith(".position_embedding"):
            name += ".weight"
        if name in result:
            raise ValueError("Duplicate converted CLIP parameter")
        result[name] = value
    return result


def config_from_json(config):
    def subset(cls, values):
        return cls(**{f.name: values[f.name] for f in fields(cls)})

    return CLIPConfig(
        subset(CLIPTextConfig, config["text_config"]),
        subset(CLIPVisionConfig, config["vision_config"]),
        config["projection_dim"],
    )


def pixels(path):
    with Image.open(path) as source:
        image = source.convert("RGB")
    # Preserve the whole screenshot; no task-specific crop or region detector.
    image = ImageOps.pad(image, (224, 224), method=Image.Resampling.BICUBIC, color=(0, 0, 0))
    array = np.asarray(image, dtype=np.float32) / 255
    array = (array - np.array([0.48145466, 0.4578275, 0.40821073], np.float32)) / np.array(
        [0.26862954, 0.26130258, 0.27577711], np.float32
    )
    return mx.array(array[None])


class FrozenCLIP:
    def __init__(self):
        from pathlib import Path

        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(CLIP_ID, revision=CLIP_REVISION, local_files_only=True))
        raw = json.loads((path / "config.json").read_text())
        raw["vision_config"].setdefault("num_channels", 3)
        self.config = config_from_json(raw)
        self.model = CLIPModel(self.config)
        weights = mx.load(str(path / "weights.npz"))
        # This historical MLX export already has NHWC convolution weights.
        self.model.load_weights(list(legacy_weights(weights).items()), strict=True)
        self.model.eval()
        self.model.freeze()
        mx.eval(self.model.parameters())
        self.tokenizer = CLIPTokenizer.from_pretrained(path)
        self.metadata = {
            "model": CLIP_ID,
            "revision": CLIP_REVISION,
            "config": asdict(self.config),
            "preprocess": "224px bicubic letterbox, CLIP normalization",
        }

    def image(self, path):
        result = normalized(self.model.get_image_features(pixels(path)))
        mx.eval(result)
        return result[0]

    def text(self, text):
        ids = self.tokenizer(text)
        if len(ids) > self.config.text_config.max_position_embeddings:
            raise ValueError("Reference caption exceeds CLIP context; shorten it explicitly")
        result = normalized(self.model.get_text_features(ids[None]))
        mx.eval(result)
        return result[0]
