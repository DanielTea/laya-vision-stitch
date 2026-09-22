"""MLX port of the pretrained Open-P2P-150M image tokenizer, not a policy.

Original implementation of EfficientNet-B0 stages 0..5 and the learned spatial
projection. No ImageNet normalization: the released gameplay model uses RGB/255.
Full policy conversion and Laya goal alignment are separate, unvalidated work.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

P2P_ID = "guaguaa/open-p2p"
P2P_REVISION = "de18b62bc8f9722bda64497600c38c8f7634d86b"


class ConvNorm(nn.Module):
    def __init__(self, inputs, outputs, kernel, stride=1, groups=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(
            inputs, outputs, kernel, stride=stride, padding=kernel // 2, groups=groups, bias=False
        )
        self.norm = nn.BatchNorm(outputs, eps=1e-5)
        self.activation = activation

    def __call__(self, x):
        x = self.norm(self.conv(x))
        return nn.silu(x) if self.activation else x


class MBConv(nn.Module):
    def __init__(self, inputs, outputs, expansion, kernel, stride):
        super().__init__()
        expanded = inputs * expansion
        self.expand = ConvNorm(inputs, expanded, 1) if expansion != 1 else None
        self.depthwise = ConvNorm(expanded, expanded, kernel, stride, expanded)
        self.se_reduce = nn.Conv2d(expanded, inputs // 4, 1)
        self.se_expand = nn.Conv2d(inputs // 4, expanded, 1)
        self.project = ConvNorm(expanded, outputs, 1, activation=False)
        self.residual = stride == 1 and inputs == outputs

    def __call__(self, x):
        y = self.expand(x) if self.expand is not None else x
        y = self.depthwise(y)
        squeeze = y.mean((1, 2), keepdims=True)
        scale = mx.sigmoid(self.se_expand(nn.silu(self.se_reduce(squeeze))))
        y = self.project(y * scale)
        return x + y if self.residual else y


class OpenP2PVision(nn.Module):
    """Return both spatial features and the original single 1024D image token."""

    def __init__(self):
        super().__init__()
        self.stem = ConvNorm(3, 32, 3, 2)
        stages, inputs = [], 32
        for expansion, outputs, kernel, stride, repeats in (
            (1, 16, 3, 1, 1),
            (6, 24, 3, 2, 2),
            (6, 40, 5, 2, 2),
            (6, 80, 3, 2, 3),
            (6, 112, 5, 1, 3),
        ):
            stage = []
            for index in range(repeats):
                stage.append(
                    MBConv(inputs, outputs, expansion, kernel, stride if index == 0 else 1)
                )
                inputs = outputs
            stages.append(stage)
        self.stages = stages
        self.projection = nn.Linear(112 * 12 * 12, 1024)
        self.norm = nn.LayerNorm(1024, eps=1e-5)
        self.eval()

    def __call__(self, pixels):
        if pixels.ndim != 4 or pixels.shape[1:] != (192, 192, 3):
            raise ValueError("Released Open-P2P tokenizer requires 192×192 NHWC RGB")
        x = self.stem(pixels.astype(self.stem.conv.weight.dtype))
        for stage in self.stages:
            for block in stage:
                x = block(x)
        # Preserve the original NCHW flattening before the learned projection.
        token = self.norm(self.projection(x.transpose(0, 3, 1, 2).reshape(x.shape[0], -1)))
        return x, token

    @classmethod
    def from_state(cls, state, dtype=mx.float32):
        """Map a NumPy tensor dictionary from the restricted upstream checkpoint."""
        if dtype not in (mx.float32, mx.bfloat16):
            raise ValueError("Use FP32 or BF16; FP16 overflows the pretrained tokenizer")
        prefix = "image_tokenizer."
        mapped, used = [], set()

        def copy(target, source, conv=False):
            key = prefix + source
            array = state[key]
            used.add(key)
            if conv:
                array = array.transpose(0, 2, 3, 1)
            mapped.append((target, mx.array(array).astype(dtype)))

        def convnorm(target, source):
            copy(target + ".conv.weight", source + ".0.weight", True)
            for name in ("weight", "bias", "running_mean", "running_var"):
                copy(target + ".norm." + name, source + ".1." + name)
            used.add(prefix + source + ".1.num_batches_tracked")

        model = cls()
        convnorm("stem", "efficientnet_preprocess.0")
        for i, stage in enumerate(model.stages):
            for j, block in enumerate(stage):
                target, source = f"stages.{i}.{j}", f"efficientnet_preprocess.{i + 1}.{j}.block"
                depth = 1 if block.expand is not None else 0
                if depth:
                    convnorm(target + ".expand", source + ".0")
                convnorm(target + ".depthwise", f"{source}.{depth}")
                for ours, theirs in (("se_reduce", "fc1"), ("se_expand", "fc2")):
                    copy(f"{target}.{ours}.weight", f"{source}.{depth + 1}.{theirs}.weight", True)
                    copy(f"{target}.{ours}.bias", f"{source}.{depth + 1}.{theirs}.bias")
                convnorm(target + ".project", f"{source}.{depth + 2}")
        for target, source in (("projection", "mlp.0"), ("norm", "mlp.1")):
            for name in ("weight", "bias"):
                copy(f"{target}.{name}", f"{source}.{name}")
        unexpected = {k for k in state if k.startswith(prefix)} - used
        if unexpected:
            raise ValueError(f"Unexpected image tokenizer weights: {sorted(unexpected)}")
        model.load_weights(mapped, strict=True)
        model.freeze()
        mx.eval(model.parameters())
        return model


def preprocess(image):
    """Full-frame Hamming resize; parity with the upstream Rust resizer is untested.

    Numerical model parity compares identical input tensors. Pillow downsampling
    need not be byte-identical to fast_image_resize's interpolation mode.
    """
    return (
        np.asarray(
            image.convert("RGB").resize((192, 192), Image.Resampling.HAMMING), dtype=np.float32
        )[None]
        / 255
    )
