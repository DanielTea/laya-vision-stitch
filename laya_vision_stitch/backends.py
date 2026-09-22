"""Local frozen model interfaces. No capture, keyboard or mouse APIs."""

import json
import time

import numpy as np
from PIL import Image

LAYA_ID = "aac6fef/laya-multilingual-coreml-ane"
LAYA_REVISION = "39d6a9b3d0f67f06da74fbade6121ea134cbdb21"
QWEN_ID = "mlx-community/Qwen3.5-4B-4bit"
QWEN_REVISION = "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c"
CHOICES = {
    "A": "Wait if weapon is cooling down.",
    "B": "Select monster if target is unknown.",
    "C": "Approach if range warning is yes.",
    "F": "Retreat if health is low.",
    "G": "Attack if monster alive and weapon ready.",
}

QUESTION = {
    "action": {"type": "choice", "instructions": "Choose next game action.", "criteria": CHOICES}
}


class LayaEmbeddings:
    def __init__(self):
        import laya_coreml
        from laya_coreml.inputs import collate_items

        self.agent = laya_coreml.load(LAYA_ID, revision=LAYA_REVISION, local_files_only=True)
        item = self.agent.prepare("", QUESTION)[0][0]
        self.prefix = item["ids"][:-1]
        self.start = len(self.prefix)
        self.slots = 19
        if self.start + self.slots + 1 > self.agent.shape["max_length"]:
            raise ValueError("Not enough visual-state slots in pinned ANE export")
        item["ids"] = (
            self.prefix + [self.agent.tok.pad_token_id] * self.slots + [self.agent.tok.sep_token_id]
        )
        self.batch = collate_items([item], self.agent.tok.pad_token_id, shape=self.agent.shape)
        self.inputs = self.agent.model_inputs(self.batch)
        self.width = self.agent.embedding.shape[1]
        self.metadata = {
            "model": LAYA_ID,
            "revision": LAYA_REVISION,
            "source_weights_sha256": self.agent.manifest["source_weights_sha256"],
            "state_slots": self.slots,
            "width": self.width,
            "prefix_tokens": self.start,
            "choices": CHOICES,
        }

    def state_ids(self, text):
        ids = self.agent.tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) != self.slots:
            raise ValueError(f"State has {len(ids)} tokens; budget {self.slots}")
        return ids

    def encode(self, text):
        return self.agent.embedding[self.state_ids(text)].astype(np.float32)

    def predict(self, embeddings):
        start = time.perf_counter()
        state = np.asarray(embeddings, np.float32).reshape(self.slots, self.width)
        if not np.isfinite(state).all() or np.max(np.abs(state)) > 65000:
            raise ValueError("Non-finite or FP16-overflowing state embeddings")
        inputs = dict(self.inputs)
        inputs["embeddings"] = self.inputs["embeddings"].copy()
        inputs["embeddings"][0, :, 0, self.start : self.start + self.slots] = state.T.astype(
            np.float16
        )
        out = self.agent.model.predict(inputs)
        logits = next(v for v in out.values() if v.shape[1] == 1).reshape(-1)[: len(CHOICES)]
        if not np.isfinite(logits).all():
            raise ValueError("Non-finite decision logits")
        return logits.astype(np.float32), (time.perf_counter() - start) * 1000

    def parity(self, text):
        batch = {k: v.copy() for k, v in self.batch.items()}
        batch["input_ids"][0, self.start : self.start + self.slots] = self.state_ids(text)
        expected, _ = self.agent.forward(batch)
        actual, _ = self.predict(self.encode(text))
        error = float(np.max(np.abs(expected[0, : len(CHOICES)] - actual)))
        if error > 0.01:
            raise RuntimeError(f"Embedding injection parity failed: {error}")
        return error

    def ordinary_choice(self, text):
        return list(CHOICES).index(
            self.agent.predict(text, QUESTION)["answers"]["action"]["choice"]
        )


class QwenVision:
    def __init__(self, width=512):
        import mlx.core as mx
        from huggingface_hub import snapshot_download
        from mlx_vlm import load

        self.mx = mx
        self.width = width
        local = snapshot_download(QWEN_ID, revision=QWEN_REVISION, local_files_only=True)
        self.model, self.processor = load(local, lazy=True)
        self.model.eval()
        mx.eval(self.model.vision_tower.parameters())
        self.metadata = {
            "model": QWEN_ID,
            "revision": QWEN_REVISION,
            "max_image_width": width,
            "features": "merged visual tokens: global mean plus 2x2 spatial means",
        }

    def inputs(self, path, prompt="Describe the game screenshot."):
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import prepare_inputs

        with Image.open(path) as source:
            image = source.convert("RGB")
        if image.width > self.width:
            image = image.resize((self.width, round(image.height * self.width / image.width)))
        formatted = apply_chat_template(
            self.processor, self.model.config, prompt, num_images=1, enable_thinking=False
        )
        return prepare_inputs(
            self.processor,
            images=[image],
            prompts=formatted,
            image_token_index=self.model.config.image_token_index,
        )

    def encode(self, path):
        start = time.perf_counter()
        data = self.inputs(path)
        tower = self.model.vision_tower
        visual, _ = tower(
            data["pixel_values"].astype(tower.patch_embed.proj.weight.dtype), data["image_grid_thw"]
        )
        self.mx.eval(visual)
        grid = np.asarray(data["image_grid_thw"])[0]
        t, h, w = map(int, grid)
        merge = tower.spatial_merge_size
        if t != 1 or h // merge < 2 or w // merge < 2:
            raise ValueError("Expected a single image with at least 2x2 merged patches")
        features = np.asarray(visual.astype(self.mx.float32)).reshape(h // merge, w // merge, -1)
        pools = [features.mean((0, 1))]
        for ys in np.array_split(features, 2, axis=0):
            for region in np.array_split(ys, 2, axis=1):
                pools.append(region.mean((0, 1)))
        return np.concatenate(pools), (time.perf_counter() - start) * 1000

    def choose(self, path):
        """One multimodal prefill; score existing single-token action labels."""
        start = time.perf_counter()
        prompt = (
            "Choose the next game action from this screenshot. "
            "Attack only a selected living monster when the weapon is ready and in range. "
            "Retreat at low health. Wait when uncertain. "
            + " ".join(f"{k}={v}." for k, v in CHOICES.items())
            + " Answer with exactly one capital letter."
        )
        data = self.inputs(path, prompt)
        label_ids = [self.processor.tokenizer.encode(k, add_special_tokens=False) for k in CHOICES]
        if any(len(ids) != 1 for ids in label_ids):
            raise ValueError("Action labels must each be a single tokenizer token")
        output = self.model(data.pop("input_ids"), mask=data.pop("attention_mask", None), **data)
        logits = output.logits[0, -1]
        selected = logits[self.mx.array([ids[0] for ids in label_ids])]
        self.mx.eval(selected)
        return int(self.mx.argmax(selected).item()), (time.perf_counter() - start) * 1000


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
