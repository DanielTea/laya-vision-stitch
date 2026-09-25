"""Jev-Omni decision classifier in MLX (Gemma 4 12B with an option-scoring head).

Port of the reference CUDA loader in `akhilaaa3/Jev-Omni` (Apache-2.0). One forward pass
of the merged multimodal Gemma 4 model reads an optional image and a prompt listing the
options; a linear head on the final-norm hidden state of the last token scores up to 256
options. Nothing is generated. Only `unified/` and the head are needed; the reference
loader's FP32 backbone and separate base-model download are not.
"""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

REPO = "akhilaaa3/Jev-Omni"
REVISION = "c050d51354147985d13286cf4acf90f562f2c631"
FILES = ["unified/*", "head.pt", "decision_config.json", "verification.json"]


def prompt(state, question, options):
    """The reference prompt, verbatim."""
    choices = "\n".join(f"{i + 1}. {value}" for i, value in enumerate(options))
    return (
        f"{state}\n\n---\n\nQUESTION: {question}\n\nOPTIONS:\n{choices}\n\n"
        f"Reply with only the number of the correct option (1-{len(options)}).\n"
        "Output a single number and nothing else."
    )


class JevOmni:
    def __init__(self, path=None, bits=None):
        import torch
        from huggingface_hub import snapshot_download
        from mlx_vlm import load

        path = Path(path or snapshot_download(REPO, revision=REVISION, allow_patterns=FILES))
        self.model, self.processor = load(str(path / "unified"))
        if bits:
            # Quantize the text decoder only; vision and the head stay unquantized.
            nn.quantize(self.model.language_model, group_size=64, bits=bits)
        head = torch.load(path / "head.pt", map_location="cpu", weights_only=True)
        self.mu = mx.array(head["mu"].float().numpy())
        self.sd = mx.array(head["sd"].float().numpy())
        self.weight = mx.array(head["linear.weight"].float().numpy())
        self.bias = mx.array(head["linear.bias"].float().numpy())
        mx.eval(self.model.parameters())

    def inputs(self, text, image=None):
        from mlx_vlm.utils import prepare_inputs

        content = ([{"type": "image"}] if image is not None else []) + [
            {"type": "text", "text": text}
        ]
        rendered = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        images = None if image is None else [image.convert("RGB")]
        return prepare_inputs(self.processor, images=images, prompts=[rendered])

    def scores(self, state, question, options, image=None):
        """Option logits (before softmax) for one question."""
        if not 2 <= len(options) <= 256:
            raise ValueError("Jev-Omni needs 2-256 options")
        inputs = dict(self.inputs(prompt(state, question, options), image))
        input_ids = inputs.pop("input_ids")
        pixel_values = inputs.pop("pixel_values", None)
        inputs.pop("attention_mask", None)
        embedded = self.model.get_input_embeddings(
            input_ids=input_ids, pixel_values=pixel_values, **inputs
        )
        types = inputs.get("mm_token_type_ids", inputs.get("token_type_ids"))
        hidden = self.model.language_model.model(
            None,
            inputs_embeds=embedded.inputs_embeds,
            per_layer_inputs=embedded.per_layer_inputs,
            mm_token_type_ids=types,
            logits_to_keep=1,
        )
        last = hidden[:, -1].astype(mx.float32)
        logits = ((last - self.mu) / self.sd) @ self.weight.T + self.bias
        return logits[0, : len(options)]

    def predict(self, state, question, options, image=None):
        """{prediction, prediction_index, confidence, probabilities}, like the reference."""
        probs = mx.softmax(self.scores(state, question, options, image), axis=-1)
        values = np.asarray(probs).tolist()
        best = int(np.argmax(values))
        return {
            "prediction": options[best],
            "prediction_index": best,
            "confidence": values[best],
            "probabilities": dict(zip(options, values, strict=True)),
        }
