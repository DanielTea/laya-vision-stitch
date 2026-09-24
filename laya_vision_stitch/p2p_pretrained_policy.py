"""Native MLX Open-P2P-150M policy with its pretrained temporal/action weights.

This is a conversion baseline, not yet a Laya stitch or qualified live agent.
State is explicit; no frame/action lookup bank or hand-written gameplay policy.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from .p2p_pretrained_vision import OpenP2PVision

STEP_TOKENS = 12
KEY_NAMES = (
    None,
    "space",
    "1",
    "2",
    "3",
    "4",
    "a",
    "d",
    "e",
    "f",
    "q",
    "w",
    "s",
    "z",
    "down",
    "up",
    "left",
    "right",
    "shift",
    "shift",
)
MOUSE_NAMES = (None, "mouse_left", "mouse_right", "mouse_middle")
MOUSE_X = (
    -501,
    -322,
    -110,
    -61,
    -38,
    -24,
    -15,
    -9,
    -5,
    -2,
    -1,
    0,
    1,
    2,
    5,
    9,
    15,
    24,
    38,
    61,
    110,
    322,
    501,
)
MOUSE_Y = (-151, -87, -18, -10, -6, -4, -2, -1, 0, 1, 2, 4, 6, 10, 18, 87, 151)


def policy_mask(query_positions, key_positions):
    """Released mask: image/text/thinking share a frame; action-out sees no targets."""
    q, k = query_positions[:, None], key_positions[None, :]
    qr, kr, qs, ks = q % STEP_TOKENS, k % STEP_TOKENS, q // STEP_TOKENS, k // STEP_TOKENS
    past = (ks < qs) & (kr != 3) & (qs - ks <= 200)
    same = (qs == ks) & (((qr < 3) & (kr < 3)) | ((qr == 3) & (kr <= 3)) | ((qr > 3) & (kr != 3)))
    return past | same


class RMS(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.scale = mx.ones(width)

    def __call__(self, x):
        xf = x.astype(mx.float32)
        return (xf * mx.rsqrt((xf * xf).mean(-1, keepdims=True) + 1e-5)).astype(
            x.dtype
        ) * self.scale


class Attention(nn.Module):
    def __init__(self, width, heads, sinks=0):
        super().__init__()
        self.heads, self.sinks = heads, sinks
        self.qkv, self.output = (
            nn.Linear(width, width * 3, bias=False),
            nn.Linear(width, width, bias=False),
        )
        self.q_norm, self.k_norm = RMS(width), RMS(width)
        if sinks:
            self.k_sinks, self.v_sinks = (
                mx.zeros((1, heads, sinks, width // heads)),
                mx.zeros((1, heads, sinks, width // heads)),
            )

    def __call__(self, x, offset, cache, mask):
        b, n, d = x.shape
        q, k, v = mx.split(self.qkv(x), 3, -1)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = [
            a.reshape(b, n, self.heads, d // self.heads).transpose(0, 2, 1, 3) for a in (q, k, v)
        ]
        q, k = [
            mx.fast.rope(a, d // self.heads, traditional=True, base=10000, scale=1.0, offset=offset)
            for a in (q, k)
        ]
        if cache is not None:
            k, v = mx.concatenate([cache[0], k], 2), mx.concatenate([cache[1], v], 2)
        new_cache = (k, v)
        if self.sinks:
            sink_shape = (b, self.heads, self.sinks, d // self.heads)
            k, v = (
                mx.concatenate([mx.broadcast_to(self.k_sinks, sink_shape), k], 2),
                mx.concatenate([mx.broadcast_to(self.v_sinks, sink_shape), v], 2),
            )
            if mask is not None:
                mask = mx.concatenate([mx.ones((n, self.sinks), mx.bool_), mask], -1)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=(d // self.heads) ** -0.5, mask=mask
        )
        return self.output(out.transpose(0, 2, 1, 3).reshape(b, n, d)), new_cache


class Layer(nn.Module):
    def __init__(self, width=1024, heads=16, sinks=0):
        super().__init__()
        hidden = ((int(2 * (4 * width) / 3) + 7) // 8) * 8
        self.attention = Attention(width, heads, sinks)
        self.attention_norm, self.ffn_norm = RMS(width), RMS(width)
        self.w13, self.w2 = (
            nn.Linear(width, 2 * hidden, bias=False),
            nn.Linear(hidden, width, bias=False),
        )

    def __call__(self, x, offset, cache, mask):
        out, cache = self.attention(self.attention_norm(x), offset, cache, mask)
        x = x + out
        a, b = mx.split(self.w13(self.ffn_norm(x)), 2, -1)
        return x + self.w2(nn.silu(a) * b), cache


class Stack(nn.Module):
    def __init__(self, depth, heads, sinks=0):
        super().__init__()
        self.layers = [Layer(heads=heads, sinks=sinks) for _ in range(depth)]

    def __call__(self, x, offset=0, caches=None, mask=None):
        updated = []
        for i, layer in enumerate(self.layers):
            x, cache = layer(x, offset, None if caches is None else caches[i], mask)
            updated.append(cache)
        return x, updated


class OpenP2PPolicy(nn.Module):
    def __init__(self, vision=None, depth=10):
        super().__init__()
        self.vision = vision if vision is not None else OpenP2PVision()
        # Released 150M/300M checkpoints differ only in policy depth (10/20 at width 1024).
        self.policy = Stack(depth, 16)
        self.decoder = Stack(3, 8, sinks=1)
        self.text_projection = nn.Linear(768, 1024, bias=False)
        self.decoder_projection = nn.Linear(1024, 1024)
        self.text_position, self.no_text = mx.zeros((1, 1, 1024)), mx.zeros((1, 1, 1024))
        self.image_position, self.thinking, self.action_start = [
            mx.zeros((1, 1, 1024)) for _ in range(3)
        ]
        self.action_position, self.decoder_position = mx.zeros((1, 8, 1024)), mx.zeros((9, 1024))
        self.embeddings = [nn.Embedding(n, 1024) for n in (20, 4, 23, 17)]
        self.outputs = [nn.Linear(1024, n) for n in (20, 4, 23, 17)]
        self.eval()

    @staticmethod
    def action_type(index):
        return 0 if index < 4 else 1 if index < 6 else index - 4

    def action_embeddings(self, tokens):
        return mx.stack([self.embeddings[self.action_type(i)](tokens[:, i]) for i in range(8)], 1)

    def prefix(self, image_token, text=None, spatial=None, target=None):
        batch = image_token.shape[0]
        if image_token.ndim != 2 or image_token.shape[1] != 1024:
            raise ValueError("Expected a batch of 1024D image tokens")
        if hasattr(self, "visual_adapter"):
            image_token = self.visual_adapter(image_token)
        if hasattr(self, "spatial_adapter"):
            if spatial is None or text is None:
                raise ValueError("Spatial checkpoint requires spatial features and a goal")
            image_token = self.spatial_adapter(spatial, text, image_token)
        language = (
            mx.broadcast_to(self.no_text, (batch, 1, 1024))
            if text is None
            else self.text_projection(text.reshape(batch, 1, 768))
        )
        language = mx.where(mx.any(language != 0, axis=-1, keepdims=True), language, self.no_text)
        thinking = mx.broadcast_to(self.thinking, (batch, 1, 1024))
        if target is not None and hasattr(self, "target_encoder"):
            # Optional screen point to act on (NaN rows: none), on the constant thinking slot.
            thinking = thinking + self.target_encoder(target.reshape(batch, 2))[:, None].astype(
                thinking.dtype
            )
        return mx.concatenate(
            [
                language + self.text_position,
                image_token[:, None] + self.image_position,
                thinking,
                mx.broadcast_to(self.action_start, (batch, 1, 1024)),
            ],
            1,
        ).astype(self.text_projection.weight.dtype)

    def context(self, prefix, actions=None, caches=None, position=0):
        if position < 0 or position % STEP_TOKENS:
            raise ValueError("Policy position must start a frame")
        depth = len(getattr(self.policy, "layers", caches or ()))
        if caches is not None and len(caches) != depth:
            raise ValueError("Expected one cache per policy layer")
        previous = 0 if caches is None else caches[0][0].shape[2]
        if previous % STEP_TOKENS or previous > 200 * STEP_TOKENS:
            raise ValueError("Cache must contain at most 200 complete frames")
        if previous > position:
            raise ValueError("Cache cannot extend before the start of the episode")
        if caches is not None and any(
            k.ndim != 4 or k.shape != v.shape or k.shape[2] != previous for k, v in caches
        ):
            raise ValueError("Policy cache layers disagree")
        if previous == 200 * STEP_TOKENS:
            caches = [(k[:, :, STEP_TOKENS:], v[:, :, STEP_TOKENS:]) for k, v in caches]
            previous -= STEP_TOKENS
        target = (
            mx.zeros((prefix.shape[0], 8, 1024), prefix.dtype)
            if actions is None
            else self.action_embeddings(actions) + self.action_position
        )
        x = mx.concatenate([prefix, target], 1)
        qpos, kpos = (
            mx.arange(position, position + STEP_TOKENS),
            mx.arange(position - previous, position + STEP_TOKENS),
        )
        output, updated = self.policy(x, position, caches, policy_mask(qpos, kpos))
        return output[:, 3:4], updated

    def decode(self, context, forced=None, temperature=0.0):
        """Greedy or categorical model sampling; forced tokens enable parity audits."""
        if temperature < 0:
            raise ValueError("Temperature must be nonnegative")
        x = self.decoder_projection(context) + self.decoder_position[None, 0:1]
        caches, tokens, logits = None, [], []
        for i in range(8):
            h, caches = self.decoder(x, i, caches)
            kind = self.action_type(i)
            scores = self.outputs[kind](h[:, 0])
            token = (
                forced[:, i]
                if forced is not None
                else mx.argmax(scores, -1)
                if temperature == 0
                else mx.random.categorical(scores / temperature)
            )
            logits.append(scores)
            tokens.append(token)
            x = self.embeddings[kind](token)[:, None] + self.decoder_position[None, i + 1 : i + 2]
        return mx.stack(tokens, 1), tuple(logits)

    def teacher_logits(self, context, tokens):
        """Parallel causal decoding for training; targets are shifted by one token."""
        start = self.decoder_projection(context)
        previous = self.action_embeddings(tokens)[:, :7]
        x = mx.concatenate([start, previous], 1) + self.decoder_position[None, :8]
        mask = mx.arange(8)[:, None] >= mx.arange(8)[None, :]
        h, _ = self.decoder(x, mask=mask)
        return tuple(self.outputs[self.action_type(i)](h[:, i]) for i in range(8))

    def step(self, pixels, text=None, caches=None, position=0, forced=None, temperature=0.0):
        spatial, image = self.vision(pixels)
        prefix = self.prefix(image, text, spatial)
        context, _ = self.context(prefix, caches=caches, position=position)
        tokens, logits = self.decode(context, forced=forced, temperature=temperature)
        _, updated = self.context(prefix, tokens, caches, position)
        return tokens, logits, updated, context

    @classmethod
    def from_state(cls, state, dtype=mx.float32):
        vision = OpenP2PVision.from_state(state, mx.float32)
        prefix = "bc_transformer._transformer.transformer_layers."
        depth = len({k.removeprefix(prefix).split(".")[0] for k in state if k.startswith(prefix)})
        model, mapped, used = cls(vision, depth=depth), [], set()

        def copy(target, source):
            used.add(source)
            mapped.append((target, mx.array(state[source]).astype(dtype)))

        for ours, theirs in (
            ("text_projection", "bc_transformer.text_embed_mlp"),
            ("decoder_projection", "bc_transformer.action_decoder.input_proj"),
        ):
            copy(ours + ".weight", theirs + ".weight")
            if ours == "decoder_projection":
                copy(ours + ".bias", theirs + ".bias")
        for ours, theirs in (
            ("text_position", "text_pos_tokens"),
            ("no_text", "text_embedding_for_no_text_input"),
            ("image_position", "img_pos_tokens"),
            ("thinking", "thinking_pos_tokens"),
            ("action_start", "action_out_token"),
            ("action_position", "action_pos_tokens"),
            ("decoder_position", "action_decoder.pos_tokens"),
        ):
            copy(ours, "bc_transformer." + theirs)
        for stack, source, depth, sinks in (
            ("policy", "bc_transformer._transformer", depth, False),
            ("decoder", "bc_transformer.action_decoder", 3, True),
        ):
            for i in range(depth):
                target, origin = f"{stack}.layers.{i}", f"{source}.transformer_layers.{i}"
                for ours, theirs in (
                    ("attention.qkv.weight", "self_attention.c_attn.weight"),
                    ("attention.output.weight", "self_attention.c_proj.weight"),
                    ("attention.q_norm.scale", "self_attention.q_norm.scale"),
                    ("attention.k_norm.scale", "self_attention.k_norm.scale"),
                    ("attention_norm.scale", "self_attention_norm.scale"),
                    ("ffn_norm.scale", "ffn_norm.scale"),
                    ("w13.weight", "ffn.w13.weight"),
                    ("w2.weight", "ffn.w2.weight"),
                ):
                    copy(target + "." + ours, origin + "." + theirs)
                if sinks:
                    for name in ("k_sinks", "v_sinks"):
                        copy(f"{target}.attention.{name}", f"{origin}.self_attention.{name}")
        for i, name in enumerate(("key_action", "mouse_button", "mouse_delta_x", "mouse_delta_y")):
            copy(f"embeddings.{i}.weight", name + "_embedding.weight")
            output = "keyboard" if i == 0 else name
            for field in ("weight", "bias"):
                copy(f"outputs.{i}.{field}", f"{output}_out_logits.{field}")
        for name, value in tree_flatten(vision.parameters()):
            mapped.append(("vision." + name, value))
        # The upstream module registers the same image tokenizer twice.
        for key in state:
            if key.startswith("bc_transformer.image_tokenizer."):
                original = key.removeprefix("bc_transformer.")
                if original not in state or not np.array_equal(state[key], state[original]):
                    raise ValueError("Shared image tokenizer aliases differ")
                used.add(key)
            if key.startswith("image_tokenizer."):
                used.add(key)
        if set(state) != used:
            raise ValueError(f"Unmapped policy weights: {sorted(set(state) - used)}")
        model.load_weights(mapped, strict=True)
        model.freeze()
        mx.eval(model.parameters())
        return model


def physical_action(tokens, key_names=KEY_NAMES):
    tokens = np.asarray(tokens).reshape(-1).tolist()
    if len(tokens) != 8 or any(
        not isinstance(t, int) or not 0 <= t < n
        for t, n in zip(tokens, (len(key_names),) * 4 + (4, 4, 23, 17), strict=True)
    ):
        raise ValueError("Invalid pretrained action tokens")
    buttons = [key_names[t] for t in tokens[:4]] + [MOUSE_NAMES[t] for t in tokens[4:6]]
    return {
        "buttons": sorted({b for b in buttons if b is not None}),
        "mouse_delta": [MOUSE_X[tokens[6]] / 512, MOUSE_Y[tokens[7]] / 512],
        "duration_seconds": 0.05,
        "sampling": "categorical tokens; mouse bin centers",
    }
