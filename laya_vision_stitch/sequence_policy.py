"""Teacher-forced parallel sequence forward for the P2P policy; training/evaluation only.

The parallel form equals streaming `OpenP2PPolicy.context` when each recorded action is
committed after its frame. Nothing here adds gameplay logic to inference.
"""

import mlx.core as mx

from .p2p_pretrained_policy import STEP_TOKENS, policy_mask


def sequence_prefix(
    policy, images, goals=None, spatial=None, residual=None, language=None, target=None
):
    """images [B,T,1024]; goals [B,T,768] (zero rows mean no goal) -> [B,T,4,1024].

    `residual` adds to the image token and `language` to the goal token; both are
    optional learned inputs (for example elapsed-time or slow-path intent embeddings).
    `target` [B,T,2] is a screen point to act on (NaN rows: none), for a target encoder.
    """
    b, t = images.shape[:2]
    flat = images.reshape(b * t, -1)
    if residual is not None:
        flat = flat + residual.reshape(b * t, -1).astype(flat.dtype)
    g = None if goals is None else goals.reshape(b * t, 768)
    s = None if spatial is None else spatial.reshape(b * t, *spatial.shape[2:])
    if target is None:
        prefix = policy.prefix(flat, g, s)
    else:
        prefix = policy.prefix(flat, g, s, target.reshape(b * t, 2))
    prefix = prefix.reshape(b, t, 4, 1024)
    if language is not None:
        extra = mx.zeros_like(prefix)
        extra = mx.concatenate(
            [language.reshape(b, t, 1, 1024).astype(prefix.dtype), extra[:, :, 1:]], 2
        )
        prefix = prefix + extra
    return prefix


def sequence_hidden(policy, prefix, tokens):
    """prefix [B,T,4,1024], tokens [B,T,8] recorded actions -> policy outputs [B,T,12,1024]."""
    b, t = prefix.shape[:2]
    if tokens.shape != (b, t, 8):
        raise ValueError("Expected one 8-token action per frame")
    if t > 200:
        raise ValueError("Sequences longer than the 200-frame memory need streaming")
    target = policy.action_embeddings(tokens.reshape(b * t, 8)).reshape(b, t, 8, 1024)
    target = target + policy.action_position
    x = mx.concatenate([prefix, target.astype(prefix.dtype)], 2).reshape(b, t * STEP_TOKENS, 1024)
    positions = mx.arange(t * STEP_TOKENS)
    out, _ = policy.policy(x, 0, None, policy_mask(positions, positions))
    return out.reshape(b, t, STEP_TOKENS, 1024)


def sequence_contexts(policy, images, goals, tokens, **inputs):
    """Action-out contexts [B,T,1024]; frame t sees actions up to t-1 only."""
    prefix = sequence_prefix(policy, images, goals, **inputs)
    return sequence_hidden(policy, prefix, tokens)[:, :, 3]


def guided_decode(policy, context, unconditional=None, scale=1.0, temperature=0.0, key=None):
    """Autoregressive decoding with optional classifier-free guidance on the goal.

    logits = uncond + scale * (cond - uncond), applied at every decoder position with the
    same previously chosen tokens. scale=1 or unconditional=None is ordinary decoding.
    """
    contexts = [context] if unconditional is None else [context, unconditional]
    xs = [
        policy.decoder_projection(c.reshape(-1, 1, 1024)) + policy.decoder_position[None, 0:1]
        for c in contexts
    ]
    caches = [None] * len(xs)
    tokens, logits = [], []
    for i in range(8):
        kind = policy.action_type(i)
        scores = []
        for j, x in enumerate(xs):
            h, caches[j] = policy.decoder(x, i, caches[j])
            scores.append(policy.outputs[kind](h[:, 0]).astype(mx.float32))
        score = scores[0] if len(scores) == 1 else scores[1] + scale * (scores[0] - scores[1])
        if temperature == 0:
            token = mx.argmax(score, -1)
        else:
            if key is not None:
                key, sub = mx.random.split(key)
                token = mx.random.categorical(score / temperature, key=sub)
            else:
                token = mx.random.categorical(score / temperature)
        tokens.append(token)
        logits.append(score)
        embedded = (
            policy.embeddings[kind](token)[:, None] + policy.decoder_position[None, i + 1 : i + 2]
        )
        xs = [embedded for _ in xs]
    return mx.stack(tokens, 1), tuple(logits)
