# Local latency and gameplay diagnosis

Measured on 2026-09-22 on the local Apple M3 Max (48 GiB). The checkpoint is
`artifacts/gameplay-buttons-002/bundle`. This is the strongest small training-set
fit, not a validated gameplay policy. No game inputs were sent.

## Latency

`scripts/profile_latency.py` measures 32 recorded validation examples after three
warmups. A second run added MLX compilation of the Laya/action subgraph. Each
variant processes the same examples in a fixed order; this is a diagnostic, not
a randomized sustained-load benchmark. Cached variants pretokenize the unchanged
prompt and reuse the older frame's already computed visual features. The current
frame is freshly preprocessed and encoded on every prediction. Its history age
and frame position remain identical to the original request.

| Variant | Median ms | p95 ms |
|---|---:|---:|
| Original, images loaded from disk | 109.05 | 114.33 |
| Both images already in memory | 105.79 | 109.05 |
| Reuse historical visual features and prompt tokens | 61.69 | 65.34 |
| Also omit unused choice output | 60.13 | 63.05 |
| Also compile Laya/action subgraph | 59.84 | 62.20 |

All evaluated output tensors matched the original exactly in these samples.
Omitting choices is valid here because actions read encoder states; checkpoints
using decision-head states cannot skip that head. MLX lazy evaluation eliminates
the unused choice computation when it is absent from evaluated outputs.

An independent first run measured 59.95 ms median / 61.30 ms p95 for the cache plus
action-only variant without compilation. Compilation's small difference in the
second run is insufficient evidence of a material improvement.

Stage timings with explicit synchronization fences put each vision encoding at
about 43–44 ms, connector at 1.2 ms and Laya/actions at 15 ms. Stage medians are not
additive guarantees. Vision dominates the remaining cost. The older 120.4 ms
baseline came from a separate 63-sample run under different runtime conditions;
use the within-run comparison above to assess the cache benefit.

These timings exclude capture, HTTP/base64 transport, action decoding and OS input
posting, model loading and compilation warmup. They do not establish performance
while actively rendering/playing Hordes. The cache models steady state only; the
first request still needs its entire history. Production caching must invalidate
on changed image/model/preprocessing and reconstruct age/order coordinates for
each request. Changed prompts must be retokenized.

**Conclusion:** approximately 60 ms offline inference is feasible with unchanged
weights, but a reliable sub-60 ms live median has not been demonstrated. Aim for
45–50 ms offline to create headroom, then measure sustained p50/p95 and frame age
with the game running. Reducing fresh-frame vision cost is the next performance
experiment: test lower precision or a smaller visual token budget with action
quality checks, and if necessary distill into a smaller visual encoder. Neither
Core ML nor quantization should be assumed faster without a matched benchmark.

## Why the agent does not play reliably

- The model uses Qwen's vision tower, not its language/reasoning decoder. A new
  connector does not automatically transfer the omitted decoder's capabilities.
- Current gameplay experiments use 64–190 clips from three other games, one
  training recording per game. They contain no Hordes demonstrations and only
  generic goals. Some buttons appear once. The best-fit checkpoint achieves
  95.3% training exact matches but only 14.1% on separate recording sessions.
- Two images without actual task intent or previous controls can admit several
  reasonable next actions. Recorded human controls are not necessarily optimal.
- Button-only losses do not establish camera movement, pointer targeting,
  attack outcomes, retreat, loot collection or recovery in a closed loop.

## Next capability experiment

Keep one inference graph: screenshots + goal + recent actions -> pretrained
vision -> learned temporal connector -> Laya with small adapters -> parallel
keyboard, camera and pointer outputs. Train the connector, adapters and heads;
retain the pretrained backbones. Supply previous controls and meaningful temporal
history during training and deployment consistently.

Collect reviewed, successful Hordes trajectories with explicit short goals:
select a monster, approach, attack, retreat, turn the camera, collect a drop and
recover after losing a target. Include unsuccessful attempts paired with expert
corrections. Existing controller traces are candidate material, not ground truth.
Use a larger Qwen teacher offline to propose goal/state annotations and action
distributions only where they can be validated; it adds no deployment latency.
Retain multi-game replay and held-out games to test whether the learned mapping
generalizes rather than only adapting to Hordes.

Separate training and evaluation by complete sessions and locations. Measure
actual episode outcomes on unseen sessions in addition to button prediction:
target acquisition, kills, survival, confirmed pickups and recovery. Advance from
offline validation to observation and bounded live trials only as evidence
supports it. Faster execution cannot repair an incorrect policy; more updates on
the present tiny ambiguous dataset are not the justified next experiment.

## Reproduce

From the repository root:

```sh
PYTHONPATH=. .venv/bin/python scripts/profile_latency.py
```

The script writes `artifacts/latency-profile-002/report.json`. The recorded result
is also stored as `docs/latency-profile-002.json`. The script changes no model
weights or deployed controller. MLX compilation behavior is documented in the
[official MLX documentation](https://ml-explore.github.io/mlx/build/html/usage/compile.html).
