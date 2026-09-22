# Scaling results

The learned stitch now performs basic visual movement and factual tasks in a
procedural environment. This is not evidence that it plays WoW, Hordes or unseen
games. All final results below use **the same checkpoint**,
`artifacts/scaled-robust-001/bundle`.

Machine-readable summary: [scaling-001.json](scaling-001.json).
Reproduction and API: [SCALING.md](SCALING.md).

## Model and training

- Frozen Qwen3.5 vision encoder and visual merger; no Qwen language decoder.
- Learned 4×4 spatial connector aligned to Laya embedding sequences.
- Laya with rank-8 LoRA on the final two attention layers.
- Parallel button, pointer, mouse and duration heads, plus Laya choice scores.
- One safetensors checkpoint, with tokenizer/processor/config files.
- 6,054,416 trainable parameters out of 760,862,486 tensor elements (~0.796%).
  SHA-256 checks verified original Qwen and Laya weights were unchanged.
- 512 unique training screenshots, 12,288 final-stage examples including goal
  and wording variants. The final lineage is 8,000 canonical training steps plus
  4,000 steps with paraphrases and shuffled choices. Optimizer state resets
  between stages. Earlier failed runs are not part of that lineage.

Training uses procedural action labels and descriptions, not Qwen-generated
teacher labels. Descriptions provide an auxiliary embedding-alignment loss and
initialization using training data only. They never enter the inference prompt.
No detector, reference lookup or fallback controller selects deployed actions.
Reusing Qwen vision does not transfer all of Qwen's language reasoning.

## Fresh image tests

The final evaluation generator used seed **2083**, after the checkpoint was
trained. Each split contains 64 distinct scenes with two opposite target goals
per scene. Image hashes and episode separation were checked against the complete
training manifest. The generator and task families are shared with training;
these are distribution-specific synthetic tests, not external benchmarks.

| Test | Standard choices | Zero vision | Shuffled images | Reversed choices |
|---|---:|---:|---:|---:|
| Fresh familiar-composition scenes | 128/128 | 50.0% | 50.0% | 90.6% |
| Withheld color/shape combinations | 128/128 | 50.0% | 51.6% | 89.1% |
| Withheld visual style | 128/128 | 50.0% | 53.1% | 85.9% |

The two excluded combinations are red triangles and blue squares. Of the 128
composition questions, 66 directly target an excluded combination; all 66 were
correct. The other questions target the companion object in the same scene.
Both goals were correct for all 192 scenes under standard choice ordering.
The learned button head also matched all standard direction labels.

Perfect finite-sample scores do not establish perfect population accuracy.
The recorded empirical bootstrap intervals collapse to 100% because every
sampled scene succeeded; they must not be interpreted as certainty.

## Goals and wording

Each row below uses 64 fresh scenes. Move-away, color and shape are trained task
families, so their ordinary scores are not zero-shot task transfer.

| Question family | Ordinary wording | Alternative test wording |
|---|---:|---:|
| Move away from named object | 57/64 (89.1%) | 42/64 (65.6%) |
| Color of leftmost object | 64/64 | 64/64 |
| Shape of leftmost object | 64/64 | 64/64 |

The alternative movement goal uses “Retreat from…”, which was not in the
augmentation phrases. Its weak result exposes limited instruction robustness.
Laya receiving the exact textual scene description scored 41/64 on that same
wording. That diagnostic suggests a language/instruction limitation as well as
possible visual errors; it does not prove their relative contributions.

## Closed-loop sandbox

The model completed **40/40 target-approach episodes**, seed **9406**. The best
constant direction succeeded in 60%. Mean episode length was 3.425 decisions.
The centered player sees a scrolling scene with two colored shapes; A/D outputs
move it left/right. The environment ends the episode within a fixed target
distance or after ten steps. The model has not learned when to stop.

Pixels, goal, control descriptions and previous-action text are the only model
inputs. Simulator coordinates are used only for rendering, physics and scoring.
The model's button head controls movement; the environment does not correct its
direction. This is a short one-dimensional task without obstacles or combat.

On an Apple M3 Max with 48 GiB unified memory, warm screenshot-processing plus
model inference was **57.5 ms median / 68.0 ms p95** for variable 224–320px input
images. An earlier export/reload benchmark of this same checkpoint measured
74.5/87.5 ms; timings depend on inputs and machine state. Neither includes screen
capture, desktop event posting or game response. The GIF preview is an animation
of sampled frames, not a real-time gameplay recording.

Local evidence:

```text
artifacts/scaled-robust-001/
  report.json
  holdouts-fresh.json
  questions-fresh.json
  paraphrases-fresh.json
  api-smoke.json
  sandbox-fresh/report.json
  sandbox-fresh/preview.gif
```

The persistent local HTTP API was also tested with a fresh screenshot and goal.
All four requests returned the correct direction and D button; the three warmed
HTTP round trips took approximately 51 ms each. This is a smoke check, not a
separate accuracy or load benchmark. No OS inputs were sent.

## What made the difference

The original action-only connector stayed at chance even with more steps and
spatial pooling. Description alignment and training-description initialization
enabled grounding. Mixing panned views improved short closed-loop movement.
Simply adding objectives and more steps then overfit; inconsistent description
ordering was an unhelpful alignment target.

The successful fresh run used consistent left-to-right descriptions and explicit
leftmost questions. Paraphrases and choice-order augmentation improved robustness,
although the tests above show remaining weaknesses. These were sequential
development experiments, not isolated, controlled ablations; their individual
causal effects are not established. Development seeds were replaced for final
evaluation, and earlier artifacts are retained locally.

## What scaling still requires

The implementation now supports mixed manifests, bounded disk feature caching,
small-parameter training, reproducible holdouts and a persistent inference API.
Training remains batch-size-one on one Mac; this is not distributed training.

The next capability bottleneck is data coverage: varied real screenshots paired
with goals, grounded descriptions and synchronized expert controls. Hold out
entire games and sessions, test camera/mouse and temporal decisions, and compare
closed-loop success against simple baselines. More repetitions of colored-shape
training cannot establish general gameplay. Mouse/pointer outputs, arbitrary UI
understanding, combat, loot and real-game transfer remain unvalidated.
