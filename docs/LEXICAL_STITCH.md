# No-training stitch with Laya retained

The required architecture is a single local neural model with visual input and a
fast Laya decision stage. There must be no new training, regression fitting or
game-specific state extraction. This experiment keeps those constraints. It
does **not** yet meet the capability requirement: the bridge failed basic visual
tests.

## Implemented graph

```text
Screenshot pixels                    Instruction + supplied choices
       |                                          |
Frozen Qwen vision tower                   Laya token embeddings
       |
Existing Qwen visual merger
       |
4 x 4 spatially pooled visual vectors
       |
Fixed shared-vocabulary attention
       |                                          |
16 vectors in Laya's input dimensions -------------+
                                                  |
                                    Frozen Laya encoder and decision head
                                                  |
                                            Choice scores
```

All neural stages reside inside one `FrozenStitch` MLX module. The checkpoint
has one `model.safetensors` file plus tokenization, preprocessing and configuration
files. It contains Qwen's vision tower, the fixed bridge tensors, and Laya's
original weights. **It contains no Qwen language decoder.** No text is generated.
This is a composite model assembled from pretrained components, not a newly
pretrained foundation model.

The bridge uses 86,547 alphabetic token strings selected deterministically from
Qwen's vocabulary, each representable with one to four Laya tokens. This includes
word fragments; it is not a curated dictionary or game-specific label list.
Qwen's existing input embeddings provide keys; the mean of the corresponding
Laya subword embeddings provides values. For each visual vector, cosine
similarity selects eight keys and softmax with fixed temperature 0.05 combines
their Laya values. No image/text examples, labels, optimizer, regression solve or
parameter search construct this bridge. Normalization, pooling and the attention
settings are fixed design choices.

The central unproven assumption is that visual vectors before Qwen's language
decoder are semantically comparable with its input word embeddings. Their
matching dimensions alone do not establish this. This is inspired by the general
idea of shared anchors, but **is not an implementation of ASIF**. ASIF uses paired
cross-modal examples; this experiment uses existing lexical weights instead.
[ASIF paper](https://arxiv.org/abs/2210.01738)

## Measured probe, 2026-09-22

Apple M3 Max, 48 GiB memory. Eight synthetic images vary red/blue, left/right and
square/circle. Each image receives three questions in both option orders: 48
correlated decisions, not 48 independent images. Expected answers are generated
by the fixture renderer and used only for evaluation. Settings were not selected
against these answers.

| Method | Color | Position | Shape | Overall |
| --- | ---: | ---: | ---: | ---: |
| Frozen lexical stitch | 50% | 50% | 50% | 50% |
| Same model given a blank image | 50% | 50% | 50% | 50% |
| Same model with word correspondences reversed | 50% | 50% | 50% | 50% |
| Original Laya given the correct text description | 100% | 62.5% | 100% | 87.5% |

The real visual vectors mostly retrieved strings such as `and` and `in` in the
initial inspection. The bridge does not demonstrate useful semantic alignment.
The text baseline also has a spatial-reasoning weakness, so the bridge is not
necessarily the only obstacle.

Measured after loading the exported checkpoint:

- Image-to-scores median: **46.14 ms**; 95th percentile: **47.10 ms**.
- First inference: 87.58 ms; checkpoint load: 1.21 seconds.
- Input images: 384 x 256 before processor patch-grid sizing.
- Trainable tensors: zero. Generated tokens: zero. Game inputs sent: zero.
- Text-embedding forward parity with the original MLX Laya graph: zero maximum
  logit difference on the probe descriptions.
- The local fused weights file is approximately 2.3 GiB. No weights are in Git.

A separate replay of one recorded Hordes screenshot at a maximum width of 512
pixels measured **70.11 ms median** over four warm repetitions. It has no action
correctness labels and does not establish gameplay ability. Exporting and
reloading the checkpoint preserved the pre-export probe scores exactly. Its only
parameter roots are `vision`, `bridge` and `laya`.

Timing includes image reading, preprocessing, both neural stages and the bridge;
it excludes live capture, event posting and game response. Larger screenshots or
questions can cost more. The earlier 65.8 ms fitted pilot used a different
resolution and runtime and is not a controlled speed comparison.

Aggregate evidence: [lexical-001.json](lexical-001.json). Raw predictions and the
single-checkpoint bundle are under the ignored local `artifacts/lexical-001/`.

## What remains unresolved

This checkpoint must not be described as a general game-playing model. It accepts
one screenshot and supplied choice descriptions; it has no temporal memory and
does not independently emit unrestricted keyboard/mouse trajectories. No
cross-game evaluation is warranted until basic visual distinctions beat controls.

The next no-training research direction is a bridge based on a pretrained
contrastively aligned image/text encoder, or paired general image/text anchors
without parameter fitting. Laya would remain the decision stage. Either would
need a fresh evaluation; neither is proven by this negative result. The Qwen-only
path is not the project's active architecture.

## Attribution

Laya and its weights originate with Convai Innovations; the native MLX runtime
is [laya-mlx](https://github.com/mizorewww/laya-mlx). The embedding-forward
adaptation in `lexical_stitch.py` is Apache-2.0 licensed; its source license and
notice are preserved under `third_party/`. Qwen vision code is provided by
`mlx-vlm`. Pretrained weights retain their upstream licenses. Other original
project code remains under the root MIT license.
