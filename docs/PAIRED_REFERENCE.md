# Paired references: useful visual signal, limited composition

On 2026-09-22 the no-training paired-reference model scored **67.9%** on a
three-choice synthetic visual benchmark, versus **33.3%** with blank images and
**41.4%** with shuffled image/description correspondences. It measured **24.2 ms**
median image-to-scores latency on the local M3 Max. This is evidence of visual
grounding in a small controlled task, not evidence of general game-playing.

The four-reference mixture does **not** outperform the nearest-reference
baseline (68.5%). Withheld color/shape combinations lower accuracy to 47.2%.
Reference coverage and weak spatial retrieval remain major limitations.

## One frozen composite model

```text
Pixels -> frozen CLIP vision encoder + projection
                      |
              cosine similarity to fixed reference-image vectors
                      |
              top four reference caption token sequences
                      |                         instruction + choices
                      +----------------------------------+
                                                         |
                                               frozen English Laya
                                               (four rows in one batch)
                                                         |
                                               weighted choice scores
```

Both backbones remain frozen. The reference bank is stored memory; there is no
optimizer, regression, connector fitting or generated text. This is a
retrieval-based composite model, not newly learned native multimodality. It is
packaged as one MLX module and one `model.safetensors` file plus configuration and
a tokenizer. Laya is the final decision stage.

Unlike the failed lexical bridge, this version preserves each retrieved
description as a coherent token sequence. It does not average unrelated token
embeddings. Laya answers the caller's question independently for each retrieved
description; fixed similarity weights combine its answer distributions. No
game-specific detector, state parser, action mask or rule selects the answer.

CLIP's text encoder is used only while building an optional caption-key control.
Neither CLIP's text encoder nor Qwen's language decoder is in the deployed graph.
Caption tokenization and reference image encoding happen once during bank
construction. No images or raw source paths are needed inside the deployed bank.
The model still needs supplied choice descriptions; it does not generate
unrestricted keyboard/mouse trajectories or maintain temporal memory.

This is inspired by paired-anchor approaches such as
[ASIF](https://arxiv.org/abs/2210.01738), **not an ASIF reproduction**. Here,
similarity-weighted Laya decisions replace ASIF's relative-representation scoring.
[CLIP](https://arxiv.org/abs/2103.00020) provides the pretrained visual features.

## Fixed protocol

- 36 reference images: two variations of 18 color/shape/position combinations.
- 54 previously unseen evaluation images: all 27 combinations under two new
  render styles, with new sizes and positions. No identical image files cross
  reference/evaluation boundaries.
- Three color/shape pairs never appear in reference memory: red triangles,
  blue squares and green circles. Their 18 test images measure limited
  compositional transfer. The other 36 test images measure familiar combinations
  under changed rendering.
- Three questions per image (color, shape, horizontal position), each asked in
  two option orders. This produces 324 correlated decisions, not 324 independent
  images. Each question has three balanced answer classes.
- Top-k = 4 and reference temperature = 0.02 were fixed before evaluation. No
  reference selection or parameters were optimized against evaluation labels.
- English Laya was designated the active checkpoint before seeing results. Both
  English and multilingual text-only baselines are reported; a predeclared
  alternate English question format is measured without choosing between them.
- The images are procedural geometric fixtures. They are not game screenshots,
  natural photographs or a broad visual-language benchmark.

## Results

| Method | All | Familiar combinations | Withheld combinations |
| --- | ---: | ---: | ---: |
| Four-reference image-key mixture + Laya | 67.9% | 78.2% | 47.2% |
| Nearest image reference + Laya | 68.5% | 80.6% | 44.4% |
| Direct fact from nearest reference, evaluation-only baseline | 68.5% | 80.6% | 44.4% |
| Four-reference caption-key mixture + Laya | 62.7% | 71.3% | 45.4% |
| Blank image + Laya mixture | 33.3% | 33.3% | 33.3% |
| Shuffled correspondence + Laya mixture | 41.4% | 47.2% | 29.6% |
| English Laya with correct text descriptions | 100% | 100% | 100% |
| Multilingual Laya with correct text descriptions | 98.8% | 99.1% | 98.1% |

The alternate English oracle question format also scored 100%. These are
oracle descriptions of very simple scenes; they do not imply general reasoning
or gameplay competence. The protocol differs from earlier two-choice tests, so
oracle scores should not be compared as a controlled checkpoint improvement.

For the active mixture, color scored 83.3%, shape 77.8%, and horizontal position
42.6%. Laya exactly reproduces the nearest-reference facts in the hard-retrieval
baseline. The experiment therefore demonstrates that reference retrieval can
supply useful visual information to Laya; it does not demonstrate reasoning that
improves upon retrieval for these simple questions.

An exploratory bootstrap resampling the 54 images (keeping each image's six
questions together) gives a 95% accuracy interval of 62.0–73.8% for the active
mixture. Its paired difference from the nearest-reference baseline spans
−4.9 to +3.7 percentage points. The small procedural test family limits what
these intervals say about other visual domains.

## Runtime and verification

- Apple M3 Max, 48 GiB, native arm64 Python 3.13.9.
- CLIP ViT-B/32, 224-pixel bicubic letterboxing and standard CLIP normalization.
  No task-dependent crop. Small HUD text will be challenging at this resolution.
- English Laya MLX checkpoint, original FP16 weights. CLIP weights retain the
  original converted FP32 values; only parameter names are translated.
- Median image-to-scores: **24.19 ms**; 95th percentile: **25.19 ms**.
- First inference: 605.26 ms. Initial bank/model construction: 1.19 seconds in
  this cached run, excluding checkpoint download and later export.
- The measurements include reading the image, preprocessing, CLIP, retrieval,
  batched Laya and score combination. They exclude live capture, event posting
  and game response. Different workloads and larger banks require new timing.
- Saved checkpoint: 1,194,180,213 bytes, approximately 1.11 GiB.
- Zero trainable tensors, generated tokens and game input events.
- Export/reload reproduced six checked decisions and their attention weights
  exactly. Unit tests verify pairing, frozen parameters, split isolation,
  duplicate rejection and value-preserving CLIP parameter renaming.

Pinned sources:

- `mlx-community/clip-vit-base-patch32` at
  `b0d393a1f061c5bdcbaa4bfba8682091f04d0c0d`.
- `aac6fef/laya-mlx` at `20aed815fc6acde75733882e7ec0e3f28aeb9717`.
- `aac6fef/laya-multilingual-mlx` at
  `f2b4faf51023039425946074e2cf1361d2db11d5` for the text control.
- Apple MLX CLIP implementation at
  `09b641aaa74f9737f747b62ad8c628405e7e25be`; local compatibility changes and
  upstream licenses are recorded under `third_party/`.

Aggregate evidence is in [reference-001.json](reference-001.json). Locally,
`artifacts/reference-001/` contains the paired data, all predictions, original
summary, additional checks and the deployable bundle. Artifacts are ignored by
Git. The original ScreenQuest repository is unchanged.

## Next gate

The architecture can accept a broader bank without retraining, but it cannot
invent descriptions absent from that bank. Before claiming cross-game utility,
build a bank with diverse natural scenes and gameplay from several games, then
hold out complete games and recording sessions. Include spatial relationships,
object interactions, HUD details and motion; evaluate these separately.

The current 42.6% spatial score and composition drop do not justify live game
control. Increasing bank size alone is not a demonstrated solution. A next
experiment should compare more expressive frozen visual encoders and spatial
reference representations while preserving the same held-out-game evaluation.
