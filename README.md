# Laya Vision Stitch

Research toward a fast, local visual decision model that retains **Laya** and
requires **no new training or connector fitting**. The active experiment combines
Qwen's frozen vision tower, a fixed lexical bridge and frozen Laya in one MLX
module and one weights file. Qwen's language decoder is excluded.

**Current result: fast, but not visually competent.** The exported model measured
46 ms median on small synthetic probes, but scored 50%—the same as blank-image
and scrambled-correspondence controls. It is not a working general game agent.
See [architecture, results and limitations](docs/LEXICAL_STITCH.md).

## Active no-training experiment

Use native arm64 Python 3.13 on Apple Silicon:

```bash
uv sync --extra models --extra stitch --python /path/to/arm64/python3.13
uv run --no-sync hf download mlx-community/Qwen3.5-4B-4bit --revision 0e7ffd5c629ef7719d4cbc04069232580bfa9d9c
uv run --no-sync hf download aac6fef/laya-multilingual-mlx --revision f2b4faf51023039425946074e2cf1361d2db11d5
```

Prepare a JSON object of choice labels and descriptions, for example
`{"left": "The object is on the left.", "right": "The object is on the right."}`.
Build and save the frozen model from cached checkpoints:

```bash
uv run --no-sync python -m laya_vision_stitch.lexical_stitch \
  --image /path/to/image.png --question 'Which side contains the object?' \
  --choices /path/to/choices.json --save artifacts/my-stitch
```

Future inference loads that single stitched checkpoint:

```bash
uv run --no-sync python -m laya_vision_stitch.lexical_stitch \
  --bundle artifacts/my-stitch --image /path/to/image.png \
  --question 'Which side contains the object?' --choices /path/to/choices.json

uv run --no-sync python -m laya_vision_stitch.lexical_probe \
  --bundle artifacts/my-stitch --output artifacts/my-probe
```

These commands output scores only. There is no live capture, keyboard/mouse
execution, OCR, game-state detector, optimizer or action policy outside the model.
Choices and instructions are caller inputs, not a built-in Hordes action set.
No generalization claim follows from packaging the components together.

## Historical fitted experiment

The earlier ridge-regression pilot below predates the no-training constraint. It
is retained for reproducibility and is **not the active approach**. Its fitted
connector and scripts are not used by `lexical_stitch`.

An offline research experiment connecting frozen Qwen visual features directly to
frozen Laya input embeddings. A ridge-regression connector is fitted from paired
screenshots and text state. Neither model's weights are updated. This is **not**
a pretrained multimodal Laya release or a working game controller.

**First pilot: fast, but no demonstrated visual decision benefit.** The stitched
path's summed offline median was 65.8 ms on an M3 Max. It chose attack on every
held-out image and did not beat constant or shuffled-data controls. See the
[measured results and limitations](docs/RESULTS.md).

```text
Screenshot -> frozen Qwen vision tower -> pooled visual features
                                            |
                                  fitted linear connector
                                            |
                         19 state vectors + fixed question/options
                                            |
                             frozen Laya Core ML graph -> scores
```

The existing ANE export exposes an `embeddings` tensor internally. Its public
API accepts text, but this experiment replaces only the state segment of that
tensor. Question tokens, choice markers and attention masks stay fixed. A parity
check verifies that injecting the original token embeddings reproduces the
ordinary graph path for the same prepared batch.

## Run on Apple Silicon

Use native **arm64 Python 3.13**, not Intel Python under Rosetta. The live game,
screen recording and Accessibility permissions are not needed.

```bash
uv sync --extra models
# Download once if these pinned snapshots are not already cached:
uv run --no-sync hf download aac6fef/laya-multilingual-coreml-ane --revision 39d6a9b3d0f67f06da74fbade6121ea134cbdb21
uv run --no-sync hf download mlx-community/Qwen3.5-4B-4bit --revision 0e7ffd5c629ef7719d4cbc04069232580bfa9d9c

uv run --extra models python -m laya_vision_stitch \
  --runs /path/to/screenquest/runs \
  --output artifacts/pilot-001 \
  --qwen-baseline
```

Inference resolves cached snapshots only. `--reuse` reuses extracted features if
the dataset and configuration match. Artifacts contain private source paths and
game state, and are ignored by Git. The source ScreenQuest checkout is read only.

If uv selects Intel Python, supply the path to a native interpreter explicitly:
`uv sync --extra models --python /path/to/arm64/python3.13`.

After fitting, evaluate a saved connector on an individual screenshot:

```bash
uv run --no-sync python -m laya_vision_stitch.infer \
  --artifact artifacts/pilot-001 --image /path/to/screenshot.jpg --repeat 5
```

This path takes only an image and the fitted artifact at inference time. It does
not read OCR state or ask Qwen to generate a caption. The first measurement
includes model warm-up; checkpoint loading is reported separately. It prints
scores and never executes the selected action.

## What the pilot measures

- The last eight eligible recording runs, split chronologically into fitting,
  validation and test groups; no run crosses a split.
- At most 32 screenshots per run, at least one second apart. Identical image
  files are deduplicated across the entire dataset.
- Supervision from recorded OCR/heuristic state. Policy phase, recorded action
  and allowed-action sets are excluded from input descriptions.
- Five fixed action choices for every example, so the answer is not supplied
  through a one-option mask.
- Regularization selected on validation only. Both backbones remain frozen.
- Agreement with Laya's text-state decisions, balanced agreement and class
  counts. There are no human correctness labels.
- Mean-state, training-majority, nearest-image and shuffled-pair controls.
  Optional Qwen control scores single-token actions in one multimodal prefill.
- Separate visual encoding, connector and Laya timings. Their sum is an offline
  estimate, not measured live screenshot-to-keypress latency.

A positive result must beat the controls and survive a later independently
labeled, scene-disjoint evaluation. Adjacent recording runs can still depict
similar scenes. Agreement with a weak or biased teacher does not prove visual
understanding, correct action selection, or useful gameplay.

## Checks

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest -q
```

Tests cover connector behavior, split isolation and the embedding injection
boundary without loading model weights. See [research protocol](docs/PROTOCOL.md).

## Sources

- [Laya](https://github.com/NandhaKishorM/laya)
- [Laya Core ML](https://github.com/mizorewww/laya-coreml)
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Latent Space Translation via Semantic Alignment](https://arxiv.org/abs/2311.00664)
- [ASIF](https://arxiv.org/abs/2210.01738)

This pilot uses fitted ridge regression; it is not an implementation or
reproduction of either paper. "No backbone retraining" does not mean "no fitting."
Upstream models and dependencies retain their own licenses. Game recordings are
not included. Original experiment code is MIT licensed.
