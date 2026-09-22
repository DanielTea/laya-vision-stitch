# Laya Vision Stitch

Research toward a fast local visual decision model with **Laya retained**.

The current path combines **frozen Qwen vision, a trainable goal-conditioned
connector, Laya, and keyboard/mouse action outputs**, with optional small LoRA
adapters. Full Qwen can supply offline teacher targets; it does not decode text
during deployment. Original backbone weights stay frozen. See the
[training guide and data format](docs/TRAINABLE_STITCH.md) and
[initial training results](docs/TRAINING_RESULTS.md).

The pipeline trains and exports a single model, but the initial synthetic pilots
remain at chance on goal-dependent choices. This is research infrastructure,
not a trained general game agent. The previous no-training experiments remain
available below.

The previous experiment combines frozen CLIP, paired image–description reference
memory, and frozen Laya in one MLX module and one weights file. No autoregressive
model generates captions. No game-specific detector or action rule selects the
answer.

**Measured result:** 67.9% on a three-choice synthetic visual test, versus 33.3%
with blank images and 41.4% with shuffled references. Median inference was 24.2 ms
on an M3 Max. Accuracy drops to 47.2% on withheld combinations; the mixture does
not outperform the nearest-reference baseline. **This is not a general game
agent.** [Full protocol, results and limits](docs/PAIRED_REFERENCE.md).

```text
Screenshot -> CLIP vision -> similarities to reference images
                                      |
                         four stored description sequences
                                      |       instructions + choices
                                      +-----------------+
                                                        |
                                                frozen Laya
                                                        |
                                          weighted decision scores
```

## Setup on Apple Silicon

Use native arm64 Python 3.13. If uv chooses an Intel interpreter, pass the path
to an arm64 interpreter with `--python /path/to/arm64/python3.13`.

```bash
uv sync --extra models --extra stitch
uv run --no-sync hf download mlx-community/clip-vit-base-patch32 --revision b0d393a1f061c5bdcbaa4bfba8682091f04d0c0d
uv run --no-sync hf download aac6fef/laya-mlx --revision 20aed815fc6acde75733882e7ec0e3f28aeb9717
# Needed for the multilingual baseline in the experiment:
uv run --no-sync hf download aac6fef/laya-multilingual-mlx --revision f2b4faf51023039425946074e2cf1361d2db11d5
```

Model loading uses pinned, cached snapshots. The game, live screen recording and
Accessibility permission are not required for these offline experiments.

## Reproduce the paired-reference experiment

```bash
uv run --no-sync python -m laya_vision_stitch.reference_probe \
  --output artifacts/reference-new
```

This creates 36 reference images and 54 separate evaluation images, constructs a
frozen reference bank, compares both Laya text baselines, runs negative controls,
and saves a single-checkpoint bundle. It performs **no training**. Use a fresh
output directory to preserve previous evidence.

For the existing local result, the bundle is `artifacts/reference-001/bundle`.
The evaluation contains colors, shapes and spatial positions, not gameplay.
Reference and test images are disjoint; three color–shape pairs are entirely
absent from reference memory. Full details are in the linked report.

## Build a bank from your own paired references

Provide a JSON array with at least four records. Each record has this form:

```json
{
  "id": "scene-001",
  "image": "images/scene-001.png",
  "caption": "A blue door is in the middle of a corridor."
}
```

Image paths are relative to the manifest. Descriptions should be accurate and
concise. The bank supplies the model's visual reference coverage; an unrelated
bank does not make it capable of understanding a new game.

```bash
uv run --no-sync python -m laya_vision_stitch.reference_stitch \
  --references /path/to/references.json --save artifacts/my-reference-model
```

For inference, put a JSON object of choice labels and descriptions in a file,
for example `{"left":"Move left.","right":"Move right.","wait":"Wait."}`:

```bash
uv run --no-sync python -m laya_vision_stitch.reference_stitch \
  --bundle artifacts/my-reference-model \
  --image /path/to/screenshot.png \
  --question 'Which action would move toward the described doorway?' \
  --choices /path/to/choices.json
```

The command prints scores and retrieved reference IDs. It sends no keyboard or
mouse events. Choice descriptions are caller inputs; no built-in Hordes action
set is used. Inference accepts one screenshot and has no temporal memory. The
model cannot invent descriptions absent from its reference bank.

## Checks

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest -q
```

Tests cover frozen pairing, reference/test separation, duplicate rejection,
value-preserving CLIP weight conversion and the earlier embedding experiments.
Large checkpoint tests and measured runs remain local. Model files, references
and screenshots under `artifacts/` are ignored by Git. ScreenQuest is unchanged.

## Earlier experiments

- [Paired references: previous no-training experiment](docs/PAIRED_REFERENCE.md).
- [Fixed lexical bridge: fast, but failed basic visual tests](docs/LEXICAL_STITCH.md).
- [Historical fitted ridge bridge: failed constant-action controls](docs/RESULTS.md).
  Its [protocol](docs/PROTOCOL.md) is retained for reproducibility.

## Sources and licensing

- [Laya](https://github.com/NandhaKishorM/laya) and
  [Laya MLX](https://github.com/mizorewww/laya-mlx).
- [CLIP](https://github.com/openai/CLIP) and
  [Apple MLX CLIP example](https://github.com/ml-explore/mlx-examples/tree/main/clip).
- [ASIF: paired anchors without parameter training](https://arxiv.org/abs/2210.01738).
  Our decision-mixture experiment is not an ASIF reproduction.

Original project code is MIT licensed, except the Apache-2.0 embedding-forward
adaptations identified in `lexical_stitch.py` and `trainable_model.py`. Vendored CLIP code retains Apple's
MIT license; modifications and upstream notices are recorded under
`third_party/`. Pretrained weights retain their upstream licenses and are not
included in this Git repository.
