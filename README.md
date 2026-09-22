# Laya Vision Stitch

One local screenshot-and-goal model combining **pretrained vision, a learned visual
connector, Laya with small LoRA adapters, and action outputs**. It exports as one
checkpoint and runs without Qwen language decoding, generated captions or a
reference-image bank. Original pretrained backbone weights stay frozen.

**Latest experiment:** Laya goal features now condition the pretrained Open-P2P
vision, memory and action decoder in one checkpoint. Fresh screenshot inference
takes **31.7 ms median** offline. Image interventions show visual dependence on
public games, but Hordes combat and robust goal following remain unsolved.
[Architecture, conversion checks and limitations](docs/LAYA_P2P.md).

**Previous faster-vision experiment:** a native MLX C-RADIOv3-B visual encoder and learned visual
action branch reduce full fresh-image prediction to **31.0 ms median** offline.
A 32-image fitting test passes and fails under image permutation, but held-out
Hordes gameplay remains unsolved. **No reliable Hordes agent or sub-60 ms live
reaction has been demonstrated.** [Architecture and evidence](docs/RADIO_STITCH.md) ·
[Record human demonstrations](docs/HUMAN_DEMONSTRATIONS.md).

**Previous temporal experiment:** expanded training to **7,520 frames across nine games
and 38 goals**, with random sequence crops, visual-token masking and validation-based
checkpoint selection. A regularized attention adapter improves button F1 on newly
held-out recordings from **16.2% to 32.2%**, but regresses on the previous held-out
experiences from **57.8% to 43.0%**. Goal-swap tests change no actions. It is still
**not a validated Hordes/general-game agent** and has not replaced the default model.

Fresh-image inference measures **49.9 ms median** on an M3 Max, excluding capture
and game contention. Qwen and Laya remain frozen; about 0.13% of parameters train.
[Expanded data, comparisons and limitations](docs/TEMPORAL_EXPANSION.md) ·
[Temporal architecture and streaming usage](docs/TEMPORAL_MEMORY.md).

A subsequent **60-second live Hordes trial** measured **52.7 ms median inference**
and produced movement and a selected monster, but **no ability presses or
demonstrated combat**. Slow frames caused 111 temporal-memory resets.
[Live results, recording details and experimental runner](docs/HORDES_TEMPORAL_LIVE.md).

The earlier visual action adapter with numeric control history reached **82.0%**
validation button F1 and **84.4%** on withheld experiences on a different sample
set. It still lost to repeating the previous action. These percentages are not
directly comparable with the new consecutive-sequence experiment.
[Earlier visual adapter](docs/VISUAL_ACTION_ADAPTER.md) ·
[Recorded-action results](docs/GAMEPLAY_BUTTONS.md) ·
[Instruction robustness](docs/ROBUST_DECODER.md) ·
[Reasoning-transfer experiment](docs/REASONING_TRANSFER.md) ·
[Public P2P data](docs/P2P_TRAINING.md).

The synthetic-training checkpoint reached its target in **40/40 fresh synthetic sandbox episodes**,
using its learned A/D button outputs. Warm inference took **58 ms median** on an
M3 Max. It answered 384/384 standard move-toward questions across fresh scenes,
withheld object combinations and a withheld visual style; shuffled images
reduced accuracy to 50–53%.

This demonstrates basic visual movement, not general gameplay. Unseen “retreat”
wording scored 66%, and reversing answer order reduced accuracy to 86–91%.
Combat, camera control, clicking and transfer to actual games remain unvalidated.
[Full results and limits](docs/SCALING_RESULTS.md) ·
[Reproduce training and inference](docs/SCALING.md) ·
[Training data format](docs/TRAINABLE_STITCH.md).

```text
Screenshot → frozen Qwen vision → learned connector → Laya + small LoRA → actions
                                                        ↑
                                             goal, controls, recent actions
```

The synthetic recipe trains **6.05M parameters (~0.80%)**. The P2P recipe trains
about **8.89M (~1.16%)**, keeping original backbone weights frozen.
The measured synthetic recipe uses 512
training images, 12,288 examples including different goals/wordings, and 12,000
optimization steps. Description supervision exists only during training;
inference receives pixels and the ordinary prompt.

## Run the current local model

The subsequent [real-game training pilot](docs/REAL_GAME_TRAINING.md) imports
synchronized human controls from Grounded, Raft and Minecraft, with Satisfactory
and Barony held out. It adds two-frame inputs, a larger physical-key vocabulary,
and offline action-agreement evaluations with visual and persistence controls.
These demonstrations use D2E's CC BY-NC 4.0 data. Offline imitation must not be
confused with successful closed-loop gameplay. **The first trained real-game
checkpoint failed the transfer checks:** Barony button F1 was 0.332, versus 0.841
for repeating the previous action, and shuffling screenshots scored 0.338.
[Results and failure analysis](docs/REAL_GAME_RESULTS.md). It has not replaced
the default model below.

A subsequent [learning diagnostic](docs/LEARNING_GATE.md) adds a temporal connector,
parallel action chunks, reviewed real-image grounding and explicit goal probes.
**All five diagnostic runs failed the small-fit and separate-session gates.**
The final checkpoint predicts no buttons; it remains experimental and has not
replaced the default.

A subsequent [decoder-only experiment](docs/DECODER_PROBE.md) freezes the entire
visual/language path and improves a two-action, paired-goal diagnostic from 50%
to 82% exact action match on separate sessions. Balanced accuracy is 71%, and
reworded goals score 34%; its strict gates still fail. This is progress in action
decoding, not validated gameplay. The checkpoint remains experimental.

An earlier continuation corrects saturated action attention, reads Laya encoder
features directly, and runs seven wording experiments plus four recorded-button
experiments. Small-adapter training fixes the 64-clip button-fitting failure;
neither instruction robustness nor gameplay transfer passes. New-game scoring
and live control remain gated, and no experimental checkpoint replaces the
default model below. All experiments remain one neural model at inference.

The full-Qwen teacher audit now tests scene recognition, paired instructions and
recorded button candidates. Neither direct generation nor bounded thinking
qualifies as an instruction teacher: 62.5% / 60.4% accuracy versus the stitched
model's 79.2% on the same cases. A guarded feature/decision distillation recipe is
implemented, but no transfer training or deployment ran with these failed labels.

With the locally trained checkpoint available:

```bash
./scripts/start-model.command --port 8767
```

This keeps the model loaded at `http://127.0.0.1:8767`. Send screenshots and goals
to `POST /predict`; see the [Python request example](docs/SCALING.md#persistent-local-api).
The API returns proposed actions and sends no desktop inputs. Model weights are
not committed to Git; a fresh checkout needs the [setup and training recipe](docs/SCALING.md#reproduce).

To run the simple screenshot-driven sandbox with a new seed:

```bash
uv run --no-sync python -m laya_vision_stitch.policy_sandbox \
  --bundle artifacts/scaled-robust-001/bundle --output artifacts/my-sandbox \
  --episodes 40 --seed 9501
```

## Historical no-training experiment

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

Tests cover training/inference separation, feature caching, held-out data,
goal augmentation, the persistent API and earlier embedding experiments.
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
- [Qwen3.5](https://github.com/QwenLM/Qwen3.5); the current model uses its vision
  encoder through MLX VLM, with pinned weights listed in the training guide.
- [CLIP](https://github.com/openai/CLIP) and
  [Apple MLX CLIP example](https://github.com/ml-explore/mlx-examples/tree/main/clip).
- [ASIF: paired anchors without parameter training](https://arxiv.org/abs/2210.01738).
  Our decision-mixture experiment is not an ASIF reproduction.

Original project code is MIT licensed, except the Apache-2.0 embedding-forward
adaptations identified in `lexical_stitch.py`, `trainable_model.py` and the
Mamba-3 recurrence in `temporal_memory.py`. Vendored CLIP code retains Apple's
MIT license; modifications and upstream notices are recorded under
`third_party/`. Pretrained weights retain their upstream licenses and are not
included in this Git repository.
