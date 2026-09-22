# Diagnosing real-game learning

This experiment asks whether the stitched model can first fit a small real-game
sample, then transfer to different recording sessions. It is an offline training
experiment, not a gameplay demonstration. **All five runs failed the training-fit
and separate-session gates. The latest checkpoint collapses to no-button output.**
The default checkpoint is unchanged.

## Neural changes

The experimental variant remains one exported model:

```text
2 screenshots -> frozen Qwen vision -> frame-separated spatial tokens
                                      -> temporal attention + goal-conditioned readout
                                      -> aligned visual tokens
Goal + controls ---------------------> Laya with rank-8 LoRA in its last 2 layers
                                      -> attention over the full Laya token sequence
                                      -> 4 parallel keyboard/mouse steps, 100 ms apart
```

The temporal path and action readout begin as zero residuals around the source
model. Camera movement uses 17 categorical bins per axis; inference selects a
mode instead of averaging incompatible turns. Single-step labels also supervise
this categorical head; a missing loss for those examples was corrected after
the first grounding diagnostic, then tested in a fresh run. Continuous mouse regression remains
an auxiliary training target. Pointer and duration heads remain present, but this
experiment does not validate clicking. Qwen language decoding and an external
planner are absent at inference. No game-specific action rules are added.

Only the connector, action heads and small Laya adapters train: 8,901,982 of
763,710,052 parameters (1.17%). Original Qwen and Laya backbone hashes are checked
after training. Checkpoints are reloaded and their button outputs compared. The runtime exports
all four neural action proposals; no desktop executor or camera-control loop is
introduced by this experiment.

## Data and fixed gates

64 training clips span Grounded, Minecraft and Raft. Another 64 clips come from
separate recording sessions in those same games. Training selection covers
button-set/mouse-direction strata; validation selection is uniform within each
session. Each example has two past screenshots and four future control labels.
Future actions are never model inputs. Previous-action text is removed to force
the experiment to use vision; recorded history is retained only for a baseline.

The D2E-480p source revision is
`f075f7e25df6f6d385840a836f86bf92dfb877ff` (CC BY-NC 4.0). The existing source
import and synchronization procedure is in [REAL_GAME_TRAINING.md](REAL_GAME_TRAINING.md).
Image, recording-session and example overlap checks protect the split. The same
validation sessions are reused across these diagnostics, so they are development
validation, not an untouched final benchmark.

Visual review found a browser/YouTube screenshot incorrectly grouped with a
Grounded recording. The first diagnostic contains it; later runs replace it with
a reviewed clip from the same session. The tracked curation file pins both IDs.
All 128 current frames were subsequently reviewed. Window metadata alone did not
catch that contamination. Labels and review are by the assistant, not an expert
human annotator.

Criteria were fixed before these runs:

- Tiny training fit: button micro-F1 >= 0.95, exact button-set match >= 0.90,
  moving-axis mouse-direction accuracy >= 0.80.
- Separate-session validation: button F1 >= 0.70, at least 0.10 above shuffled
  images, and above the recorded previous-action baseline.
- Cross-game and live trials remain blocked until these criteria pass.

Shuffling pairs of screenshots stays within each game. Blank-image controls and
synthetic-skill retention are also measured. Grounding audits additionally balance
answers within each goal wording: predicting the common menu-closed state can
otherwise score above 80% without understanding the screenshot. Four-step aggregate metrics are
available in the final diagnostic, in addition to first-action metrics.

## Visual grounding and goal probes

The final diagnostic adds reviewed menu-state labels on the real screenshots:
menu present/absent questions and their negations. It also adds paired synthetic
instructions, such as holding W only with a menu open versus only with it closed.
Each identical image therefore receives opposing goals and opposing targets.
These are controlled instruction probes, **not inferred player intentions**.
No claim of general gameplay goal following follows from their accuracy.

Each split contains 256 probes derived from its own 64 images. Current-image
hashes bind them to the tracked review. Old action targets and descriptions are
removed before producing new supervision. The training mixture has 384 real
imitation slots, 128 preservation-replay slots and 256 grounding slots.
An unchanged source checkpoint supplies preservation targets offline; it is not
loaded during model inference. The source's original synthetic holdout is reused
as a retention diagnostic, not a new untouched test.

## Measured results

Each run performed 4,000 updates from the same synthetic source checkpoint.
Values below are first-action button micro-F1; the shuffle stays within game.

| Diagnostic | Training F1 | Separate-session F1 | Shuffled F1 | Synthetic choice retention |
|---|---:|---:|---:|---:|
| Aligned + preservation (one contaminated clip) | 0.310 | 0.225 | 0.155 | 78.1% |
| Aligned, pure 64-clip fit | 0.186 | 0.248 | 0.229 | 48.4% |
| Temporal/chunks, pure 64-clip fit | 0.325 | 0.122 | 0.024 | 58.6% |
| Temporal + grounding + preservation | 0.291 | 0.257 | 0.257 | 50.0% |
| Same recipe, categorical mouse-loss correction | 0.000 | 0.000 | 0.000 | 49.2% |

The clean-split recorded previous-action baseline scores 0.821 on training clips
and 0.901 on separate-session validation. The final model's all-four-step F1 is
also 0.000. First-step camera direction accuracy is 0% on moving validation axes;
the four-step aggregate is 3.1%. No run meets the prespecified gates.

The first grounding run answered 88.3% of separate-session conditional questions,
but its button-set accuracy was 50%. It did not turn answer selection into useful
actions. After the categorical mouse-loss correction, the fresh run regressed to
50% conditional choices and 50% button-set accuracy. Factual menu questions scored
82.8%, exactly the majority-state baseline; answer-balanced accuracy was 50%, and
shuffling screenshots left the score unchanged. This is not evidence of robust
visual grounding or goal control. Preservation losses also failed to retain the
source's 100% synthetic choice accuracy: the final run retained only 49.2%.

The final two-image inference median was **126 ms over five warm calls** on the
M3 Max. It includes image preprocessing and vision encoding, excludes desktop
capture, input posting and game response, and is too small a timing sample for
a performance guarantee. Earlier run timings ranged substantially. Fast failed
action predictions do not establish a useful control rate.

44 tests and Ruff passed. Frozen backbone hashes were unchanged and exported
button predictions matched after reload in all five runs. Rebuilding the curated
manifests reproduced both splits exactly after normal manifest validation,
including image hashes. No live input events were sent, no additional held-game
benchmark was consumed, and the original controller repository was unchanged.

### What the failure narrows down

Removing replay did not solve the tiny-set fit failure. Adding temporal attention,
a full-token action decoder and chunk targets did not solve it either. The
categorical/continuous mouse-head mismatch was a real implementation error in the
first grounding run, but correcting it did not solve the learning problem.
These observations do not identify a single cause, and the changed architectures
and losses do not isolate individual component effects.

The next defensible experiment is a staged diagnostic: freeze the existing
visual/language path and train only the action decoder on a balanced, explicitly
goal-labelled real-image task. Check action loss scale and whether opposing goals
produce opposing button outputs before reintroducing joint imitation losses.
Then align the connector on diverse real-image semantic labels. This is a
proposed next experiment, **not completed work or a promised fix**. More gameplay
recordings alone are not justified until the tiny fitting gate passes. Human
intent and task-level goals remain absent from the raw demonstration source.

## Reproduction

Run from the repository root after the existing D2E source import and synthetic
source training. Use new output directories to preserve evidence:

```bash
uv sync --extra models --extra stitch --extra data
uv run --no-sync python -m laya_vision_stitch.learning_gate \
  --curation annotations/gameplay-curation-001.json \
  --output artifacts/my-learning-gate
uv run --no-sync python -m laya_vision_stitch.grounding_data \
  --data artifacts/my-learning-gate \
  --annotations annotations/menu-grounding-001.json
uv run --no-sync python -m laya_vision_stitch.fit_gate \
  --data artifacts/my-learning-gate --temporal --grounding \
  --replay-count 64 --steps 4000 --learning-rate 0.0001 \
  --output artifacts/my-grounding-run
```

For the aligned capacity control omit `--temporal --grounding` and set
`--replay-count 0`. For the temporal capacity control omit only `--grounding`
and set replay to zero. Each run uses seed 17, single-example AdamW updates,
gradient clipping at 1.0 and frozen feature caching. The source checkpoint is
`artifacts/scaled-robust-001/bundle`.

The first four diagnostics predate the single-step categorical mouse-loss fix;
`fit-gate-grounding-002` starts fresh from the same synthetic source and includes
that correction. The commands above use the corrected implementation. The
grounding report is expanded after training with shuffled-image and balanced
metrics, without updating checkpoint weights.

Four thousand steps are about 62.5 passes over 64 clips without replay, 31.25
passes with 64 replay examples, or 5.2 passes over the final weighted mixture.
Different objectives and mixtures are not a controlled attribution of each
architectural change. These failures cannot prove that the architecture is
incapable with every optimization recipe or dataset.

Machine-readable measurements: [learning-gate-001.json](learning-gate-001.json).
