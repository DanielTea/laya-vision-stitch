# Isolating the action decoder

This experiment freezes Qwen vision, the temporal connector and all Laya weights,
including the previously trained LoRA adapters. Only the button decoder trains.
The exported checkpoint remains one neural model taking screenshots and goals.
There is no inference-time lookup, menu rule or conversion from answer labels to
keyboard events.

## Scope

The controlled task has two possible actions: hold W or release all buttons.
Each of 64 real screenshot pairs receives two opposing instructions. Another 64
pairs come from separate recording sessions in Grounded, Minecraft and Raft.
Menu labels were reviewed by the assistant, and the conditional instructions are
synthetic. They are not records of what a human player intended. This small
probe does not measure gameplay, combat, mouse control or transfer to new games.

The source is `artifacts/fit-gate-grounding-001/bundle`, selected because its
conditional answer accuracy was 89.8% on these training images and 88.3% on these
previously used development sessions. Its button accuracy was 50%. This source
selection and subsequent comparisons use development validation; none is an
untouched final benchmark. The source's limitations remain visible: rewording
validation goals reduces frozen answer accuracy to 45.3%, and reversing answer
options reduces it to 10.9%.

## What is isolated

The decoder has 843,059 trainable parameters, about 0.11% of the full model.
The experiment:

- Caches the final Laya token sequence for each screenshot/goal input. Cache
  equivalence with normal forward inference is tested. No target labels enter
  these features.
- Trains the existing attention readout, normalization and button outputs.
- Removes language, camera, preservation and gameplay-imitation losses.
- Gives changing buttons and suppression of unused buttons equal loss weight,
  so 50 always-off keys cannot dilute the changing key's supervision.
- Samples goal/action strata equally, balancing menu-open/menu-closed examples
  and both opposing goals. The batch variant includes every stratum in each
  update, then averages gradients before clipping and updating parameters.

SHA-256 checks cover the complete frozen visual/language path, including the
connector and LoRA. Mouse, pointer and duration head **weights** are also checked
unchanged. Their outputs are not guaranteed unchanged because the shared action
readout changes; this experiment does not validate those outputs or future steps.

The saved model still encodes images and goals normally at inference. Training
context caches are not used for deployment. Each invocation of this experiment
reapplies its freeze mask; exporting weights does not serialize MLX optimizer
state or guarantee that other training programs use the same freeze mask.

## Fixed evaluation criteria

Before the first update, the experiment records its protocol and manifest hashes.
The first-step threshold remains 0.5 for every key.

- Training: at least 95% exact button sets, 95% accuracy balanced over goal/action
  strata, and 90% of image pairs correct for **both** opposing goals.
- Separate sessions: at least 85% exact button sets, 80% balanced accuracy, 75%
  both-goal success, and a 15-point balanced-accuracy margin over shuffled images.

Checks include blank visual features, within-game image shuffling, reworded goals
and reversed answer options. Goals/options never receive evaluation labels as
inputs. The menu-closed prior can reach about 83% ordinary accuracy; balanced
accuracy and paired-goal results are needed to interpret this task.

## Results

Decoder-only training with balanced minibatches improved action prediction, but
**all four runs still failed the fixed gates**. No joint training or live control
was started, and the default model was not replaced.

| Run | Updates × batch | Training exact / balanced | Separate-session exact / balanced | Both validation goals correct |
|---|---:|---:|---:|---:|
| Single-example control | 4,000 × 1 | 50.0% / 50.0% | 50.0% / 50.0% | 0.0% |
| Matched presentations, balanced batches | 500 × 8 | 68.0% / 61.1% | 69.5% / 60.0% | 40.6% |
| Larger decoder-only budget | 4,000 × 8 | 93.8% / 86.5% | 83.6% / 70.3% | 71.9% |
| Lower-rate refinement | 2,000 × 8 | 89.8% / 93.8% | 82.0% / 71.1% | 67.2% |

The fourth run continues the third for 2,000 updates at 0.00001 with a new AdamW
optimizer. It has 48,000 cumulative decoder-only presentations. The refinement
improves balanced training accuracy but reduces ordinary accuracy; it is not an
across-the-board improvement. All results are retained rather than selecting one
checkpoint by its most favorable metric.

Final-checkpoint separate-session controls:

| Inputs | Exact action match | Balanced action match | Both opposing goals correct |
|---|---:|---:|---:|
| Original screenshots and goals | 82.0% | 71.1% | 67.2% |
| Blank visual features | 4.7% | 13.6% | 0.0% |
| Screenshots shuffled within game | 70.3% | 51.5% | 54.7% |
| Reworded goals | 34.4% | 33.4% | 0.0% |
| Reversed answer options | 43.0% | 45.8% | 20.3% |

The final decoder gets both goals right on **43/64 validation scenes**, versus
0/64 initially. It uses visual information: the balanced score falls by 19.7
percentage points with shuffled images. But its 71.1% balanced accuracy remains
below the 80% threshold; reworded goals score only 34.4% exact. An always-menu-closed
strategy can score 82.8% ordinary accuracy on this data, so the final 82.0% ordinary
score alone would not establish visual learning. Balanced and paired controls
provide the more informative evidence. Blank-feature inputs are outside the
training distribution and can trigger extra keys, explaining scores below 50%.

The frozen path's answer scores remain unchanged throughout. Exact full-component
hashes confirm this, including the connector and LoRA. Exported/reloaded button
probabilities match. **48 tests and Ruff pass**, including cache/forward equivalence,
frozen-weight protection for single-example and minibatch updates, and preservation
of changing-key gradient scale as unused-key count grows.

This narrows the problem: the existing frozen representations can support better
button decoding when optimization is isolated and balanced. It does not establish
a general game agent or prove that all earlier failures were caused by a single
optimizer issue. Visual-state coverage and goal-format robustness remain unresolved.
The next experiment should train goal paraphrases and option-order variation with
new wording held out, while retaining the frozen-path controls. That experiment
has not been performed here.

[Machine-readable measurements](decoder-probe-001.json). Full predictions and
checkpoints remain under `artifacts/decoder-probe-001` through `decoder-probe-004`.

## Reproduce

Use the existing source checkpoint and curated data from
[the earlier learning diagnostic](LEARNING_GATE.md). Run from the repository root:

```bash
# Single-example control: 4,000 presentations.
uv run --no-sync python -m laya_vision_stitch.decoder_probe \
  --steps 4000 --batch-size 1 --output artifacts/my-decoder-single

# Balanced batches, matched 4,000 presentations.
uv run --no-sync python -m laya_vision_stitch.decoder_probe \
  --steps 500 --batch-size 8 --output artifacts/my-decoder-batch

# Larger decoder-only optimization budget: 32,000 presentations.
uv run --no-sync python -m laya_vision_stitch.decoder_probe \
  --steps 4000 --batch-size 8 --output artifacts/my-decoder-long

# Refine that decoder without updating the frozen path.
uv run --no-sync python -m laya_vision_stitch.decoder_probe \
  --bundle artifacts/my-decoder-long/bundle --steps 2000 --batch-size 8 \
  --learning-rate 0.00001 --output artifacts/my-decoder-refined
```

The first three start fresh from the same checkpoint with seed 17 and AdamW at
0.0001; the fourth continues the third at 0.00001. All use zero weight decay and
gradient clipping at 1.0. Balanced sampling uses replacement.
The matched comparison equalizes presentation count, not optimizer updates or
wall time. It does not isolate every aspect of the minibatch optimization change.

Dataset: D2E-480p revision `f075f7e25df6f6d385840a836f86bf92dfb877ff`,
CC BY-NC 4.0. Source media, feature caches and model weights remain local.
