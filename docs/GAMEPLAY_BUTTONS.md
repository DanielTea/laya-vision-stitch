# Recorded gameplay: fitting succeeds, transfer remains unresolved

The normalized encoder readout can now learn recorded physical button sets from
real screenshots. With the connector and small Laya adapters trainable, it fits
61/64 training clips exactly (95.3%), with button micro-F1 of 97.6%. The original
Qwen and Laya backbone weights remain unchanged.

**This is a fitting result, not a working gameplay agent.** On separate recording
sessions, the same checkpoint scores only 20.3% button F1. Repeating the recorded
previous action scores 90.1%. The experimental model is not deployed, no game
inputs are sent, and camera, clicking, future action chunks and goal robustness
are not validated by these button-only experiments.

## Protocol

The source `robust-decoder-003` provides the normalized action readout over Laya
encoder tokens. Qwen vision is frozen throughout. Run 001 trains only the button
decoder, with the entire upstream path frozen. Subsequent runs also update the
temporal connector and rank-8 LoRA in the last two Laya encoder layers. They train
8,859,955 parameters (1.16% of the exported model), without updating the original
backbones. Neither Qwen text generation nor an external controller decides the
actions at inference.

Only first-step button BCE is optimized. Language, camera and preservation losses
are absent. Mouse, pointer and duration head weights are frozen, but their outputs
can change through shared representations. The experiments do not validate those
outputs. Decoder contexts are recomputed after adapter updates; frozen-only runs
can reuse contexts. Every completed run checks frozen weight hashes and compares
button probabilities after export/reload.

Training and development use separate recording sessions in Grounded, Minecraft
and Raft. The previous action is excluded from the model's prompt and retained
only as a comparison. Goals remain generic “continue playing” instructions, not
expert annotations of intent. Predicting a recorded player's exact next control
from this input may be ambiguous; these measurements do not isolate all reasons
for the generalization failure.

The button-only fitting gate requires F1 >= 0.95 and exact button sets >= 0.90.
This is a **subset** of the earlier full gameplay fitting gate, which also required
camera accuracy. The transfer gate requires F1 >= 0.70, a 0.10 margin over shuffled
images, and performance above recorded-action persistence. Thresholds remain fixed.

## Results

| Run | Training clips | Updated components | Updates | Training F1 / exact | Development F1 / exact |
|---|---:|---|---:|---:|---:|
| 001 | 64 | Decoder | 4,000 | 40.6% / 29.7% | 32.8% / 21.9% |
| 002 | 64 | Connector, LoRA, decoder | 1,000 | 97.6% / 95.3% | 20.3% / 14.1% |
| 003 | 190 | Same small components | 2,000 | 46.9% / 23.2% | 29.1% / 21.9% |
| 004 | 190 | Same, positive-key loss balance | 1,000 | 62.0% / 22.1% | 25.7% / 4.7% |

001 starts from `robust-decoder-003`; each subsequent run continues the previous
gameplay checkpoint. Learning rates are 0.0001, 0.00005 and 0.00003. Every run uses
a new AdamW optimizer, no weight decay, gradient clipping at 1.0, and seed 41.
The decoder-only run cycles through game-goal/button-set strata one sample at a
time. Adapter runs average four samples drawn from uniformly selected strata.
Run 004 also uses 0.00003. Extra training and a changed objective are confounded;
there is no matched-step unweighted control for that continuation.

For run 002, shuffling training images reduces F1 from 97.6% to 26.8%, showing
that its fit depends on the visual input. On development sessions, real versus
shuffled F1 is 20.3% versus 9.4%. A visual margin is insufficient when the absolute
score is poor. Run 003 gives 29.1% versus 9.1%, still below both the fixed F1
threshold and the 90.1% persistence baseline.

Run 003 tests data expansion after run 002 passes fitting. An additional 128 clips
were selected from the same training sessions using label strata, independently
of model predictions. All 128 current frames were visually reviewed by the
assistant. Two browser/YouTube frames were rejected, leaving 190 training clips;
the 64 development clips are unchanged. Preceding frames were not separately
reviewed. The review verifies content, not whether every recorded human action
was correct. Exact screenshot/goal input pairs have no conflicting button labels
in this selected set, but near-duplicate scenes and unknown intent remain concerns.

The expanded set contains 18 button classes; some occur once. An optional
`--balance-positive-keys` loss weights presses versus releases using their
frequency under the uniform-stratum training sampler. This information affects
training gradients only. It is neither a runtime button rule nor a threshold
adjustment. Unit tests check the weighting against the sampler's distribution.
In run 004 it raises training F1 but worsens transfer and exact action sets.
Development empty-output rate falls from 54.7% to 3.1%, so the change trades missed
presses for excess presses without solving generalization. All four runs fail the
transfer gate. Only run 002 passes the button-only training-fit gate.

Full two-frame inference for run 002 took **120.4 ms median / 132.0 ms p95** over
63 warm calls on an Apple M3 Max with 48 GiB memory. A separate first call was
127.3 ms. These calls include local image loading, image preprocessing, vision
encoding, Laya and action outputs. They exclude model loading, desktop capture,
event posting and game response. No training job was active during this timing
run. It is an offline model benchmark, not measured closed-loop gameplay latency.

## Reproduce

Use the data and source checkpoints described in [ROBUST_DECODER.md](ROBUST_DECODER.md).
Run from the repository root and use fresh output directories:

```bash
uv run --no-sync python -m laya_vision_stitch.gameplay_buttons \
  --output artifacts/my-gameplay-decoder
uv run --no-sync python -m laya_vision_stitch.gameplay_buttons \
  --bundle artifacts/my-gameplay-decoder/bundle --train-adapters \
  --steps 1000 --learning-rate .00005 --output artifacts/my-gameplay-adapters

# Prepare candidates and contact sheets without creating a training manifest.
uv run --no-sync python -m laya_vision_stitch.gameplay_data \
  --output artifacts/my-expanded-data
# The pinned review applies only to this deterministic candidate selection.
uv run --no-sync python -m laya_vision_stitch.gameplay_data \
  --output artifacts/my-expanded-data --review annotations/gameplay-expanded-001.json
uv run --no-sync python -m laya_vision_stitch.gameplay_buttons \
  --bundle artifacts/my-gameplay-adapters/bundle --data artifacts/my-expanded-data \
  --train-adapters --steps 2000 --learning-rate .00003 \
  --output artifacts/my-expanded-buttons
uv run --no-sync python -m laya_vision_stitch.gameplay_buttons \
  --bundle artifacts/my-expanded-buttons/bundle --data artifacts/my-expanded-data \
  --train-adapters --balance-positive-keys --steps 1000 --learning-rate .00003 \
  --output artifacts/my-balanced-buttons

# Full inference, no feature-cache shortcut and no desktop inputs.
uv run --no-sync python -m laya_vision_stitch.policy predict \
  --bundle artifacts/my-gameplay-adapters/bundle \
  --manifest artifacts/learning-gate-002/validation.jsonl \
  --output artifacts/my-inference-timings.json
```

The original D2E import is pinned to revision
`f075f7e25df6f6d385840a836f86bf92dfb877ff`, CC BY-NC 4.0. Media and checkpoints remain
local. Content review is hash-bound and rejects missing, duplicate or changed
entries. Split checks cover example IDs, recording sessions and all frame hashes.

## Pretrained action knowledge

The current stitch uses pretrained vision and language weights but a newly
learned physical-action readout. Better pretrained action knowledge is therefore
a reasonable alternative to repeatedly extending a tiny imitation set. It is
not an implemented or validated fix here.

[NVIDIA NitroGen's model card](https://huggingface.co/nvidia/NitroGen) describes a
493M-parameter SigLIP2/DiT vision-to-gamepad policy trained on 40,000 hours across
over 1,000 games. Its [official repository](https://github.com/MineDojo/NitroGen)
supports Windows game execution and explicitly limits claims about unseen games,
long-horizon planning and end-to-end gameplay. It does not provide the requested
Laya-conditioned keyboard/mouse model or a verified Mac runtime. Reusing its
weights would require a separately measured conversion, control representation
and goal-conditioning experiment, not just adding it to the current server.

[Machine-readable results and source-report hashes](action-learning-002.json).
The code passes 60 tests and Ruff, including frozen-weight protection, attention
normalization gradients, balanced sampling, review integrity and evaluation gates.
