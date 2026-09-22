# First small-parameter training pilots

Measured locally on an Apple M3 Max with 48 GiB unified memory using native
arm64 Python 3.13.9 and MLX. [Machine-readable evidence](training-001.json).

## What was implemented and verified

- A goal-conditioned learned connector from Qwen3.5 visual patches into Laya.
- Named-choice supervision and parallel keyboard/mouse/pointer/duration heads.
- Recent-frame and previous-action inputs, with two-frame raw-image inference
  exercised separately after export.
- Full-Qwen offline teacher generation and optional first-token soft targets.
- Optional rank-8 LoRA on the last two Laya attention layers.
- A single exported weights file with tokenizer/config/image processor; reload
  matches outputs within 1e-6. Export excludes Qwen's language decoder.
- Nonzero training gradients, changed new parameters, and exact SHA-256 checks
  showing the original Qwen vision and Laya weights did not change.

The full optional-LoRA checkpoint is 1,513,326,485 bytes. Local bundles and input
data are under ignored `artifacts/`; weights are not committed to Git.

## Data and supervision

Eight training scenes and four validation scenes contain a red and blue circle
at different horizontal locations. Each scene receives two different color
goals with opposite left/right answers: 16 training decisions and 8 validation
decisions. Images and episode IDs are disjoint, with procedural background/location
variation. This is one synthetic environment, not a held-out game evaluation.

Action labels are procedural: left/right button, zero mouse delta, target pointer
coordinate, and a fixed 0.1-second duration. Lower action loss can result from
learning these constants. It does not establish useful control.

The full Qwen teacher's restricted first-token scores answered **9/16** training
questions correctly. Allowing a brief generated visual explanation followed by
an explicit final answer answered **16/16**. Both are recorded; generated labels
remain unreviewed model outputs, not proof of general teacher reliability.

## Results

| Run | Steps | Trainable parameters | Validation choice accuracy | Zeroed-feature accuracy | Warm p50 / p95 |
|---|---:|---:|---:|---:|---:|
| Connector/heads, ground truth | 24 | 814,608 | 50% | 50% | 47.3 / 48.9 ms |
| Connector/heads, ground truth | 512 | 814,608 | 50% | 50% | 45.0 / 46.8 ms |
| Connector/heads, first-token teacher + ground truth | 200 | 814,608 | 50% | 50% | 45.3 / 47.8 ms |
| Connector/heads + LoRA, generated teacher + ground truth | 512 | 912,912 | 50% | 50% | 45.4 / 47.2 ms |

The connector/heads alone comprise ~0.108% of the 755.6M combined parameters;
with LoRA, ~0.121% of 755.7M. Base weights remain frozen in all runs.

The 24-step run reduced mean supervised training loss from 3.986 to 0.966 while
choice accuracy stayed at chance. The final LoRA model predicted left on every
validation example, so **0/4 same-image, different-goal pairs were both correct**.
Its exact button match was 50%. The synthetic pilot has not established learned
visual grounding or goal-following behavior. LoRA support is functional, but
this run does not show that those adapters solve the problem.

Timings are ten repeated warm predictions on one 256×256 image, including image
loading/preprocessing, Qwen vision, connector, Laya and output heads. They exclude
screen capture, event posting and game response. They do not establish the same
latency for 720p images or a four-frame history. Runs are exploratory and differ
in training length/teacher method; this is not a controlled LoRA benefit study.

## Interpretation and next gate

The code path supports small-parameter learning without full-backbone updates.
The initial learned model is still unusable as a game policy. More training time
on these tiny fixtures alone is not justified by a decreasing scalar loss.

The next research gate is an explicit small-data overfit/grounding study: inspect
visual-token scale and alignment, compare direct spatial-token projection with
the query bottleneck, separate grounding from constant action targets, and check
that both image changes and goal changes alter correct decisions. Only after
that gate should varied visual grounding data and multi-game demonstrations be
used for broader training. Complete games must be held out to claim transfer.

No live game inputs were sent. The original ScreenQuest repository was not
changed by this work.
