# Trainable Qwen–Laya model

This path follows the user's updated requirement: reuse pretrained knowledge and
allow small-parameter training, without fully fine-tuning the backbones. Earlier
frozen/retrieval experiments remain reproducible in separate modules.

**Status:** the original pilots below remained at chance. The subsequent
[scaling work](SCALING.md) adds aligned initialization, description supervision,
mixed goals, bounded feature caching and a persistent local API. Use that guide
for the current recipe. [Original pilot results](TRAINING_RESULTS.md) remain
available; no cross-game gameplay capability has been established.

## Model

```text
up to four recent screenshots
  -> frozen Qwen3.5 vision encoder and existing visual merger
  -> spatial patch features + x/y + frame age/order
  -> learned cross-attention connector (16 visual tokens)
       ^ goal / controls / previous actions via frozen Laya embeddings
  -> Laya encoder + original decision head
       ^ same goal / controls / previous actions as ordinary text tokens
  -> pretrained named-choice scorer
  -> new parallel button, mouse, pointer and duration heads
```

Qwen's language decoder is absent from the deployed graph. There is no reference
bank or generated caption at inference. Goals can change between calls. Visual
tokens occupy active, unmasked sequence slots. Prompts/options that overflow the
Laya context are rejected rather than silently truncated. Text uses Laya's
existing tokenizer, choice-question format and embeddings.

The connector attends to prompt embeddings, then to all spatial image features.
It emits vectors in Laya's embedding dimension. Backpropagation passes through
the frozen Laya encoder and decision head into this connector. Frozen vision
features are cached in memory during training, separately for each split.
Caching is an optimization of the same computation; deployment runs vision.

That paragraph describes the original `queries` connector. The active `aligned`
connector instead preserves a fixed spatial grid and learns scene vectors near
Laya's text embeddings. Goal conditioning occurs inside Laya. A `description`
field can supervise these vectors during training; it never enters inference.
The current configuration has ~6.05M trainable parameters, including small LoRA,
rather than the smaller counts for the original defaults below.

Default training changes 814,608 of 755,622,678 parameters (~0.108%). Optional
rank-8 LoRA on QKV and attention-output projections in the last two Laya encoder
layers adds 98,304 trainable parameters (~0.121% total). Original backbone tensors
stay frozen. LoRA currently remains as small residual modules in the single
checkpoint; it is not merged into the original weights.

## Setup and historical smoke run

Use a native Apple Silicon Python interpreter. From this repository:

```bash
uv sync --extra models --extra stitch
uv run --no-sync hf download mlx-community/Qwen3.5-4B-4bit --revision 0e7ffd5c629ef7719d4cbc04069232580bfa9d9c
uv run --no-sync hf download aac6fef/laya-mlx --revision 20aed815fc6acde75733882e7ec0e3f28aeb9717

uv run --no-sync python -m laya_vision_stitch.policy fixtures --output artifacts/my-data
uv run --no-sync python -m laya_vision_stitch.policy teacher \
  --manifest artifacts/my-data/train.jsonl \
  --output artifacts/my-data/train-teacher.jsonl
uv run --no-sync python -m laya_vision_stitch.policy train \
  --train artifacts/my-data/train-teacher.jsonl \
  --validation artifacts/my-data/validation.jsonl \
  --steps 200 --output artifacts/my-training
```

These generated fixtures are small paired red/blue-circle scenes, not gameplay.
Each image appears with two different goals and opposite correct actions. Train
and validation images/episodes are distinct. This is a wiring and optimization
test, not evidence of cross-game generalization.

Teacher mode defaults to `generate`: full Qwen generates short visual evidence
and an explicit final choice. The evidence and exact prompt are retained for
review; only the final choice supervises the student. It is a hard target, not a
calibrated confidence distribution. `--mode logits --temperature 2` instead
exports softened distributions from restricted first-token option logits. It
was substantially worse on the pilot. No teacher output replaces existing
ground-truth answers or recorded actions. Invalid generated final answers fail
with a saved `.error.json` response instead of guessing a label.

Training sums any available choice cross-entropy, temperature-scaled teacher KL,
button binary cross-entropy, mouse/pointer MSE, pointer-active binary
cross-entropy and duration cross-entropy. The original loss terms have equal weight. Optional description alignment adds
a normalized embedding MSE weighted by 10; teacher mistakes can conflict with
ground truth. Review teacher quality first.
There is no hidden automated pseudo-label filtering or claimed transfer of all
Qwen knowledge.

## Data contract

Manifests are JSONL. Image paths are relative to the manifest (absolute paths
also work). Each record describes one decision; no future frames are inputs.

```json
{
  "id": "example-001",
  "game": "my-game",
  "episode": "session-001",
  "frames": [
    {"image": "images/before.png", "age_seconds": 0.1},
    {"image": "images/now.png", "age_seconds": 0}
  ],
  "goal": "Move toward the doorway while avoiding the obstacle.",
  "controls": "WASD moves. Right-drag rotates the camera.",
  "previous_actions": [{"buttons": ["w"], "duration_seconds": 0.1}],
  "choices": {"left": "Move left.", "right": "Move right."},
  "answer": "left",
  "action": {
    "buttons": ["a"],
    "mouse_delta": [0, 0],
    "pointer_xy": null,
    "duration_seconds": 0.1
  }
}
```

- `goal`, `id`, `game`, `episode`, and `frames` are required. Frames are oldest
  first with distinct nonnegative ages; the latest frame has age zero.
- `controls` and `previous_actions` are optional text/list context. Previous
  actions must describe actions before the current decision, not target labels.
- `choices` maps 2–32 labels to descriptions. `answer` names one label. Teacher
  generation supports at most 26 choices because it uses A–Z response labels.
- Training accepts any combination of `answer`, named `teacher_probs`, and
  `action`. Grounding-only data does not train the action heads. Action-only data
  uses a fixed act/wait question to provide Laya's sequence format; those two
  scores are not supervised or used as a hard-coded controller.
- Buttons are simultaneous desired holds, including mouse buttons. Missing
  buttons are negative labels. Mouse deltas are fractions of viewport width and
  height in [-1, 1]. Pointer coordinates are [0, 1]; `null` means inactive.
  Duration must match a configured bin. These conventions require matching
  normalization when collecting demonstrations and decoding outputs.
- New datasets should contain varied games, goals, camera motion and recovery.
  Use `--holdout-games` to require disjoint game identities. ID, episode and exact
  image hash overlap are always rejected across train/validation. This does not
  detect mislabeled episodes or near-duplicate images.

The generic default button vocabulary is intentionally small. For another set,
supply `--config config.json` when creating a fresh training run, for example:

```json
{
  "buttons": ["w", "a", "s", "d", "space", "1", "2", "mouse_left", "mouse_right"],
  "durations": [0.05, 0.1, 0.2, 0.4],
  "visual_slots": 16,
  "connector_width": 128,
  "max_frames": 4,
  "image_width": 320,
  "lora_rank": 8,
  "lora_layers": 2
}
```

Leave `lora_rank` at zero for connector/head-only training. LoRA is optional, not
a full-model fine-tune. Button vocabulary/architecture are checkpoint properties;
control meanings and goals are runtime text inputs. Supporting a new action
dimension requires changing/training the output head, not just a prompt.

## Export, inference and evidence

Every successful training run produces one `bundle/model.safetensors` containing
vision, connector, Laya and action outputs, plus tokenizer, image processor and
configuration. Loading this bundle requires no Qwen language decoder, reference
images or original model cache. No command sends keyboard or mouse events.

```bash
uv run --no-sync python -m laya_vision_stitch.policy predict \
  --bundle artifacts/my-training/bundle \
  --manifest artifacts/my-data/validation.jsonl \
  --output artifacts/my-training/predictions.json

uv run --no-sync python -m laya_vision_stitch.policy evaluate \
  --bundle artifacts/my-training/bundle \
  --manifest artifacts/my-data/validation.jsonl \
  --output artifacts/my-training/evaluation.json
```

Inference manifests may omit supervision. Outputs include choice/button
probabilities, normalized mouse movement, pointer position/activation and a
duration bin. Probabilities are uncalibrated. A threshold of 0.5 produces the
reported button list; it has not been validated for live control.

`before.json`, `steps.jsonl` and `report.json` retain losses, accuracy, exact
button agreement, mouse error, image-zeroing controls, frozen-weight hashes,
parameter counts, checkpoint reload comparisons and warm raw-image timings.
Evaluation also checks all goals in same-scene counterfactual groups together.
Latency excludes capture/input posting and is not a gameplay reaction-time test.

`train --bundle existing/bundle` continues learned weights with a fresh optimizer;
it is not exact optimizer-state resume. Use a fresh output directory for every
run. Training uses batch size one and caches visual features in RAM, so this is
a small-experiment trainer, not a streaming dataset/distributed training system.

## Next research gate

Before collecting expensive game data, demonstrate that the model can overfit a
small visual/goal dataset and outperform image-zeroing and constant baselines.
Then expand supervised grounding, compare connector-only versus selected LoRA,
and evaluate on entire unseen games. The current pilots have not passed the
first semantic gate. Full-backbone fine-tuning is neither implemented nor needed
to run the next experiments.
