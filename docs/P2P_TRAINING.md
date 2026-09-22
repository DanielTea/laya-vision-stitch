# Public gameplay data: P2P pilot

Follow-up: [visual action residuals, numeric history and ten more public
recordings](VISUAL_ACTION_ADAPTER.md). That experiment improves offline button
prediction but still fails visual/camera qualification.

This experiment imports [Elefant AI's public P2P sample](https://huggingface.co/datasets/elefantai/p2p-toy-examples)
into the existing Qwen-vision → connector → Laya → action model. It adds human
mouse-delta supervision alongside physical buttons. It does not use Qwen language
generation at inference, train the full pretrained backbones, or send live inputs.

## Reproduce

```sh
uv sync --extra models --extra stitch --extra data
uv run python -m laya_vision_stitch.p2p_data \
  --download --output artifacts/p2p-pilot-data-002
uv run python -m laya_vision_stitch.p2p_training \
  --bundle artifacts/gameplay-buttons-002/bundle \
  --data artifacts/p2p-pilot-data-002 \
  --output artifacts/p2p-pilot-002
# Follow-up: balance button states; isolate camera fitting from the button path.
uv run python -m laya_vision_stitch.p2p_training \
  --recipe separate --adapter-steps 1500 --decoder-steps 1500 \
  --data artifacts/p2p-pilot-data-002 --output artifacts/p2p-pilot-004
```

The starting bundle is a local research artifact from the recorded-button
experiment, not a separately published downloadable checkpoint. Output directories
must be fresh. The importer downloads approximately 729 MB of raw videos and
annotations; it excludes duplicate 192×192 videos. Large files stay in ignored
`artifacts/` directories.

That parent bundle already learned from D2E's CC BY-NC 4.0 demonstrations.
Its provenance remains part of this experiment; adding P2P does not make this a
checkpoint trained exclusively on P2P data.

Pinned dataset revision: `305ebc891474580f2da295b42ed46ddfad053d96`.
Vendored upstream protobuf schema revision: `a329d98cbe62119679a254d71bea6446773541bc`.
Each imported recording retains its annotation/video SHA-256 hashes. The original
dataset card, including its modified MIT terms and third-party audiovisual-rights
notice, is copied with the imported data. Schema code has its own MIT license in
`laya_vision_stitch/vendor/p2p/LICENSE`.

## What is supervised

| Recording | Source duration | Selected examples | Role |
|---|---:|---:|---|
| Call of Duty Mobile | 122.8 s | 84 | Training |
| GTA: San Andreas | 939.5 s | 256 | Training |
| Roblox: A Dusty Trip | 1,096.5 s | 192 | Whole-game holdout |

The sample has 43,176 annotated frames at 20 FPS. Each example supplies a current
screenshot, a screenshot 200 ms earlier, a weak instruction label, and the previous
recorded action. The target is the next **50 ms** of human buttons and raw mouse
movement. The upstream loader's alignment is preserved: `image[t]` predicts
`annotation[t+1]`; `annotation[t]` is history. Video frame counts must match the
protobuf counts. PTS checks allow the source MP4's 4.5 ms initial mux offset,
but reject a whole missing/duplicated frame.

Instruction segments were generated retrospectively by
`gemini-2.5-flash-thinking-0905`. They are weak labels, not verified human intent
or evidence of successful reasoning. They propagate only within their declared
span. Missing, overlapping, and boundary-crossing labels are excluded. Unknown
human actions, conflicting system actions, scroll, gamepad, unsupported keys and
out-of-range motion are excluded rather than converted into idle actions.
Absolute screen coordinates are never treated as camera deltas.

The training split contains only two left-click examples; the test game also uses
keys absent from training. This sample can exercise the data/training pipeline,
but cannot establish general combat skills. There is one recording per game,
so it cannot provide independent same-game session validation.

## Training and evaluation

1. Warm up the button/camera action decoder for 400 updates on cached frozen
   upstream contexts.
2. Train the connector, last-two-layer rank-8 Laya attention adapters and action
   decoder together for 800 updates, with pretrained weights frozen.
3. Evaluate the final checkpoint, without selecting it by Roblox performance.

Minibatches contain four uniformly sampled training examples. The loss combines
varying-key button BCE and categorical mouse-bin cross entropy. Mouse-bin weights
come only from training frequencies. About 8.89 million parameters are trainable,
approximately 1.16% of the model. Exact frozen-weight hashes and export/reload
equivalence are checked.

Reports include button F1, exact button sets, action-transition accuracy, raw
mouse error, moving-camera error, and camera-direction agreement. Controls include
repeating the previous action, doing nothing, shuffling screenshots within each
game, zeroing visual features, changing goals, and removing action history.
Changing goals measures sensitivity only; it is not a counterfactual correctness
test because the sample has no paired expert demonstrations for alternative goals.

The preregistered pilot check requires held-out button F1 ≥0.70, a ≥0.05 F1
advantage over shuffled screenshots, and improvement over persistence for both
buttons and moving-camera error. This is an offline screen, not a live-game
qualification. Future chunks, pointer positions and duration outputs are not
supervised by this pilot and must not be used to control a game from its bundle.
No new model is automatically deployed, even if this offline check passes.

### Corrective experiment: separate objectives

The first joint run collapsed to W-only buttons and zero camera movement.
Shuffling screenshots, changing goals, or removing action history did not change
its discrete outputs. Held-out button F1 was 0.555, compared with 0.904 for action
persistence. Moving-camera error was 34.54 pixels per axis, compared with 15.49
for persistence. [Full compact report](p2p-pilot-002.json).

The follow-up `separate` recipe changes **training**, not inference rules:

1. Sample uniformly over game/button-set groups and train button BCE only for
   1,500 updates. Freeze camera/pointer/duration heads during this stage.
2. Freeze the entire learned button path, connector and backbones. Train only
   the categorical camera head for 1,500 updates, balancing mouse-bin pairs.

The camera stage checks that upstream and button-head weights are unchanged.
A small-model regression test also checks exact button-logit equality across
camera updates. This follow-up reuses the same Roblox development recording;
it is exploratory, not a new untouched test set. No gradients or checkpoint
selection use Roblox examples.

### Measured follow-up result

| Metric | Training: 340 examples | Withheld Roblox: 192 examples |
|---|---:|---:|
| Button F1 | 0.896 | 0.762 |
| Exact button set | 80.6% | 69.8% |
| Button F1 with shuffled screenshots | 0.846 | 0.746 |
| Button F1 without action history | 0.044 | 0.000 |
| Previous-action baseline F1 | 0.915 | 0.904 |
| Moving-camera MAE, pixels per axis | 12.86 | 53.99 |
| Previous-action moving-camera MAE | 12.49 | 15.49 |
| Zero-motion moving-camera MAE | 23.80 | 34.54 |

The revised training removed the W-only collapse and learned rare training button
sets (93.5% action-set-balanced exact agreement). It did **not** pass the offline
qualification. On the withheld game, the screenshot-shuffle margin is only 1.63
F1 percentage points. Removing action history produces idle predictions on every
example. Camera predictions transfer poorly and are worse than zero motion.
Sensitivity to zeroed features alone is insufficient evidence of useful visual
understanding: real-image shuffling is the more informative comparison here.

Exact pretrained-weight hashes stayed unchanged, camera training preserved the
button path, and export/reload outputs matched. No live inputs were sent and no
controller default was changed. The bundle is
`artifacts/p2p-pilot-004/bundle`; it remains an offline experimental checkpoint.
[Compact report](p2p-pilot-004.json).

This experiment establishes a working public-data training/evaluation path, not
successful Hordes play, full Qwen reasoning transfer, or general game control.
Scaling the same recipe without addressing history dependence, independent
camera learning and demonstration coverage is not yet justified by these results.

### Local inference measurement

On the M3 Max, 32 held-out examples after three warm-ups measured:

| Inference path | Median | p95 |
|---|---:|---:|
| Existing path, images from disk | 99.05 ms | 109.78 ms |
| Both images in memory | 96.83 ms | 107.82 ms |
| Cached historical frame and prompt | 58.31 ms | 67.31 ms |
| Same cache, omit unused choice-head computation | **56.78 ms** | **66.62 ms** |
| Same cache, compiled Laya action path | 57.10 ms | 67.14 ms |

The current screenshot is freshly encoded on every timed prediction. Cached-path
outputs matched the reference exactly in this test. These are **offline inference
benchmarks**, excluding screenshot capture, HTTP, action decoding and event
posting. The cached paths also reuse prepared prompt tokens. They are diagnostic
implementations in the profiler, not a claim that the existing live controller
now runs this model below 60 ms. The complete screenshot-to-input target remains
unverified. [Latency report](p2p-latency-004.json).

```sh
PYTHONPATH=. uv run python scripts/profile_latency.py \
  --bundle artifacts/p2p-pilot-004/bundle \
  --manifest artifacts/p2p-pilot-data-002/validation.jsonl \
  --output artifacts/p2p-latency-004 --samples 32
```

Repository verification after these changes: **76 tests passed**, Ruff passed,
and `git diff --check` passed. These software checks are separate from the failed
gameplay qualification.

## Scaling boundary

The full dataset is packaged in 545 compressed archives; the smallest inspected
archive is approximately 5.60 GB. Its metadata identifies games and recordings,
but does not map recordings to archives. This original pilot downloads only the small
public sample. The follow-up uses explicitly bounded archive prefixes rather than
fetching the 20 TB collection. Meaningful scaling
requires more independent sessions, attack/camera/interaction coverage, and
separate validation recordings, not more adjacent frames from the same clips.

The pinned full metadata contains 108,851 recordings and 133 distinct environment
names (these names are not necessarily 133 distinct commercial games). There are
1,230 Call of Duty Mobile recordings and 270 GTA: San Andreas recordings, which
could support genuine same-game session splits. No environment name matches
Hordes or Warcraft. See the [metadata catalog](p2p-dataset-catalog.json). Metadata
alone does not verify instruction coverage, demonstration quality, or archive
membership.
