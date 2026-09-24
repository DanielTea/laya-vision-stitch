# Production readiness: latency, memory, shadow mode and promotion gates

This covers runtime engineering for the `laya-p2p-1` checkpoint
(`artifacts/laya-p2p-bridge-001/bundle`). It changes scheduling and caching; the
controls are unchanged and there are still no gameplay rules. **It does not make the
agent a competent Hordes player.** No keyboard or mouse input was sent: every
measurement below is an offline replay of recorded frames.

**All latencies are a contended smoke test.** Another process was training on the
same GPU. The p95 values mostly reflect that contention. The scripts are meant to be
rerun on an idle GPU (see [Reproduce](#reproduce)). Numbers are in
`docs/production-readiness-results.json`.

## What changed

Every behaviour change is opt-in. With no new flags, the serial runner and stream
behave as before.

| Component | Option | Default | Effect |
|---|---|---|---|
| `LayaP2PStream` | goal cache | on | Laya+bridge goal vector computed once per goal string (≤16 cached); bitwise-identical outputs |
| | `max_gap_seconds` / `--max-memory-gap` | 0.1 s | largest screenshot gap that keeps memory (≤10 s) |
| | policy `time_gap_adapter` | absent → no-op | elapsed time residual on the image token |
| | `compile` / `--compile` | off | `mx.compile` with parity self-check and eager fallback |
| | `commit(action)` / `--precommit` | unused | memory update before the next screenshot instead of after it |
| | `temperature` | 1.0 | decoding temperature; 0 is used for parity audits only |
| `temporal_live` | `--pipeline`, `--guard-interval` | off, 50 ms | pipelined scheduler in `live_pipeline.py` |

One error path also changed. An exception after the memory update starts, such as an
unsupported applied action or nonfinite logits, now resets memory. Before, it could
leave a stale pending frame.

## Model latency

`scripts/benchmark_live_pipeline.py model` uses 260 distinct frames from
`hordes-p2p-live-003` and feeds back that trial's recorded applied actions. The four
variants run interleaved, frame by frame, in rotating order. Decoding is greedy, so
tokens and logits can be compared per frame. The timed region covers preprocessing,
goal, memory update, vision, policy and decoding.

| Variant (FP32 bundle) | Growing cache p50 / p95 ms | Full cache (frames 200–259) p50 / p95 ms | Greedy tokens equal | Max \|Δlogit\| |
|---|---:|---:|---:|---:|
| Original path | 52.4 / 378.5 | 39.2 / 236.5 | reference | – |
| Goal cache | 37.8 / 295.8 | 29.5 / 174.3 | 260/260 | 0 |
| Goal cache + compile | 35.7 / 364.1 | 24.7 / 113.6 | 260/260 | 1.6e-5 |
| … + precommit, screenshot path only | 26.1 / 199.3 | 20.4 / 195.8 | 260/260 | 1.6e-5 |

Precommit moves the memory update off the screenshot path. That update took
7.3 ms p50 at full cache (9.0 ms while the cache grows), and it still runs each frame.
A seeded temperature-1 run matched the original path's tokens on all 30 frames.

The compiled path has three parts:

- **Vision+prefix and decoding** are compiled once. Sampling threads MLX's random
  state through the graph, so seeded draws match eager decoding.
- **The policy context** is compiled only for the full 200-frame cache. A
  full-cache mirror of `OpenP2PPolicy.context` passes RoPE an array offset. The
  released mask depends only on relative positions, so one graph serves every later
  frame. While the cache grows, its length changes every frame, and each new shape
  would need a new trace (≈0.9 s contended). That phase stays eager.
- **All graphs are traced at construction**, at live shapes, and compared with eager
  outputs. Construction took 0.7 s (FP32) or 3.3 s (BF16). Relative errors: vision
  prefix 6.4e-7 (FP32) and 3.2e-4 (BF16); context and decoding 0. Any exception or
  parity miss falls back to eager execution and records the reason in `compile_status`.

The remaining non-bitwise difference comes from fused vision kernels reordering
floating-point operations. The per-frame memory update still recomputes the four
prefix tokens. A split key/value update could remove that work, but it would need
model-code changes and was not attempted.

## Pipelined live loop

`laya_vision_stitch/live_pipeline.py` implements three stages:

- **Capture:** a newest-frame slot that drops, never queues.
- **Inference worker:** picks the newest frame, predicts and applies the same
  `bounded_action`.
- **Dispatch worker:** hands the action to a non-waiting transport.

Guards run on the main thread every `--guard-interval`. That thread also pumps the
Cocoa run loop that NSWorkspace focus state and ScreenCaptureKit rely on. Capture,
preparation, prediction, guards and dispatch are all injected.

| Safety property | Serial runner | `--pipeline` |
|---|---|---|
| STOP file, focus/layout, text input, HUD | checked in every iteration and before posting | checked every 50 ms; any failure stops the run and releases inputs immediately |
| Posting permission | synchronous checks | fail-closed veto: posts only if no guard failed and the last full pass began ≤100 ms earlier; otherwise logged as not applied |
| Allowed keys, ±64 px mouse, playfield clip | yes | yes (same functions) |
| Stale frame (>180 ms at pickup / apply) | stop / not applied | same |
| Previous pulse | waits for its 50 ms release | released at once, then the new pulse (`PreemptingPulse`); a stale timer cannot end the new pulse |
| Model memory | commits the applied action | same; inference waits for dispatch feedback or precommits it |

Residual risk: the pipeline can notice a focus, layout or text-input change up to one
guard interval plus one guard pass (about 62 ms) later than the serial runner.
ScreenQuest's mouse-button helpers still re-check focus for each event. Only allowed
keys can be posted, so Enter cannot send chat text.

`scripts/benchmark_live_pipeline.py pipeline` replays frames at 60 FPS. Guards are
sleeps using the live-003 medians: 11.6 ms focus/layout/text, 0.4 ms HUD and 3.8 ms
previous-pulse wait. The serial emulation also charges 11.6 ms for its loop-top guard;
that cost is an assumption. A recording dispatcher posts nothing. Each mode ran for
20 s.

| Model | Mode | Decisions/s | Screenshot→dispatch start p50 / p95 ms | Frame age at inference p50 ms |
|---|---|---:|---:|---:|
| Recorded live trial `hordes-p2p-live-003` | serial | 15.1 | 60.5 / 73.6 | 9.4 |
| Stub, 34.3 ms fixed | serial (emulated) | 16.1 | 59.0 / 66.5 | 8.8 |
| | pipelined | 29.1 | 42.7 / 50.1 | 8.4 |
| | pipelined + precommit (7.3 ms assumed) | 29.1 | 35.1 / 42.8 | 8.1 |
| Real FP32 stream (contended) | serial (emulated) | 17.6 | 53.6 / 62.9 | 9.1 |
| | pipelined | 33.0 | 39.2 / 47.5 | 8.9 |
| | pipelined + precommit | 31.5 | 30.2 / 39.1 | 7.3 |

With a stub at the live inference median, the serial emulation reproduces the recorded
trial within 1.5 ms (p50) and 1 decision/s. Pipelining removes about 16 ms of guard
and pulse wait from each decision. It also roughly doubles the decision rate, because
guards no longer occupy the loop.

A separate `--compile` run showed no clear difference (pipelined 39.7 / 56.3 ms). Those
runs were sequential, not interleaved, and the compiled context only applies after
the first 200 decisions (7–11 s into each 20 s run). The model benchmark above is the
controlled comparison.

Limitations of this benchmark:

- It does not model capture latency, the CPU and GIL cost of native guard calls, event
  delivery or game response.
- Plain `time.sleep` in this process overshot by about 50%, capped near 10 ms
  (34.3 → 44.3 ms). The benchmark therefore uses a precise sleep. Whether the live
  runner's 50 ms release timer overshoots the same way is unmeasured.
- Pipelining keeps the GPU busy nearly all the time, where the serial loop left it
  idle about half the time. The browser game competes for that GPU.
- **The pipelined runner has never run against the game.** Its posting path is tested
  only with fake targets.

## Time-gap-aware memory

By default, a screenshot gap above 0.1 s resets memory. `--max-memory-gap 1.0` (or
`LayaP2PStream(max_gap_seconds=1.0)`) keeps memory across longer gaps.

The time-gap adapter hook works as follows:

- If the policy has `time_gap_adapter`, it receives `dt`: the time since the previous
  frame in memory, as `mx.array([B])`.
- The first frame after a reset gets the nominal 0.05 s. This matches `gaps()` in
  `scripts/train_time_gap_adapter.py`.
- The returned `[B, 1024]` residual is shape-checked and added to the image token
  before `policy.prefix`.

`LayaP2PRuntime.load` does not install the adapter yet. A bundle whose metadata
declares `time_gap_adapter` therefore fails loudly: either strict weight loading
rejects it, or the stream raises "declares a time-gap adapter that is not installed".

Why it matters: `hordes-p2p-live-001` (30 FPS capture) had 161 decision gaps above
100 ms and 162 memory resets. `hordes-p2p-live-003` (60 FPS) had 1 gap and 2 resets.
Keeping memory across gaps without a trained adapter has not been evaluated offline.

## Shadow mode

The live runner's default, without `--execute`, already is a shadow mode:

```sh
PYTHONPATH=. .venv/bin/python -m laya_vision_stitch.temporal_live \
  --bundle artifacts/laya-p2p-bridge-001/bundle \
  --screenquest-root /path/to/jev_wow_control --window CURRENT_WINDOW_ID \
  --reference artifacts/hordes-live-calibration-001/reference.png \
  --crop 0 143 1280 720 --capture-fps 60 --seconds 60 \
  --output artifacts/prod-shadow-NEW          # add --pipeline for the pipelined scheduler
```

- **What it does:** it captures the window, checks the HUD calibration and runs the
  model. It logs every proposal and bounded action with `applied: false`, and saves
  frames plus `summary.json`.
- **What it never does:** it posts no key or mouse events and skips the neutral
  pointer move.
- **Requirements:** Screen Recording permission, a visible Chrome window whose layout
  matches the reviewed calibration, and the ScreenQuest controller lock. Accessibility
  permission and window focus are not required.
- **Memory caveat:** no action is applied, so memory resets on every decision
  (`state_reset: true`). Shadow mode therefore evaluates a memoryless policy.
  Screenshot-to-dispatch latency is empty.
- **Scorecard:** it reads shadow trials through `proposed_buttons`; idle fraction is
  1.0 by definition.

## Closed-loop scorecard

`scripts/closed_loop_scorecard.py` reads trial logs only
(`artifacts/prod-scorecard-001/scorecard.{json,md}`).

- **Control counts:** applied decisions that contained the control.
- **Idle fraction:** decisions that posted no input.
- **Dispatch latency:** measured to dispatch start. The first trial's legacy
  `screenshot_to_post_ms` field holds the same timestamp.

| Trial | Span s | Stop | Decisions | With events | Idle | WASD | Tab | 1–4 | Mouse btn | Cursor | First 1–4 s | Resets | Inference p50/p95 | Dispatch p50/p95 | First event p50/p95 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| temporal-live-001 | 19.9 | complete | 305 | 37 | 0.88 | 17 | 0 | 0 | 5 | 20 | – | 4 | 51.9 / 54.4 | 72.4 / 92.0 | – |
| temporal-live-002 | 59.9 | complete | 840 | 209 | 0.75 | 95 | 0 | 0 | 39 | 113 | – | 111 | 52.7 / 95.8 | 75.7 / 122.8 | – |
| p2p-live-001 | 53.2 | focus lost | 688 | 365 | 0.47 | 273 | 0 | 5 | 59 | 141 | 16.6 | 162 | 33.5 / 63.4 | 78.4 / 103.5 | 81.3 / 108.0 |
| p2p-live-002 | – | no decisions logged | – | – | – | – | – | – | – | – | – | – | – | – | – |
| p2p-live-003 | 59.9 | complete | 905 | 294 | 0.68 | 253 | 0 | 0 | 10 | 52 | – | 2 | 34.3 / 36.9 | 60.5 / 73.6 | 60.7 / 73.8 |

`hordes-p2p-live-002` contains only `config.json` and an empty `frames/` directory.

No decision log records health, XP, target health or kills. The only outcome
signals are manual reviews, from the trials' `summary.json` and
`docs/hordes-temporal-live-002.json`. Every reviewed trial shows no progress: target
health 50/50 at the reviewed points (70/70 for temporal-live-002's final target, the
only value it recorded), player health 244/244, XP 466/1600 and zero verified kills. The only ability presses were in p2p-live-001: 4× "1" and 1× "2",
followed by "Out of range" feedback.

## Promotion gate

`scripts/promotion_gate.py` reads reports shaped `{split: {model_key: {"macro": ...}}}`.
It averages list entries (sampling seeds) and applies these checks to every split
that contains `--model-key`:

- **Button F1:** macro button F1 must be above the same report's `repeat_previous`
  (+ `--button-margin`, default 0).
- **Onset F1:** must be above the baseline report's model (`--baseline-model-key`,
  default the same key) + `--onset-margin` (0.02).
- **Idle false positives:** may be at most the baseline's + `--idle-margin` (0.10).
- **Image shuffle (optional):** a `--shuffle-key` control must score below the
  candidate by more than `--shuffle-margin`.

Optional closed-loop gates use `--scorecard` and `--trial`:

- ability presses ≥ 1;
- idle fraction within [0, 0.9];
- screenshot→dispatch p50 < 60 ms;
- no safety stop.

Exit status is 0 to promote, 1 to reject and 2 for missing or malformed input.

Example (`artifacts/prod-gate-001/`): `cfg_4_greedy` against `greedy`, both from
`seqdecode-p2p-001`. The candidate passes validation and test but fails fresh-test
onset F1 (0.164 versus 0.187). The closed-loop gates, read from live-003, fail on
ability presses (0) and p50 latency (60.5 ms). That pairing only demonstrates the
mechanics: live-003 ran sampling on the BF16 bundle, not `cfg_4_greedy`.

## Reproduce

```sh
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_prod_*.py
.venv/bin/ruff check laya_vision_stitch/laya_p2p_stream.py laya_vision_stitch/temporal_live.py \
  laya_vision_stitch/live_pipeline.py scripts/benchmark_live_pipeline.py \
  scripts/closed_loop_scorecard.py scripts/promotion_gate.py tests/test_prod_*.py
# Idle-GPU rerun: confirm no other MLX process is running, then use new output directories.
PYTHONPATH=. .venv/bin/python scripts/benchmark_live_pipeline.py model \
  --bundle artifacts/laya-p2p-bridge-001/bundle --frames artifacts/hordes-p2p-live-003/frames \
  --events artifacts/hordes-p2p-live-003/events.jsonl --label "idle GPU" \
  --output artifacts/prod-model-latency-002
PYTHONPATH=. .venv/bin/python scripts/benchmark_live_pipeline.py pipeline \
  --bundle artifacts/laya-p2p-bridge-001/bundle --frames artifacts/hordes-p2p-live-003/frames \
  --fps 60 --seconds 20 --label "idle GPU" --output artifacts/prod-pipeline-002/real-fp32
PYTHONPATH=. .venv/bin/python scripts/benchmark_live_pipeline.py pipeline \
  --frames artifacts/hordes-p2p-live-003/frames --stub-model-ms 34.3 --stub-commit-ms 7.3 \
  --output artifacts/prod-pipeline-002/stub-34ms
PYTHONPATH=. .venv/bin/python scripts/closed_loop_scorecard.py artifacts/hordes-temporal-live-00{1,2} \
  artifacts/hordes-p2p-live-00{1,2,3} \
  --review hordes-temporal-live-002=docs/hordes-temporal-live-002.json#visual_review \
  --output artifacts/prod-scorecard-002
PYTHONPATH=. .venv/bin/python scripts/promotion_gate.py \
  --candidate artifacts/seqdecode-p2p-001/report.json --baseline artifacts/seqdecode-p2p-001/report.json \
  --model-key cfg_4_greedy --baseline-model-key greedy
```

None of these commands sends input.

## Idle-GPU rerun

The benchmarks above were repeated after all training jobs finished, with no other MLX
process running: screenshot-to-dispatch p50/p95 is 51.7/60.3 ms serial, 36.2/43.1 ms
pipelined and 29.9/38.1 ms pipelined with precommitted memory (150M FP32). Full-memory
model latency is 42.7 ms original, 32.9 ms with goal caching, 29.9 ms compiled and 23.2 ms
with precommit. Reports: `artifacts/prod-model-latency-002`,
`artifacts/prod-model-latency-300m-001` and `artifacts/prod-pipeline-002`. See
[IMPROVEMENT_EXPERIMENTS.md](IMPROVEMENT_EXPERIMENTS.md#latency-on-an-idle-gpu).

