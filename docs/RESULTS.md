# Pilot 001: low latency, no demonstrated decision benefit

Measured on 2026-09-22 using an Apple M3 Max with 48 GiB unified memory and
native arm64 CPython 3.13.9. Both neural backbones remained frozen. A ridge
connector was fitted; no game inputs were sent.

The model interface works: Qwen visual features can enter Laya as continuous
state embeddings, without generating text or running OCR at inference time.
This pilot **does not demonstrate useful visual decision making**. Every tested
method picked attack on every held-out frame, including a connector fitted to
shuffled image/state pairs.

## Data and checks

218 existing Hordes screenshots from eight recording runs were split
chronologically: 97 fitting frames from four runs, 64 validation frames from two
runs, and 57 test frames from two runs. Exact image-file duplicates were removed.
Runs are disjoint; locations, camera views and game sessions are not guaranteed
to be independent.

There are only five distinct weak state descriptions. Every frame reports high
health, no frame reports a dead enemy, and only three report an out-of-range
warning. These data cannot evaluate retreat, kill recognition, loot or recovery.
Descriptions were derived from historical OCR/heuristics, not human labels.

Injecting original text embeddings produced **zero maximum logit difference**
from the normal prepared-batch graph. The fixed 19-token protocol also agreed
with the ordinary public Laya text API on all 218 examples. Reloading the saved
connector reproduced the freshly fitted connector's predictions exactly.

The text teacher chose attack for 215 frames and approach for three. It also
chose attack when the weak description reported no target or ability cooldown.
This teacher is not an action-correctness oracle.

## Held-out decisions

The test teacher labels were 55 attack and two approach. All methods below chose
attack 57 times, missed both approach examples, and had the same scores.

| Method | Teacher agreement | Mean recall over present teacher classes |
| --- | ---: | ---: |
| Fitted visual-to-Laya connector | 96.49% | 50.00% |
| Nearest fitting image | 96.49% | 50.00% |
| Constant mean-state embedding | 96.49% | 50.00% |
| Most common fitting teacher label | 96.49% | 50.00% |
| Connector fitted with shuffled pairs | 96.49% | 50.00% |
| Qwen direct multimodal action scores | 96.49% | 50.00% |

The validation teacher labels were all attack. All four regularization values
therefore tied at 100% agreement. The predefined tie-break selected alpha 10.
This is a limitation of validation coverage, not evidence that the selected
regularization generalizes. Qwen is also scored against the weak Laya teacher;
these numbers do not measure either model's gameplay accuracy.

## Warm offline latency

| Component | Median | 95th percentile |
| --- | ---: | ---: |
| Qwen vision features, including image load/preprocessing | 61.03 ms | 61.69 ms |
| Fitted connector | 0.24 ms | 0.28 ms |
| Laya graph and input preparation | 4.46 ms | 4.54 ms |
| Per-frame sum of the three measured stages | 65.78 ms | 66.63 ms |
| Qwen single multimodal prefill, scoring action tokens | 273.76 ms | 275.58 ms |

The summed path is approximately 4.2 times faster in this offline measurement.
Feature extraction and connector/head evaluation were performed in separate
phases. Their per-frame sum excludes live capture, model startup, event posting
and game response. It is not measured screenshot-to-keypress latency or proof of
sustained gameplay frame rate. Input width was capped at 512 pixels and the
processor then applied its patch-grid sizing. No autoregressive text generation
was used in either timing.

A separate saved-artifact replay measured the actual sequential image-to-scores
call on one held-out screenshot. Four warm repetitions took 64.96–66.23 ms;
the first call took 165.03 ms after 28.04 seconds of model/artifact initialization.
This confirms the standalone path runs, but four repetitions of one image are
not an independent performance benchmark.

## Interpretation and next experiment

This establishes a runnable interface and a promising compute budget, but the
stitched model does not beat controls. Simply attaching compatible tensor shapes
has not transferred useful visual semantics in this test. The experiment neither
proves that training-free multimodal fusion works nor rules out other mappings.

Before another connector comparison, assemble separately labeled, diverse scenes
covering missing targets, range, cooldown, low health and enemy death. Verify the
text-only decision task across those conditions first. Freeze a new test set and
then compare a fitted linear connector with a small trained connector while
keeping both backbones frozen. That would be connector training, not a claim of
zero training. No live controller integration is warranted by this pilot.

Aggregate results and dependency versions are in [pilot-001.json](pilot-001.json).
Private screenshots, source paths, extracted features and fitted weights stay
under the ignored local `artifacts/pilot-001/` directory.
