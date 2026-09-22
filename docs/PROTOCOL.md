# Frozen-model stitching pilot

The question is whether a small algebraically fitted map from frozen visual
features into Laya's input space carries enough state information to reproduce
its decisions. This first study does not establish native general multimodality.

## Interface

The pinned ANE runtime normally performs embedding lookup on the host and passes
`embeddings`, `full_mask`, `local_mask`, `type_vectors` and `marker_map` to Core ML.
The experiment retains that graph and its trained parameters. State tokens occupy
a fixed 19-slot segment, with the existing text choices and question unchanged.
Every state must tokenize to exactly 19 tokens; otherwise the experiment rejects
it. There is no state-dependent length or padding signal. The report compares
this text protocol with the ordinary public Laya text API.

The Qwen vision tower outputs merged image-patch features. The connector sees a
global average and four spatial averages (2x2), concatenated into 12,800 features.
It never runs Qwen's language decoder during stitched inference.

## Fitting and controls

Feature mean and scale are estimated using the fitting partition only. Target
state embeddings are flattened; ridge regression uses a dual solution to avoid a
large dense visual-to-token matrix. Hyperparameter candidates are fixed before
test evaluation. Nearest-image and shuffled-pair controls help expose memorization
and predictions driven mostly by the teacher's most common answer.

The native-Qwen control has the same fixed choices, an image and a static game
instruction. It uses existing output-token logits for the five action labels,
with no explanatory text generation. It still pays for the visual tower and full language prefill.
These scores are not calibrated action probabilities.

## Interface preflight

Before fitting, an initial 32-slot state template exposed padding positions to
attention and changed text decisions. It was discarded. The recorded state
descriptions all tokenize to 19 tokens, so this pilot uses exactly 19 active
positions. Short seven-action options were also replaced with five conditional
options during text-only preflight. These are development choices, not evidence
of improved held-out performance. All five unique text descriptions were checked
for injection parity before fitting; the text labels are therefore not a blind
test of the teacher itself.

The preflight teacher selects attack even for some missing-target and cooldown
states. Its decisions are an imitation target, not a reliable action oracle. A
constant-action control matching it is a failed demonstration of useful visual
grounding, even if raw teacher agreement is high.

## Boundaries

All computation is offline. There are no game inputs, screenshot capture, online
learning, modifications to the original checkpoints or changes to ScreenQuest.
Model artifacts are resolved from pinned cached snapshots. The Laya loader checks
its distributed checksums. Benchmark metadata records revisions and source hashes.

Supervision consists of logged visual state, not human labels. Grounding errors,
range-warning ambiguity and class imbalance remain. The split holds out runs;
it does not ensure distinct locations, days, characters or camera viewpoints.
Any promising result needs independent labeling and new recordings before a live
trial. A failure is still useful evidence against this particular interface and
feature representation; it does not rule out a trained visual connector.
