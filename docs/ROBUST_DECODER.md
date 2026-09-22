# Action learning and instruction robustness

This continuation isolates button learning, tests numerical conditioning, and
then trains the small connector and Laya adapters alongside the decoder. The
original Qwen vision and Laya backbone weights remain frozen. All checkpoints
are experimental; the default model and ScreenQuest controller are unchanged.

## What changed

The old action readout received very large Laya activations. An initial sampled
input had hidden RMS 141 and attention logits above five million, with mean
maximum attention probability 1.0. Functional normalization reduced the logits
to about 145 and the mean maximum probability to 0.379. This is an activation
diagnostic, not proof that saturation explains every learning failure.

`normalize_action_context` adds learned LayerNorm before action attention.
`action_context_source="encoder"` reads the last Laya encoder representation
instead of the later choice-scoring representation. Both are opt-in; existing
checkpoints retain their behavior. Laya's choice scores are still computed and
are unchanged by selecting the earlier action context. These experiments do not
validate the choice scores as an alternative action policy.

The resulting inference path is one exported neural model:

```text
Two screenshots -> frozen Qwen vision -> learned temporal connector
                                          |
Goal and controls ----------------------> Laya encoder + small LoRA
                                          |
                                normalized action attention -> physical keys
```

No caption generator, menu detector, prompt rewriting, nearest-reference lookup
or handwritten action rule runs at inference. Training labels and balancing
metadata are excluded from model inputs. The training script checks frozen
backbone hashes and exported/reloaded button probabilities.
Decoder-only normalization runs train 845,107 parameters. The adapter runs
train 8,859,955 of 763,712,100 parameters (1.16%); original backbone weights
remain byte-identical. Frozen upstream representations are cached only in
decoder-only runs. Adapter runs recompute Laya contexts after updating weights.

## Data and interpretation

The controlled task is still **hold W or release all buttons depending on the
visible menu and the requested goal**. It is not gameplay imitation. There are
64 training screenshot pairs and 64 separate-session development pairs from
Grounded, Minecraft and Raft. Current frames have assistant-reviewed menu labels,
bound to image hashes. The image pairs and development sessions are reused from
earlier diagnostics; they are not an untouched benchmark.

Every scene is paired with opposite goals and both orders of generic act/wait
options. Options contain no semantic hold/release answer. Original demonstration
actions, descriptions, answer labels and action history are removed from inputs.
Training uses four equally weighted menu-state/goal strata. Button BCE gives
changing keys and suppression of unused keys equal weight. Mouse, pointer and
duration heads receive no new supervision. Their fixed weights do not imply
fixed outputs because the shared representation changes.

Curriculum versions:

| Version | Training wordings | Training examples | Development wordings | Reserved wordings |
|---|---:|---:|---:|---:|
| 1 | 7, hold-condition first | 1,792 | 3 | 2 |
| 2 | 14, both clause orders | 3,584 | 5 | 2 new |
| 3 | 26, including unless/negation | 6,656 | 5 | same 2 new |

Expanded examples share screenshots and are not independent demonstrations.
Version 1's reserved wordings became development material after their first
evaluation. Versions 2 and 3 use a new reserved pair, evaluated only after
training and development gates pass. Reserved wording uses the same development
images, so it tests wording novelty only.

The optional matched sampler forms each minibatch from a single wording and
option order: one open-menu scene, one closed-menu scene, and both opposing
goals for each. This changes training sampling, not the inference model.

## Gates

Training requires 95% exact action sets, 95% balanced accuracy and 90% both-goal
success. Every development wording requires 85% exact, 80% balanced and 75%
both-goal success. Overall balanced accuracy must also exceed within-game
shuffled-image accuracy by 15 percentage points. Reserved wording must meet the
same per-wording thresholds before a new-game audit is allowed.

All thresholds remain fixed. Report ordinary accuracy alongside balanced and
both-goal metrics: menu-closed frames dominate the old development split.
Blank visual features are an out-of-distribution ablation. Neither blank nor
shuffled results alone prove causal understanding.

## Experiments

All runs use AdamW, zero weight decay, clipping at 1.0 and seed 29. Decoder-only
runs use batch 8; adapter runs use batch 4. Each run starts a new optimizer.
Source checkpoints and manifests are recorded before training.

| Run | Change | Updates | Train exact / balanced | Development exact / balanced |
|---|---|---:|---:|---:|
| 001 | Frozen upstream, original readout | 4,000 | 60.7% / 58.1% | 44.3% / 45.0% |
| 002 | Normalize decoder input | 4,000 | 66.2% / 60.1% | 58.7% / 55.3% |
| 003 | Encoder context, fresh normalized decoder | 4,000 | 57.9% / 64.7% | 56.9% / 60.5% |
| 004 | Train connector + LoRA + decoder, buttons only | 1,000 | 99.1% / 99.5% | 93.6% / 92.5% |
| 005 | Continue 004 with inverse-clause curriculum | 2,000 | 99.7% / 99.8% | 53.5% / 49.1% |
| 006 | Continue 005, 26 wording templates | 2,000 | 61.3% / 73.9% | 59.5% / 58.8% |
| 007 | Continue 006, matched scene/goal minibatches | 4,000 | 99.4% / 99.7% | 68.4% / 69.6% |

Runs 001–003 independently start from `decoder-probe-004`; 004 starts from 003;
005 starts from 004. Learning rates are 0.00003, 0.00003, 0.0001, 0.0001 and
0.00005 respectively. These changes are exploratory, not a factorial ablation.
Runs 006 and 007 use 0.00005 and 0.00003 respectively. Their batch size remains
four. The matched-sampler run also changes learning rate and training budget, so
its improvement cannot be attributed solely to matching.

Run 004 passes training and development gates and collapses to 51.4% balanced
accuracy with shuffled images. However, its two reserved wordings score **75.0%
and 7.0% exact**. The second puts the release condition first, unlike all seven
training templates. That result exposes a clause-order shortcut and disqualifies
the checkpoint from new-game/live control evaluation.

Run 005 fits the larger training set but fails development wording, despite
95.7% exact accuracy on the familiar template. More training examples alone did
not produce reliable instruction following. None of these results establishes
combat, camera control, clicking, navigation or general game understanding.

Run 007 recovers training fit across the 26 templates, and the familiar wording
gets 94.9% exact / 91.5% balanced accuracy on separate sessions. However, its
four development wordings get only 57.0%, 82.4%, 51.2% and 56.6% exact accuracy.
Both-goal success over all development prompts is 37.8%. This is still wording
memorization, not robust instruction following. The two new reserved wordings
and the fresh-game test remain unscored; no quality threshold was lowered.

### Numerical and text-state audits

The reproducible eight-input attention audit confirms saturation on every
sample: raw mean maximum attention probability is 1.0 and entropy 0.0. Input
normalization brings maximum-probability means into 0.295–0.487 and entropies
into 1.796–2.131. Raw logit magnitudes reach 1.27–5.95 million. The audit applies
functional normalization to the original checkpoint; it does not measure trained
checkpoint accuracy or claim normalization alone solved learning.

`goal_oracle` supplies the correct menu state as **text**, removing visual errors.
It tests nine non-reserved wordings, both menu states, both opposing goals and
both option orders (72 cases). Native English Laya scores 54.2%; the pinned
Qwen3.5-4B quantized model scores 70.8% by single-prefill A/B logits with thinking
disabled. The models receive their own native input formats, so this is not an
architecture-controlled comparison. It does not establish what Qwen would score
with generation, reasoning, different prompts or larger models. Neither model
is a reliable ground-truth teacher in this specific configuration.

An additional representation probe encodes each instruction through frozen Qwen,
averages the last four final-layer language states and fits a linear ridge readout
to which menu condition should trigger W. The ridge coefficient is fixed at 0.01;
there is no development-set parameter search. It scores 92.3% on 52 training
instructions and 62.5% on eight development instructions. No images or actions
are tested. This small probe does not establish that a nonlinear Qwen-to-Laya
goal bridge would fail, but it gives no evidence for adopting this simple one.
The new reserved wording pair remains untouched.

```bash
uv run --no-sync python -m laya_vision_stitch.goal_oracle \
  --backend laya --output artifacts/my-text-state-laya.json
uv run --no-sync python -m laya_vision_stitch.goal_oracle \
  --backend qwen --output artifacts/my-text-state-qwen.json
uv run --no-sync python -m laya_vision_stitch.goal_feature_probe \
  --output artifacts/my-goal-feature-probe.json
```

## Fresh-game protocol

`configs/d2e-fresh-games.json` pins two recordings from Monster Hunter Wilds and
Core Keeper, games absent from this project's training. Foundation-model
pretraining overlap is unknown. Each recording contributes 32 uniformly spaced
current frames with preceding screenshots. The assistant reviewed all 64 current
frames before any model scoring; exact current-frame duplicates are removed by
the preparation tool. Nearby frames remain strongly correlated. There are 54
menu-open and 10 menu-closed scenes, so balanced scores are essential.

The tracked review is `annotations/fresh-games-menu-001.json`. Each game must
reach 85% balanced action accuracy and 80% both-goal success. This is a small
cross-game menu diagnostic, not an evaluation of playing either game. The audit
refuses to run if the source checkpoint has failed earlier gates.

## Reproduction

Use the source data and checkpoints from [DECODER_PROBE.md](DECODER_PROBE.md).
From the repository root, use fresh output directories:

```bash
uv run --no-sync python -m laya_vision_stitch.goal_curriculum \
  --version 1 --output artifacts/my-goals-v1
uv run --no-sync python -m laya_vision_stitch.robust_decoder \
  --data artifacts/my-goals-v1 --encoder-context --normalize-action-context \
  --reset-decoder --steps 4000 --learning-rate .0001 \
  --output artifacts/my-encoder-decoder
uv run --no-sync python -m laya_vision_stitch.robust_decoder \
  --bundle artifacts/my-encoder-decoder/bundle --data artifacts/my-goals-v1 \
  --train-adapters --steps 1000 --learning-rate .0001 \
  --output artifacts/my-adapter-buttons
uv run --no-sync python -m laya_vision_stitch.attention_probe \
  --output artifacts/my-attention-audit.json
```

Generate versions 2/3 with `goal_curriculum --version 2` / `--version 3`.
Continue small-adapter training with `--train-adapters`; add `--matched-pairs`
for the counterfactual minibatch sampler. Do not treat a command completing as
passing quality gates: inspect its report, including reserved wording.

Fresh-game import and gated scoring:

```bash
uv run --no-sync python -m laya_vision_stitch.d2e_data \
  --config configs/d2e-fresh-games.json --download \
  --output artifacts/fresh-games-001
uv run --no-sync python -m laya_vision_stitch.game_goal_audit \
  --source artifacts/fresh-games-001 --prepare-only
uv run --no-sync python -m laya_vision_stitch.game_goal_audit \
  --bundle /path/to/passing-run/bundle \
  --training-manifest artifacts/goal-curriculum-003/train.jsonl \
  --output artifacts/my-fresh-game-audit
```

D2E source revision `f075f7e25df6f6d385840a836f86bf92dfb877ff`, CC BY-NC 4.0.
Source media, model weights and full local reports are ignored by Git.

[Machine-readable results, report hashes and timings](action-learning-002.json).
The continuation also includes [recorded gameplay button experiments](GAMEPLAY_BUTTONS.md).
