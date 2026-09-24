# Latent action models and latent world models

Offline experiments on frozen Open-P2P features: a small LAPA/Genie-style latent action
model (LAM), a scarce-label utility test, a probe on unlabeled Hordes live-trial frames and
action-conditioned latent world models. **No keyboard or mouse input was sent, no live
trial was run and no policy-improvement loop was attempted.** Validation selected every
checkpoint, penalty and action lag; test splits and held-out trials never did.
Numbers are in [`latent-models-results.json`](latent-models-results.json).

**Summary.** The LAM latents detect *that* the view is changing (mouse motion on P2P,
movement keys on Hordes) but not *which* action caused it. Pretraining on them does not
help scarce-label action learning. No world model shows meaningful action controllability
on held-out data, so RL in imagination is not justified here.

## Data

- **P2P:** `artifacts/seqcache-p2p-001`. This is public human gameplay at 20 FPS in
  32-frame windows. Train has 7,520 frames from 9 games; validation has 512 (games also
  seen in train); test has 768 from 3 held-out games; fresh_test has 1,024 from 2 unseen
  games. z is the frozen 1024-D image token, standardized with train statistics only.
  One-step copy-last MSE is about 0.15 per dimension (unit variance).
- **Hordes:** `artifacts/latent-hordes-cache-001`, built from three live trials:
  `hordes-p2p-live-001` (688 frames), `hordes-temporal-live-002` (840) and
  `hordes-p2p-live-003` (905), each captured at about 14 Hz.
  - **Frames:** the runner saves the logged 1280×720 viewport after cropping, so no
    further crop is needed.
  - **Actions:** each frame's action is the dispatched `bounded_action` when `applied`,
    otherwise empty. `Pulse.apply` mutates it in place (playfield mouse clipping) before
    the record is written, as checked in `temporal_live.py` for both runner versions.
  - **Features:** frozen vision from `laya-p2p-bridge-001`.
  - **Contexts:** teacher-forced `sequence_contexts` over non-overlapping 32-frame windows,
    with the dispatched tokens as history and the trial goal passed through the bridge.
    This matches the P2P cache windowing.
  - **Evaluation:** two folds hold out live-003 or live-001 entirely. Validation is the
    last 20% of each training trial, and fit rows end before it.

## Architecture

| Model | Design | Parameters |
|---|---|---:|
| LAM | Encoder LN(z_t, z_t+1, Δz) → MLP 512 → 4 tokens × 16-D. One 8-code VQ codebook per token (4,096 combinations). Straight-through; codebook + 0.25·commitment; dead codes (running usage < 2%) replaced by batch encodings every 100 steps. Decoder MLP(latents) → Δz, residual on z_t, zero-initialized (starts at copy-last). 10% latent dropout makes "zeroed latents" a fitted action-free predictor. | 2.70 M |
| LAPA head | LN → 1024→256 → 256; outputs are a 24-D action vector (BCE on 22 controls + MSE on tanh symlog mouse) or 4×8 code logits | 0.34 M |
| World model, MLP | s₀ from (c_t, z_t) (`mlp_full_content`) or learned (`mlp_action_only`); s_h = s_h−1 + g(s_h−1, a_t+h−1); ẑ_t+h = z_t + out(s_h); horizon 4 | 2.38 M / 1.33 M |
| World model, ridge | Closed form per horizon: Δz_t+h = [c_t, z_t, a_t..a_t+h−1]·W_h, one penalty chosen on validation | linear |

Every action-conditioned world model has an **action-blind twin** trained on the same rows.

The decoder sees **no z_t**. Validation chose this after two failures on the 235 training
clips. With full z_t (`latent-actions-001`), validation MSE rose from 0.145 to 0.198 after
250 steps and to 0.531 by the end. Validation kept step 0, and the codes collapsed to two
combinations. A 32-D projection of z_t also overfit. Two diagnostics explain why: ridge
from the true action vector to Δz gains nothing (0.1452 → 0.1452), and the top 4
principal components of Δz hold only 5% of its variance.

## Experiment A: LAM on P2P (`latent-actions-002`)

The LAM trained 3,000 steps and validation selected step 2,250. It uses 1,321 code
combinations on train and 217–336 on the evaluation splits.

| Split | Copy-last MSE | Zeroed latents | Full LAM | Full / copy |
|---|---:|---:|---:|---:|
| validation | 0.1452 | 0.1452 | 0.1406 | 0.968 |
| test | 0.1499 | 0.1499 | 0.1479 | 0.987 |
| fresh_test | 0.1601 | 0.1601 | 0.1568 | 0.979 |

**NMI** is computed between the code combination and the action at t (lag 0, chosen on
validation). Values are shown as NMI / NMI after permuting the codes; the permutation
estimates the bias from many small clusters.

| Split | Button set | Mouse direction (none + 8 sectors) |
|---|---:|---:|
| validation | 0.296 / 0.214 | 0.305 / 0.205 |
| test | 0.273 / 0.214 | 0.275 / 0.179 |
| fresh_test | 0.377 / 0.235 | 0.287 / 0.172 |

**Linear probes** are trained on train and evaluated on 64-D quantized embeddings. Cells
show balanced accuracy, with accuracy / majority-class accuracy in brackets. The reference
column is the same probe on 256 PCA components of Δz.

| Fact | Validation | Test | Fresh | Δz reference (val / test / fresh) |
|---|---|---|---|---|
| Any mouse motion | **0.72** (0.77/0.69) | **0.67** (0.73/0.65) | **0.68** (0.72/0.67) | 0.58 / 0.57 / 0.56 |
| Sign of mouse x (moving frames only) | 0.52 | 0.52 | 0.50 | 0.52 / 0.57 / 0.51 |
| W held | 0.70 (0.81/0.79) | 0.48 | 0.49 | 0.54 / 0.51 / 0.47 |
| mouse_left held | 0.50 | 0.50 | 0.50 | 0.50 / 0.50 / 0.51 |
| Any key onset | 0.50 | 0.50 | 0.50 | 0.60 / 0.50 / 0.51 |

The latents encode "the camera is moving", and do so better than linear Δz, but not its
direction. W transfers only to the validation games, which also appear in train.

### LAPA scarce-label utility (`latent-lapa-001`)

Two setups are compared:

- **Scratch:** a head trained on 10% of train labels. That is 730–736 frames, about 10%
  of each game's sequences.
- **LAM-pretrained:** the same head, first pretrained to predict LAM codes on all train
  frames without labels, then fine-tuned on the same labels.

Cells show button F1 / onset F1, as the mean of 3 seeds with greedy thresholding. Seed
std is 0.01–0.08.

| Input, labels | Method | Validation | Test | Fresh |
|---|---|---|---|---|
| contexts, 10% | scratch | 0.916 / 0.104 | 0.866 / 0.122 | 0.689 / 0.122 |
| contexts, 10% | LAM-pretrained | 0.887 / 0.044 | 0.842 / 0.075 | 0.597 / 0.198 |
| contexts, 100% | scratch | 0.962 / 0.158 | 0.916 / 0.080 | 0.841 / 0.144 |
| contexts, 100% | LAM-pretrained | 0.959 / 0.118 | 0.918 / 0.105 | 0.845 / 0.155 |
| image token, 10% | scratch | 0.629 / 0.035 | 0.596 / 0.023 | 0.161 / 0.067 |
| image token, 10% | LAM-pretrained | 0.623 / 0.034 | 0.614 / 0.021 | 0.204 / 0.070 |
| — | repeat previous action | 0.955 / 0.000 | 0.916 / 0.000 | 0.859 / 0.000 |

With contexts and 10% labels, pretraining lowers button F1 on every split. Its onset F1
is lower on validation and test and higher on fresh, but within about one seed std. The
image-token variant is truly label-free, because it uses no teacher-forced action
history. Its only gain is +0.04 button F1 on fresh_test, about one std. Code-prediction
pretraining stopped improving on validation after 100–300 steps. Repeating the previous
action still beats every head on button F1.

## Experiment B: unlabeled Hordes frames (`latent-hordes-actions-001`)

Validation chose lag 1 in both folds: the latent for pair (t, t+1) is compared with the
action dispatched after frame t−1. This matches dispatch timing, where the first event
is posted about 60–80 ms after capture and frames arrive about 70 ms apart. Cells show
test balanced accuracy for the Δz reference / P2P LAM / Hordes-fine-tuned LAM.

| Held-out trial | Test MSE / copy (P2P LAM, fine-tuned) | Any movement key | W | A | S, D, camera drag | Mouse motion | Key onset |
|---|---|---|---|---|---|---|---|
| live-003 (904 pairs) | 1.011, 0.927 | 0.59 / **0.68** / **0.78** (acc 0.80/0.84 vs 0.72) | 0.53 / 0.59 / 0.63 | ≤ 0.52 | 0.50 | ≤ 0.51 | ≤ 0.54 |
| live-001 (687 pairs) | 0.994, 0.936 | 0.56 / **0.67** / **0.63** (acc 0.72/0.70 vs 0.60) | ≤ 0.54 | ≤ 0.53 | ≤ 0.53 | ≤ 0.51 | ≤ 0.54 |

Held-out-trial results:

- **Movement:** the P2P LAM, which never saw Hordes, detects "some movement key is held"
  from frame pairs of the held-out trial.
- **Fine-tuning** on about 1.2–1.4k unlabeled Hordes pairs lowers next-token MSE 6–7%
  below copy-last. It raises movement detection on one fold and lowers it on the other.
- **At chance:** individual keys, mouse motion, camera drag and key onsets. Most dispatched
  mouse motion only moves the cursor, which is not in the capture. Camera drags occurred
  on only 31 frames.
- **NMI** exceeds the permutation reference for button sets (e.g. 0.336 vs 0.180), but not
  for mouse direction.

## Experiment C: action-conditioned world models

Controllability compares MSE under true actions with MSE under the action sequence of a
random other frame of the same game or trial (5 draws). Relative gain is
(shuffled − true) / shuffled, and the 95% CI comes from a sequence bootstrap.

**P2P** (`latent-world-model-002`). Validation selected ridge penalty 3·10⁴. Both MLP
variants kept step 0, the identity, which equals copy-last.

| Split | Horizon | Copy-last | Ridge, true actions | Ridge, shuffled | Ridge, action-blind | Relative gain |
|---|---|---:|---:|---:|---:|---:|
| validation | h1 / h4 | 0.1438 / 0.3785 | 0.1415 / 0.3634 | 0.1415 / 0.3633 | 0.1415 / 0.3634 | −0.00001 / −0.00009 |
| test | h1 / h4 | 0.1499 / 0.4479 | 0.1473 / 0.4263 | 0.1473 / 0.4263 | 0.1473 / 0.4263 | 0.00000 / 0.00004 |
| fresh_test | h1 / h4 | 0.1601 / 0.4787 | 0.1571 / 0.4524 | 0.1571 / 0.4525 | 0.1571 / 0.4525 | 0.00001 / 0.00014 |

Actions do change how much z moves. One-step copy-last error is 3.8–4.5× larger on frames
with mouse motion than without. But no model predicts the direction of the change from
actions.

**Hordes** (`latent-hordes-world-model-001`, held-out trial, h1 / h4). Hordes contexts
come from 32-frame teacher-forced windows (see Data).

| Held-out | Copy-last | P2P ridge zero-shot | Hordes ridge (penalty 10⁵) | Relative gain, Hordes ridge | MLP variants |
|---|---:|---:|---:|---:|---|
| live-003 | 0.0272 / 0.0610 | 0.0288 / 0.0722 | 0.0271 / 0.0610 | 0.00000 / 0.00004 | step 0 selected (identity) |
| live-001 | 0.0350 / 0.0810 | 0.0369 / 0.0953 | 0.0349 / 0.0810 | 0.00000 / 0.00008 | step 0 selected (identity) |

Dispatched movement keys make the next change 4.4–4.9× larger, and mouse motion
1.5–1.8×. The largest Hordes gain, 0.008%, is statistically nonzero at h2–h4 but
negligible.

### Is RL in imagination (Dreamer-4 style) justified?

No:

- **P2P:** the best world model (validation-selected ridge) improves on copy-last by only
  1.6–5.5% on held-out games. Replacing the executed actions with another frame's changes
  its error by at most 0.014%. The MLP variants never beat copy-last on validation.
- **Hordes:** on held-out trials the action gain is at most 0.008%, and the P2P model
  transferred zero-shot is worse than copy-last.
- **Consequence for imagination:** actions visibly change how much the frozen token moves
  (4–5× larger steps), but not in any predictable direction. An imagined rollout would
  return nearly the same future for every action sequence. A policy optimizer would get no
  signal about which action is better, and could only exploit model error.
- **What would be needed first:** a representation in which action effects are predictable
  (for example the 12×12 spatial grid or pixels), far more action-labeled target-game
  data than 2.4k frames from three failed-policy trials, and a controllability gain that
  holds up on held-out trials.

## Limitations

- **Representation:** only the pooled 1024-D token was modeled; spatial features were not
  used. MSE is averaged over all 1024 standardized dimensions, most of which change
  unpredictably.
- **Input history:** P2P contexts are teacher-forced with recorded previous actions, so
  the context-input LAPA variant is not strictly label-free. The image-token variant is.
- **Hordes data:** the trials are about 14 Hz (P2P is 20 Hz), come from rollouts of a
  failed policy, and have narrow action distributions. `temporal-002` holds only W and
  left-click, and camera drags are rare. JPEG frames and a one-frame dispatch lag add
  noise.
- **Statistics:** one LAM seed; LAPA has three. NMI with hundreds of code combinations is
  biased upward, so compare it with the permutation reference. "Chance" is the evaluation
  split's own majority class.
- **Tuning:** decoder context (LAM) and content (world model) were chosen on validation
  after the full-view runs. `latent-actions-001` and `latent-world-model-001` are those
  superseded full-view runs, kept as ablations. The Hordes ridge penalty sits at the top
  of its grid; larger penalties would only move it closer to copy-last.

## Reproduce

These commands send no input; each finishes in minutes on the M3 Max.

```sh
PYTHONPATH=. .venv/bin/python scripts/train_latent_actions.py p2p \
  --cache artifacts/seqcache-p2p-001 --output artifacts/latent-actions-new
# full-z_t decoder ablation: add --context -1
PYTHONPATH=. .venv/bin/python scripts/train_latent_actions.py lapa \
  --cache artifacts/seqcache-p2p-001 --lam artifacts/latent-actions-002 --output artifacts/latent-lapa-new
PYTHONPATH=. .venv/bin/python scripts/train_latent_actions.py encode-hordes \
  --bundle artifacts/laya-p2p-bridge-001/bundle --output artifacts/latent-hordes-cache-new \
  --trials artifacts/hordes-p2p-live-001 artifacts/hordes-temporal-live-002 artifacts/hordes-p2p-live-003
PYTHONPATH=. .venv/bin/python scripts/train_latent_actions.py hordes \
  --lam artifacts/latent-actions-002 --hordes-cache artifacts/latent-hordes-cache-001 \
  --output artifacts/latent-hordes-actions-new \
  --trials hordes-p2p-live-001 hordes-temporal-live-002 hordes-p2p-live-003 \
  --test-trials hordes-p2p-live-003 hordes-p2p-live-001
PYTHONPATH=. .venv/bin/python scripts/train_latent_world_model.py p2p \
  --cache artifacts/seqcache-p2p-001 --output artifacts/latent-world-model-new
PYTHONPATH=. .venv/bin/python scripts/train_latent_world_model.py hordes \
  --init artifacts/latent-world-model-002 --hordes-cache artifacts/latent-hordes-cache-001 \
  --output artifacts/latent-hordes-world-model-new \
  --trials hordes-p2p-live-001 hordes-temporal-live-002 hordes-p2p-live-003 \
  --test-trials hordes-p2p-live-003 hordes-p2p-live-001
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_latent_models.py
```

Code: `laya_vision_stitch/latent_actions.py`, `laya_vision_stitch/latent_world_model.py`,
`scripts/train_latent_actions.py`, `scripts/train_latent_world_model.py`.
