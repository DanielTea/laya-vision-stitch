# Training with recorded PC gameplay

This pilot continues the learned Qwen-vision → connector → Laya model on recorded
human controls from actual PC games. It measures offline imitation on entirely
withheld games. Agreement with a recorded action is not a gameplay success score.
No real game is installed or controlled by these commands.

**Measured outcome:** the first run failed the transfer checks. Read the
[results and limitations](REAL_GAME_RESULTS.md) before using the checkpoint.

## Source and usage

The source is [D2E-480p](https://huggingface.co/datasets/open-world-agents/D2E-480p),
revision `f075f7e25df6f6d385840a836f86bf92dfb877ff`. It pairs video with timestamped
keyboard/mouse events. The dataset is **CC BY-NC 4.0**; this research checkpoint
retains that noncommercial training-data provenance. Original project code
remains under its existing license. Recorded game content and pretrained weights
retain their respective upstream rights. No video, screenshots or weights are
committed to this repository.

Attribution: Suhwan Choi et al., *D2E: Scaling Vision-Action Pretraining on Desktop
Data for Transfer to Embodied AI* (2025). The source card is retained alongside
the downloaded files. The pinned manifest and source SHA-256 checksums are saved
with every import.

The pilot takes two distinct recording sessions each from Grounded, Raft and
Minecraft. Satisfactory is validation; Barony is the final test. Neither held-out
game contributes gradients. Recordings were selected by file size for a bounded
first run, not sampled representatively from the full dataset. The videos total
about 628 MB; MCAP files add input metadata.

| Split | Games | Examples | Source recording duration |
|---|---|---:|---:|
| Training | Grounded, Raft, Minecraft | 1,907 | 52.93 min |
| Validation | Satisfactory | 159 | 1.62 min |
| Test | Barony | 192 | 23.53 min |

Training additionally replays 192 earlier synthetic examples. These are clearly
identified as synthetic; they do not increase the real-game sample count. Entire
sessions and games, as well as exact screenshot hashes, are checked for leakage.
There is only one recording for each held-out game, so frames are correlated and
these numbers cannot estimate broad cross-game reliability.

## Screenshot/action alignment

- Samples are drawn from stratified timeline intervals without selecting on the
  target action. Human play includes menus, stationary periods and mistakes;
  there are no expert-quality or reward labels.
- MCAP log time is the common clock. Screen records supply video presentation
  timestamps; embedded absolute event timestamps are not mixed into this clock.
- The decoder selects a video frame at or before the recorded timestamp, never
  after it. Samples with either image more than 20 ms old are rejected. Maximum
  accepted offset in this import is about 17 ms.
- Input contains the current screenshot and one about 200 ms earlier, resized
  to at most 320 px on either dimension, plus the preceding 100 ms of actions.
  The next 100 ms of controls is supervision only.
- A button label means held for at least half of the prediction interval. Raw
  mouse deltas are summed within the interval, divided by 512 and clipped to
  [-1, 1]. The audit counts clipped examples. Short taps can be lost with this
  representation; it does not reconstruct every event.
- Mouse buttons come from the dedicated mouse channels. Windows mouse-button
  codes appearing in keyboard snapshots are excluded there to avoid mixing two
  state sources. Keyboard-state audit counts retain these ignored codes.
- Absolute pointer positioning and wheel events are not supervised in this pilot.
  Relative mouse motion may correspond to a camera or cursor, depending on the
  visible game state. Predicted controls are not sent to the OS.

The source has no task-level intent annotations. Each sample therefore receives
`Continue playing <game> and make progress.` This is explicit generic task text,
not recovered human intent. Physical-key descriptions avoid assuming that E,
Shift or other keys mean the same thing across games. This dataset alone cannot
teach arbitrary goal following.

## Training recipe

The input checkpoint is `artifacts/scaled-robust-001/bundle`, from the synthetic
scaling experiment. The button vocabulary is expanded while preserving existing
head rows exactly. Qwen and original Laya tensors remain frozen; the connector,
rank-8 LoRA in the last two Laya layers, and action heads receive gradients.

The pilot fixes 3,000 steps at learning rate 0.00003, seed 17, with batch size one.
Button-positive weights use training frequencies only: square root of negative /
positive frequency, bounded to [1, 10]; unseen training keys get weight one. This
reduces the easy all-negative solution without tuning thresholds on test labels.
The 0.5 inference threshold stays fixed. Original choice/description losses remain
active on replay examples. There are no generated game captions or Qwen language
teacher labels in this run.

```bash
uv sync --extra models --extra stitch --extra data

uv run --no-sync python -m laya_vision_stitch.d2e_data \
  --download --config configs/d2e-pilot.json --source artifacts/d2e-source \
  --output artifacts/my-game-data \
  --replay-manifest artifacts/scaled-recovery-data-001/canonical-train.jsonl

uv run --no-sync python -m laya_vision_stitch.policy train \
  --bundle artifacts/scaled-robust-001/bundle \
  --train artifacts/my-game-data/mixed-train.jsonl \
  --validation artifacts/my-game-data/validation.jsonl \
  --buttons-file configs/desktop-buttons.json --balance-buttons --holdout-games \
  --feature-cache artifacts/vision-feature-cache \
  --steps 3000 --learning-rate 0.00003 --train-eval-limit 128 \
  --output artifacts/my-game-model

uv run --no-sync python -m laya_vision_stitch.game_eval \
  --bundle artifacts/my-game-model/bundle \
  --manifest artifacts/my-game-data/test.jsonl \
  --training-manifest artifacts/my-game-data/mixed-train.jsonl \
  --feature-cache artifacts/vision-feature-cache \
  --reference-bundle artifacts/scaled-robust-001/bundle \
  --output artifacts/my-game-model/test
```

A fresh checkout first needs the [base training recipe](SCALING.md). Each output
directory must be new. The decoder requires PyAV/FFmpeg support provided by the
`data` extra. Downloading is limited to the manifest's files; the full D2E corpus
is not fetched. Frozen features are shared by content hash across runs.

## Evaluation

The report compares button exact match and positive-class micro/macro F1. It also
measures mouse MAE and sign accuracy on axes whose target normalized motion is at
least 0.02. Sparse idle buttons must not inflate the headline score through
per-bit accuracy. Rare and absent controls are visible in per-key support counts.

Controls include zero visual features, shuffled two-frame histories, and removal
of prior-action text. Baselines are no input, training-frequency buttons, repeating
the previous action, and the pre-game-training checkpoint. A strong persistence
baseline is expected at a 100 ms horizon; beating a no-action baseline alone does
not establish useful visual decisions.

The report separately scores button transitions where the next recorded button
set differs from the previous interval. This exposes failures that are hidden
when most controls simply persist.

Inference timings include preprocessing and both screenshot encodings, but exclude
screen capture, input posting and game response. All this evaluation is offline.
Further useful evidence requires task/goal labels, larger session diversity,
short-tap and wheel modeling, and actual closed-loop tasks with reward/success.
