# Status and research findings (2026-09-24)

This page is the single summary of where the project stands and what the experiments
showed. Details, tables and reproduction commands are in the linked reports. Compact
numbers for this round are in
[improvement-experiments-results.json](improvement-experiments-results.json).

## Where things stand

- **Goal.** One local, general game-playing model: screenshot and goal text in,
  keyboard and mouse out, fast enough for real-time play, and not specialized to one
  game. Hordes.io is used only as an unseen live test.
- **Current best setup.** Three parts:
  - a fast controller (stitched Laya–Open-P2P 150M with LoRA trained on 24 public games,
    20 ms per decision);
  - a slow planner (Molmo-7B, about 3 s per answer) that points at goal-relevant objects;
  - a RADIO feature tracker that keeps the planner's target up to date between answers.

  The planner selects a target with one click. With `--planner-act` it also walks to the
  target and clicks the leftmost skill on the skill bar.
- **What is demonstrated.**
  - Imitation of recorded human play improves on unseen games as public training data
    grows.
  - The live runtime reaches 33 ms median from screenshot to first input.
  - In one live Hordes trial the planner selected a monster by clicking it.
- **What is not demonstrated.**
  - The agent has never damaged or killed a monster.
  - The planner's approach and skill actions have not been exercised in play: in both
    live runs Molmo found no monsters.
  - No part of the planner stack has been tested outside Hordes (see
    [Generality](#generality-what-was-tuned-on-hordes)).
- **Main bottlenecks.** (1) Molmo's recall of small, distant objects; (2) acting on a
  selected target, which imitation of human data did not teach; (3) evidence that the
  planner stack generalizes beyond Hordes.

## Current architecture

```text
                       ┌──────────── slow path (own process, ~3 s per answer) ────────────┐
screenshot ────────────┤ Molmo-7B-D 4-bit: "Point to the <goal target>." / skill-bar icons │
   │                   └───────────────┬──────────────────────────────────────────────────┘
   │                                   │ target point, skill point (two answers must agree)
   │                     RADIO template tracker (1024×576, 3×3 patches) keeps the target
   │                                   │
   │                     planner actions: select click · W/A/S/D approach · skill click
   ▼                                   ▼
EfficientNet image token ─► Open-P2P temporal policy (10 layers, 200-frame memory, LoRA)
Laya goal features ─bridge─►            │
                                        ▼
                          action decoder (LoRA) ─► keys, mouse buttons, mouse motion
                          RADIO click head ─► absolute click position on new presses
```

- **Checkpoint.** `artifacts/laya-p2p-general-001` combines the policy and decoder LoRA
  from `policy-lora-captions-001`, the RADIO click head and FP16 C-RADIOv3-B in one
  bundle. Artifacts are local only and ignored by Git.
- **Runtime.** `laya_vision_stitch/laya_p2p_stream.py` and
  `laya_vision_stitch/temporal_live.py` provide goal caching, `mx.compile` with a parity
  self-check, precommitted memory, a pipelined runner with asynchronous guards, a pointer
  mode, the planner and planner actions.
- **Live command** (sends input only with `--execute`):

  ```sh
  PYTHONPATH=. .venv/bin/python -m laya_vision_stitch.temporal_live \
    --bundle artifacts/laya-p2p-general-001 --screenquest-root ../jev_wow_control \
    --window <id> --reference artifacts/hordes-live-calibration-002/reference.png \
    --crop 0 87 1280 720 --capture-fps 60 \
    --goal "Defeat nearby monsters. Avoid attacking players. Retreat when health is low." \
    --pipeline --precommit --compile --pointer --planner --planner-act \
    --wait-for-focus 60 --seconds 60 --execute --output artifacts/<new-dir>
  ```

  The Hordes Chrome window must be 1280×807, so the game viewport is exactly 1280×720
  and includes the skill bar. The window was resized to this on 2026-09-24.

## Findings

### 1. Evaluation

- **Button F1 flatters persistence.** Repeating the previous action scores 0.86–0.96.
  Onset F1 scores only new presses, where the repeat baseline scores 0. Both are
  reported.
- **Public P2P splits are not clean holdouts** for Open-P2P, which was pretrained on the
  same dataset family. On D2E games outside that source, the pretrained policy loses to
  repeating the previous action (button F1 0.851 vs 0.894).
- **Offline evidence is imitation with recorded history** (teacher forcing). None of it
  measures closed-loop success.
- **Live frames need a different kind of check.** With no ground truth on live frames,
  forward–backward tracking drift is a usable consistency check, while Molmo re-pointing
  is too inconsistent to serve as a reference.

### 2. Data and scaling

The data is public D2E-480p (CC BY-NC 4.0) plus Open-P2P replay: 197,632 training frames
from 24 games, with five games held out entirely. All scaling runs use decoder LoRA
selected on validation. Loss is recorded-action NLL on unseen games.

| Training data | Frames | Core Keeper | Stardew Valley |
|---|---:|---:|---:|
| Pretrained | 0 | 2.394 | 2.874 |
| 4 games | ~65k | 2.341 | 2.893 |
| 8 games | ~92k | 2.328 | 2.884 |
| 24 games | ~205k | 2.295 | 2.826 |
| 24 games, LoRA in policy layers too | ~205k | 2.266 | 2.801 |

- **Core Keeper improves roughly log-linearly**, by about 0.028 per doubling of frames.
- **Stardew needed game variety.** It got worse with 4 and 8 games and improved only
  at 24.
- **Policy-layer LoRA** gained about as much as tripling the data.
- **On all five unseen games** the final model lowers NLL by 3–12%. Button F1 is level
  with the persistence baseline.
- **This is not a scaling law.** There are only three sizes, game count and frame count
  move together, and the base is a frozen 150M model with small adapters that overfit
  after about 1,000 updates. The 300M checkpoint gave no gain.
- **Extrapolation is discouraging.** Continuing the Core Keeper trend to a loss near 2.0
  would take about 1000× the data.

### 3. Architecture ideas (offline)

| Idea | Verdict |
|---|---|
| Decoder LoRA on public D2E data | **Positive:** held-out-session NLL −24%, level with persistence |
| Policy-layer LoRA over 24 games | **Positive:** all 5 unseen games improve |
| Open-P2P 300M instead of 150M | No gain |
| Goal classifier-free guidance | Negative; goals carry almost no action information |
| Flow-matching action chunks | Better onsets, worse holds; below the pretrained decoder |
| Real-time chunking | Smoother chunk switches, slightly less accurate |
| Learned slow planner (RADIO residuals) | Negative; overfits, ignores its input |
| Elapsed-time memory | Negative; resetting memory after gaps is better |
| Foveated central crop | No information beyond a duplicate-token control |
| Latent action pretraining (LAPA-style) | Negative; latents detect motion, not the action |
| Latent world model | Negative; true vs shuffled actions change error by ≤0.014% |
| Outcome-weighted imitation (AWR) on Hordes logs | Negative; the damage signal is at chance |
| Hindsight captions as goals | No goal effect; swapping captions changes NLL by ≤0.004 |
| Hindsight click targets as controller input | **Negative, twice** (see section 5) |

Reports: [IMPROVEMENT_EXPERIMENTS.md](IMPROVEMENT_EXPERIMENTS.md),
[LATENT_MODELS.md](LATENT_MODELS.md), [HORDES_AWR.md](HORDES_AWR.md),
[GENERAL_MODEL.md](GENERAL_MODEL.md).

### 4. Pointing and grounding

- **Click-position head** (C-RADIOv3-B patches, 24k real clicks). It beats clicking the
  screen center on 4 of 5 unseen games at a 5% radius (20.0% vs 15.4% overall). On Hordes
  it picks the player character, because shooter data teaches crosshair-center clicks.
- **Grounding teachers.** Qwen3.5-4B and OWLv2 found no Hordes monsters in 3 frames; the
  detector labels the player "an enemy". Molmo-7B found them in 2 of 3 and abstains
  instead of pointing at the player.
- **Goal-conditioned pointer** (3,622 Molmo labels). Agreement with the teacher on unseen
  games is 51% with the correct goal, 27% with a wrong goal and 27% for the screen center.
  The goal text steers pointing. On Hordes it mostly chose the quest arrow.

### 5. Dual system: Molmo planner

- **Planning on demand.** Molmo is asked only when no target is being followed.
  Candidates within 0.08 of the screen center count as the avatar and are skipped.
- **Tracker.** The live click offsets (about 25 px) came from tracking, not from Molmo,
  whose points were on the monsters. Matching a 3×3 patch template at 1024 px instead of
  the mean patch at 768 px changes the numbers as follows:
  - forward–backward drift on real Hordes frames: median 104 px → 13 px (0/7 → 6/7 within
    24 px);
  - synthetic shifts: median 36 px → 9 px;
  - cost: p95 controller latency 80 → 113 ms in replay.
- **Hindsight target conditioning.** The next human click, within 2 s or 10 s, was fed to
  the controller as a point.
  - The model uses whether a target is present, not where it is: another sequence's
    target scores as well as the correct one on unseen games, including at decision
    points.
  - On training games the position effect is small (≤0.013 nats).
  - Human actions toward the next click are already predicted by the image and the
    previous action, so imitation cannot learn "go there". Neither run was bundled.
- **Planner actions.** Asking Molmo to name the attack key fails ("Enter", "X", "J",
  "Click"; the key is 1), and named-skill prompts drift between slots. Pointing to each
  skill icon at the bottom of the screen and taking the leftmost is stable (slot 1 on
  both frames). A skill button is used only after two answers agree, and clicks outside
  the safe area are discarded; one dry run caught Molmo confirming a "Skill Books" quest
  panel.

### 6. Latency

On an idle GPU in replay, screenshot-to-dispatch dropped from 51.7 / 60.3 ms (p50 / p95)
to 29.9 / 38.1 ms with goal caching, compilation, precommit and pipelining. In live runs
with Molmo sharing the GPU, the median is 33–36 ms and the p95 is 150–170 ms.
[PRODUCTION_READINESS.md](PRODUCTION_READINESS.md).

### 7. Live Hordes trials (this round)

| Trial | Setup | Screenshot → first input p50 / p95 | Outcome |
|---|---|---:|---|
| `hordes-general-live-001` | 24-game LoRA, pointer | 35 / 66 ms | Ability 1 selected a Mature Grub ("Out of range"); no approach, no damage |
| `hordes-planner-live-001` | + Molmo planner (select click) | 36 / 153 ms | 2 planner clicks, 1 selected a Mature Grub; health stayed 70/70; a center click later reselected the player |
| `hordes-act-live-001` | + planner actions, full viewport | 33 / 156 ms | Molmo found no monsters (0/15 answers); no planner action |
| `hordes-act-live-002` | same | 33 / 171 ms | Stopped at 48 s (focus lost); 0/12 answers with monsters |
| `hordes-hold-live-001` | camera-trained bundle `general-002`, `--hold --all-keys`, no planner | 38 / 56 ms | Left-button drags rotated the camera; no zoom or attack; opening click selected self |
| `hordes-hold-live-002` | same | 38 / 58 ms | Left drag tilted the camera to a third-person view; walked toward the village; no zoom or attack |

Earlier rounds ran live trials at 34–53 ms median inference. They produced movement and
occasional selection, never attacks with damage
([LAYA_P2P.md](LAYA_P2P.md), [HORDES_TEMPORAL_LIVE.md](HORDES_TEMPORAL_LIVE.md)).

### 8. Generality: what was tuned on Hordes

**Evidence for generality:**
- the 24-game LoRA (all five unseen D2E games improve);
- the click head (4 of 5 unseen games);
- the goal-conditioned pointer (unseen games);
- deriving the target type from goal text.

**Chosen while looking at Hordes frames, and never tested elsewhere:**
- the skill-bar prompt and the leftmost-skill rule;
- the tracker threshold (0.75) and resolution;
- the avatar radius (0.08) and "near" radius (0.12);
- W/A/S/D-as-screen-direction movement;
- replanning timing.

The planner stack contains no game names or rules, but its generality is a design goal,
not a result.

### 9. Camera control and strategy games

Camera control now has a general path: [CAMERA_CONTROL.md](CAMERA_CONTROL.md).
- **Vocabulary.** The action vocabulary gains mouse-wheel notches and 20 more keys.
- **Runtime.** A stateful live transport (`--hold`) keeps buttons down, sends drags,
  clutches the cursor and posts wheel events.
- **Data.** Training adds Crusader Kings III (CC-BY-4.0) and three privately used
  "other"-licensed sets: Baldur's Gate 3, Civilization VI and Diablo II. About half of
  those sets' sessions have encrypted input logs and were skipped. Training windows are
  also placed around rare drag and wheel events.

Against the same recipe on the existing data, the model trained with all four:
- has lower NLL on all five held-out D2E games;
- beats repeating the previous action on held-out Baldur's Gate 3, Civilization VI and
  Diablo II sessions (for example Diablo II button F1 0.715 vs 0.698);
- improves click timing in Diablo II (onset F1 0.22 to 0.30).

Zoom timing and drag detection improve little, and middle-button camera rotation could
not be tested offline.

**Live (`hordes-hold-live-001`).** With `--hold`, the model held the left button for up to
3.8 s while moving the mouse, and the Hordes camera rotated. This is the first live camera
control. Screenshot to first input took 38 ms median and 56 ms p95. There was no zoom and
no attack, and the drag's opening click selected the player's own character. Presses
predicted on the avatar now move 0.15 off it in `--hold` mode (not yet tried live).

### 10. Jev-Omni: a native multimodal decision model instead of stitching?

[`akhilaaa3/Jev-Omni`](https://huggingface.co/akhilaaa3/Jev-Omni) (revision `c050d51`,
Apache-2.0) is Gemma 4 12B-it with a linear head. It reads text, image, audio or video,
takes a question and 2–256 options, and returns one probability per option from a single
forward pass. This is the interface Laya provides, with vision built in.

`laya_vision_stitch/jev_omni.py` ports the reference CUDA loader to MLX. It uses only
`unified/` (bf16, 23.9 GB) and the 4 MB head. `scripts/evaluate_jev_omni.py` evaluates
it; the report is at `artifacts/jev-omni-eval-001`.

| Check | Result |
|---|---|
| Parity with the repository's reference probabilities | max difference 0.023 (the repository's own two versions differ by 0.019) |
| Latency on M3 Max, bf16 | 270 ms per text question; 753 ms per question on a 1280×720 screenshot |
| Hordes: "Is a monster selected?" (80 frames; ground truth from the target panel color) | AUC 0.67–0.79 depending on wording; best wording catches 14 of 40 selections with no false yes |
| Held-out D2E games: which of 9 regions is clicked (100 click frames) | 40%, answering "middle center" 86% of the time; always-center scores 43%, the RADIO click head 48% |

**Verdict.** It runs locally and is faster than Molmo's answers, but it reads game UI
state unreliably and has no spatial sense of where to act.

It cannot replace the controller either: it has no key or mouse outputs, no memory and
no coordinates. It also would not remove any stitching that currently matters. In the
current model Laya only supplies goal features, and those carry almost no action
information.

It is not integrated. It could return as a planner question-answerer if its UI reading
improves, for example through fine-tuning on labelled game-state questions.

## Earlier rounds, in brief

- **No-training stitches.** CLIP/Qwen → Laya with paired references or a lexical bridge
  reached 67.9% on a 3-choice synthetic test and failed withheld combinations.
  [PAIRED_REFERENCE.md](PAIRED_REFERENCE.md), [LEXICAL_STITCH.md](LEXICAL_STITCH.md).
- **Trainable stitch on a synthetic sandbox.** 40/40 episodes reached their target;
  there was no real-game transfer. [SCALING_RESULTS.md](SCALING_RESULTS.md).
- **Real-game imitation from scratch** (D2E, P2P) failed persistence and shuffle
  controls. [REAL_GAME_RESULTS.md](REAL_GAME_RESULTS.md),
  [LEARNING_GATE.md](LEARNING_GATE.md), [TEMPORAL_EXPANSION.md](TEMPORAL_EXPANSION.md).
- **Stitching Laya goals into the pretrained Open-P2P policy** is the base of the
  current model. [LAYA_P2P.md](LAYA_P2P.md).
- **Visual residual adapters and spatial goal attention** gave no replacement model.
  [P2P_VISUAL_ADAPTER.md](P2P_VISUAL_ADAPTER.md),
  [SPATIAL_EXPERIMENTS.md](SPATIAL_EXPERIMENTS.md).

## Recommended next steps

1. **Cross-game planner benchmark.** Freeze all prompts and thresholds first. On the
   five held-out D2E games plus combat-heavy training games at full 480p, measure:
   Molmo target quality against where humans click or act next (compared with the
   screen center and the click head), tracker drift, and skill-bar hits. This needs a
   re-download of a few recordings per game, roughly 5–10 GB.
2. **Molmo recall.** Try tiled or zoomed queries and object-phrase variants, scored on
   recorded frames that contain monsters.
3. **Live test near monsters,** so the approach and skill actions are actually exercised.
   Then run a second live game chosen for a real closed-loop generality check.
4. **Acting on targets by learning.** Imitation does not teach target-seeking. This
   needs outcome feedback, such as online reinforcement learning in a game or
   simulator, or object-centric relabeling from full-resolution video.

## Housekeeping

- **Weights and data.** Model weights, datasets, caches and trial recordings stay under
  the ignored `artifacts/` directory (about 1.7 GB for the current bundle).
- **Data license.** D2E is CC BY-NC 4.0, so adapters trained on it inherit a
  non-commercial restriction.
- **External code.** The live transport comes from ScreenQuest
  (`../jev_wow_control`), unchanged. Hordes trials were short and supervised; the game's
  terms of service apply to automated play.
