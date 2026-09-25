# Camera control and strategy games

Goal: camera control that works in any game from the same fast controller, under 60 ms,
with no per-game rules. Camera control means:
- mouse-look;
- rotating the view by holding a button and dragging;
- dragging the map and box-selecting units;
- scroll-wheel zoom;
- panning with keys or the screen edge.

Strategy games with unit selection join the training data.

## Why camera control did not work

1. **The runtime could not hold.** The pulse transports in `temporal_live.py` pressed
   every control for 50 ms per step, then released it, and clamped the cursor to a central
   box. A "hold and drag" became a series of clicks.

   In the first 24-game live Hordes trial, the model held a mouse button in 27 separate
   runs, the longest 19 steps (about 0.8 s), with motion on 23 steps. None of these could
   rotate the camera.
2. **No scroll output.** D2E records wheel notches (for example 131 in one Core Keeper
   session), but the released action vocabulary has no wheel, so zoom was never learned.
3. **Little drag-camera data.** D2E's 29 games are mostly mouse-look shooters or games
   without a camera, and include no strategy game. Roblox in the P2P data uses
   right-button drag.

## Design

**One learned action vocabulary covers every camera scheme.**
- **Mouse-look:** relative mouse motion.
- **Drag camera, map drag and box selection:** a mouse button held across steps plus
  motion.
- **Zoom:** the new `scroll_up` / `scroll_down` controls.
- **Panning:** W/A/S/D and arrow keys.

`p2p_adaptation.EXTENDED_KEYS` appends 22 controls after the released keys and Tab, so
earlier token ids are unchanged:
- mouse-wheel notches (`scroll_up`, `scroll_down`);
- Ctrl, Alt, Escape, Enter;
- the digits 5–0;
- R, C, X, V, G, I, M, B, T, H.

New output rows start at a −8 logit bias and new embeddings at the mean key embedding
(`install_control_adapter(extra_keys=…)`). `scripts/retokenize_caches.py` re-encodes
every existing D2E and P2P cache without re-encoding images. It adds wheel notches from
the kept D2E input logs.

| Re-tokenized existing data (246,144 frames) | Before | After |
|---|---:|---:|
| Frames flagged incomplete (dropped controls) | thousands (every Ctrl, R, Alt, Tab, M… frame) | 238 (mostly Backspace) |
| Steps with a wheel notch | 0 (dropped) | 934 |
| Newly labelled controls | — | Ctrl 3,629 · Tab 1,802 · Alt 1,389 · R 1,386 · C 397 · X 390 · Escape 294 · M 155 |

**Stateful transport** (`laya_vision_stitch/hold_transport.py`, live runner `--hold`;
`--all-keys` allows the extended vocabulary):
- Keys and the left, right and middle buttons stay down until the model stops outputting
  them. Events go out only when the state changes.
- Motion while a button is held is a drag event for that button. It carries the model's
  full delta in the event's delta fields, which pointer-locked cameras read.
- **Clutch:** when a drag would leave the allowed area, the runtime releases, re-centers
  the cursor and presses again, like lifting a physical mouse.
- Wheel controls post one notch per step.
- A 250 ms watchdog releases everything if actions stop arriving.
- Planner clicks re-press the left button at their target.

Posting events adds no model time. The controller stays at 20–35 ms per decision.

## Data

| Dataset | License | Used | Camera and selection content |
|---|---|---|---|
| D2E-480p, 24 games (existing) | CC BY-NC 4.0 | all training sessions | mouse-look, some wheel |
| Open-P2P replay (existing) | P2P terms | as before | Roblox right-drag camera |
| **Crusader Kings III** (`Ethosoft/ck3-gameplay-mouse-keyboard-dataset`, rev `95a3856`) | CC-BY-4.0 | 209 sessions, 35 GB (145 train / 24 val / 40 test by the published date split) | map zoom, left/right drag, army and UI selection by click |
| **Baldur's Gate 3** (`yinhuankuang/rl-game-traces-baldurs-gate-3`, rev `1bcda3b`) | "other" (private research use) | 5 readable of 8 downloaded (of 16) | middle-button drag rotates the camera, left-drag party selection, WASD, zoom |
| **Civilization VI** (`yinhuankuang/rl-game-traces-civilization-6`, rev `9ef528a`) | "other" (private research use) | 6 readable of 9 downloaded (of 30) | map drag and zoom, unit selection by click |
| **Diablo II Resurrected** (`yinhuankuang/rl-game-traces-diablo-2-resurrected`, rev `9c7a4a2`) | "other" (private research use) | 6 of 18 sessions | click-to-move and click-to-attack combat |

Format notes:
- **CK3:** the dataset card claims 1080p and 610+ hours. The videos are 2560×1440 and
  average about 27 FPS because frames are dropped, and the byte sizes imply about 21
  hours in total.
- **Baldur's Gate 3, Civilization VI and Diablo:** these share one recorder. Its log
  lists input events for every 60 FPS video frame, including raw mouse motion, absolute
  cursor, buttons, wheel and virtual-key codes.

In one Baldur's Gate 3 hour there are:
- 226 middle-button presses, with 2,216 frames of middle drag (camera rotation);
- 2,092 left clicks;
- 50 wheel events;
- 715 W presses.

In one CK3 minute and a half there are 686 wheel notches and 42 left clicks.

`laya_vision_stitch/recorded_sequences.py` converts both recorders into the D2E sequence
format:
- 20 FPS, 32-frame windows at the released 192×192 preprocessing;
- majority-time holds and raw motion ÷ 512;
- wheel notches as controls;
- pointer labels (cursor and press position) for the click head.

A frame may be up to 70 ms old, one dropped 30 FPS frame, compared with 35 ms for D2E.
`scripts/prepare_camera_data.py` builds extended-vocabulary caches and then deletes the
videos, keeping the input logs.

No real-time strategy game such as StarCraft or Age of Empires with screen video and raw
input logs was found on the Hub. Several World of Warcraft and one other Civilization VI
set require an access request.

## Data actually used

About half of the "other"-licensed sessions store their input logs encrypted: the file
starts with a `BGI_JSH_V1` header followed by ciphertext. It appears in sessions from
roughly 9 May 2026 onward. Readability was checked per session, either locally or by
fetching the first 16 bytes of the log. Encrypted sessions are skipped and their videos
deleted.

**Event-aware sampling.** Uniform windows at 2 per minute caught only 82 Baldur's Gate 3
middle-drag frames. Training splits now also get up to 3 windows per minute starting
0.25–1 s before a drag or a wheel burst (`recorded_sequences.control_events`):
- a drag is a mouse button held at least 150 ms with at least 20 px of motion;
- rows from these windows are tagged `"sampling": "event"`;
- validation and test stay uniform;
- test sessions get a separate `test_events` split in its own cache, a targeted test
  for camera controls.

| Game | Train frames | Val / test frames | Train content |
|---|---:|---:|---|
| Crusader Kings III (209 sessions, 409 min) | 58,240 | 7,328 / 7,936 | left click 5.9%, wheel 6.7%, Ctrl 2.6%, Esc 1.7% |
| Baldur's Gate 3 (5 readable sessions) | 19,648 (7,008 event) | 3,936 / 4,224 (+768 event) | Alt 22.9%, middle-button drag 7.3%, left click 4.7%, WASD |
| Civilization VI (3 + 3 readable sessions) | 44,896 | — / 3,904 | left click 9.1% (unit and city selection) |
| Diablo II Resurrected (6 sessions) | 46,464 | 4,960 / 4,288 (+6,432 event) | left button held 42% (click-to-move and attack), right-click skills 3.6%, Shift 3.1% |

The Diablo 4 repository (whose files are labelled Diablo 2) and the later Civilization VI
and Baldur's Gate 3 sessions were fully encrypted.

## Results

All three runs use the same recipe:
- extended vocabulary;
- policy and decoder LoRA rank 8;
- 3,000 updates;
- hindsight captions.

Only the training data differs:

| Run | Training data | Checkpoint selection |
|---|---|---|
| `policy-lora-extended-base-001` | 24 D2E games and P2P | D2E validation |
| `policy-lora-extended-ck3-001` | + Crusader Kings III | D2E validation |
| `policy-lora-extended-camera-001` | + CK3, Baldur's Gate 3, Civilization VI and Diablo II, each weighted 2 in game-balanced sampling | mean of D2E, CK3, Baldur's Gate 3 and Diablo II validation |

Recorded-action NLL on the five held-out D2E games (never trained):

| | Core Keeper | Raft | Rainbow Six | Satisfactory | Stardew Valley |
|---|---:|---:|---:|---:|---:|
| Existing data | 2.383 | 3.499 | 2.112 | 3.145 | 2.935 |
| + CK3 | 2.377 | 3.503 | 2.091 | 3.116 | 2.890 |
| **+ all four** | **2.362** | **3.484** | **2.051** | **3.100** | **2.882** |

Greedy button F1 on these games stays within ±0.01 of the persistence baseline.

Held-out sessions of the new games. Columns: greedy button F1, and sampled onset F1 (new
presses matched within ±2 steps).

| Split | Existing data | + CK3 | + all four | Repeat previous |
|---|---:|---:|---:|---:|
| CK3 test | 0.377 / 0.098 | 0.480 / 0.129 | **0.537 / 0.146** | 0.549 / 0 |
| Baldur's Gate 3 test | 0.655 / 0.089 | 0.730 / 0.093 | **0.803 / 0.109** | 0.785 / 0 |
| Civilization VI test | 0.712 / 0.056 | 0.706 / 0.100 | **0.726 / 0.133** | 0.717 / 0 |
| Diablo II test | 0.656 / 0.223 | 0.647 / 0.203 | **0.715 / 0.302** | 0.698 / 0 |
| Diablo II drag and wheel events | 0.618 / 0.274 | 0.612 / 0.254 | **0.729 / 0.372** | 0.681 / 0 |
| Baldur's Gate 3 drag and wheel events | 0.641 / 0.108 | 0.731 / 0.122 | **0.797 / 0.142** | 0.813 / 0 |

Camera-control details (`sequence_metrics.control_report`, sampled decoding):
- **Drags.** Drag-step F1 on the Diablo II event split rises from 0.50 to 0.57. When the
  model drags where the human drags, its motion points the same way (cosine 0.67–0.91).
  Drag F1 on the Baldur's Gate 3 event split stays about 0.57 in all runs.
- **Clicks.** Left-click onset F1 on the Diablo II events rises from 0.37 to 0.46.
- **Zoom.** Wheel onsets remain weak: F1 0.09–0.13 on CK3, Civilization VI and Baldur's
  Gate 3, and 0 on the five D2E games.
- **Middle-button rotation not tested.** The one held-out Baldur's Gate 3 session contains
  no middle-button use, so middle-button camera rotation, the drag scheme closest to
  Hordes, could not be tested offline.

**Reading.**
- More and more varied games lower NLL on every held-out game.
- On held-out sessions of Baldur's Gate 3, Civilization VI and Diablo II the model now
  beats repeating the previous action on button F1. Before, only the persistence baseline
  reached that level.
- Click timing improves most in the click-heavy action RPG.
- Zoom timing and drag detection improve little. When to scroll or start a drag is not
  predictable from one frame and the previous action.
- These are teacher-forced offline metrics. Whether camera dragging works in a live game
  needs a live test with `--hold`.

**Bundle.** `artifacts/laya-p2p-general-002` packages the "+ all four" run with the RADIO
click head: 43 keys, reload bit-identical. On a replay of 20 s of recorded Hordes frames it
ran at 29.6 ms p50 and 55 ms p95 (GPU shared). It proposed movement, 28 steps holding the
right button and 18 left clicks.

## Live Hordes trial with holds

`hordes-hold-live-001`: 60 s, bundle `laya-p2p-general-002`, `--hold --all-keys --pointer
--pipeline --precommit --compile`, no planner. The game viewport was the calibrated
1280×720 in a 1280×807 window.
- **Latency.** The trial completed with 1,678 decisions (28/s). Inference took 22 ms
  median (40 ms p95); screenshot to first input took 38 ms median and **56 ms p95**, the
  best tail so far and under the 60 ms target.
- **Holds.** The model held the left button in 36 runs, the longest 106 steps (3.8 s,
  612 px of horizontal motion), and the right button in 4 short runs. 112 steps were drags
  (a button held while moving).

It pressed Space on 18 steps, W/A/S/D on 253, and never used the wheel.

**The camera rotated.** During the long left-button drag (6.2–10.7 s) the view turned
around the centered character: the village and tree gave way to the path, then to
cobblestones. This is the first camera control observed live; the pulse transport could
not produce it.

The press that started the drag went to the screen center (pointer head), which selected
the player's own character.

Not seen live:
- no wheel zoom;
- no middle-button use;
- no attacks.

**Second run** (`hordes-hold-live-002`, same setup):
- **Latency.** 1,755 decisions; screenshot to first input took 38 ms median and 58 ms p95.
- **Holds.** 20 left-button runs, the longest 104 steps, and 96 drag steps.
- **Camera.** The longest drag (25.2–30.3 s) tilted the camera from a steep top-down view
  to a horizontal third-person view with the village in sight.
- **Behavior.** The character then walked toward the village, holding W on 410 steps, and
  passed close to another player.
- **Missing.** No zoom, no ability keys and no attack. The opening center click again
  selected the player's own character.

## Reproduce

```sh
export PYTHONPATH=.
.venv/bin/python scripts/download_strategy_data.py            # CK3 + Baldur's Gate 3
.venv/bin/python scripts/download_bgi_sessions.py --datasets \
  yinhuankuang/rl-game-traces-diablo-2-resurrected:6
.venv/bin/python scripts/retokenize_caches.py --caches artifacts/seqcache-*:train ...
.venv/bin/python scripts/prepare_camera_data.py --dataset diablo-2 \
  --source artifacts/strategy-source/diablo-2-resurrected --game Diablo_II \
  --data artifacts/seq-diablo2-new --cache artifacts/seqcache-diablo2-new \
  --windows-per-minute 2 --event-windows-per-minute 3
.venv/bin/python scripts/train_policy_lora_sequences.py ... --vocabulary extended \
  --game-weight Crusader_Kings_III=2 Baldurs_Gate_3=2 Civilization_VI=2 Diablo_II=2 \
  --validation <several splits> --output artifacts/policy-lora-extended-camera-new
.venv/bin/python scripts/build_general_bundle.py --base artifacts/laya-p2p-bridge-001/bundle \
  --policy artifacts/policy-lora-extended-camera-new --pointer artifacts/pointer-radio-001 \
  --output artifacts/laya-p2p-general-new
```

Live (sends input; needs the 1280×807 Hordes window and calibration-002):
`temporal_live ... --bundle artifacts/laya-p2p-general-002 --hold --all-keys --pointer`.
