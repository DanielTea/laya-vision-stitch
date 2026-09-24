# Toward a general model: more games, pointing and goal grounding

This round pursues one general model rather than Hordes-specific behavior. Nothing is
trained on Hordes, and nothing game-specific enters the model: Hordes is only an unseen
test, together with five D2E games held out entirely. Numbers are in
[improvement-experiments-results.json](improvement-experiments-results.json)
(`general_round`).

**Summary.** Scaling public training data from 4 to 24 games gives consistent gains on
unseen games: LoRA in the temporal policy and decoder lowers recorded-action NLL on all
five held-out games (−3% to −12%) and brings greedy button F1 level with or above the
persistence baseline on each. A general click-position head, trained on 24k real clicks
with RADIO features, beats clicking the screen center on four of five unseen games at the
5% radius, but on Hordes it points at the player character. A goal-conditioned pointer
trained on Molmo teacher points doubles pointing agreement on unseen games and responds
to the goal text, but is not yet reliable on Hordes. Hindsight captions did not
make goal text steer actions. A live Hordes trial ran at 35 ms median screenshot-to-input
and selected a monster through an ability press, but did not approach it or deal damage.
With Molmo-7B as a slow planner, the agent selected a monster by clicking it, and a
template tracker now keeps planner targets within about 13 pixels over seconds of motion.
Conditioning the controller on hindsight click targets from D2E did not teach it to act
on a given point: human actions there are already predicted without the target.

## Data

D2E-480p (`open-world-agents/D2E-480p`, CC BY-NC 4.0, revision `f075f7e2…`) was streamed game
by game: recordings were downloaded (up to a per-game budget), cut into 20 FPS windows at
the released 192×192 preprocessing, and the videos deleted; input logs are kept.
`laya_vision_stitch/d2e_pointer.py` adds the absolute cursor position and button presses
normalized to the game window, and `laya_vision_stitch/d2e_press_frames.py` saves a
384-pixel frame at up to 1,500 sampled presses per game.

| | Frames | Games |
|---|---:|---:|
| Training windows | 197,632 | 24 |
| Validation / test windows (later sessions of training games) | 38,784 / 34,528 | 24 |
| Held-out games (never trained) | 21,280 | Core Keeper, Raft, Rainbow Six, Satisfactory, Stardew Valley |
| Click frames with position labels | 35,170 | 28 |

Frames that are exactly identical across splits must be near-blank (loading screens);
anything else fails the cache build. One black frame per affected game was exempted.

## Actions on unseen games

LoRA (rank 8) was trained on cached frozen features from all training games, with games
sampled uniformly and P2P replay; validation NLL on later sessions selected checkpoints.

| Unseen game, recorded-action NLL | Pretrained | Decoder LoRA | + policy LoRA | + policy LoRA, caption goals |
|---|---:|---:|---:|---:|
| Core Keeper | 2.395 | 2.295 | 2.266 | 2.261 |
| Raft | 3.575 | 3.465 | 3.395 | 3.380 |
| Rainbow Six | 2.043 | 1.900 | 1.863 | 1.860 |
| Satisfactory | 3.206 | 2.856 | 2.852 | 2.829 |
| Stardew Valley | 2.877 | 2.826 | 2.801 | 2.789 |

Greedy button F1 of the final model is 0.896 / 0.911 / 0.946 / 0.914 / 0.878 against
0.895 / 0.903 / 0.944 / 0.908 / 0.877 for repeating the previous action; the pretrained
policy was below that baseline on four of five. On Core Keeper, NLL fell steadily as
training games increased (2.394 pretrained, 2.341 with 4 games, 2.328 with 8, 2.266 with
24 and policy LoRA). P2P public games did not regress. Greedy onset F1 falls after
adaptation: the adapted decoder holds controls more and starts fewer new presses.

**Goals.** 2,780 clips were captioned by Qwen3.5-4B (hindsight instructions). On captioned
held-out clips, replacing a clip's caption with another clip's changes NLL by at most
0.004. Captions from 192-pixel frames were mostly generic ("move toward the enemy", "aim at
the enemy"); the captions run's small NLL edge is not evidence of goal following.

## Where to click

The policy decides whether a mouse button is pressed; a separate head predicts the
absolute press position. At runtime the live runner moves the cursor there only on a new
press (`--pointer`); held buttons keep relative drags.

| Pointer head | Seen games, later sessions: median distance | Unseen games: hit within 5% | Screen center |
|---|---:|---:|---:|
| 12×12 P2P grid + policy context, 4–8 games | 0.086–0.095 | worse than center | — |
| C-RADIOv3-B 24×14 patches, image only, 22 games | 0.100 (center 0.155) | **20.0%** | 15.4% |

Per unseen game (hit within 5%): Raft 60% vs 54% (crosshair game), Stardew Valley 19% vs
6%, Core Keeper 13% vs 10%, Satisfactory 4% vs 2%, Rainbow Six 4% vs 4% (worse median).
On 24 Hordes frames the head consistently chose the player character near the center,
which in Hordes selects oneself.

## Live trial with the general bundle

`artifacts/laya-p2p-general-001` combines the policy/decoder LoRA and the RADIO pointer in
one checkpoint (reload verified bit-identical). A 60-second trial with the pipelined
runtime, compiled model and pointer mode completed without stops: 1,527 decisions (25/s),
inference 23 ms median, screenshot-to-first-event 35 ms median and 66 ms p95. The agent
pressed ability 1 at 13.7 s, which selected a Mature Grub; the game reported "Out of range"
and the agent never approached. The target's later drop from 70 to 56 health occurred
with no agent input in the preceding two seconds, so it is not attributed to the agent.
Pointer clicks landed at the center and reselected the player. No kill or experience gain.

## Grounding teachers

Goal-conditioned pointing needs to know which region matches a goal. Candidate teachers
were tested on three Hordes frames (evaluation only) and three Stardew Valley frames:

| Teacher | Hordes monsters found | Behavior |
|---|---:|---|
| Qwen3.5-4B (4-bit) | 0 / 3 | Points at the player, the path or another player |
| OWLv2 base (detector) | 0 / 3 | Labels the player "an enemy"; finds Stardew doors |
| Molmo-7B-D (4-bit) | 2 / 3 | Points at grubs; answers "none" rather than the player when unsure |

## Goal-conditioned pointing

Molmo labelled 3,622 frame/category pairs offline (training games 2,962; held-out games
600; Hordes 60, evaluation only) for six general categories: enemies, doors, characters,
animals, vehicles and containers. `laya_vision_stitch/goal_pointer.py` lets a goal token
(the bridged Laya goal) attend with RADIO patches and predicts the goal object's cell and
whether any is visible. Scores are agreement with the teacher, within 10% of the frame:

| | Validation (training games) | Held-out games |
|---|---:|---:|
| Correct goal phrase | 68% | **51%** |
| Unseen wording of the same goal | 42% | 35% |
| Goal removed | 33% | 23% |
| Wrong category's goal | 34% | 27% |
| Screen center | 24% | 27% |
| Presence (visible or not) accuracy | 90% | 91% |

The goal text steers pointing: the correct phrase roughly doubles agreement on unseen
games, and a wrong phrase falls to the center baseline. Paraphrases transfer only
partly. On Hordes the teacher found monsters in 4 of 60 frames (1 of 4 hit with the live
goal); on 24 reviewed Hordes frames the pointer moved off the player but mostly selected
the golden quest arrow below it rather than monsters. Labels and features used 384-pixel
frames, where Hordes monsters are about 6 pixels wide; higher-resolution labels and
features are the next general step.

## Dual system: Molmo planner with the fast controller

Molmo understands scenes far better than any head trained so far, but takes 3–5 s per
answer. It therefore runs as a slow planner in its own process
(`laya_vision_stitch/molmo_planner.py`) while the 150M controller keeps acting at 20 ms:

1. Once per goal, Molmo answers from the goal text alone what to target. For the Hordes
   goal it answered "monsters".
2. When there is no live target, the newest full-resolution frame goes to Molmo with
   "Point to the monsters." Molmo answers with points or abstains.
3. Points within 8% of the screen center (the avatar in third-person games) are
   dropped, and the one nearest the center is kept.
4. `laya_vision_stitch/target_tracker.py` carries the point onto newer frames by
   matching frozen RADIO patch features (a 3×3 descriptor, cosine at least 0.78). A target
   is dropped when it is lost or 6 s old.
5. A newly acquired target gets one left click. After that, the policy's own mouse presses
   go to the tracked target, falling back to the RADIO pointer head.

Nothing in this is game-specific: the target kind comes from the goal text, and the
avatar rule assumes only a centered camera.

| | Replay of a recorded trial (`planner-replay-004`) | Live Hordes, 60 s (`hordes-planner-live-001`) |
|---|---:|---:|
| Controller inference p50 / p95 | 25 / 201 ms | 21 / 183 ms |
| Screenshot to first input p50 / p95 | — | 36 / 152 ms |
| Planner answers (with points) | 11 (5) | 12 (2) |
| Planner seconds per answer | 2.9 median | 3.0 without points, 5.0–5.2 with points |
| Planner clicks | 5, 4 on monsters | 2 |

The p95 rises from 66 ms (controller alone) to about 150–200 ms because both models
share the GPU.

**What the live clicks did.** Both clicks landed about 25 pixels (at 1280 px) to the left of Mature
Grubs:
- At 8.6 s the click fell between two grubs and selected nothing.
- At 19.3 s the click selected a Mature Grub (Lv 2, 70/70). This is the first time the
  general model selected a monster by clicking.

The grub stayed at 70/70 until 44.6 s. Then a policy click at the screen center, placed by
the RADIO pointer, selected the player's own character. The policy pressed no ability key
during the whole run, so selection did not turn into an attack. A second player nearby
was not targeted.

Two limits show up:
- **Click offset.** The tracker works on a 16-pixel RADIO grid (about 27 px at 1280), and
  targets keep moving while Molmo answers.
- **Nothing follows the selection.** The controller has no training that links "target
  selected" to "approach and use an ability".

Conditioning the controller on the next click position in D2E was tested next; see
[Acting on a target](#acting-on-a-target-hindsight-conditioning-negative).

### Tracking precision

The live click offsets came from tracking, not from Molmo: on the frames Molmo saw, its
chosen points sat on the grubs, but the targets moved about 0.18 of the screen width
during the 5 s answers. `scripts/evaluate_tracker.py` measures two things on the 7
recorded planner answers with points (33 Molmo points):

- **Synthetic shifts.** Each frame is shifted by up to 6% and scaled by 0.95–1.05, so the
  true new position of every point is known.
- **Real motion.** Each target is tracked from the frame Molmo saw to the frame where
  the answer arrived, then back again. With no ground truth on live frames, the
  distance from the starting point measures drift.

An attempt to use Molmo's own points on later frames as ground truth was dropped:
Molmo points at different monsters from frame to frame.

| Tracker | Synthetic: median error | Within 16 px | Real: forward–backward median | Within 24 px |
|---|---:|---:|---:|---:|
| Previous: mean of 3×3 patches, 768 px | 36 px | 18% | 104 px | 0 / 7 |
| 3×3 template, 768 px | 16 px | 55% | — | — |
| 3×3 template, 1024 px (runtime) | **9 px** | **94%** | **13 px** | **6 / 7** |
| Same, template updated after each match | — | — | 13 px | 4 / 7 |

Averaging the 3×3 neighborhood mostly encoded grass around a 30-pixel monster. Matching
the nine patches as a template separates the object from its surroundings. Errors are in
pixels at 1280 width.

The runtime now works as follows:
- It tracks at 1024×576, costing 59 ms per track instead of 31 ms at 768 px.
- It widens the first search after an answer to ±0.3 of the screen. That matched
  stepping through frames buffered during planning, at a fraction of the cost.
- It counts a match below 0.75 as lost. True matches score 0.88 median under synthetic
  shifts and about 0.77 after seconds of real motion; wrong matches score 0.76.
- It keeps a target for up to 30 s while it can still be found (previously 6 s).
- With the planner running, a click from the pointer head that falls on the avatar
  (screen center) no longer moves the cursor.

The recorded trial was replayed with Molmo's recorded answers, so no live Molmo process
competed for the GPU. Controller latency was p50 38 ms and p95 80 ms without planning, and
p50 35 ms and p95 113 ms with the 1024-pixel tracker. A target was live for 46% of the
replay. All 14 policy clicks went to the tracked target.

## Acting on a target: hindsight conditioning (negative)

The dual system selects a monster but never acts on it. The idea tested here was to give
the controller the target point as an input and learn from D2E what players do before
they click somewhere. `laya_vision_stitch/target_conditioning.py` encodes a screen point
and adds it to the policy's constant "thinking" token. Its output layer starts at zero,
so without a target the policy is unchanged, and the streaming runtime feeds it the
tracked planner target.

**Labels.** `scripts/add_target_labels.py` reads the kept D2E input logs. For each
frame, the target is the position of the next mouse press within a horizon, or of the
last press shortly before.

In games where the mouse turns the camera, the OS cursor is not a screen location. There
the target is the screen center, where clicks act. Which kind a game is follows from the
data: the correlation between mouse motion and the change in the next frame's image
token, with a threshold of 0.2.
- Pointer games by this test: Brotato, Dinkum, Eternal Return, MapleStory, Monster Hunter
  Wilds, Vampire Survivors, Core Keeper and Stardew Valley.
- Medieval Dynasty (0.19) is a borderline case.
- Most shooters and survival games are mouse-look games.

Two runs used the same recipe as the general model (`policy-lora-captions-001`), with
targets dropped for 15% of windows:

| | Run 1: next press within 2 s | Run 2: next press within 10 s |
|---|---:|---:|
| Training frames with a target (pointer / center) | 36.5k / 31.1k | 56.0k / 59.1k |
| Selected step (validation NLL) | 1000 (2.514) | 1000 (2.518) |
| Unseen-game NLL vs model without targets | equal (±0.01) | equal (±0.01) |

For both runs, the audit compares recorded-action NLL, on frames with a target, under
three inputs: the frame's own target, a target from another sequence of the same game,
and no target.

| Loss change on frames with a target | Run 1: no target | Run 1: target from another sequence | Run 2: no target | Run 2: target from another sequence |
|---|---:|---:|---:|---:|
| Core Keeper (unseen) | +0.023 | −0.007 | +0.006 | +0.001 |
| Stardew Valley (unseen) | +0.062 | −0.001 | +0.035 | −0.010 |
| Seen games, later sessions | +0.065 | −0.002 | +0.026 | +0.001 |

The model uses whether a target is present, which tells it a click is coming, but not
where the target is. Greedy movement and mouse direction relative to the target are the
same with the correct target, another sequence's target, or none.

The results hold up under closer checks:
- **Decision points.** Restricted to frames where the movement keys change or a click
  starts, another sequence's target still scores as well as or better than the correct one.
- **Training games.** The position effect there is small: +0.013 ± 0.005 on Dinkum,
  +0.005 ± 0.003 on Brotato, +0.002 on the four seen D2E games.

The recorded behavior explains why. Human movement toward the next click point has
cosine 0.31 in Stardew Valley, −0.03 in Core Keeper and −0.23 in the seen games. The
model reproduces these values without the target, from the image and the previous
action. In this data, knowing where the next click will be adds little to predicting the
current action, so imitation does not learn to steer toward a given point. Neither run
was bundled.

## Planner actions: select, approach, use the skill

Since imitation does not supply target-seeking, the planner now acts on its target with
three general actions. `laya_vision_stitch/planner_actions.py` implements them, and the
live runner enables them with `--planner-act`.

1. **Select.** Click the target when it is acquired. The click repeats every 4 s while
   approaching, in case it missed.
2. **Approach.** While the tracked target is more than 0.12 of the screen from the
   avatar, hold the W/A/S/D keys for its screen direction (8 sectors). These replace the
   policy's own movement keys.
3. **Use the skill.** Once the target is near, click the skill button Molmo pointed to,
   every 0.8 s.

These actions rely on stated conventions rather than learned behavior:
- the avatar is at the screen center;
- W/A/S/D move in screen directions;
- clicking a skill-bar button uses that skill;
- skill bars sit at the bottom of the screen.

Nothing names a game, key binding or layout.

**Finding the skill.** Asking Molmo for the attack key's name fails in Hordes: it
answered "Enter", "X", "J", "Click" or "Left mouse button" (the key is 1). Pointing
works. The trial crop (`[9, 87, 1280, 720]` of a 1289×872 window) cut off the Hordes
skill bar, so neither model had ever seen it. On a full-canvas screenshot, wordings
compare as follows:

| Prompt | Full canvas, bar visible | Cropped frames, bar cut off (3) |
|---|---|---|
| "…the defeat skill in the skill bar." | slot 1 (sword) | "Skill Books" quest panel ×2, top menu ×1 |
| "…skill button in the action bar." | potion slot | top menu ×3 |
| "…first ability slot in the hotbar." | slot 1 | top menu ×2, none ×1 |
| "…defeat skill icon in the skill bar at the bottom of the screen." | slot 1 | bottom edge ×3 |

Named skills were still unstable on a second full-canvas frame (the 1280×720 trial
viewport).
- "Defeat skill icon" gave the purple icon at the far right.
- "Attack skill icon" gave a potion slot on one frame and slot 1 on the other.
- "Point to each skill icon in the skill bar at the bottom of the screen." returned the
  slots from left to right on both frames.

The runtime therefore uses that prompt and takes the leftmost icon as the primary skill:
slot 1 on both frames, at (0.300, 0.923) and (0.299, 0.932).

On low-resolution frames from Eternal Return, MapleStory, Core Keeper, Stardew Valley
and Monster Hunter Wilds, an earlier wording often landed on other interface elements.
Skill pointing is reliable only when a skill bar is clearly visible.

Two safeguards apply:
- A skill button is used only after two answers on different frames agree within 4% of
  the screen.
- Planner clicks outside the safe area (x 0.03–0.97, y 0.10–0.97) are discarded rather
  than clamped. The top band holds browser-game settings and logout buttons.

Molmo asks for targets only when none is being followed (about 3 s per answer). It asks
for the skill button only while a target is followed, when it would otherwise be idle.
Asking both at once took 5–8 s, and targets went stale before they could be tracked.

**Observation-only dry runs.** These use the live runner without `--execute`: it
captures the window, proposes actions and posts no events.

In the 40-second run (`hordes-act-dryrun-002`):
- Molmo found monsters in 2 answers.
- The runtime proposed 7 selection clicks and approach keys on 264 steps (`a`, `w+a`,
  `w`).
- It asked for the skill button in the idle time.

Two agreeing answers there pointed at the "Skill Books" quest panel, which is what
prompted the bottom-of-screen wording. Clicking the skill bar needs a viewport that
includes it: a 1280×720 Chrome viewport (window 1280×807) with crop `[0, 87, 1280, 720]`
keeps the reviewed 1280×720 calibration.

**Live trial** (`hordes-act-live-001`, 60 s, `--planner --planner-act`, new viewport and
calibration `hordes-live-calibration-002`):
- The run completed with 975 observations and 846 applied steps.
- Inference took 20 ms median (197 ms p95). Screenshot to first input took 33 ms median
  (156 ms p95).
- The policy pressed w/a/d, clicked 18 times and pressed ability 4 three times.
- Monsters in view were few, small and near the screen edges. Molmo answered "none"
  in all 15 answers (about 3 s each), so no selection, approach or skill click was
  issued.

A second run with the same setup (`hordes-act-live-002`) stopped after 48 s when the window
lost focus. Molmo again answered "none" in all 12 answers (inference 20 / 166 ms, first
input 33 / 171 ms).

The action path is therefore untested in play. Molmo's recall of small, distant
monsters is now the bottleneck: it found monsters in 2 of 12 and 5 of 11 answers in
denser earlier scenes.

**Generality caveat.** The planner stack has run only on Hordes. Prompts, the tracker
threshold, the avatar and "near" radii, and the action conventions were chosen while
looking at Hordes frames. Beyond Hordes, the only checks were Molmo as a label teacher on
D2E and skill pointing on ten low-resolution frames from other games, where it mostly
failed. Its generality is an untested design goal, not a result.

## Limitations

- Offline results are imitation of recorded human controls; the live trials are short
  runs. Hordes success (approaching, damage) is not demonstrated; the dual system selected
  one monster.
- Held-out games share the D2E recording setup; five games is a small test set.
- Click frames are sampled at recorded presses, so the pointer is trained on where people
  click, not on what they intended.
- The Molmo teacher is imperfect (it missed one of three Hordes frames); its labels are
  supervision, not ground truth.

## Reproduce

```sh
export PYTHONPATH=.
.venv/bin/python scripts/stream_d2e.py --games Grounded Barony --max-gb-per-game 5 \
  --press-output artifacts/d2e-presses-new --progress artifacts/d2e-stream-new.json
.venv/bin/python scripts/build_sequence_cache.py --bundle artifacts/laya-p2p-bridge-001/bundle \
  --data artifacts/d2e-seq-grounded-001 --output artifacts/seqcache-grounded-new \
  --splits train validation test heldout --spatial-press
.venv/bin/python scripts/train_policy_lora_sequences.py --bundle artifacts/laya-p2p-bridge-001/bundle \
  --train artifacts/seqcache-grounded-new:train artifacts/seqcache-p2p-001:train \
  --validation artifacts/seqcache-d2e-new-002:validation \
  --evaluate artifacts/seqcache-core-keeper-001:heldout --output artifacts/policy-lora-new
.venv/bin/python scripts/cache_press_features.py --presses artifacts/d2e-presses-*/presses.jsonl \
  --output artifacts/press-features-new
.venv/bin/python scripts/train_pointer_radio.py --cache artifacts/press-features-new \
  --output artifacts/pointer-radio-new
.venv/bin/python scripts/build_general_bundle.py --base artifacts/laya-p2p-bridge-001/bundle \
  --policy artifacts/policy-lora-new --pointer artifacts/pointer-radio-new \
  --output artifacts/laya-p2p-general-new
```

Captioning: `scripts/caption_clips.py`; teacher points: `scripts/label_goal_points.py`;
goal-conditioned pointer: `scripts/train_goal_pointer.py`. None of these sends game input.

Tracking and target conditioning (no game input):

```sh
.venv/bin/python scripts/evaluate_tracker.py --bundle artifacts/laya-p2p-general-001 \
  --trials artifacts/hordes-planner-live-001 artifacts/planner-replay-004@artifacts/hordes-p2p-live-003 \
  --output artifacts/tracker-eval-new
.venv/bin/python scripts/add_target_labels.py --ahead 10 --behind 2 --tag targets10 \
  --caches artifacts/seqcache-stardew-valley-001:heldout  # and every training cache
.venv/bin/python scripts/train_policy_lora_sequences.py ... --targets --target-labels targets10 \
  --output artifacts/policy-lora-targets10-new
.venv/bin/python scripts/evaluate_planner_loop.py --bundle artifacts/laya-p2p-general-001 \
  --trial artifacts/hordes-p2p-live-003 --recorded-planner artifacts/planner-replay-004 \
  --output artifacts/target-replay-new
```
