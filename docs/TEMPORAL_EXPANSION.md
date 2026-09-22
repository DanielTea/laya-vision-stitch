# Broader temporal training and goal coverage

This experiment addresses two measured weaknesses of the first temporal pilot:
its 64 seconds of training clips all used a generic continuation goal, and longer
training memorized those clips without improving separate-recording performance.
The model architecture and frozen Qwen/Laya parent stay unchanged.

## Results

The regularized attention candidate improves some metrics on the new recordings,
but all three candidates fail qualification. The new dataset does not produce a
consistently better general-game policy. One seed was used, so these are exploratory
results, not a statistically established architecture ranking.

| Model | Selected update | Original validation F1 | Previous held-out F1 | Fresh held-out F1 | Fresh camera MAE, px/axis |
|---|---:|---:|---:|---:|---:|
| Previous attention, small dataset | 6,000 | 64.7% | **57.8%** | 16.2% | 53.65 |
| Expanded attention, ordinary training | 3,000 | 62.1% | 52.6% | 25.5% | 42.29 |
| Expanded attention, regularized | 4,000 | 65.2% | **43.0%** | **32.2%** | **37.86** |
| Expanded Mamba-3, regularized | 6,000 | 68.9% | 62.6% | 8.7% | 39.60 |
| Repeat previous human controls | — | 97.1% | 92.7% | **82.2%** | **28.13** |

The fresh-test improvement from 16.2% to 32.2% F1 is accompanied by a regression
from 57.8% to 43.0% on the previous held-out experiences. Fresh exact button-transition
agreement also drops from **35/189 (18.5%) to 24/189 (12.7%)**. The regularized
attention model's correct-versus-shuffled-image F1 difference is only **2.7
percentage points** on the fresh test; on original validation, shuffled observations
score better (71.6% vs 65.2%). Visual dependence remains inadequate.

The best validation-control-loss snapshots occur before the final update for both
attention recipes. Their training-set F1 is 57.3% / 61.5%, versus 98.5% for the old
small-data model. This avoids selecting a memorized final checkpoint, but these
figures are on different training sets and do not establish better capacity.
The new Mamba candidate improves the older test yet collapses on the fresh test;
there is no overall Mamba advantage.

Teacher-forced history matters. On the new test, feeding the regularized attention
model its own previous proposals reduces F1 from 32.2% to **28.7%** and raises camera
error from 37.86 to **59.10 px/axis**. The old attention model scores 12.1% F1 / 55.23
px on this same diagnostic. Recorded images still follow human gameplay, so this
is not a closed-loop outcome test.

On **224 specifically instructed validation frames**, substituting a generic or
different original goal changes **zero discrete actions** for every audited model.
The regularized attention model's largest button-probability change is under
0.0006. This demonstrates little prompt sensitivity on this restricted test,
not instruction following. These clips all come from Be a Tornado and contain no
moving-camera labels; the alternative goals do not have counterfactual human
controls. Broader conclusions about arbitrary goals would require different data.

The regularized attention checkpoint measures **49.91 ms median / 53.05 ms p95**
over 128 freshly encoded validation frames from four clips on the M3 Max. This
includes disk image loading, preprocessing, prompt preparation, Qwen, Laya,
state update and action decoding. It excludes capture, event posting and game
contention. Measurements made during the training session reached 58–70 ms, so
this warm offline result is not a latency guarantee. Memory itself takes about
0.81 ms median. No inference architecture or timing shortcut was added.

All three new runs verify unchanged parent weights, action equivalence after
reload, and fresh-image streaming agreement with the cached model graph.
The repository checks pass, with **113 unit tests**. No checkpoint was promoted
to the default controller, and no live Hordes trial ran.

[Complete metrics, selection history, goal controls and latency](temporal-expansion-results.json)
· [Dataset and source audit](temporal-expanded-data-001.json).

The next useful supervision target is **verified goal-conditioned demonstrations**:
multiple scenes for the same goal and genuinely different requested goals in
similar scenes, with corresponding human control labels. This expansion has
only 47 specifically instructed training clips covering 37 specific goals—most
goals still have approximately one clip. Another memory block cannot supply
that missing supervision. A separately pretrained gameplay-policy baseline also
remains useful; this run did not port or evaluate Open P2P weights.

## Data

The training set expands from **1,280 to 7,520 consecutive frames**: 235 windows
of 32 frames at 20 FPS, approximately 376 seconds of selected gameplay. It spans
**nine games/experiences and 38 distinct goals**. Of these frames, **1,504 (20%)**
have a specific retrospective instruction annotation. The others retain the
original generic continuation goal; no instruction labels were invented.

| Split | Frames | Windows | Purpose |
|---|---:|---:|---|
| Train | 7,520 | 235 | Expanded training and goal coverage |
| Validation | 512 | 16 | Unchanged clips; checkpoint selection |
| Previous test | 768 | 24 | Unchanged withheld experiences; reused developmental test |
| Fresh test | 1,024 | 32 | Four newly acquired recordings from two additional games |

Training includes Doom, Left 4 Dead 2, Be a Tornado, Blade Ball, Hypershot,
Warhammer: Vermintide 2, Euro Truck Simulator 2, Call of Duty Mobile and
Grand Theft Auto: San Andreas. The fresh test includes **Doom II** and **Natural
Disaster Survival**. Both experiences are excluded from training. This is not a
player-independent dataset split, and it contains no Hordes demonstrations.

The new sampler first covers distinct available instructed goals, up to half the
requested training windows. It then fills the remaining quota uniformly from
valid windows. Windows must have a constant goal, consecutive valid human
controls and regular timestamps. They cannot overlap within a recording,
including their extra future-target frame. The fresh test uses ordinary random
window selection, without instruction prioritization.

The original validation/test rows, images, goals and actions are semantically
identical after resolving paths. Source file hashes are pinned in
[the selection plan](../configs/temporal-expansion-001.json). New recordings come
from bounded prefixes of P2P archives 450, 545 and 499. Only fully extracted,
hash-audited recordings are used; truncated video members are discarded. The
COD/GTA recordings reuse the already downloaded pinned toy subset. The existing
Dusty Trip recording stays outside training.

The fresh test covers additional FPS/Roblox experiences, not Hordes or WoW.
It cannot establish transfer to a different MMO interface.

The sequence reader rejects overlap in recording groups or current/future image
hashes across splits, incorrect `image[t] -> action[t+1]` alignment, and previous
controls that differ from the preceding action label. All parent computations
are cached with model fingerprints and image/manifest hashes.

## Learning changes

The comparison uses 6,000 updates with two sequences per update, the same frozen
parent, seed, action vocabulary and auxiliary visual-change objective as the
first temporal experiment. The learned modules remain about 0.13% of the model.

- **Ordinary training:** full 32-frame windows and the existing 50% previous-control
  dropout.
- **Regularized training:** random aligned 24-frame crops of the same windows,
  plus masking 10% of the direct spatial visual tokens. Each token's mask is
  constant through the clip so masking cannot introduce artificial motion. The
  Laya context path stays available. These are training augmentations; inference
  still receives the unmodified current screenshot.
- **Checkpoint selection:** evaluate supervised control loss on the original
  validation clips every 500 updates. Restore the lowest-loss adapter before
  export. Auxiliary future-feature targets and both test sets cannot select
  weights. The final update is not automatically assumed best.

The sampler and history-dropout RNG are shared between recipes; crop/masking RNG
is separate. Both recipes use the same training example draw order. Regularized
updates see fewer frames per window, so this is not an equal-token-budget study.
The ordinary attention model and regularized attention/Mamba models use the same
validation-selection rule. Differences from the previous pilot include data
coverage and checkpoint selection, so improvements cannot all be attributed to
augmentation or memory architecture.

## Evaluation controls

The existing controls remain: reset memory every frame, remove previous controls,
shuffle observations within the same game and exact goal, reorder past frames,
feed previous model proposals on recorded screenshots, repeat human controls,
and predict no input. Per-game scores and exact button-transition agreement are
reported alongside aggregate button F1 and moving-camera error.

A separate goal audit uses only specifically instructed validation windows with
another available goal in that game. It preserves every screenshot and previous
control, then recomputes the connector and Laya with either the generic goal or
a different original instruction. It measures changed predictions and agreement
with the original recorded actions. This measures **goal sensitivity**, not
counterfactual task success: the dataset does not provide human action labels for
the altered goals, and multiple goals may justify the same next action.

A frozen-checkpoint comparison evaluates old and new adapters on the same newly
acquired test clips. No weights are changed in that comparison. Historical test
results are kept separate from the fresh test. No policy is automatically deployed
to Hordes, and all these tools return proposed controls without posting events.

## Reproduce

The commands require the audited source recordings and parent checkpoint already
present locally. Dataset videos and model weights are ignored by Git. The plan
records exact repository revisions, recording IDs and source hashes. The two new
large-recording prefixes retained fewer recordings than requested; their inventory
files list the complete usable members and bounded transport hashes.

To reconstruct the additional source folders (use fresh output folders):

```bash
.venv/bin/python -m laya_vision_stitch.p2p_stream \
  --archive 450 --count 1 --max-bytes 700000000 --output artifacts/p2p-stream-450
.venv/bin/python -m laya_vision_stitch.p2p_stream \
  --archive 545 --count 1 --max-bytes 700000000 --output artifacts/p2p-stream-545
.venv/bin/python -m laya_vision_stitch.p2p_stream \
  --archive 499 --count 4 --max-bytes 600000000 --output artifacts/p2p-stream-499
```

The source-member hashes must match the plan, even if bounded transport byte
counts differ when stopping immediately after the complete member.

```bash
.venv/bin/python -m laya_vision_stitch.sequence_expansion \
  --plan configs/temporal-expansion-001.json \
  --output artifacts/temporal-expanded-data-001

.venv/bin/python -m laya_vision_stitch.temporal_training \
  --bundle artifacts/p2p-pilot-004/bundle \
  --data artifacts/temporal-expanded-data-001 \
  --output artifacts/temporal-expanded-plain-001 \
  --steps 6000 --validation-every 500 --kinds attention

.venv/bin/python -m laya_vision_stitch.temporal_training \
  --bundle artifacts/p2p-pilot-004/bundle \
  --data artifacts/temporal-expanded-data-001 \
  --output artifacts/temporal-expanded-regularized-001 \
  --steps 6000 --validation-every 500 --regularize \
  --kinds attention mamba3
```

Each output contains the selected single-checkpoint bundle, selection history,
training log and full evaluation report. Export verifies unchanged parent weights,
reload equivalence, and fresh-image streaming agreement with the cached graph.
The stateful runtime and its reset contract are described in
[the temporal architecture guide](TEMPORAL_MEMORY.md#streaming-without-desktop-inputs).

To reproduce the goal and frozen-checkpoint comparisons:

```bash
.venv/bin/python -m laya_vision_stitch.temporal_prompt_audit \
  --data artifacts/temporal-expanded-data-001 \
  --bundles artifacts/temporal-memory-002/attention/bundle \
    artifacts/temporal-expanded-regularized-001/attention/bundle \
  --output artifacts/my-goal-audit

PYTHONPATH=. .venv/bin/python scripts/compare_temporal_holdout.py \
  --data artifacts/temporal-expanded-data-001 \
  --bundles artifacts/temporal-memory-002/attention/bundle \
    artifacts/temporal-expanded-regularized-001/attention/bundle \
  --output artifacts/my-fresh-comparison
```
