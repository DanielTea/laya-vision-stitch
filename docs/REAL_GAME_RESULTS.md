# Real-game pilot: transfer not established

The model was trained on real human gameplay from three games and evaluated on
two entirely withheld games. **This checkpoint is not a useful general game
controller.** Its image ablations and simple persistence baseline do not support
that claim. The existing local controller/default model was not replaced.

Checkpoint: `artifacts/d2e-model-002/bundle`.
[Reproduction, dataset license and protocol](REAL_GAME_TRAINING.md).
[Machine-readable evidence](real-game-001.json).

## What ran

- 1,907 synchronized screenshot/action examples from Grounded, Raft and Minecraft,
  plus 192 synthetic replay examples. Six training sessions, 52.93 minutes of
  source recordings, sparsely sampled.
- Satisfactory: 159 validation examples from a separate game/session.
- Barony: 192 final test examples from a separate game/session.
- 3,000 additional training steps, learning rate 0.00003, seed 17. No updates used
  either held-out game's examples. The checkpoint stayed fixed for evaluation.
- 6,099,516 trainable parameters (~0.802% of 760,907,586 tensor elements).
  Original Qwen/Laya weights were SHA-256 verified unchanged; exported/reloaded
  predictions matched. One composite model, with no runtime language teacher.
- Two screenshots per real-game input, previous-action text, and generic goal
  text. Labels cover physical keys, clicks and relative mouse movement over the
  following 100 ms. The source does not provide task-level goal annotations.

This is a small offline imitation experiment. No live gameplay, reward,
completion rate or closed-loop recovery was measured. Training scripts completed
successfully; that is separate from policy quality.

## Held-out results

F1 below is positive-class micro F1 over the configured button vocabulary.
The output threshold was fixed at 0.5; it was not tuned on the test game.

| Button predictor | Satisfactory F1 | Barony F1 |
|---|---:|---:|
| Before real-game training | 0.115 | 0.098 |
| Trained model, correct screenshots | 0.133 | 0.332 |
| Trained model, zeroed visual features | 0.188 | 0.285 |
| Trained model, shuffled screenshots | 0.110 | 0.338 |
| Trained model, no previous-action text | 0.129 | 0.348 |
| Repeat previous recorded action | **0.814** | **0.841** |
| No action | 0.000 | 0.000 |

The improvement over the previous synthetic model does not demonstrate effective
visual grounding. Shuffled screenshots performed slightly better on Barony, and
zeroing vision improved Satisfactory. This single run cannot prove why it failed,
but it does rule out claiming successful transfer from these measurements.

| Other metric | Satisfactory | Barony |
|---|---:|---:|
| Model exact button-set agreement | 44.7% | 19.8% |
| Repeat-previous exact agreement | 84.9% | 71.4% |
| No-action exact agreement | 71.1% | 28.1% |
| Model mouse direction accuracy on moving axes | 53.3% | 49.3% |
| Repeat-previous mouse direction accuracy | 80.4% | 78.9% |
| Model normalized mouse MAE | 0.0642 | 0.0781 |
| Zero-motion mouse MAE | 0.0441 | 0.0626 |

Stationary inputs make exact agreement alone misleading. Positive F1, mouse
errors and visual controls are needed alongside it. On the 55 Barony examples
where the next button set changed, model F1 was 0.224 versus 0.526 for repeating
the previous set. A previous set can share some correct buttons despite failing
exact agreement on every transition.

Warm two-frame inference in the held-out evaluation measured 163 ms median /
216 ms p95 on the M3 Max. Satisfactory measured 144/147 ms. The immediate export
benchmark measured 206/271 ms. These include preprocessing and both vision
encodings, exclude desktop capture/input/game response, and are not equivalent
to the earlier 58 ms single-image synthetic benchmark.

## Retention of the earlier visual tasks

On the same 128 fresh synthetic move-toward questions that the source checkpoint
answered perfectly, the game-trained checkpoint scored **99/128 (77.3%)**.
Button-set agreement was 89.8%; both opposing goals were correct for 35/64 scenes.
Zeroed vision reduced choice accuracy to 50%. Some synthetic visual grounding
remains, but 192 replay examples did not prevent substantial forgetting.
The earlier checkpoint is preserved separately.

## Artifacts and interpretation

```text
artifacts/d2e-pilot-003/audit.json
artifacts/d2e-pilot-003/{train,validation,test,mixed-train}.jsonl
artifacts/d2e-model-002/report.json
artifacts/d2e-model-002/validation/report.json
artifacts/d2e-model-002/test/report.json
artifacts/d2e-model-002/test/predictions.json
artifacts/d2e-model-002/retention.json
```

This run adds a reproducible multi-game ingestion and evaluation path and a
trained research checkpoint. It does not deliver a playable general agent.
The next controlled experiment should add visual grounding and task-goal
supervision on real frames, compare stronger temporal representations, and test
on new untouched sessions/games. Simply continuing to optimize this test result
would turn Barony into a development set.

The D2E data is CC BY-NC 4.0. The checkpoint's training-data provenance is recorded
in its configuration; source recordings and derived frames remain local.
