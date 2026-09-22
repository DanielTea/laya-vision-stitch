# Recording supervised gameplay

The passive recorder captures the selected game viewport and physical gameplay
controls. It runs no model and posts no inputs. It stops on Escape, Enter (chat),
focus/window changes, system shortcuts, the pointer leaving the viewport, a
`STOP` file, or its time limit. It records a limited gameplay key vocabulary,
mouse button edges, relative mouse motion and click coordinates; it never reads
text values from accessibility elements.

In regular Chrome, use the existing reviewed Hordes window (1280×807 including
toolbars, 1280×720 game crop starting at y=87). If resizing or changing toolbars,
recalibrate before recording. Keep chat closed. Run from Terminal:

```bash
cd /Users/danieltremer/Documents/laya-vision-stitch
./scripts/record-hordes.command --seconds 300
```

Click inside the game within 60 seconds, then play normally: select monsters,
approach, attack, turn the camera, retreat when necessary, and collect drops.
Escape finishes early. Record several separate sessions, including a different
area/camera direction for evaluation. A recording is not automatically expert
or successful; review its visible outcomes before using it as expert supervision.

For an explicitly different instruction, provide `--goal '…'` and demonstrate
that instruction. Reusing the same controls with arbitrary opposite goal labels
would teach the model to ignore goals. No game-specific rules are introduced
into model inference by this recorder.

The recorder selects exactly one Chrome window titled Hordes.io; other games
require an explicit `--window`, `--crop`, `--game`, `--controls`, and `--goal`.
Screen Recording and Input Monitoring permissions must be available to the host
application running Python. A native listen-only event tap was successfully
created on the development Mac; a full human recording remains to be validated.

## Files and timing

Each run creates `artifacts/human-hordes-<timestamp>-<pid>/`:

- `frames/` and `frames.jsonl`: viewport images and actual presentation times.
- `controls.jsonl`: physical button states/edges and mouse deltas with event times.
- `demonstrations.jsonl`: screenshots labelled with the following 50 ms of controls.
- `config.json`, `summary.json`, `audit.json`: scope, stop reason and exclusions.

Capture requests 20 FPS. Actual presentation intervals are retained; dropped or
unchanged frames are not silently synthesized. Keyboard repeats are omitted;
held-key durations and press/release edges remain available. Screenshots omit
the OS cursor to match live inference; click coordinates remain supervised targets.

An action includes any button held for part of its interval or newly pressed
within it. Exact held durations are preserved in provenance. The next action is
not simply the state of keys when the screenshot arrived. Inputs beyond the
recording boundary are excluded, and motion exceeding the model's vocabulary is
rejected rather than clipped. Previous-control inputs are populated only when
their complete labelled interval ends before the new screenshot.

Human reaction delay is not corrected automatically. Preserve the raw traces for
later sequence/latency experiments. All frames from one recording must stay in
one dataset split; splitting adjacent frames randomly would inflate evaluation.

To rebuild labels without recording again:

```bash
.venv/bin/python -m laya_vision_stitch.demonstration_recorder export artifacts/<recording>
```

This supplies data for learning. It does not itself make the current stitched
model a reliable player or validate a sub-60 ms screenshot-to-input response.
