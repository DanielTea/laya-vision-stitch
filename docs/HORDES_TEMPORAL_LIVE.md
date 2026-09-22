# Live temporal-attention trials

## Requested 60-second run

The follow-up ran the same checkpoint and goal for the full 60 seconds in Chrome.
It made **840 decisions**, of which **209 produced nonempty input events**:
95 forward pulses, 39 left-button pulses, and 113 steps with cursor motion
(these categories overlap). It ended with a **Mature Grub selected at 70/70 HP**.
It never proposed Tab, ability keys or the right mouse button. Player health
stayed 244/244 and XP stayed 466/1,600. Movement and selection occurred, but useful
combat was not demonstrated. The initial selected target was the player itself.

| Measurement | 60-second result |
|---|---:|
| Inference median / p95 | 52.70 / 95.81 ms |
| Screenshot-to-dispatch-start median / p95 | 75.71 / 122.77 ms |
| Temporal state resets | 111 |
| Consecutive screenshot gaps over 100 ms | 110 |
| Blocked button proposals | 0 |
| Mouse proposals exceeding the 64 px/axis limit | 16 |

The memory resets include the initial frame and 110 gaps exceeding the runtime's
100 ms reset threshold. Model-only median inference is below 60 ms, but full
screenshot-to-input reaction exceeds the requested 60 ms budget. This temporal
discontinuity is an additional live limitation. It does not establish the cause
of absent ability use. Of 305 nonzero raw mouse proposals, 113 resulted in nonzero
bounded cursor motion; central-playfield limits restricted the others.

[Recorded metrics and review](hordes-temporal-live-002.json).
Local video, full logs, and review page:
`artifacts/hordes-temporal-live-002/{gameplay.mp4,events.jsonl,review.html}`.
The video is reconstructed at screenshot timestamps and lasts approximately
60 seconds. No inputs remain held; the trial stopped at its duration limit.

## Initial 20-second run

On September 22, 2026, the user authorized a live trial in their signed-in regular
Chrome Hordes.io tab. The expanded regularized attention checkpoint ran for
20 seconds with screenshots and its own applied previous controls. No teacher,
OCR, targeting rules, combat policy or ScreenQuest planner supplied actions.

**Result: the checkpoint did not demonstrate useful combat.** It changed the view,
moved a short distance, and then mostly produced no input. It never proposed Tab,
an ability key or the right mouse button. Reviewed beginning/intermediate/end
frames show no selected monster target. Health stayed at 244/244 and XP at
466/1,600. No kill, loot or ability use was established.

| Measurement | Result |
|---|---:|
| Decisions | 305 |
| Decisions containing posted button/mouse movement events | 37 |
| Left-button pulses | 5 |
| Forward pulses | 17 |
| Tab / ability-key / right-button proposals | 0 |
| Inference median / p95 | 51.93 / 54.39 ms |
| Screenshot-to-dispatch-start median / p95 | 72.36 / 91.98 ms |
| Temporal state resets | 4 |
| Blocked button proposals | 0 |
| Raw mouse proposals exceeding the per-axis limit | 0 |

This is one short starting-state test, not a multi-episode success rate. Inference
met the 60 ms median target while the game was running. Useful control did not.
The failure to attack was not an ability-key filter: keys 1–4 were allowed and
none was proposed. Left clicks can rotate the view when dragged; this is not
evidence of purposeful camera control or target selection.

## What was actually executed

- Checkpoint: `artifacts/temporal-expanded-regularized-001/attention/bundle`.
- Goal: “Select a nearby Young Grub with Tab, approach with WASD, and attack with
  ability 1. Avoid other players. Retreat if health is low.”
- Screenshot: complete 1280×720 game viewport, excluding Chrome toolbars.
- Allowed controls: WASD, Space, Tab, 1–4, left/right mouse buttons. Other model
  button proposals would be logged and omitted. No buttons were substituted.
- Each control pulse releases after 50 ms independently of the next inference.
- Relative mouse units convert using the training convention (`delta × 512`),
  capped to 64 pixels per axis. Cursor position stays inside a central playfield
  rectangle. Of 63 nonzero raw mouse proposals, 20 yielded nonzero bounded cursor
  motion; the remaining proposals reached this cursor boundary. This restriction
  limits what the trial can establish about camera behavior.
- Previous-control input contains the bounded controls actually dispatched,
  including the adjusted mouse delta, rather than the original proposals.
- Window identity, size, focus, text-field focus, frame age and a calibrated HUD
  icon guard input. A STOP file or focus loss stops the trial. A shared kernel
  lock prevents concurrent ScreenQuest controllers. All inputs were released.

The runner imports only ScreenQuest's existing native window capture/input
transport and its fixed HUD layout guard. No ScreenQuest gameplay controller is
running. Native dependencies are the research project's optional `desktop` extra.

## Evidence and timing interpretation

Local evidence is in `artifacts/hordes-temporal-live-001/`: the full input
`events.jsonl`, screenshots, `before.png`, `after.png`, `summary.json`, a
timestamp-paced `gameplay.mp4`, and `review.html`. Captures exclude the cursor.
These artifacts remain ignored by Git; they include visible multiplayer chat.

The first trial's raw summary used the name `screenshot_to_post_ms` for a
timestamp taken **at dispatch start**, before posting events. The table above
uses its correct interpretation. The runner now names that field
`screenshot_to_dispatch_start_ms`; the original trial log is preserved.
This metric includes decisions that apply an empty action and is neither an
event-delivery acknowledgement nor measured game response latency.
Likewise, `applied_steps=305` means the execution gate accepted 305 decisions,
not that 305 nonempty inputs occurred. Only 37 decisions posted button/mouse
movement events. Warm-up is excluded from the reported inference figures.

## Repeating a bounded trial

The runner is intentionally specific to the reviewed 1280×807 Chrome window
layout (87 pixels of toolbar, 720 of game). A new window ID/reference is needed
after reopening Chrome or changing that layout. The reference must be reviewed
as the live game before executing; it is not an automatically trusted screenshot.

```bash
uv sync --extra desktop --extra models --extra stitch --extra data
.venv/bin/python -m laya_vision_stitch.temporal_live \
  --bundle artifacts/temporal-expanded-regularized-001/attention/bundle \
  --screenquest-root /path/to/jev_wow_control \
  --window CURRENT_WINDOW_ID \
  --reference artifacts/hordes-live-calibration-001/reference.png \
  --output artifacts/hordes-temporal-live-NEW \
  --seconds 20 --execute
```

Without `--execute`, this observes and logs proposals only. The runtime itself
still marks the checkpoint unqualified; explicitly running this experimental
harness does not promote it to the live controller.

The next learning experiment should address the missing goal-to-control mapping
with verified gameplay demonstrations, including approach/target/attack
transitions and recovery. Running this unchanged checkpoint longer is not
supported by the observed behavior.
