# Architecture, data and production experiments without new Hordes recordings

This round tests the proposed improvements to the stitched Laya–Open-P2P model without
recording any new Hordes gameplay: public datasets, action chunking, real-time chunking,
goal guidance, a slow planner with a fast controller, elapsed-time memory, foveation,
latent actions, a latent world model, outcome weighting, a larger pretrained policy and
production latency work. **No live Hordes trial ran and no input was sent to any game.**
No checkpoint replaces the default.

**Bottom line.** Most architectural additions do not help at the available data scale,
and several honest negatives replace earlier optimism. The main new evidence is about
evaluation and data: public P2P splits overstate the pretrained policy because they
likely overlap its pretraining data; on held-out D2E games it does not beat repeating the
previous action. The one clearly
positive modeling result is data: adapting the pretrained action decoder on 58k frames of
public D2E gameplay reduces held-out-session NLL by 24% and brings greedy button F1 level
with the persistence baseline, without regressing public P2P games. This is progress in
offline imitation on public games, not demonstrated Hordes play.

## Evaluation changes

- **Sequence cache.** Frozen image tokens, goals and teacher-forced policy contexts are
  cached per 32-frame, 20 FPS window (`scripts/build_sequence_cache.py`). The parallel
  sequence forward matches the streaming runtime (maximum context difference 9e-5 on real
  frames; `tests/test_sequence_experiments.py`). Deployed inference is unchanged.
- **Onset F1.** Button F1 rewards holding whatever was already held, so repeating the
  previous action scores 0.86–0.96. Onset F1 scores only newly pressed controls (matched
  within ±2 frames), where the repeat baseline scores zero
  (`laya_vision_stitch/sequence_metrics.py`).
- **Held-out D2E games.** Five local D2E games (Core Keeper, Grounded, Minecraft, Raft,
  Satisfactory; 3,168 frames) are not part of Open-P2P's training data. The new D2E games
  below are split chronologically by session within each game.

| Pretrained stitched policy (150M) | Repeat previous F1 | Greedy F1 | Greedy onset F1 | Sampled onset F1 |
|---|---:|---:|---:|---:|
| P2P validation (seen games) | 0.955 | 0.973 | 0.188 | 0.266 |
| P2P test (3 other Roblox games) | 0.916 | 0.936 | 0.112 | 0.277 |
| P2P fresh test (Doom II, Natural Disaster Survival) | 0.859 | 0.877 | 0.167 | 0.246 |
| **Held-out D2E games** | **0.894** | **0.851** | **0.059** | **0.100** |

All P2P splits come from `elefantai/p2p-full-data`, the dataset family used to pretrain
the policy, so they are not clean holdouts. On data outside that source, the pretrained
policy loses to the persistence baseline. This supersedes the impression that it beats
repetition in general.

## What each idea did

| Idea | Verdict | Key measurement |
|---|---|---|
| Public datasets (D2E new games) + decoder LoRA | **Positive offline** | Held-out D2E test NLL 1.974 → 1.505; greedy F1 0.797 → 0.852 (repeat 0.857) |
| Larger pretrained policy (Open-P2P 300M) | No meaningful gain | Held-out D2E NLL 3.159 → 3.120; greedy F1 0.851 → 0.847 |
| Goal classifier-free guidance | Changes more actions, lowers accuracy | Goal changes 6% → 19% of test actions (scale 4); held-out D2E F1 0.851 → 0.714 at scale 7 |
| Flow-matching action chunks | Better onsets, worse holds | Fresh test onset F1 0.373 vs 0.167 greedy; button F1 0.582 vs 0.877 |
| Deterministic chunk ablation | Collapses toward idle | 0.31 predicted vs 0.58 recorded controls per frame (flow: 0.45) |
| Real-time chunking | Smoother, not more accurate | Button changes at chunk switches −20–35%; button F1 −0.5 to −4 points |
| Slow planner + fast controller | Negative | Test NLL 1.873 → 1.929 (RADIO) / 1.895 (EfficientNet control) |
| Elapsed-time memory | Negative | Post-gap NLL: reset 2.65, keep 2.95, keep + time input 2.79 (fresh test) |
| Foveation (central crop) | Negative | Same gain as a duplicate full-frame control: test NLL 1.459 vs 1.462 (LoRA only 1.504) |
| Latent action pretraining | Negative | Scarce-label fresh-test F1 0.689 → 0.597 with latent pretraining |
| Latent world model | Negative | True vs shuffled actions change prediction error by ≤0.014% |
| Outcome-weighted imitation (Hordes logs) | Negative | Every run kept update 0; damage AUC 0.41–0.50 (chance) |
| Production latency (idle GPU, replay) | Positive | Screenshot-to-dispatch p50 51.7 → 29.9 ms, p95 60.3 → 38.1 ms; 17.8 → 33.0 decisions/s |

Details for latent models, outcome weighting and production are in
[LATENT_MODELS.md](LATENT_MODELS.md), [HORDES_AWR.md](HORDES_AWR.md) and
[PRODUCTION_READINESS.md](PRODUCTION_READINESS.md).

## Public data: D2E games

No public Hordes.io dataset exists. The closest licensed data with synchronized keyboard
and mouse labels is D2E-480p (`open-world-agents/D2E-480p`, CC BY-NC 4.0, revision
`f075f7e25df6f6d385840a836f86bf92dfb877ff`). Four games the project had not used were
downloaded (35.9 GB, about 52 hours): Monster Hunter Wilds (third-person monster combat),
MapleStory Worlds (MMO), Eternal Return (ability hotkeys, click-to-move) and GTA V.
Other candidates, including P2P archives with third-person games and Bleeding Edge, are
listed with licenses in the research notes but were not downloaded.

`laya_vision_stitch/d2e_sequences.py` cuts 32-frame, 20 FPS windows, stores frames after
the released 192×192 preprocessing, and splits sessions chronologically within each game
(latest session → test, second latest → validation). A frame may be at most 35 ms older
than its control step; Eternal Return captures near 52 FPS with jitter, and a 20 ms limit
rejected 76% of its windows. Overall 2,353 of 2,561 sampled windows were accepted.

| Split | Eternal Return | Monster Hunter Wilds | GTA V | MapleStory Worlds | Total |
|---|---:|---:|---:|---:|---:|
| Train | 13,312 | 18,784 | 8,256 | 17,536 | 57,888 |
| Validation | 192 | 8,704 | 896 | 1,504 | 11,296 |
| Test | 416 | 576 | 1,088 | 4,032 | 6,112 |

One pure-black loading frame occurs in several splits; the separation check now exempts
only constant-color images and records their count (1). Keys outside the released
vocabulary (for example R, Ctrl, Alt, Tab) are dropped and those frames flagged.

### Decoder LoRA on the new games

Rank-8 LoRA in the three decoder layers (239,745 trainable parameters) trains on cached
frozen-policy contexts from D2E and P2P training frames, with KL preservation of the
released decoder on P2P replay. Validation NLL selected update 1,250 of 3,000.

| Evaluation set | NLL | Greedy button F1 | Repeat previous | Sampled onset F1 |
|---|---:|---:|---:|---:|
| D2E held-out sessions, validation | 2.880 → 2.626 | 0.809 → 0.869 | 0.862 | 0.119 → 0.202 |
| D2E held-out sessions, test | 1.974 → 1.505 | 0.797 → 0.852 | 0.857 | 0.130 → 0.166 |
| Held-out D2E games (not trained) | 3.159 → 3.007 | 0.852 → 0.898 | 0.894 | 0.123 → 0.088 |
| P2P fresh test | 1.945 → 1.938 | 0.877 → 0.877 | 0.859 | 0.192 → 0.251 |

The adapted decoder closes the gap to the persistence baseline on data outside the
policy's pretraining source. On games it never saw, it becomes more persistence-like:
button F1 rises, onset F1 falls. Idle false positives and mouse error fall slightly on
every set.

### Flow head with more data

| Flow chunk head, button F1 / onset F1 | P2P training only | D2E + P2P training |
|---|---:|---:|
| D2E held-out sessions, test | 0.315 / 0.052 | 0.706 / 0.183 |
| Held-out D2E games | 0.426 / 0.076 | 0.675 / 0.084 |
| P2P fresh test | 0.558 / 0.320 | 0.625 / 0.307 |

More data roughly doubles the from-scratch head's agreement, but it remains below the
pretrained decoder with LoRA. Replacing pretrained action knowledge with a new head is
not justified at this scale; adapting the pretrained decoder is.

### Foveation

A central crop covering half of each dimension of the 448×448 D2E frame is encoded by
the same frozen vision tower and fused into the image token through a zero-initialized
residual, trained through the frozen policy together with decoder LoRA. The capacity
control feeds the full-frame token again.

| D2E held-out sessions | Validation NLL | Test NLL | Test greedy F1 | Test sampled onset F1 |
|---|---:|---:|---:|---:|
| Decoder LoRA only | 2.632 | 1.504 | 0.852 | 0.154 |
| + central crop fusion | 2.548 | 1.459 | 0.851 | 0.176 |
| + duplicate full-frame control | 2.551 | 1.462 | 0.856 | 0.192 |

The improvement comes from a trainable image-token path, not from the higher-resolution
crop. A fixed central fovea adds no measurable information here; a learned glimpse or
Hordes-specific small text (health bars, nameplates) was not tested.

## Larger pretrained policy

The Open-P2P repository publishes 150M, 300M, 600M and 1.2B checkpoints (MIT). The 300M
differs from the 150M only in policy depth (20 vs 10 layers). `OpenP2PPolicy` now infers
depth from the checkpoint, and the Laya goal bridge transfers unchanged because it targets
the shared 768D text space. The MLX conversion matches the upstream PyTorch modules with
maximum context error 1.8e-4 and no argmax disagreements. On every evaluation set the
300M is within noise of the 150M, so the 600M/1.2B downloads were not pursued.

## Goal guidance

Guidance uses the released no-goal embedding as the unconditional branch, so it needs no
retraining: `logits = uncond + s·(cond − uncond)` at every decoder position. The Laya
goal carries almost no information about recorded actions: on 224 instructed validation
frames the recorded-action NLL is 0.166 with the goal and 0.162 without it. Guidance
therefore amplifies noise; it changes more actions and lowers accuracy. Useful guidance
needs goals that actually predict behavior, which these datasets do not provide.

## Action chunks and real-time chunking

A pi0-style flow-matching head predicts eight 50 ms steps (22 binary controls in {−1, +1}
plus symlog mouse motion) from the frozen policy context; a deterministic head with the
same architecture is the ablation (`laya_vision_stitch/flow_action_head.py`). Trained on
the 7,520 public training frames, both overfit within 250–750 updates. The flow head keeps
realistic control rates and predicts new presses far better, but it loses the pretrained
decoder's knowledge of what to keep holding.

Real-time chunking follows Black et al. (2025): the new chunk is inpainted with guidance
toward the actions that execute while it is computed, with the published soft mask and
β = 5. With 1–3 steps of simulated delay it consistently reduces discontinuities at chunk
switches; accuracy falls slightly because the new chunk is anchored to the old one.

| P2P test, delay 1 step | Button F1 | Onset F1 | Button change at switch | Mouse jump at switch |
|---|---:|---:|---:|---:|
| Naive switching | 0.794 | 0.215 | 0.488 | 0.215 |
| Real-time chunking | 0.789 | 0.208 | 0.360 | 0.179 |

## Slow planner, elapsed-time memory

The planner reads frozen RADIO patches (or the fast path's own EfficientNet grid as a
matched control) plus the goal and adds zero-initialized residuals to the fast policy's
goal and image tokens. It is trained with stale plans (period 1–4 frames, latency 0–2) to
match asynchronous execution. Both variants overfit and worsen held-out NLL; shuffling the
planner's input barely matters, so it did not learn useful visual intent from this data.

The elapsed-time adapter adds a learned residual encoding the time since the previous
processed frame. On sequences with simulated 100–350 ms gaps, resetting memory (the
current runtime rule) gives lower NLL after gaps than keeping memory, with or without the
time input. Keeping memory gives slightly higher button F1 on unseen games (0.855 vs 0.839),
but the likelihood evidence does not justify changing the runtime default.

## Latency on an idle GPU

Measured after all training finished, with no other MLX process running
(`scripts/benchmark_live_pipeline.py`, 260 recorded frames from `hordes-p2p-live-003`).
Greedy tokens and logits of every variant match the original path exactly.

| Model latency, full 200-frame memory, p50 | 150M | 300M |
|---|---:|---:|
| Original runtime | 42.7 ms | 55.0 ms |
| Goal vector cached | 32.9 ms | 45.5 ms |
| + `mx.compile` | 29.9 ms | 41.5 ms |
| + memory update before the next screenshot | 23.2 ms | 28.0 ms |

| 150M, replayed at 60 FPS | Screenshot-to-dispatch p50 / p95 | Decisions per second |
|---|---:|---:|
| Serial runner (current) | 51.7 / 60.3 ms | 17.8 |
| Pipelined, asynchronous guards | 36.2 / 43.1 ms | 35.1 |
| Pipelined + precommitted memory | 29.9 / 38.1 ms | 33.0 |

The model benchmark interleaves four variants frame by frame and shows p95 spikes of
170–200 ms in every variant; inside the pipeline run, inference p95 is 30–31 ms, so the
spikes appear to come from the interleaved benchmark rather than the model. This remains
unverified live.

## Limitations

- All learning results are offline imitation with teacher-forced history. None measures
  closed-loop success, and none uses Hordes human demonstrations.
- Several evaluation sets are small (P2P validation 16 windows; Eternal Return validation
  6 windows) and some were reused across experiments. One seed per configuration unless
  stated.
- D2E labels are recorded human controls at 20 FPS with majority-time key holds; D2E
  games use keys outside the released vocabulary, which are dropped and flagged.
- Latency is measured by replaying recorded Hordes frames with emulated guard costs and
  a stub dispatcher. Capture, native guard CPU cost, event delivery and game response are
  not included, and the pipelined runner has never run against the game.

## Reproduce

Model weights, datasets and reports stay under ignored `artifacts/`; compact numbers are
in [improvement-experiments-results.json](improvement-experiments-results.json)
(`scripts/summarize_improvement_experiments.py`). Output directories must be new.

```sh
export PYTHONPATH=.
B=artifacts/laya-p2p-bridge-001/bundle
# Caches and decoding comparisons (greedy, sampling, goal guidance)
.venv/bin/python scripts/build_sequence_cache.py --bundle $B \
  --data artifacts/temporal-expanded-data-001 --output artifacts/seqcache-p2p-new --spatial
.venv/bin/python scripts/evaluate_sequence_decoding.py --bundle $B \
  --cache artifacts/seqcache-p2p-new --output artifacts/seqdecode-new
# D2E games (download the pinned folders first; see above)
.venv/bin/python -m laya_vision_stitch.d2e_sequences --games Monster_Hunter_Wilds \
  --output artifacts/d2e-seq-mhw-new --windows-per-minute 2 --fovea
.venv/bin/python scripts/build_sequence_cache.py --bundle $B \
  --data artifacts/d2e-seq-new --output artifacts/seqcache-d2e-new --splits train validation test
# Decoder LoRA on D2E + P2P with replay KL
.venv/bin/python scripts/train_decoder_lora_sequences.py --bundle $B \
  --train artifacts/seqcache-d2e-new:train artifacts/seqcache-p2p-new:train \
  --replay artifacts/seqcache-p2p-new:train --validation artifacts/seqcache-d2e-new:validation \
  --evaluate artifacts/seqcache-d2e-new:test --output artifacts/decoder-lora-new
# Chunk heads, real-time chunking, planner, elapsed time, foveation
.venv/bin/python scripts/train_chunk_heads.py --cache artifacts/seqcache-p2p-new \
  --output artifacts/chunk-flow-new --kind flow
.venv/bin/python scripts/evaluate_realtime_chunking.py --head artifacts/chunk-flow-new \
  --cache artifacts/seqcache-p2p-new --output artifacts/rtc-new
.venv/bin/python scripts/cache_radio_sequences.py --source artifacts/radio-source \
  --data artifacts/temporal-expanded-data-001 --cache artifacts/seqcache-p2p-new
.venv/bin/python scripts/train_dual_system.py --bundle $B --cache artifacts/seqcache-p2p-new \
  --output artifacts/dual-new --features radio
.venv/bin/python scripts/train_time_gap_adapter.py --bundle $B \
  --cache artifacts/seqcache-p2p-new --output artifacts/time-gap-new
.venv/bin/python scripts/train_fovea_adapter.py --bundle $B --cache artifacts/seqcache-d2e-new \
  --output artifacts/fovea-new --source fovea
# Open-P2P 300M: convert with upstream parity, then reuse the fitted goal bridge
.venv/bin/python scripts/profile_p2p_policy.py \
  --checkpoint "artifacts/open-p2p-pretrained/300M/checkpoint-step=00500000.ckpt" \
  --upstream artifacts/p2p-upstream --frames artifacts/hordes-temporal-live-002/frames \
  --output artifacts/p2p-policy-300m-new
.venv/bin/python scripts/build_laya_p2p_variant.py --policy artifacts/p2p-policy-300m-new \
  --bridge-bundle $B --output artifacts/laya-p2p-300m-new
```

None of these commands sends game input.
