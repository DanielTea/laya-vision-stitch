"""Offline latency benchmarks for the live runtime; replays recorded frames, posts no input.

`model`: per-frame stitched-model latency on recorded screenshots for the original path,
goal caching, and `mx.compile`, interleaved frame by frame so GPU contention affects all
variants alike. Every variant receives the same recorded applied actions as feedback, so
greedy tokens and logits are compared frame by frame; a seeded sampling run checks
compiled decoding at temperature 1.

`pipeline`: screenshot-age-to-dispatch-start and throughput for an emulation of the serial
runner versus `live_pipeline.LivePipeline`. Frames are replayed at a fixed rate through a
newest-frame slot, a stub dispatcher only records timestamps, and guards are sleeps with the
median costs measured in live trial hordes-p2p-live-003. The model is the real stream, or a
fixed-latency stub (`--stub-model-ms`) to isolate scheduling from GPU contention.

Neither mode measures capture latency, native guard CPU cost, event delivery or gameplay.
"""

import argparse
import json
import platform
import threading
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from laya_vision_stitch.laya_p2p import LayaP2PRuntime
from laya_vision_stitch.laya_p2p_stream import LayaP2PStream
from laya_vision_stitch.live_pipeline import Frame, LatestFrameSlot, LivePipeline, percentiles
from laya_vision_stitch.temporal_live import bounded_action

GOAL = "Defeat nearby monsters. Avoid attacking players. Retreat when health is low."
IDLE = {"buttons": [], "mouse_delta": [0.0, 0.0]}
MEMORY_FRAMES = 200


def precise_sleep(seconds):
    """Sleep without macOS timer leeway (up to ~50% overshoot, capped near 10 ms, was
    measured for plain `time.sleep` in this process); short sleeps, then a brief spin."""
    end = time.perf_counter() + seconds
    while (remaining := end - time.perf_counter()) > 0:
        time.sleep(min(0.5 * remaining, 0.001) if remaining > 0.0002 else 0)


def load_frames(directory, count):
    paths = sorted(Path(directory).glob("*.jpg"))[:count]
    if len(paths) != count:
        raise ValueError(f"Need {count} frames in {directory}; found {len(paths)}")
    return [Image.open(p).convert("RGB") for p in paths]


def recorded_feedback(events, count):
    """Applied bounded actions from the trial log; idle where none was applied."""
    if events is None or not Path(events).exists():
        return [IDLE] * count
    rows = [json.loads(s) for s in Path(events).read_text().splitlines() if s.strip()]
    out = []
    for row in rows[:count]:
        action = row["bounded_action"]
        out.append(
            {"buttons": action["buttons"], "mouse_delta": action["mouse_delta"]}
            if row["applied"]
            else IDLE
        )
    return out + [IDLE] * (count - len(out))


def environment(label):
    return {
        "label": label,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "mlx": mx.__version__,
        "device": str(mx.default_device()),
    }


def model_benchmark(args):
    total = args.warm_frames + args.timed_frames
    if args.warm_frames < 6 or args.timed_frames < 1 or args.sample_frames > total:
        raise ValueError("Need at least 6 warm frames, 1 timed frame and enough sampled frames")
    images = load_frames(args.frames, total)
    feedback = recorded_feedback(args.events, total)
    runtime = LayaP2PRuntime.load(args.bundle)
    variants = {
        "original": LayaP2PStream(runtime, cache_goal=False, temperature=0.0),
        "goal_cache": LayaP2PStream(runtime, temperature=0.0),
        "goal_cache_compiled": LayaP2PStream(runtime, temperature=0.0, compile=True),
        "compiled_precommit": LayaP2PStream(runtime, temperature=0.0, compile=True),
    }
    names = list(variants)
    times = {n: [] for n in names}
    commit_times, agreement = [], {n: [] for n in names}
    logit_error = {n: [] for n in names}
    for i, image in enumerate(images):
        outputs = {}
        # Rotate order so no variant always runs first after the previous frame.
        for name in names[i % len(names) :] + names[: i % len(names)]:
            stream = variants[name]
            previous = [feedback[i - 1]] if i and name != "compiled_precommit" else []
            row = {"frames": [{"image": image, "age_seconds": 0}], "goal": GOAL}
            row["previous_actions"] = previous
            result = stream.predict(row, session_id="bench", timestamp_seconds=0.05 * i)
            times[name].append(result["image_to_outputs_ms"])
            outputs[name] = (stream.last_tokens, stream.last_logits)
            if name == "compiled_precommit":
                started = time.perf_counter()
                stream.commit(feedback[i])
                commit_times.append(1000 * (time.perf_counter() - started))
        ref_tokens, ref_logits = outputs["original"]
        for name in names:
            tokens, logits = outputs[name]
            agreement[name].append(bool(mx.array_equal(tokens, ref_tokens).item()))
            logit_error[name].append(
                max(
                    mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()
                    for a, b in zip(ref_logits, logits, strict=True)
                )
            )
    # The first frames include lazy allocation; the cache is full from frame 200 on.
    warm, timed = slice(5, min(MEMORY_FRAMES, total)), slice(MEMORY_FRAMES, total)
    report = {
        "environment": environment(args.label),
        "bundle": str(args.bundle),
        "frames": str(args.frames),
        "scope": "Per-frame stitched-model latency (preprocessing, goal, memory commit, vision, "
        "policy, decoding) on recorded screenshots with recorded applied actions as feedback. "
        "Greedy decoding for parity. No capture, guards, dispatch or gameplay.",
        "growing_cache_frames": [5, min(MEMORY_FRAMES, total)],
        "full_cache_frames": [MEMORY_FRAMES, total],
        "variants": {},
    }
    for name in names:
        stream = variants[name]
        report["variants"][name] = {
            "compile_status": stream.compile_status,
            "compile_parity_relative_error": stream.compile_parity,
            "growing_cache_ms": percentiles(times[name][warm]),
            "full_cache_ms": percentiles(times[name][timed]),
            "greedy_token_agreement_with_original": float(np.mean(agreement[name])),
            "max_abs_logit_difference_growing": max(logit_error[name][warm], default=None),
            "max_abs_logit_difference_full": max(logit_error[name][timed], default=None),
        }
    report["variants"]["compiled_precommit"]["commit_off_path_ms"] = {
        "growing_cache": percentiles(commit_times[warm]),
        "full_cache": percentiles(commit_times[timed]),
    }
    report["sampled_parity"] = sampled_parity(runtime, images[: args.sample_frames], feedback)
    return report


def sampled_parity(runtime, images, feedback, seed=20260923):
    """Same seed, temperature 1: compiled decoding must draw the same tokens."""
    streams = {
        "original": LayaP2PStream(runtime, cache_goal=False),
        "goal_cache_compiled": LayaP2PStream(runtime, compile=True),
    }
    tokens = {}
    for name, stream in streams.items():
        mx.random.seed(seed)
        tokens[name] = []
        for i, image in enumerate(images):
            row = {"frames": [{"image": image, "age_seconds": 0}], "goal": GOAL}
            row["previous_actions"] = [feedback[i - 1]] if i else []
            stream.predict(row, session_id="sampled", timestamp_seconds=0.05 * i)
            tokens[name].append(stream.last_tokens.tolist()[0])
    same = [a == b for a, b in zip(tokens["original"], tokens["goal_cache_compiled"], strict=True)]
    return {"frames": len(images), "seed": seed, "token_agreement": float(np.mean(same))}


class FrameReplay(threading.Thread):
    """Publish recorded frames at a fixed rate into a newest-frame slot."""

    def __init__(self, slot, images, fps):
        super().__init__(daemon=True)
        self.slot, self.images, self.period = slot, images, 1 / fps
        self.stopped = threading.Event()

    def run(self):
        start, i = time.perf_counter(), 0
        while not self.stopped.is_set():
            wait = start + i * self.period - time.perf_counter()
            if wait > 0:
                precise_sleep(wait)
            self.slot.put(Frame(self.images[i % len(self.images)], time.perf_counter(), i))
            i += 1


class RecordingDispatcher:
    """Stub transport: records the dispatch-start timestamp, posts nothing, never waits."""

    def __init__(self):
        self.applied, self.releases = [], 0

    def apply(self, action, deadline):
        start = time.perf_counter()
        self.applied.append((start, action))
        return {
            "dispatch_start": start,
            "first_event_posted": None,
            "pulse_wait_ms": 0.0,
            "dispatch_guard_ms": 0.0,
        }

    def release(self):
        self.releases += 1

    def finish(self):
        self.release()


class StubModel:
    """Fixed-latency stand-in for scheduling comparisons (sleeps; no GPU work)."""

    def __init__(self, ms, commit_ms=0.0):
        if not 0 <= commit_ms <= ms:
            raise ValueError("Stub commit time must be within the stub inference time")
        self.ms, self.commit_ms, self.committed = ms, commit_ms, False

    def predict(self, image, frame, previous):
        started = time.perf_counter()
        precise_sleep((self.ms - (self.commit_ms if self.committed else 0)) / 1000)
        reset = not previous and not self.committed
        self.committed = False
        return {
            "buttons": ["w"],
            "mouse_delta": [0.0, 0.0],
            "image_to_outputs_ms": 1000 * (time.perf_counter() - started),
            "state_reset": reset,
        }

    def commit(self, applied):
        precise_sleep(self.commit_ms / 1000)
        self.committed = True


def stream_model(stream, session):
    def predict(image, frame, previous):
        row = {"frames": [{"image": image, "age_seconds": 0}], "goal": GOAL}
        row["previous_actions"] = previous
        return stream.predict(row, session_id=session, timestamp_seconds=frame.captured_at)

    return predict


def run_serial(slot, predict, args, deadline, start):
    """Emulation of the serial `temporal_live` loop order with sleep-based guard costs."""
    records, previous, last = [], [], -1
    stop_reason = "duration_complete"
    while time.perf_counter() < deadline:
        precise_sleep(args.loop_guard_ms / 1000)  # run loop, STOP, layout and focus at loop top
        frame = slot.latest()
        if frame is None or frame.sequence == last:
            time.sleep(0.002)
            continue
        last = frame.sequence
        if time.perf_counter() - frame.captured_at > args.max_frame_age:
            stop_reason = "Stale capture"
            break
        precise_sleep(args.hud_ms / 1000)  # HUD check on the model input
        started = time.perf_counter()
        proposal = predict(frame.image, frame, previous)
        action = bounded_action(proposal)
        finished = time.perf_counter()
        if finished >= deadline:
            break
        applied = finished - frame.captured_at <= args.max_apply_age
        precise_sleep(args.hud_ms / 1000)  # HUD check on a fresh frame after inference
        dispatch = None
        if applied:
            precise_sleep(args.pulse_wait_ms / 1000)  # previous pulse must finish
            precise_sleep(args.guard_ms / 1000)  # focus, layout and text-input guards
            dispatch = time.perf_counter()
        previous = (
            [{"buttons": action["buttons"], "mouse_delta": action["mouse_delta"]}]
            if applied
            else []
        )
        records.append(
            {
                "elapsed_s": frame.captured_at - start,
                "applied": applied,
                "proposal": proposal,
                "frame_age_before_inference_ms": 1000 * (started - frame.captured_at),
                "screenshot_to_dispatch_start_ms": None
                if dispatch is None
                else 1000 * (dispatch - frame.captured_at),
            }
        )
    return records, stop_reason


def sleeper(ms):
    def check():
        precise_sleep(ms / 1000)
        return None

    return check


def pipeline_benchmark(args):
    images = load_frames(args.frames, args.replay_frames)
    runtime = None if args.stub_model_ms else LayaP2PRuntime.load(args.bundle)
    options = {"compile": args.compile, "max_gap_seconds": args.max_memory_gap}
    report = {
        "environment": environment(args.label),
        "bundle": None if args.stub_model_ms else str(args.bundle),
        "model": f"stub {args.stub_model_ms} ms" if args.stub_model_ms else "real stream",
        "stream_options": None if args.stub_model_ms else options,
        "settings": {
            k: v
            for k, v in vars(args).items()
            if k not in ("func", "output", "frames", "bundle", "label")
        },
        "scope": "Scheduling comparison with replayed frames, sleep-based guard costs and a "
        "recording dispatcher. Capture latency, native guard CPU/GIL cost, event delivery "
        "and game response are not measured.",
        "modes": {},
    }
    for mode in args.modes:
        if runtime is None:
            model = StubModel(args.stub_model_ms, args.stub_commit_ms)
            predict, commit = model.predict, model.commit
        else:
            stream = LayaP2PStream(runtime, **options)
            predict, commit = stream_model(stream, "bench-" + mode), stream.commit
            warm = {"frames": [{"image": images[0], "age_seconds": 0}], "goal": GOAL}
            for i in range(3):
                stream.predict(warm, session_id="warmup", timestamp_seconds=0.05 * i)
            stream.reset()
        slot = LatestFrameSlot()
        replay = FrameReplay(slot, images, args.fps)
        replay.start()
        time.sleep(0.05)
        start = time.perf_counter()
        deadline = start + args.seconds
        if mode == "serial":
            records, stop_reason = run_serial(slot, predict, args, deadline, start)
            extra = {}
        else:
            pipeline = LivePipeline(
                capture=slot,
                prepare=lambda frame: frame.image,
                predict=predict,
                dispatcher=RecordingDispatcher(),
                guards=[("focus_layout", sleeper(args.guard_ms)), ("hud", sleeper(args.hud_ms))],
                deadline=deadline,
                execute=True,
                commit=commit if mode == "pipelined_precommit" else None,
                guard_interval=args.guard_interval,
                max_frame_age=args.max_frame_age,
                max_apply_age=args.max_apply_age,
                start=start,
            )
            summary = pipeline.run()
            records, stop_reason = pipeline.records, summary["stop_reason"]
            extra = {"guard_pass_ms": summary["guard_pass_ms"], "skipped": summary["skipped"]}
        replay.stopped.set()
        replay.join()
        slot.close()
        elapsed = [r["elapsed_s"] for r in records]
        span = elapsed[-1] - elapsed[0] if len(elapsed) > 1 else None
        report["modes"][mode] = {
            "stop_reason": stop_reason,
            "decisions": len(records),
            "applied": sum(r["applied"] for r in records),
            "decisions_per_second": (len(records) - 1) / span if span else None,
            "screenshot_to_dispatch_start_ms": percentiles(
                [r["screenshot_to_dispatch_start_ms"] for r in records if r["applied"]]
            ),
            "frame_age_before_inference_ms": percentiles(
                [r["frame_age_before_inference_ms"] for r in records]
            ),
            "inference_ms": percentiles([r["proposal"]["image_to_outputs_ms"] for r in records]),
            "state_resets": sum(bool(r["proposal"].get("state_reset")) for r in records),
            "frames_published": slot.published,
            **extra,
        }
        print(json.dumps({mode: report["modes"][mode]}), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(required=True)
    m = sub.add_parser("model", help="model latency and output parity")
    m.add_argument("--bundle", type=Path, required=True)
    m.add_argument("--frames", type=Path, required=True)
    m.add_argument("--events", type=Path, help="trial events.jsonl for recorded feedback")
    m.add_argument("--warm-frames", type=int, default=200)
    m.add_argument("--timed-frames", type=int, default=60)
    m.add_argument("--sample-frames", type=int, default=30)
    m.set_defaults(func=model_benchmark)
    q = sub.add_parser("pipeline", help="serial versus pipelined scheduling")
    q.add_argument("--bundle", type=Path, default=Path("artifacts/laya-p2p-bridge-001/bundle"))
    q.add_argument("--frames", type=Path, required=True)
    q.add_argument("--replay-frames", type=int, default=120)
    q.add_argument("--fps", type=float, default=60)
    q.add_argument("--seconds", type=float, default=20)
    q.add_argument("--modes", nargs="+", default=["serial", "pipelined", "pipelined_precommit"])
    q.add_argument("--guard-ms", type=float, default=11.6, help="focus/layout/text guard")
    q.add_argument("--loop-guard-ms", type=float, default=11.6, help="serial loop-top guard")
    q.add_argument("--hud-ms", type=float, default=0.4)
    q.add_argument("--pulse-wait-ms", type=float, default=3.8)
    q.add_argument("--guard-interval", type=float, default=0.05)
    q.add_argument("--max-frame-age", type=float, default=0.18)
    q.add_argument("--max-apply-age", type=float, default=0.18)
    q.add_argument("--stub-model-ms", type=float)
    q.add_argument("--stub-commit-ms", type=float, default=0.0)
    q.add_argument("--compile", action="store_true")
    q.add_argument("--max-memory-gap", type=float, default=0.1)
    q.set_defaults(func=pipeline_benchmark)
    for parser in (m, q):
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--label", default="contended smoke test")
    args = p.parse_args()
    if getattr(args, "modes", None) and not set(args.modes) <= {
        "serial",
        "pipelined",
        "pipelined_precommit",
    }:
        raise ValueError("Modes are serial, pipelined and pipelined_precommit")
    args.output.mkdir(parents=True, exist_ok=False)
    report = args.func(args)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
