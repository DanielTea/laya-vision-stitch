"""Import synchronized D2E human gameplay; no environment state enters the model."""

import argparse
import bisect
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

VK = {ord(c.upper()): c for c in "abcdefghijklmnopqrstuvwxyz0123456789"}
VK.update(
    {
        32: "space",
        9: "tab",
        13: "enter",
        27: "escape",
        8: "backspace",
        16: "shift",
        160: "shift",
        161: "shift",
        17: "ctrl",
        162: "ctrl",
        163: "ctrl",
        18: "alt",
        164: "alt",
        165: "alt",
        37: "left",
        38: "up",
        39: "right",
        40: "down",
    }
)
BUTTONS = tuple(
    dict.fromkeys(
        ["w", "a", "s", "d", "space", "mouse_left", "mouse_right", *VK.values(), "mouse_middle"]
    )
)
CONTROLS = (
    "Outputs are physical keyboard keys and mouse buttons held over the next 100 ms. "
    "Mouse delta is raw relative movement divided by 512, clipped to [-1, 1]. "
    "Key names identify physical controls, not universal game-specific abilities."
)


def download(config_path, source):
    """Fetch only the pinned manifest's selected recordings, with a size ceiling."""
    from huggingface_hub import hf_hub_download

    config = json.loads(Path(config_path).read_text())
    if sum(r["bytes"] for r in config["recordings"]) > 2_000_000_000:
        raise ValueError("Pilot video manifest exceeds the 2 GB download ceiling")
    for rec in config["recordings"]:
        for name in (rec["video"], str(Path(rec["video"]).with_suffix(".mcap"))):
            hf_hub_download(
                config["repo_id"],
                name,
                repo_type="dataset",
                revision=config["revision"],
                local_dir=source,
            )
    hf_hub_download(
        config["repo_id"],
        "README.md",
        repo_type="dataset",
        revision=config["revision"],
        local_dir=source,
    )


def file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_events(path):
    from mcap.reader import make_reader

    screens, states, mouse = [], [], []
    keys, clicks = set(), set()
    unknown = Counter()
    with Path(path).open("rb") as f:
        reader = make_reader(f)
        for _, channel, message in reader.iter_messages(log_time_order=True):
            t = message.log_time / 1e9
            data = json.loads(message.data)
            topic = channel.topic
            if topic == "screen":
                if Path(data["media_ref"]["uri"]).name != Path(path).with_suffix(".mkv").name:
                    raise ValueError("Screen refers to a different video than its recording")
                screens.append((t, data["media_ref"]["pts_ns"] / 1e9))
            elif topic in ("keyboard/state", "keyboard", "mouse/state", "mouse"):
                if topic == "keyboard/state":
                    keys = set(data["buttons"])
                elif topic == "keyboard":
                    if data["event_type"] == "press":
                        keys.add(data["vk"])
                    else:
                        keys.discard(data["vk"])
                elif topic == "mouse/state":
                    clicks = set(data["buttons"])
                elif data["event_type"] == "click":
                    if data["pressed"]:
                        clicks.add(data["button"])
                    else:
                        clicks.discard(data["button"])
                else:
                    continue
                unknown.update(k for k in keys if k not in VK)
                buttons = {VK[k] for k in keys if k in VK}
                buttons.update("mouse_" + b for b in clicks if b in ("left", "right", "middle"))
                states.append((t, sorted(buttons)))
            elif topic == "mouse/raw":
                mouse.append((t, data["last_x"], data["last_y"]))
    return screens, states, mouse, dict(unknown)


class Timeline:
    def __init__(self, states, mouse):
        self.states, self.mouse = states, mouse
        self.times = [x[0] for x in states]
        self.mouse_times = [x[0] for x in mouse]

    def action(self, start, duration, scale):
        """Majority-time button holds and summed raw delta in [start, end)."""
        end = start + duration
        i = bisect.bisect_right(self.times, start) - 1
        held = self.states[i][1] if i >= 0 else []
        cursor, occupancy = start, Counter()
        for t, next_held in self.states[i + 1 : bisect.bisect_left(self.times, end)]:
            for b in held:
                occupancy[b] += t - cursor
            cursor, held = t, next_held
        for b in held:
            occupancy[b] += end - cursor
        lo, hi = (
            bisect.bisect_left(self.mouse_times, start),
            bisect.bisect_left(self.mouse_times, end),
        )
        delta = np.asarray([sum(x[k] for x in self.mouse[lo:hi]) for k in (1, 2)]) / scale
        return {
            "buttons": sorted(b for b, d in occupancy.items() if d >= duration / 2),
            "mouse_delta": np.clip(delta, -1, 1).tolist(),
            "pointer_xy": None,
            "duration_seconds": duration,
        }, bool(np.any(np.abs(delta) > 1))


def extract_frames(video, requested, directory):
    """Use only frames at/before recorded PTS, at most 20 ms old; skip gaps."""
    import av

    paths, indices, offsets = {}, {}, {}
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for pts in sorted(set(requested)):
            container.seek(max(0, int(pts / float(stream.time_base))), stream=stream)
            nearest, error = None, float("inf")
            for frame in container.decode(stream):
                stamp = float(frame.pts * frame.time_base)
                distance = abs(stamp - pts)
                if stamp <= pts and distance < error:
                    nearest, error = frame, distance
                if stamp >= pts:
                    break
            if nearest is None or error > 0.020:
                continue
            offsets[pts] = error
            if nearest.pts in indices:
                paths[pts] = indices[nearest.pts]
                continue
            image = nearest.to_image()
            image.thumbnail((320, 320))
            name = f"{nearest.pts}.png"
            image.save(directory / name)
            paths[pts] = directory / name
            indices[nearest.pts] = directory / name
    return paths, offsets


def build(config_path, source, output):
    config = json.loads(Path(config_path).read_text())
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(config["seed"])
    splits = {s: [] for s in ("train", "validation", "test")}
    audit = {"config": config, "recordings": [], "buttons": list(BUTTONS)}
    owners = {}
    for rec in config["recordings"]:
        game, split = rec["game"], rec["split"]
        if game in owners and owners[game] != split:
            raise ValueError("An entire game must belong to one split")
        owners[game] = split
        video = Path(source) / rec["video"]
        screens, states, mouse, unknown = load_events(video.with_suffix(".mcap"))
        if not states or not mouse or len(screens) < 3:
            raise ValueError("Recording lacks synchronized state or movement")
        times = [x[0] for x in screens]
        timeline = Timeline(states, mouse)
        history, horizon = config["history_seconds"], config["horizon_seconds"]
        # Stratified timeline sampling independent of recorded action values.
        low = max(times[0] + history, states[0][0] + history, 2.0)
        high = min(times[-1], states[-1][0], mouse[-1][0]) - horizon
        count = config["per_recording"][split]
        edges = np.linspace(low, high, count + 1)
        pending, requested = [], []
        seen_times = set()
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            t = screens[max(0, bisect.bisect_right(times, rng.uniform(a, b)) - 1)][0]
            if t in seen_times:
                continue
            seen_times.add(t)
            current = screens[bisect.bisect_left(times, t)]
            previous = screens[max(0, bisect.bisect_right(times, t - history) - 1)]
            if previous[0] >= t:
                continue
            label, clipped = timeline.action(t, horizon, config["mouse_scale"])
            past, _ = timeline.action(t - horizon, horizon, config["mouse_scale"])
            pending.append((current, previous, label, past, clipped))
            requested.extend([current[1], previous[1]])
        ident = hashlib.sha256(rec["video"].encode()).hexdigest()[:12]
        directory = output / "images" / game / ident
        directory.mkdir(parents=True)
        paths, offsets = extract_frames(video, requested, directory)
        clipped_count = 0
        accepted_count = 0
        for index, (current, previous, label, past, clipped) in enumerate(pending):
            if current[1] not in paths or previous[1] not in paths:
                continue
            accepted_count += 1
            row = {
                "id": f"d2e-{ident}-{index}",
                "game": game,
                "episode": rec["session"],
                "frames": [
                    {
                        "image": str(paths[previous[1]].relative_to(output)),
                        "age_seconds": current[0] - previous[0],
                    },
                    {"image": str(paths[current[1]].relative_to(output)), "age_seconds": 0},
                ],
                "goal": f"Continue playing {game.replace('_', ' ')} and make progress.",
                "controls": CONTROLS,
                "previous_actions": [past],
                "action": label,
                "provenance": {
                    "dataset": config["repo_id"],
                    "revision": config["revision"],
                    "license": config["license"],
                    "recording": rec["video"],
                    "log_seconds": current[0],
                    "video_seconds": current[1],
                    "goal_source": "generic task text, not human intent annotation",
                    "action_source": "recorded human controls, not an expert oracle",
                    "mouse_scale": config["mouse_scale"],
                },
            }
            splits[split].append(row)
            clipped_count += clipped
        audit["recordings"].append(
            {
                **rec,
                "examples": accepted_count,
                "rejected_frame_timing": len(pending) - accepted_count,
                "duration_seconds": times[-1] - times[0],
                "maximum_video_age_ms": max(offsets.values()) * 1000,
                "clipped_mouse_examples": clipped_count,
                "unknown_key_state_counts": unknown,
                "video_sha256": file_digest(video),
                "mcap_sha256": file_digest(video.with_suffix(".mcap")),
            }
        )
        print(json.dumps(audit["recordings"][-1]), flush=True)
    # Exact duplicate screenshots across games/sessions would invalidate the holdout.
    hashes = {}
    for split, rows in splits.items():
        for row in rows:
            for frame in row["frames"]:
                digest = hashlib.sha256((output / frame["image"]).read_bytes()).hexdigest()
                if digest in hashes and hashes[digest] != split:
                    raise ValueError("Cross-split duplicate screenshot; inspect before training")
                hashes[digest] = split
        (output / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def add_replay(output, manifest, count=192):
    from .policy_data import check_separation, read_manifest
    from .trainable_model import PolicyConfig

    config = PolicyConfig(buttons=BUTTONS)
    output = Path(output)
    rows = read_manifest(output / "train.jsonl", config)
    replay = read_manifest(manifest, config)
    if not 0 <= count <= len(replay):
        raise ValueError("Replay count exceeds available examples")
    indices = np.random.default_rng(17).choice(len(replay), count, replace=False)
    rows.extend(replay[i] for i in indices)
    for split in ("validation", "test"):
        check_separation(rows, read_manifest(output / f"{split}.jsonl", config), holdout_games=True)
    with (output / "mixed-train.jsonl").open("x") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/d2e-pilot.json"))
    parser.add_argument("--source", type=Path, default=Path("artifacts/d2e-source"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Fetch pinned recordings first")
    parser.add_argument("--replay-manifest", type=Path)
    parser.add_argument("--replay-count", type=int, default=192)
    args = parser.parse_args()
    if args.download:
        download(args.config, args.source)
    build(args.config, args.source, args.output)
    if args.replay_manifest:
        add_replay(args.output, args.replay_manifest, args.replay_count)


if __name__ == "__main__":
    main()
