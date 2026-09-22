"""Pinned P2P import: human controls, causal frame alignment, whole-game holdout."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .d2e_data import BUTTONS

REPOSITORY = "elefantai/p2p-toy-examples"
REVISION = "305ebc891474580f2da295b42ed46ddfad053d96"
SCHEMA_REVISION = "a329d98cbe62119679a254d71bea6446773541bc"
ANNOTATOR = "gemini-2.5-flash-thinking-0905"
CONTROLS = "Physical keys and mouse buttons; mouse_delta is raw motion / 512. Act for 50 ms."
ALIASES = {
    "Space": "space",
    "Tab": "tab",
    "Enter": "enter",
    "Escape": "escape",
    "Backspace": "backspace",
    "LeftShift": "shift",
    "RightShift": "shift",
    "LeftControl": "ctrl",
    "RightControl": "ctrl",
    "LeftAlt": "alt",
    "RightAlt": "alt",
    "Left": "left",
    "Right": "right",
    "Up": "up",
    "Down": "down",
}
MOUSE = {"0": "mouse_left", "1": "mouse_right", "2": "mouse_middle"}


def read_annotation(path):
    from .vendor.p2p.video_annotation_pb2 import VideoAnnotation

    return VideoAnnotation.FromString(Path(path).read_bytes())


def human_action(frame):
    """Unknown is not idle. Reject correction conflicts and unsupported modalities."""
    action = frame.user_action
    if not action.is_known or frame.system_action.is_known:
        raise ValueError("unknown_or_system_action")
    if action.HasField("game_pad"):
        raise ValueError("gamepad")
    if not action.mouse.HasField("mouse_delta_px"):
        raise ValueError("missing_raw_mouse")
    if action.mouse.scroll_delta_px.x or action.mouse.scroll_delta_px.y:
        raise ValueError("scroll")
    keys = [ALIASES.get(k, k) for k in action.keyboard.keys]
    try:
        keys += [MOUSE[k] for k in action.mouse.buttons_down]
    except KeyError as exc:
        raise ValueError("unknown_mouse_button") from exc
    if set(keys) - set(BUTTONS):
        raise ValueError("unknown_keyboard_key")
    delta = [action.mouse.mouse_delta_px.x, action.mouse.mouse_delta_px.y]
    if max(map(abs, delta)) > 512:
        raise ValueError("mouse_out_of_range")
    return {
        "buttons": sorted(set(keys)),
        "mouse_delta": [x / 512 for x in delta],
        "pointer_xy": None,
        "duration_seconds": 0.05,
    }


def instructions(annotation):
    """Retrospective VLM labels are weak goals, never human intent ground truth.

    Propagate only forward from the declared start, through the declared duration.
    Do not borrow the next segment's text to label an earlier screenshot.
    """
    goals = [None] * len(annotation.frame_annotations)
    for i, frame in enumerate(annotation.frame_annotations):
        for text in frame.frame_text_annotation:
            if text.frame_text_annotator.version != ANNOTATOR or not text.instruction.strip():
                continue
            if not np.isfinite(text.duration) or text.duration <= 0:
                raise ValueError("Invalid text duration")
            end = min(len(goals), i + int(text.duration * 20))
            for j in range(i, end):
                if goals[j] is not None:
                    # Overlapping retrospective labels are ambiguous. Exclude
                    # these frames instead of choosing an intent after the fact.
                    goals[j] = False
                else:
                    goals[j] = {
                        "text": text.instruction.strip(),
                        "start": i,
                        "end": end,
                        "annotator": ANNOTATOR,
                    }
    return goals


def candidates(annotation, stride=10):
    if annotation.metadata.frames_per_second != 20:
        raise ValueError("This importer requires 20 FPS")
    goals, rows, excluded = instructions(annotation), [], Counter()
    for i in range(4, len(goals) - 1, stride):
        goal = goals[i]
        if not goal or i + 1 >= goal["end"] or goals[i + 1] != goal:
            excluded["no_active_instruction"] += 1
            continue
        try:
            previous = human_action(annotation.frame_annotations[i])
            # Upstream trains frame[t] against annotation[t+1].
            action = human_action(annotation.frame_annotations[i + 1])
        except ValueError as exc:
            excluded[str(exc)] += 1
            continue
        compact_previous = {k: previous[k] for k in ("buttons", "mouse_delta")}
        rows.append(
            {
                "frame_index": i,
                "goal": goal["text"],
                "action": action,
                "previous_actions": [compact_previous],
                "recorded_previous_actions": [previous],
                "instruction_provenance": goal,
            }
        )
    return rows, dict(excluded)


def extract_frames(video, indices, destination, expected_count):
    import av

    destination.mkdir(parents=True, exist_ok=True)
    wanted, count, first_pts = set(indices), 0, None
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        if abs(float(stream.average_rate) - 20) > 0.01:
            raise ValueError("Video is not 20 FPS")
        for i, frame in enumerate(container.decode(stream)):
            count += 1
            if frame.pts is None:
                raise ValueError("Missing video PTS")
            stamp = float(frame.pts * stream.time_base)
            if first_pts is None:
                first_pts = stamp
            # Source MP4s have a 4.5 ms first-frame mux offset, then
            # exact 50 ms spacing. Never tolerate a dropped/duplicated frame.
            if abs((stamp - first_pts) - i / 20) > 0.005:
                raise ValueError("Nonuniform video timing; index alignment unsafe")
            if i in wanted:
                image = frame.to_image().convert("RGB")
                image.thumbnail((320, 320))
                image.save(destination / f"{i:07d}.png")
    if count != expected_count or (wanted and max(wanted) >= count):
        raise ValueError(f"Annotation/video count mismatch: {expected_count}/{count}")
    return count


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def build(source, output, train_per_game=256, validation_count=192, seed=41):
    output.mkdir(parents=True, exist_ok=False)
    splits, audit = {"train": [], "validation": []}, []
    rng = np.random.default_rng(seed)
    for path in sorted(source.glob("dataset/*/annotation.proto")):
        annotation = read_annotation(path)
        episode, game = path.parent.name, annotation.metadata.env.env
        if game not in {"call-of-duty-mobile", "grand-theft-auto-san-andreas", "roblox"}:
            raise ValueError(f"Unexpected sample game: {game}")
        split = "validation" if game == "roblox" else "train"
        rows, excluded = candidates(annotation)
        limit = validation_count if split == "validation" else train_per_game
        selected = sorted(rng.choice(len(rows), min(limit, len(rows)), replace=False))
        rows = [rows[i] for i in selected]
        indices = {i for r in rows for i in (r["frame_index"] - 4, r["frame_index"])}
        video = path.parent / "video.mp4"
        count = extract_frames(
            video, indices, output / "frames" / episode, len(annotation.frame_annotations)
        )
        for row in rows:
            index = row["frame_index"]
            row.update(
                {
                    "id": f"p2p-{episode}-{index}",
                    "game": game,
                    "episode": episode,
                    "controls": CONTROLS,
                    "frames": [
                        {"image": f"frames/{episode}/{i:07d}.png", "age_seconds": (index - i) / 20}
                        for i in (index - 4, index)
                    ],
                    "choices": {"act": "Take the next action.", "wait": "Wait."},
                    "action_supervision": ["buttons", "relative_mouse"],
                    "source": {
                        "repository": REPOSITORY,
                        "revision": REVISION,
                        "action_annotation_index": index + 1,
                    },
                }
            )
            splits[split].append(row)
        audit.append(
            {
                "episode": episode,
                "game": game,
                "split": split,
                "frames": count,
                "duration_seconds": count / 20,
                "selected": len(rows),
                "excluded_candidates": excluded,
                "annotation_sha256": sha256(path),
                "video_sha256": sha256(video),
                "human_frames": sum(f.user_action.is_known for f in annotation.frame_annotations),
                "instruction_segments": sum(
                    len(f.frame_text_annotation) for f in annotation.frame_annotations
                ),
            }
        )
    if len(audit) != 3 or not all(splits.values()):
        raise ValueError("Expected all three sample recordings")
    for split, rows in splits.items():
        (output / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    # Preserve the exact dataset license/card alongside downloaded material.
    (output / "SOURCE_DATASET_CARD.md").write_bytes((source / "README.md").read_bytes())
    report = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "schema_revision": SCHEMA_REVISION,
        "seed": seed,
        "recordings": audit,
        "alignment": "image[t] -> human_action[t+1]; previous_action[t]",
        "split": "Two training games; entire Roblox recording/game held out",
        "limitations": [
            "One session per game; no independent within-game validation",
            "Goals are retrospective Gemini labels, not verified human intent",
            "Only first 50 ms buttons/raw mouse are supervised",
            "No pointer, scroll, future-chunk or reward supervision",
            "Not a Hordes dataset or proof of live gameplay",
        ],
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/p2p-source"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--train-per-game", type=int, default=256)
    parser.add_argument("--validation-count", type=int, default=192)
    args = parser.parse_args()
    if min(args.train_per_game, args.validation_count) < 1:
        parser.error("Sample counts must be positive")
    if args.download:
        from huggingface_hub import snapshot_download

        snapshot_download(
            REPOSITORY,
            repo_type="dataset",
            revision=REVISION,
            allow_patterns=["**/annotation.proto", "**/video.mp4", "README.md"],
            local_dir=args.source,
        )
    build(args.source, args.output, args.train_per_game, args.validation_count)


if __name__ == "__main__":
    main()
