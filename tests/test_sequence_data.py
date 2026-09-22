import json

import pytest
from PIL import Image

from laya_vision_stitch.temporal_training import read_sequences
from laya_vision_stitch.trainable_model import PolicyConfig


@pytest.fixture
def sequence_files(tmp_path):
    action = {"buttons": ["w"], "mouse_delta": [0, 0], "duration_seconds": 0.05, "pointer_xy": None}
    for number, split in enumerate(("train", "validation", "test")):
        for i in range(3):
            Image.new("RGB", (4, 4), (number * 60 + i, 0, 0)).save(tmp_path / f"{split}-{i}.png")
        rows = [
            {
                "id": f"{split}-{i}",
                "game": "held" if split == "test" else "shared",
                "episode": split,
                "goal": "move",
                "sequence": split,
                "sequence_step": i,
                "sequence_length": 2,
                "timestamp_seconds": i * 0.05,
                "frame_index": i,
                "frames": [{"image": f"{split}-{i}.png", "age_seconds": 0}],
                "future_image": f"{split}-{i + 1}.png",
                "action": action,
                "recorded_previous_actions": [action],
                "source": {"action_annotation_index": i + 1},
            }
            for i in range(2)
        ]
        (tmp_path / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


def change(path, edit):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    edit(rows)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_validated_consecutive_windows(sequence_files):
    rows, sequences = read_sequences(sequence_files, PolicyConfig())
    assert len(rows["train"]) == 2 and len(sequences["test"]) == 1


def test_future_only_target_cannot_leak_across_splits(sequence_files):
    change(
        sequence_files / "train.jsonl",
        lambda rows: rows[-1].update(future_image="validation-0.png"),
    )
    with pytest.raises(ValueError, match="Current/future image leakage"):
        read_sequences(sequence_files, PolicyConfig())


def test_misaligned_action_labels_rejected(sequence_files):
    change(
        sequence_files / "train.jsonl",
        lambda rows: rows[0]["source"].update(action_annotation_index=0),
    )
    with pytest.raises(ValueError, match="current/action alignment"):
        read_sequences(sequence_files, PolicyConfig())


def test_history_must_be_preceding_label(sequence_files):
    change(
        sequence_files / "train.jsonl",
        lambda rows: rows[1]["recorded_previous_actions"][0].update(buttons=[]),
    )
    with pytest.raises(ValueError, match="Previous control"):
        read_sequences(sequence_files, PolicyConfig())
